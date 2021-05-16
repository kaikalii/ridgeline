#![warn(missing_docs)]

/*!
`ridgeline` is a crate for simplifying frequency spectrum analysis of audio streamed from input devices.
*/

use std::{collections::VecDeque, ops::RangeBounds, sync::mpsc, usize};

use crate::cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, InputCallbackInfo, Sample, SampleFormat, Stream, StreamConfig, SupportedStreamConfig,
};
use cpal::{BuildStreamError, PlayStreamError, SupportedStreamConfigsError};
use rustfft::{num_complex::Complex, FftPlanner};

#[doc(inline)]
/// Alias for cpal
pub use cpal;

/// A error encountered when trying to build a [`Spectrometer`]
#[derive(Debug, thiserror::Error)]
pub enum BuildSpectrometerError {
    /// An error building the audio stream
    #[error("{0}")]
    Stream(#[from] BuildStreamError),
    /// An error starting the audio stream
    #[error("{0}")]
    Play(#[from] PlayStreamError),
    /// An error querying stream configurations
    #[error("{0}")]
    Configs(#[from] SupportedStreamConfigsError),
    /// A device has no available stream configurations
    #[error("No config available for device")]
    NoConfig,
    /// No default input device is available
    #[error("No device available")]
    NoDevice,
}

/// A result type for trying to build a [`Spectrometer`]
pub type BuildSpectrometerResult = Result<Spectrometer, BuildSpectrometerError>;

/// The freqency spectrum of the input at some time
pub struct Spectrum {
    amps: Vec<f32>,
    sample_rate: u32,
}

impl Spectrum {
    /// Create a new silent spectrum
    pub fn silent(sample_rate: u32, max_freq: f32) -> Self {
        Spectrum {
            amps: vec![0.0; (max_freq * 2.0).round() as usize],
            sample_rate,
        }
    }
    fn bucket_width(&self) -> f32 {
        self.sample_rate as f32 / self.amps.len() as f32
    }
    fn frequency_at(&self, i: usize) -> f32 {
        (i + 1) as f32 * self.bucket_width()
    }
    /// Get the amplitude at some frequency
    ///
    /// Frequencies between FFT bucket bounds are interpolated
    pub fn amplitude(&self, freq: f32) -> f32 {
        let delta = self.bucket_width();
        let ratio = freq / delta;
        if ratio as usize >= self.amps.len() {
            return 0.0;
        }
        let floor = ratio.floor();
        let l = floor as usize;
        let r = ratio.ceil() as usize as usize;
        let left = self.amps[l];
        let right = self.amps[r];
        let param = ratio - floor;
        (1.0 - param) * left + param * right
    }
    /// Get the frequency with the maximum amplitude within some range of frequencies
    pub fn dominant_in_range(&self, range: impl RangeBounds<f32>) -> f32 {
        let bucket = self
            .amps
            .iter()
            .take(self.amps.len() / 2)
            .enumerate()
            .filter(|&(i, _)| range.contains(&self.frequency_at(i)))
            .map(|(i, a)| (i, a))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        bucket as f32 * self.bucket_width()
    }
    /// Get the frequency with the maximum amplitude
    pub fn dominant(&self) -> f32 {
        self.dominant_in_range(..)
    }
}

/// An iterator that produces frequency spectra from an audio input device
///
/// At any moment, [`Spectrometer::read`] can be called to get the [`Spectrum`] of the current input.
pub struct Spectrometer {
    _stream: Stream,
    recv: mpsc::Receiver<f32>,
    buffer: VecDeque<Complex<f32>>,
    sample_rate: u32,
    channels: u16,
    max_freq: f32,
    calibration: Option<Spectrum>,
}

/**
A builder for a [`Spectrometer`]

Created with [`Spectrometer::builder`]

At any moment, [`Spectrometer::read`] may be called to get the [`Spectrum`] of
the current audio input.
*/
pub struct SpectrometerBuilder<'a> {
    /// The maximum frequency can be detected
    pub max_freq: f32,
    /// The input device to use. If not set, the default device will be used.
    pub device: Option<&'a Device>,
    /// The stream configuration to be used. If not set, the default will be used.
    pub config: Option<SupportedStreamConfig>,
}

impl<'a> SpectrometerBuilder<'a> {
    /// Set the maximum frequency that can be detected
    pub fn max_freq(self, max_freq: f32) -> Self {
        SpectrometerBuilder { max_freq, ..self }
    }
    /// Set the input device
    pub fn device(self, device: &'a Device) -> Self {
        SpectrometerBuilder {
            device: Some(device),
            ..self
        }
    }
    /// Set the stream configuration
    pub fn config(self, config: SupportedStreamConfig) -> Self {
        SpectrometerBuilder {
            config: Some(config),
            ..self
        }
    }
    /// Build the [`Spectrometer`]
    pub fn build(self) -> BuildSpectrometerResult {
        let default_device;
        let device = if let Some(device) = self.device {
            device
        } else {
            let host = cpal::default_host();
            default_device = host
                .default_input_device()
                .ok_or(BuildSpectrometerError::NoDevice)?;
            &default_device
        };
        let config = if let Some(config) = self.config {
            config
        } else {
            let mut supported_configs_range = device.supported_input_configs()?;
            supported_configs_range
                .next()
                .ok_or(BuildSpectrometerError::NoConfig)?
                .with_max_sample_rate()
        };
        let err_fn = |err| eprintln!("An error occurred on the input audio stream: {}", err);
        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();
        let (send, recv) = mpsc::channel();
        macro_rules! input_stream {
            ($sample:ty) => {
                device.build_input_stream(
                    &config,
                    move |data: &[$sample], _: &InputCallbackInfo| {
                        for &s in data {
                            let _ = send.send(s.to_f32());
                        }
                    },
                    err_fn,
                )
            };
        }
        let stream = match sample_format {
            SampleFormat::F32 => input_stream!(f32),
            SampleFormat::I16 => input_stream!(i16),
            SampleFormat::U16 => input_stream!(u16),
        }?;

        stream.play()?;

        let mut input = Spectrometer {
            _stream: stream,
            recv,
            buffer: VecDeque::new(),
            channels: config.channels,
            sample_rate: config.sample_rate.0,
            max_freq: self.max_freq,
            calibration: None,
        };
        input
            .buffer
            .resize(input.buffer_size(), Complex::new(0.0, 0.0));
        Ok(input)
    }
}

impl Spectrometer {
    /// Create a new [`SpectrometerBuilder`]
    pub fn builder<'a>() -> SpectrometerBuilder<'a> {
        SpectrometerBuilder {
            max_freq: 4000.0,
            device: None,
            config: None,
        }
    }
    /// Create a spectrometer using the default input device
    pub fn from_default_device(max_freq: f32) -> BuildSpectrometerResult {
        Self::builder().max_freq(max_freq).build()
    }
    /// Get the number of samples used for frequency analysis
    pub fn sample_size(&self) -> usize {
        (self.max_freq * 2.0).round() as usize
    }
    /// The amount of time in seconds used to perform frequency analysis
    ///
    /// Note that [`Spectrometer::read`] can be called much more often that this
    pub fn sample_period(&self) -> f32 {
        self.sample_size() as f32 / self.sample_rate as f32
    }
    fn buffer_size(&self) -> usize {
        self.sample_size() * 2
    }
    /// Use the next few spectra to calibrate a definition of "silence"
    ///
    /// All spectra created after calling this function will have the "silence" spectrum subtracted.
    pub fn calibrate(&mut self) {
        let mut new_calibration = self.raw_next();
        for _ in 0..self.sample_size() / 10 {
            let frame = self.raw_next();
            for (c, a) in new_calibration.amps.iter_mut().zip(&frame.amps) {
                *c = c.max(*a);
            }
        }
        self.calibration = Some(new_calibration)
    }
    /// Clear the calibration set by [`Spectrometer::calibrate`]
    pub fn uncalibrate(&mut self) {
        self.calibration = None;
    }
    fn raw_next(&mut self) -> Spectrum {
        for (i, s) in self.recv.try_iter().enumerate() {
            if i % self.channels as usize == 0 {
                self.buffer.push_back(Complex::new(s, 0.0));
            }
        }
        let buffer_size = self.buffer_size();
        let sample_size = self.sample_size();
        while self.buffer.len() > buffer_size {
            self.buffer.pop_front();
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(sample_size);
        let buffer = self.buffer.make_contiguous();
        let input_start = buffer.len() - sample_size;
        let mut amps = buffer[input_start..].to_vec();
        fft.process(&mut amps);
        Spectrum {
            amps: amps
                .into_iter()
                .map(Complex::norm)
                .map(|s| if s.is_nan() { 0.0 } else { s })
                .collect(),
            sample_rate: self.sample_rate,
        }
    }
    /// Get the [`Spectrum`] at the moment this function called
    pub fn read(&mut self) -> Spectrum {
        let mut spectrum = self.raw_next();
        if let Some(min) = &self.calibration {
            for (s, min) in spectrum.amps.iter_mut().zip(&min.amps) {
                *s = (*s - min).max(0.0);
            }
        }
        spectrum
    }
}

impl Default for Spectrometer {
    /// Uses the default device and a maximum frequency of 4000 hz
    fn default() -> Self {
        Self::from_default_device(4000.0).unwrap()
    }
}

impl Iterator for Spectrometer {
    type Item = Spectrum;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.read())
    }
}
