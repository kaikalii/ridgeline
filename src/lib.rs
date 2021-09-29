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
pub type BuildSpectrometerResult<const SIZE: usize> =
    Result<Spectrometer<SIZE>, BuildSpectrometerError>;

/// The freqency spectrum of the input at some time
pub struct Spectrum<const SIZE: usize> {
    amps: [f32; SIZE],
    sample_rate: u32,
}

impl<const SIZE: usize> Spectrum<SIZE> {
    /// Create a new silent spectrum
    pub fn silent(sample_rate: u32) -> Self {
        Spectrum {
            amps: [0.0; SIZE],
            sample_rate,
        }
    }
    /// Get the sample rate of the time-domain data being sampled
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Get the raw array of FFT buckets
    pub fn buckets(&self) -> &[f32; SIZE] {
        &self.amps
    }
    /// Get the width in Hz of an FFT bucket in the spectrum
    pub fn bucket_width(&self) -> f32 {
        self.sample_rate as f32 / self.amps.len() as f32
    }
    /// Get the frequency that corresponds to a certain FFT bucket index
    pub fn frequency_at(&self, index: usize) -> f32 {
        (index + 1) as f32 * self.bucket_width()
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
pub struct Spectrometer<const SIZE: usize> {
    _stream: Stream,
    recv: mpsc::Receiver<f32>,
    buffer: VecDeque<Complex<f32>>,
    sample_rate: u32,
    channels: u16,
    calibration: Option<Spectrum<SIZE>>,
}

/**
A builder for a [`Spectrometer`]

Created with [`Spectrometer::builder`]

At any moment, [`Spectrometer::read`] may be called to get the [`Spectrum`] of
the current audio input.
*/
pub struct SpectrometerBuilder<'a, const SIZE: usize> {
    /// The input device to use. If not set, the default device will be used.
    pub device: Option<&'a Device>,
    /// The stream configuration to be used. If not set, the default will be used.
    pub config: Option<SupportedStreamConfig>,
}

impl<'a, const SIZE: usize> SpectrometerBuilder<'a, SIZE> {
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
    pub fn build(self) -> BuildSpectrometerResult<SIZE> {
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
            calibration: None,
        };
        input
            .buffer
            .resize(input.buffer_size(), Complex::new(0.0, 0.0));
        Ok(input)
    }
}

impl<const SIZE: usize> Spectrometer<SIZE> {
    /// Create a new [`SpectrometerBuilder`]
    pub fn builder<'a>() -> SpectrometerBuilder<'a, SIZE> {
        SpectrometerBuilder {
            device: None,
            config: None,
        }
    }
    /// Create a spectrometer using the default input device
    pub fn from_default_device() -> BuildSpectrometerResult<SIZE> {
        Self::builder().build()
    }
    /// The amount of time in seconds used to perform frequency analysis
    ///
    /// Note that [`Spectrometer::read`] can be called much more often that this
    pub fn sample_period(&self) -> f32 {
        SIZE as f32 / self.sample_rate as f32
    }
    fn buffer_size(&self) -> usize {
        SIZE * 2
    }
    /// Use the next few spectra to calibrate a definition of "silence"
    ///
    /// All spectra created after calling this function will have the "silence" spectrum subtracted.
    pub fn calibrate(&mut self) {
        let mut new_calibration = self.raw_next();
        for _ in 0..SIZE / 10 {
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
    fn raw_next(&mut self) -> Spectrum<SIZE> {
        for (i, s) in self.recv.try_iter().enumerate() {
            if i % self.channels as usize == 0 {
                self.buffer.push_back(Complex::new(s, 0.0));
            }
        }
        let buffer_size = self.buffer_size();
        while self.buffer.len() > buffer_size {
            self.buffer.pop_front();
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(SIZE);
        let buffer = self.buffer.make_contiguous();
        let input_start = buffer.len() - SIZE;
        let mut complex_amps = [Complex::new(0f32, 0.0); SIZE];
        for (i, amp) in buffer[input_start..].iter().enumerate() {
            complex_amps[i] = *amp;
        }
        fft.process(&mut complex_amps);
        let mut amps = [0.0; SIZE];
        for (i, amp) in complex_amps.iter().enumerate() {
            let mut amp = amp.norm();
            if amp.is_nan() {
                amp = 0.0;
            }
            amps[i] = amp;
        }
        Spectrum {
            amps,
            sample_rate: self.sample_rate,
        }
    }
    /// Get the [`Spectrum`] at the moment this function called
    pub fn read(&mut self) -> Spectrum<SIZE> {
        let mut spectrum = self.raw_next();
        if let Some(min) = &self.calibration {
            for (s, min) in spectrum.amps.iter_mut().zip(&min.amps) {
                *s = (*s - min).max(0.0);
            }
        }
        spectrum
    }
}

impl<const SIZE: usize> Iterator for Spectrometer<SIZE> {
    type Item = Spectrum<SIZE>;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.read())
    }
}
