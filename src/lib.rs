#![warn(missing_docs)]
#![doc = include_str!("../readme.md")]

#[cfg(feature = "input")]
use std::sync::mpsc::{self, TryRecvError};
use std::{
    collections::VecDeque,
    ops::{Bound, RangeBounds},
    usize,
};

#[cfg(feature = "input")]
use crate::cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, InputCallbackInfo, PlayStreamError, Sample, SampleFormat, Stream,
    StreamConfig, SupportedStreamConfig, SupportedStreamConfigsError,
};
use rustfft::{num_complex::Complex, FftPlanner};

#[doc(inline)]
#[cfg(feature = "input")]
#[cfg_attr(docsrs, doc(cfg(feature = "input")))]
/// Alias for cpal
pub use cpal;

/// A error encountered when trying to build a [`SystemAudio`]
#[cfg(feature = "input")]
#[cfg_attr(docsrs, doc(cfg(feature = "input")))]
#[derive(Debug, thiserror::Error)]
pub enum BuildSystemAudioError {
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

/// A result type for trying to build a [`SystemAudio`]
#[cfg(feature = "input")]
#[cfg_attr(docsrs, doc(cfg(feature = "input")))]
pub type BuildSystemAudioResult = Result<SystemAudio, BuildSystemAudioError>;

/// The freqency spectrum of the input at some time
#[derive(Clone)]
pub struct Spectrum {
    amps: Vec<f32>,
    sample_rate: u32,
}

impl Spectrum {
    /// Create a new silent spectrum
    pub fn silent(size: usize, sample_rate: u32) -> Self {
        Spectrum {
            amps: vec![0.0; size],
            sample_rate,
        }
    }
    /// Get the sample rate of the time-domain data being sampled
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Get the raw array of FFT buckets
    pub fn buckets(&self) -> &[f32] {
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
    fn bucket_at(&self, freq: f32) -> usize {
        ((freq / self.bucket_width()) as usize).max(1) - 1
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
    /// Get the spectrum point with the maximum amplitude within some range of frequencies
    pub fn dominant_in_range(&self, range: impl RangeBounds<f32>) -> SpectrumPoint {
        self.points_in_range(range)
            .max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap())
            .unwrap_or_default()
    }
    /// Get the spectrum point with the maximum amplitude
    pub fn dominant(&self) -> SpectrumPoint {
        self.dominant_in_range(..)
    }
    /// Get all spectrum points
    pub fn points(&self) -> impl DoubleEndedIterator<Item = SpectrumPoint> + '_ {
        self.points_in_range(..)
    }
    /// Get all spectrum points within a frequency range
    pub fn points_in_range(
        &self,
        range: impl RangeBounds<f32>,
    ) -> impl DoubleEndedIterator<Item = SpectrumPoint> + '_ {
        let skip = match range.start_bound() {
            Bound::Included(b) | Bound::Excluded(b) => self.bucket_at(*b),
            Bound::Unbounded => 0,
        };
        let take = match range.end_bound() {
            Bound::Included(b) => self.bucket_at(*b) - skip + 1,
            Bound::Excluded(b) => self.bucket_at(*b) - skip,
            Bound::Unbounded => usize::MAX,
        };
        self.amps
            .iter()
            .enumerate()
            .skip(skip)
            .take(take)
            .map(move |(i, &amplitude)| SpectrumPoint {
                frequency: (i + 1) as f32 * self.bucket_width(),
                amplitude,
            })
    }
    /// Get all peaks in the spectrum that have greater amplitude that the given minimum
    pub fn peaks(&self, min_amp: f32) -> Vec<SpectrumPoint> {
        self.peaks_in_range(min_amp, ..)
    }
    /// Get all peaks in the spectrum within a frequency range that
    /// have greater amplitude that the given minimum
    pub fn peaks_in_range(&self, min_amp: f32, range: impl RangeBounds<f32>) -> Vec<SpectrumPoint> {
        let mut peaks = Vec::new();
        let mut points = self.points_in_range(range);
        let (mut last, mut last_last) = if let Some(point) = points.next() {
            (point, point)
        } else {
            return peaks;
        };
        for point in points {
            if point.amplitude < last.amplitude && last.amplitude >= min_amp {
                let amp_sum = last_last.amplitude + last.amplitude + point.amplitude;
                let peak = SpectrumPoint {
                    frequency: (last_last.frequency * last_last.amplitude
                        + last.frequency * last.amplitude
                        + point.frequency * point.amplitude)
                        / amp_sum,
                    amplitude: amp_sum / 3.0,
                };
                peaks.push(peak);
            }
            last_last = last;
            last = point;
        }
        peaks
    }
}

/// A point on a [`Spectrum`]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct SpectrumPoint {
    /// The frequency
    pub frequency: f32,
    /// The amplitude
    pub amplitude: f32,
}

/// The result of querying a [`SignalSource`] for a sample
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalResult {
    /// A sample is available
    Sample(f32),
    /// A signal will be available
    Wait,
    /// The signal has finished
    End,
}

/// A single-channel signal source for providing input to a [`Spectrometer`]
pub trait SignalSource {
    /// The sample rate of the signal
    ///
    /// This is used to derive [`Spectrum`] frequencies
    fn sample_rate(&self) -> u32;
    /// Get the next sample
    fn next(&mut self) -> SignalResult;
    /// Create a [`Spectrometer`] from this source
    fn analyze(self, size: usize) -> Spectrometer<Self>
    where
        Self: Sized,
    {
        Spectrometer::new(size, self)
    }
}

/// A [`SignalSource`] that receives audio samples from the system audio input
#[cfg(feature = "input")]
#[cfg_attr(docsrs, doc(cfg(feature = "input")))]
pub struct SystemAudio {
    _stream: Stream,
    recv: mpsc::Receiver<f32>,
    sample_rate: u32,
    channels: u16,
    curr_channel: u16,
}

#[cfg(feature = "input")]
impl SignalSource for SystemAudio {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn next(&mut self) -> SignalResult {
        loop {
            match self.recv.try_recv() {
                Ok(s) => {
                    let should_break = self.curr_channel == 0;
                    self.curr_channel = (self.curr_channel + 1) % self.channels;
                    if should_break {
                        break SignalResult::Sample(s);
                    }
                }
                Err(TryRecvError::Empty) => break SignalResult::Wait,
                Err(TryRecvError::Disconnected) => break SignalResult::End,
            }
        }
    }
}

#[cfg(feature = "input")]
impl SystemAudio {
    /// Create a new [`SystemAudioBuilder`]
    pub fn builder<'a>() -> SystemAudioBuilder<'a> {
        SystemAudioBuilder {
            device: None,
            config: None,
        }
    }
    /// Create a spectrometer using the default input device
    pub fn from_default_device() -> BuildSystemAudioResult {
        Self::builder().build()
    }
}

/**
A builder for a [`SystemAudio`]

Created with [`SystemAudio::builder`]
*/
#[cfg(feature = "input")]
#[cfg_attr(docsrs, doc(cfg(feature = "input")))]
#[derive(Default)]
pub struct SystemAudioBuilder<'a> {
    /// The input device to use. If not set, the default device will be used.
    pub device: Option<&'a Device>,
    /// The stream configuration to be used. If not set, the default will be used.
    pub config: Option<SupportedStreamConfig>,
}

#[cfg(feature = "input")]
impl<'a> SystemAudioBuilder<'a> {
    /// Set the input device
    pub fn device(self, device: &'a Device) -> Self {
        SystemAudioBuilder {
            device: Some(device),
            ..self
        }
    }
    /// Set the stream configuration
    pub fn config(self, config: SupportedStreamConfig) -> Self {
        SystemAudioBuilder {
            config: Some(config),
            ..self
        }
    }
    /// Build the [`SystemAudio`]
    pub fn build(self) -> BuildSystemAudioResult {
        let default_device;
        let device = if let Some(device) = self.device {
            device
        } else {
            let host = cpal::default_host();
            default_device = host
                .default_input_device()
                .ok_or(BuildSystemAudioError::NoDevice)?;
            &default_device
        };
        let config = if let Some(config) = self.config {
            config
        } else {
            let mut supported_configs_range = device.supported_input_configs()?;
            supported_configs_range
                .next()
                .ok_or(BuildSystemAudioError::NoConfig)?
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

        Ok(SystemAudio {
            _stream: stream,
            recv,
            channels: config.channels,
            sample_rate: config.sample_rate.0,
            curr_channel: 0,
        })
    }
}

/// An iterator that produces frequency spectra from an audio input device
///
/// At any moment, [`Spectrometer::next`] can be called to get the [`Spectrum`] of the current input.
#[derive(Clone, Default)]
pub struct Spectrometer<S> {
    source: S,
    buffer: VecDeque<Complex<f32>>,
    calibration: Option<Spectrum>,
    size: usize,
}

impl<S> Spectrometer<S>
where
    S: SignalSource,
{
    /// Create a new `Spectrometer` from a signal source
    pub fn new(size: usize, source: S) -> Self {
        let mut spec = Spectrometer {
            source,
            buffer: VecDeque::new(),
            calibration: None,
            size,
        };
        spec.buffer
            .resize(spec.buffer_size(), Complex::new(0.0, 0.0));
        spec
    }
    /// Get a reference to the source
    pub fn source(&self) -> &S {
        &self.source
    }
    /// Get a mutable reference to the source
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
    /// The amount of time in seconds used to perform frequency analysis
    ///
    /// Note that [`Spectrometer::next`] can be called much more often that this
    pub fn sample_period(&self) -> f32 {
        self.size as f32 / self.source.sample_rate() as f32
    }
    fn buffer_size(&self) -> usize {
        self.size * 2
    }
    /// Use the next `n` + 1 spectra to calibrate a definition of "silence"
    ///
    /// All spectra created after calling this function will have the "silence" spectrum subtracted.
    pub fn calibrate_n(&mut self, n: usize) {
        if let Some(mut new_calibration) = self.raw_next() {
            for _ in 0..n {
                if let Some(frame) = self.raw_next() {
                    for (c, a) in new_calibration.amps.iter_mut().zip(&frame.amps) {
                        *c = c.max(*a);
                    }
                } else {
                    break;
                }
            }
            self.calibration = Some(new_calibration)
        }
    }
    /// Use the next `size` / 10 + 1 spectra to calibrate a definition of "silence"
    ///
    /// All spectra created after calling this function will have the "silence" spectrum subtracted.
    pub fn calibrate(&mut self) {
        self.calibrate_n(self.size / 10)
    }
    /// Clear the calibration set by [`Spectrometer::calibrate`] or [`Spectrometer::calibrate_n`]
    pub fn uncalibrate(&mut self) {
        self.calibration = None;
    }
    fn raw_next(&mut self) -> Option<Spectrum> {
        let buffer_size = self.buffer_size();
        for _ in 0..buffer_size {
            match self.source.next() {
                SignalResult::Sample(s) => self.buffer.push_back(Complex::new(s, 0.0)),
                SignalResult::Wait => break,
                SignalResult::End => return None,
            }
        }
        while self.buffer.len() > buffer_size {
            self.buffer.pop_front();
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.size);
        let buffer = self.buffer.make_contiguous();
        let input_start = buffer.len() - self.size;
        let mut complex_amps = vec![Complex::new(0f32, 0.0); self.size];
        for (i, amp) in buffer[input_start..].iter().enumerate() {
            complex_amps[i] = *amp;
        }
        fft.process(&mut complex_amps);
        let mut amps = vec![0.0; self.size];
        for (i, amp) in complex_amps.iter().enumerate() {
            let mut amp = amp.norm();
            if amp.is_nan() {
                amp = 0.0;
            }
            amps[i] = amp;
        }
        Some(Spectrum {
            amps,
            sample_rate: self.source.sample_rate(),
        })
    }
}

impl<S> Iterator for Spectrometer<S>
where
    S: SignalSource,
{
    type Item = Spectrum;
    fn next(&mut self) -> Option<Self::Item> {
        let mut spectrum = self.raw_next()?;
        if let Some(min) = &self.calibration {
            for (s, min) in spectrum.amps.iter_mut().zip(&min.amps) {
                *s = (*s - min).max(0.0);
            }
        }
        Some(spectrum)
    }
}
