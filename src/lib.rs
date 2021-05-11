use std::{collections::VecDeque, sync::mpsc, usize};

use crate::cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, InputCallbackInfo, Sample, SampleFormat, Stream, StreamConfig, SupportedStreamConfig,
};
use cpal::{BuildStreamError, PlayStreamError, SupportedStreamConfigsError};
use rustfft::{num_complex::Complex, FftPlanner};

pub use cpal;

#[derive(Debug, thiserror::Error)]
pub enum BuildSpectrometerError {
    #[error("{0}")]
    Stream(#[from] BuildStreamError),
    #[error("{0}")]
    Play(#[from] PlayStreamError),
    #[error("{0}")]
    Configs(#[from] SupportedStreamConfigsError),
    #[error("No config available for device")]
    NoConfig,
    #[error("No device available")]
    NoDevice,
}

pub type BuildSpectrometerResult = Result<Spectrometer, BuildSpectrometerError>;

pub struct Spectrum {
    amps: Vec<f32>,
    sample_rate: u32,
}

impl Spectrum {
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
        i as f32 * self.bucket_width()
    }
    pub fn amplitude(&self, freq: f32) -> f32 {
        let delta = self.bucket_width();
        let ratio = freq / delta;
        if ratio as usize >= self.amps.len() {
            return 0.0;
        }
        let floor = ratio.floor();
        let l = floor as usize;
        let r = ratio.ceil() as usize as usize;
        let left = self.amps[l] / self.frequency_at(l);
        let right = self.amps[r] / self.frequency_at(r);
        let param = ratio - floor;
        (1.0 - param) * left + param * right
    }
    pub fn max(&self) -> f32 {
        let bucket = self
            .amps
            .iter()
            .take(self.amps.len() / 2)
            .enumerate()
            .map(|(i, a)| (i, a / self.frequency_at(i)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        bucket as f32 * self.bucket_width()
    }
}

pub struct Spectrometer {
    _stream: Stream,
    recv: mpsc::Receiver<f32>,
    buffer: VecDeque<Complex<f32>>,
    sample_rate: u32,
    channels: u16,
    max_freq: f32,
    calibration: Option<Spectrum>,
}

pub struct SpectrometerBuilder<'a> {
    pub max_freq: f32,
    pub device: Option<&'a Device>,
    pub config: Option<SupportedStreamConfig>,
}

impl<'a> SpectrometerBuilder<'a> {
    pub fn max_freq(self, max_freq: f32) -> Self {
        SpectrometerBuilder { max_freq, ..self }
    }
    pub fn device(self, device: &'a Device) -> Self {
        SpectrometerBuilder {
            device: Some(device),
            ..self
        }
    }
    pub fn config(self, config: SupportedStreamConfig) -> Self {
        SpectrometerBuilder {
            config: Some(config),
            ..self
        }
    }
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
    pub fn builder<'a>() -> SpectrometerBuilder<'a> {
        SpectrometerBuilder {
            max_freq: 4000.0,
            device: None,
            config: None,
        }
    }
    pub fn from_default_device(max_freq: f32) -> BuildSpectrometerResult {
        Self::builder().max_freq(max_freq).build()
    }
    pub fn sample_size(&self) -> usize {
        (self.max_freq * 2.0).round() as usize
    }
    fn buffer_size(&self) -> usize {
        self.sample_size() * 2
    }
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
        let mut spectrum = self.raw_next();
        if let Some(min) = &self.calibration {
            for (s, min) in spectrum.amps.iter_mut().zip(&min.amps) {
                *s = (*s - min).max(0.0);
            }
        }
        Some(spectrum)
    }
}
