use std::{collections::VecDeque, sync::mpsc, usize};

use crate::cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, InputCallbackInfo, Sample, SampleFormat, Stream, StreamConfig, SupportedStreamConfig,
};
use rustfft::{num_complex::Complex, FftPlanner};

pub use cpal;

pub struct Frquencies {
    amps: Vec<Complex<f32>>,
    sample_rate: u32,
}

impl Frquencies {
    pub fn bucket_width(&self) -> f32 {
        self.sample_rate as f32 / self.amps.len() as f32
    }
    pub fn amplitude(&self, freq: f32) -> f32 {
        let delta = self.bucket_width();
        let ratio = freq / delta;
        if ratio as usize >= self.amps.len() {
            return 0.0;
        }
        let floor = ratio.floor();
        let left = self.amps[floor as usize].norm();
        let right = self.amps[ratio.ceil() as usize].norm();
        let param = ratio - floor;
        (1.0 - param) * left + param * right
    }
    pub fn max(&self) -> f32 {
        let bucket = self
            .amps
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.norm() / (i + 3) as f32))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        bucket as f32 * self.bucket_width()
    }
}

pub struct InputSpectrum {
    _stream: Stream,
    recv: mpsc::Receiver<f32>,
    buffer: VecDeque<Complex<f32>>,
    sample_rate: u32,
    channels: u16,
    max_freq: f32,
}

impl InputSpectrum {
    pub fn from_default_device(max_freq: f32) -> Self {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no input device available");
        Self::from_device(max_freq, &device)
    }
    pub fn from_device(max_freq: f32, device: &Device) -> Self {
        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();
        Self::from_device_and_config(max_freq, device, supported_config)
    }
    pub fn from_device_and_config(
        max_freq: f32,
        device: &Device,
        config: SupportedStreamConfig,
    ) -> Self {
        let err_fn = |err| eprintln!("an error occurred on the input audio stream: {}", err);
        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();
        let (send, recv) = mpsc::channel();
        macro_rules! input_stream {
            ($sample:ty) => {
                device.build_input_stream(
                    &config,
                    move |data: &[$sample], _: &InputCallbackInfo| {
                        for &s in data {
                            send.send(s.to_f32()).unwrap()
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
        }
        .unwrap();

        stream.play().unwrap();

        let mut input = InputSpectrum {
            _stream: stream,
            recv,
            buffer: VecDeque::new(),
            channels: config.channels,
            sample_rate: config.sample_rate.0,
            max_freq,
        };
        input
            .buffer
            .resize(input.buffer_size(), Complex::new(0.0, 0.0));
        input
    }
    pub fn sample_size(&self) -> usize {
        (self.max_freq * 2.0).round() as usize
    }
    fn buffer_size(&self) -> usize {
        self.sample_size() * 2
    }
}

impl Default for InputSpectrum {
    /// Uses a default maximum frequency of 4000 hz
    fn default() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no input device available");
        Self::from_device(4000.0, &device)
    }
}

impl Iterator for InputSpectrum {
    type Item = Frquencies;
    fn next(&mut self) -> Option<Self::Item> {
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
        Some(Frquencies {
            amps,
            sample_rate: self.sample_rate,
        })
    }
}
