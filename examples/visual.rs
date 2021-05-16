use std::{thread::sleep, time::Duration};

use ridgeline::*;

fn main() {
    const SEGMENTS: usize = 20;
    const MAX_FREQ: f32 = 1000.0;
    const SAMPLE_SIZE: usize = MAX_FREQ as usize * 2;
    let input = Spectrometer::<SAMPLE_SIZE>::from_default_device().unwrap();
    let mut silence = Vec::new();
    for spectrum in input {
        let calibrate = silence.is_empty();
        println!("-----------------------------------------");
        println!("max: {}", spectrum.dominant());
        for i in 0..SEGMENTS {
            let freq = i as f32 * MAX_FREQ / SEGMENTS as f32;
            let amp = spectrum.amplitude(freq);
            if calibrate {
                silence.push(amp);
            }
            println!("{:#^1$}", "", ((amp - silence[i]) * 100.0) as usize);
        }
        for _ in 0..5 {
            println!();
        }
        sleep(Duration::from_millis(200));
    }
}
