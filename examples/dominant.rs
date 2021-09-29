use std::{thread::sleep, time::Duration};

use ridgeline::*;

fn main() {
    // Stream audio from the default input device
    let audio_input = SystemAudio::from_default_device().unwrap();
    // Create a `Spectrometer` from the audio input stream with 10000 FFT buckets
    // The actual FFT buffer uses SIZE * 2 buckets, but only the lower half is usable
    let spectrometer = audio_input.analyze::<10000>();
    // Print the frequency with the highest amplitude
    for spectrum in spectrometer {
        println!("{}", spectrum.dominant());
        sleep(Duration::from_millis(10));
    }
}
