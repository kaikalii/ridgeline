use std::{thread::sleep, time::Duration};

use ridgeline::*;

fn main() {
    let input = SystemAudio::from_default_device()
        .unwrap()
        .analyze::<10000>();
    for spectrum in input {
        println!("{}", spectrum.dominant());
        sleep(Duration::from_millis(10));
    }
}
