use std::{thread::sleep, time::Duration};

use ridgeline::*;

fn main() {
    let input = Spectrometer::from_default_device(10000.0).unwrap();
    for spectrum in input {
        println!("{}", spectrum.max());
        sleep(Duration::from_millis(10));
    }
}
