use std::{thread::sleep, time::Duration};

use ridgeline::*;

fn main() {
    let input = Spectrometer::<10000>::from_default_device().unwrap();
    for spectrum in input {
        println!("{}", spectrum.dominant());
        sleep(Duration::from_millis(10));
    }
}
