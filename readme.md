# Description

Ridgeline is a crate for simplifying frequency spectrum analysis of streamed signals.

For more information, check out [the documentation](https://docs.rs/ridgeline).

# Usage

The [`SignalSource`](trait.SignalSource.html) trait defines behavior for a signal. [`SystemAudio`](struct.SystemAudio.html) is a
`SignalSource` implementation that streams audio samples from a system audio device.

[`Spectrometer`](struct.Spectrometer.html) is an iterator that wraps a `SignalSource` and yields `Spectrum`s.

[`Spectrum`](struct.Spectrum.html) contains frequency data for a signal at a single point in time.
The aplitude of a given frequency can be queried with [`Spectrum::amplitude`](struct.Spectrum.html#method.amplitude).

# Example: Get the dominant frequency
```rust
use std::{thread::sleep, time::Duration};

use ridgeline::*;

// Stream audio from the default input device
let audio_input = SystemAudio::from_default_device().unwrap();

// Create a `Spectrometer` from the audio input stream with 5000 FFT buckets
// The actual FFT buffer uses SIZE * 2 buckets, but only the lower half is usable
let spectrometer = audio_input.analyze::<5000>();

// Print the frequency with the highest amplitude
for spectrum in spectrometer {
    println!("{} Hz", spectrum.dominant());
    sleep(Duration::from_millis(10));
}
```