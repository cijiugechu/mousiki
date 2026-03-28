use mousiki::libopusenc::encoder::{OggOpusComments, OggOpusEnc, OpeError, strerror};
use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::process;

const CHANNELS: usize = 2;
const SAMPLE_RATE: i32 = 44_100;
const FAMILY: i32 = 0;
const READ_SIZE: usize = 256;

fn main() {
    if let Err(err) = run() {
        report_error(err);
        process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let input = args
        .next()
        .ok_or_else(|| ExampleError::Usage(program.clone()))?;
    let output = args
        .next()
        .ok_or_else(|| ExampleError::Usage(program.clone()))?;
    if args.next().is_some() {
        return Err(ExampleError::Usage(program));
    }

    let input_path = Path::new(&input);
    let output_path = Path::new(&output);

    let mut input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;

    let mut comments = OggOpusComments::create().map_err(ExampleError::Encode)?;
    comments
        .add("ARTIST", "Someone")
        .map_err(ExampleError::Encode)?;
    comments
        .add("TITLE", "Some track")
        .map_err(ExampleError::Encode)?;

    let output_string = output_path.to_string_lossy();
    let mut encoder = OggOpusEnc::create_file(
        &output_string,
        &comments,
        SAMPLE_RATE,
        CHANNELS as i32,
        FAMILY,
    )
    .map_err(ExampleError::Encode)?;

    let mut input_bytes = [0u8; READ_SIZE * CHANNELS * 2];
    let mut pcm = [0i16; READ_SIZE * CHANNELS];

    loop {
        let bytes_read = input_file
            .read(&mut input_bytes)
            .map_err(|err| ExampleError::Io("read input", err.kind()))?;
        if bytes_read == 0 {
            break;
        }
        if bytes_read % (CHANNELS * 2) != 0 {
            return Err(ExampleError::PartialFrame(bytes_read));
        }

        let samples_per_channel = bytes_read / (CHANNELS * 2);
        for (sample, chunk) in pcm
            .iter_mut()
            .take(samples_per_channel * CHANNELS)
            .zip(input_bytes[..bytes_read].chunks_exact(2))
        {
            *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        encoder
            .write(&pcm[..samples_per_channel * CHANNELS], samples_per_channel)
            .map_err(ExampleError::Encode)?;
    }

    encoder.drain().map_err(ExampleError::Encode)?;
    Ok(())
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage(program) => {
            let name = Path::new(&program)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("libopusenc_example");
            eprintln!("usage: {name} <raw pcm input> <Ogg Opus output>");
            eprintln!("input is 16-bit little-endian stereo PCM at 44100 Hz");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::Encode(err) => {
            eprintln!("encoding failed: {}", strerror(err));
        }
        ExampleError::PartialFrame(bytes) => {
            eprintln!("input length is not aligned to a stereo PCM frame: {bytes} bytes");
        }
    }
}

enum ExampleError {
    Usage(std::ffi::OsString),
    Io(&'static str, io::ErrorKind),
    Encode(OpeError),
    PartialFrame(usize),
}
