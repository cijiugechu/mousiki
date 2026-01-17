use mousiki::opus_decoder::{
    opus_decode, opus_decoder_create, OpusDecodeError, OpusDecoderInitError,
};
use mousiki::opus_encoder::{
    opus_encode, opus_encoder_create, opus_encoder_ctl, OpusEncodeError, OpusEncoderCtlError,
    OpusEncoderCtlRequest, OpusEncoderInitError,
};
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const FRAME_SIZE: usize = 960;
const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: i32 = 2;
const APPLICATION: i32 = 2049; // OPUS_APPLICATION_AUDIO
const BITRATE: i32 = 64_000;

const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

fn main() {
    if let Err(err) = run() {
        report_error(err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args_os();
    let _program = args.next();
    let input = args.next().ok_or(ExampleError::Usage)?;
    let output = args.next().ok_or(ExampleError::Usage)?;
    if args.next().is_some() {
        return Err(ExampleError::Usage);
    }

    let input_path = Path::new(&input);
    let output_path = Path::new(&output);

    let mut input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;
    let mut output_file =
        File::create(output_path).map_err(|err| ExampleError::Io("create output", err.kind()))?;

    let mut encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION).map_err(ExampleError::EncoderInit)?;
    opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(BITRATE))
        .map_err(ExampleError::EncoderCtl)?;
    let mut decoder =
        opus_decoder_create(SAMPLE_RATE, CHANNELS).map_err(ExampleError::DecoderInit)?;

    let channels = CHANNELS as usize;
    let mut input_bytes = vec![0u8; FRAME_SIZE * channels * 2];
    let mut input_pcm = vec![0i16; FRAME_SIZE * channels];
    let mut output_pcm = vec![0i16; MAX_FRAME_SIZE * channels];
    let mut output_bytes = vec![0u8; MAX_FRAME_SIZE * channels * 2];
    let mut packet = vec![0u8; MAX_PACKET_SIZE];

    loop {
        match input_file.read_exact(&mut input_bytes) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(ExampleError::Io("read input", err.kind())),
        }

        for (sample, chunk) in input_pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
            *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let packet_len =
            opus_encode(&mut encoder, &input_pcm, FRAME_SIZE, &mut packet).map_err(
                ExampleError::Encode,
            )?;

        let decoded = opus_decode(
            &mut decoder,
            Some(&packet[..packet_len]),
            packet_len,
            &mut output_pcm,
            MAX_FRAME_SIZE,
            false,
        )
        .map_err(ExampleError::Decode)?;

        let total_samples = decoded * channels;
        for (chunk, &sample) in output_bytes
            .chunks_exact_mut(2)
            .take(total_samples)
            .zip(output_pcm.iter().take(total_samples))
        {
            let le = sample.to_le_bytes();
            chunk[0] = le[0];
            chunk[1] = le[1];
        }

        output_file
            .write_all(&output_bytes[..total_samples * 2])
            .map_err(|err| ExampleError::Io("write output", err.kind()))?;
    }

    Ok(())
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage => {
            eprintln!("usage: trivial_example <input.pcm> <output.pcm>");
            eprintln!("input and output are 16-bit little-endian raw files");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::EncoderInit(err) => {
            eprintln!("failed to create encoder: {err:?}");
        }
        ExampleError::DecoderInit(err) => {
            eprintln!("failed to create decoder: {err:?}");
        }
        ExampleError::EncoderCtl(err) => {
            eprintln!("failed to set bitrate: {err:?}");
        }
        ExampleError::Encode(err) => {
            eprintln!("encode failed: {err:?}");
        }
        ExampleError::Decode(err) => {
            eprintln!("decode failed: {err:?}");
        }
    }
}

enum ExampleError {
    Usage,
    Io(&'static str, io::ErrorKind),
    EncoderInit(OpusEncoderInitError),
    DecoderInit(OpusDecoderInitError),
    EncoderCtl(OpusEncoderCtlError),
    Encode(OpusEncodeError),
    Decode(OpusDecodeError),
}
