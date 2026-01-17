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

const MAGIC: [u8; 8] = *b"OPUSPKT1";
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

fn run() -> Result<(), ToolError> {
    let mut args = env::args_os();
    let _program = args.next();
    let mode = args.next().ok_or(ToolError::Usage)?;
    let input = args.next().ok_or(ToolError::Usage)?;
    let output = args.next().ok_or(ToolError::Usage)?;
    if args.next().is_some() {
        return Err(ToolError::Usage);
    }

    let input_path = Path::new(&input);
    let output_path = Path::new(&output);

    match mode.to_string_lossy().as_ref() {
        "encode" => encode_packets(input_path, output_path),
        "decode" => decode_packets(input_path, output_path),
        _ => Err(ToolError::Usage),
    }
}

fn encode_packets(input_path: &Path, output_path: &Path) -> Result<(), ToolError> {
    let mut input_file =
        File::open(input_path).map_err(|err| ToolError::Io("open input", err.kind()))?;
    let mut output_file =
        File::create(output_path).map_err(|err| ToolError::Io("create output", err.kind()))?;

    let mut encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION).map_err(ToolError::EncoderInit)?;
    opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(BITRATE))
        .map_err(ToolError::EncoderCtl)?;

    write_header(&mut output_file)?;

    let channels = CHANNELS as usize;
    let mut input_bytes = vec![0u8; FRAME_SIZE * channels * 2];
    let mut input_pcm = vec![0i16; FRAME_SIZE * channels];
    let mut packet = vec![0u8; MAX_PACKET_SIZE];

    loop {
        match input_file.read_exact(&mut input_bytes) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(ToolError::Io("read input", err.kind())),
        }

        for (sample, chunk) in input_pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
            *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let packet_len =
            opus_encode(&mut encoder, &input_pcm, FRAME_SIZE, &mut packet).map_err(
                ToolError::Encode,
            )?;
        if packet_len > u16::MAX as usize {
            return Err(ToolError::PacketTooLarge(packet_len));
        }

        output_file
            .write_all(&(packet_len as u16).to_le_bytes())
            .map_err(|err| ToolError::Io("write packet length", err.kind()))?;
        output_file
            .write_all(&packet[..packet_len])
            .map_err(|err| ToolError::Io("write packet bytes", err.kind()))?;
    }

    Ok(())
}

fn decode_packets(input_path: &Path, output_path: &Path) -> Result<(), ToolError> {
    let mut input_file =
        File::open(input_path).map_err(|err| ToolError::Io("open input", err.kind()))?;
    let mut output_file =
        File::create(output_path).map_err(|err| ToolError::Io("create output", err.kind()))?;

    read_header(&mut input_file)?;
    let mut decoder =
        opus_decoder_create(SAMPLE_RATE, CHANNELS).map_err(ToolError::DecoderInit)?;

    let channels = CHANNELS as usize;
    let mut output_pcm = vec![0i16; MAX_FRAME_SIZE * channels];
    let mut output_bytes = vec![0u8; MAX_FRAME_SIZE * channels * 2];

    loop {
        let mut len_buf = [0u8; 2];
        match input_file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(ToolError::Io("read packet length", err.kind())),
        }

        let packet_len = u16::from_le_bytes(len_buf) as usize;
        if packet_len == 0 {
            break;
        }

        let mut packet = vec![0u8; packet_len];
        input_file
            .read_exact(&mut packet)
            .map_err(|err| ToolError::Io("read packet bytes", err.kind()))?;

        let decoded = opus_decode(
            &mut decoder,
            Some(&packet),
            packet_len,
            &mut output_pcm,
            MAX_FRAME_SIZE,
            false,
        )
        .map_err(ToolError::Decode)?;

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
            .map_err(|err| ToolError::Io("write output", err.kind()))?;
    }

    Ok(())
}

fn write_header(output: &mut File) -> Result<(), ToolError> {
    output
        .write_all(&MAGIC)
        .map_err(|err| ToolError::Io("write header", err.kind()))?;
    output
        .write_all(&(SAMPLE_RATE as u32).to_le_bytes())
        .map_err(|err| ToolError::Io("write header", err.kind()))?;
    output
        .write_all(&(CHANNELS as u16).to_le_bytes())
        .map_err(|err| ToolError::Io("write header", err.kind()))?;
    output
        .write_all(&(FRAME_SIZE as u16).to_le_bytes())
        .map_err(|err| ToolError::Io("write header", err.kind()))?;
    Ok(())
}

fn read_header(input: &mut File) -> Result<(), ToolError> {
    let mut magic = [0u8; 8];
    input
        .read_exact(&mut magic)
        .map_err(|err| ToolError::Io("read header", err.kind()))?;
    if magic != MAGIC {
        return Err(ToolError::InvalidHeader("magic"));
    }

    let mut buf = [0u8; 4];
    input
        .read_exact(&mut buf)
        .map_err(|err| ToolError::Io("read header", err.kind()))?;
    let sample_rate = u32::from_le_bytes(buf) as i32;

    let mut buf = [0u8; 2];
    input
        .read_exact(&mut buf)
        .map_err(|err| ToolError::Io("read header", err.kind()))?;
    let channels = u16::from_le_bytes(buf) as i32;

    let mut buf = [0u8; 2];
    input
        .read_exact(&mut buf)
        .map_err(|err| ToolError::Io("read header", err.kind()))?;
    let frame_size = u16::from_le_bytes(buf) as usize;

    if sample_rate != SAMPLE_RATE || channels != CHANNELS || frame_size != FRAME_SIZE {
        return Err(ToolError::InvalidHeader("config"));
    }

    Ok(())
}

fn report_error(err: ToolError) {
    match err {
        ToolError::Usage => {
            eprintln!("usage: opus_packet_tool <encode|decode> <input> <output>");
            eprintln!("encode: input is 16-bit little-endian PCM");
            eprintln!("decode: input is OPUSPKT1 packet stream");
        }
        ToolError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ToolError::InvalidHeader(field) => {
            eprintln!("invalid packet header: {field}");
        }
        ToolError::PacketTooLarge(size) => {
            eprintln!("packet length too large: {size}");
        }
        ToolError::EncoderInit(err) => {
            eprintln!("failed to create encoder: {err:?}");
        }
        ToolError::DecoderInit(err) => {
            eprintln!("failed to create decoder: {err:?}");
        }
        ToolError::EncoderCtl(err) => {
            eprintln!("failed to set bitrate: {err:?}");
        }
        ToolError::Encode(err) => {
            eprintln!("encode failed: {err:?}");
        }
        ToolError::Decode(err) => {
            eprintln!("decode failed: {err:?}");
        }
    }
}

enum ToolError {
    Usage,
    Io(&'static str, io::ErrorKind),
    InvalidHeader(&'static str),
    PacketTooLarge(usize),
    EncoderInit(OpusEncoderInitError),
    DecoderInit(OpusDecoderInitError),
    EncoderCtl(OpusEncoderCtlError),
    Encode(OpusEncodeError),
    Decode(OpusDecodeError),
}
