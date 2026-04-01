use mousiki::opusfile::{OpusFile, OpusPictureFormat, OpusPictureTag, OpusfileError, OpusfileOpenError};
use std::env;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 2;
const PCM_SAMPLES: usize = 120 * 48 * CHANNELS as usize;

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

    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);

    let mut opus = OpusFile::open_file(&input_path).map_err(ExampleError::Open)?;
    print_stream_info(&opus);

    let mut wav = File::create(&output_path)
        .map_err(|err| ExampleError::Io("create output", err.kind()))?;
    let declared_total = opus.pcm_total(None).unwrap_or(0);
    write_wav_header(&mut wav, declared_total)
        .map_err(|err| ExampleError::Io("write WAV header", err.kind()))?;

    let mut pcm = vec![0i16; PCM_SAMPLES];
    let mut byte_buffer = vec![0u8; PCM_SAMPLES * 2];
    let mut written_samples = 0usize;

    loop {
        let read = opus.read_stereo(&mut pcm).map_err(ExampleError::Decode)?;
        if read.samples_per_channel == 0 {
            break;
        }
        let pcm_slice = &pcm[..read.samples_per_channel * usize::from(CHANNELS)];
        interleaved_i16_to_le_bytes(pcm_slice, &mut byte_buffer[..pcm_slice.len() * 2]);
        wav.write_all(&byte_buffer[..pcm_slice.len() * 2])
            .map_err(|err| ExampleError::Io("write WAV data", err.kind()))?;
        written_samples += read.samples_per_channel;
    }

    wav.seek(SeekFrom::Start(0))
        .map_err(|err| ExampleError::Io("rewind WAV output", err.kind()))?;
    write_wav_header(&mut wav, written_samples)
        .map_err(|err| ExampleError::Io("rewrite WAV header", err.kind()))?;
    Ok(())
}

fn print_stream_info(opus: &OpusFile) {
    eprintln!("links: {}", opus.link_count());
    if let Ok(total) = opus.pcm_total(None) {
        eprintln!("duration: {} samples @ {} Hz", total, SAMPLE_RATE);
    }
    if let Ok(total) = opus.raw_total(None) {
        eprintln!("size: {total} bytes");
    }
    for link_index in 0..opus.link_count() {
        let head = match opus.head(Some(link_index)) {
            Ok(head) => head,
            Err(_) => continue,
        };
        eprintln!("link {link_index}:");
        eprintln!("  channels: {}", head.channel_count);
        if let Ok(total) = opus.pcm_total(Some(link_index)) {
            eprintln!("  duration: {} samples", total);
        }
        if let Ok(total) = opus.raw_total(Some(link_index)) {
            eprintln!("  size: {total} bytes");
        }
        if head.input_sample_rate != 0 {
            eprintln!("  original sample rate: {} Hz", head.input_sample_rate);
        }
        if let Ok(tags) = opus.tags(Some(link_index)) {
            if let Some(vendor) = tags.vendor() {
                eprintln!("  encoded by: {vendor}");
            }
            for comment in tags.comments() {
                print_comment(comment);
            }
            if let Some(binary_suffix) = tags.binary_suffix() {
                eprintln!("  <{} bytes of binary metadata>", binary_suffix.len());
            }
        }
    }
}

fn print_comment(comment: &[u8]) {
    if !comment.starts_with(b"METADATA_BLOCK_PICTURE=") {
        eprintln!("  {}", String::from_utf8_lossy(comment));
        return;
    }
    match core::str::from_utf8(comment) {
        Ok(text) => match OpusPictureTag::parse(text) {
            Ok(picture) => {
                let format = match picture.format {
                    OpusPictureFormat::Unknown => "unknown",
                    OpusPictureFormat::Url => "url",
                    OpusPictureFormat::Jpeg => "jpeg",
                    OpusPictureFormat::Png => "png",
                    OpusPictureFormat::Gif => "gif",
                };
                eprintln!(
                    "  METADATA_BLOCK_PICTURE={} {}x{}x{} format={format} bytes={}",
                    picture.picture_type,
                    picture.width,
                    picture.height,
                    picture.depth,
                    picture.data().len()
                );
            }
            Err(_) => {
                eprintln!("  <error parsing picture metadata>");
            }
        },
        Err(_) => {
            eprintln!("  <non-utf8 comment>");
        }
    }
}

fn write_wav_header(file: &mut File, samples_per_channel: usize) -> io::Result<()> {
    let data_len = samples_per_channel
        .saturating_mul(usize::from(CHANNELS))
        .saturating_mul(2);
    let data_len_u32 = u32::try_from(data_len).unwrap_or(u32::MAX);
    let riff_len = data_len_u32.saturating_add(36);
    let byte_rate = SAMPLE_RATE * u32::from(CHANNELS) * 2;
    let block_align = CHANNELS * 2;

    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&riff_len.to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes());
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align.to_le_bytes());
    header[34..36].copy_from_slice(&16u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_len_u32.to_le_bytes());
    file.write_all(&header)
}

fn interleaved_i16_to_le_bytes(input: &[i16], output: &mut [u8]) {
    debug_assert_eq!(output.len(), input.len() * 2);
    for (sample, chunk) in input.iter().zip(output.chunks_exact_mut(2)) {
        chunk.copy_from_slice(&sample.to_le_bytes());
    }
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage(program) => {
            let name = Path::new(&program)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("opusfile_example");
            eprintln!("usage: {name} <input.opus> <output.wav>");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::Open(err) => {
            eprintln!("open failed: {err}");
        }
        ExampleError::Decode(err) => {
            eprintln!("decode failed: {err}");
        }
    }
}

enum ExampleError {
    Usage(std::ffi::OsString),
    Io(&'static str, io::ErrorKind),
    Open(OpusfileOpenError),
    Decode(OpusfileError),
}
