// Test decoding CELT mode Opus files
use mousiki::opus_decoder::{opus_decoder_create, opus_decode};
use mousiki::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process;

const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";

struct FileStream {
    file: File,
}

impl FileStream {
    fn new(file: File) -> Self {
        Self { file }
    }
}

impl OggRead for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        use std::io::Read;

        loop {
            match self.file.read(buf) {
                Ok(n) => return Ok(n),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(0),
                Err(_) => return Err(ReadError::Other),
            }
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os();
    let _program = args.next();
    let input = args.next().ok_or("Usage: decode_celt <in-file> <out-file>")?;
    let output = args.next().ok_or("Usage: decode_celt <in-file> <out-file>")?;

    let input_path = Path::new(&input);
    let output_path = Path::new(&output);

    let input_file = File::open(input_path).map_err(|e| format!("Failed to open input: {}", e))?;
    let (mut ogg_reader, ogg_header) =
        OggReader::new_with(FileStream::new(input_file)).map_err(|e| format!("OGG error: {}", e))?;

    let mut output_file =
        File::create(output_path).map_err(|e| format!("Failed to create output: {}", e))?;

    // Get preskip from header
    let preskip = ogg_header.pre_skip as usize;
    let channels = ogg_header.channels as i32;
    eprintln!("Opus header: channels={}, preskip={}, sample_rate={}",
              channels, preskip, ogg_header.sample_rate);

    // Create full OpusDecoder (supports CELT)
    let mut decoder = opus_decoder_create(48000, channels).map_err(|e| format!("Decoder init error: {:?}", e))?;

    // Output buffer for 48kHz, 120ms max frame
    let mut pcm = vec![0i16; 5760 * channels as usize];
    let mut total_samples = 0usize;
    let mut packet_count = 0usize;
    let mut samples_skipped: usize = 0;

    loop {
        let (segments, _) = match ogg_reader.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(err) => return Err(format!("OGG read error: {}", err)),
        };

        for segment in segments.into_iter() {
            if segment.is_empty() {
                continue;
            }

            // Skip Opus comment header
            if segment.starts_with(OPUS_TAGS_SIGNATURE) {
                continue;
            }

            // Decode with full OpusDecoder
            // frame_size = 960 samples (20ms at 48kHz)
            let frame_size = 960;
            match opus_decode(&mut decoder, Some(segment), segment.len(), &mut pcm, frame_size, false) {
                Ok(samples) => {
                    // Handle preskip
                    let mut start = 0;
                    if samples_skipped < preskip {
                        let to_skip = (preskip - samples_skipped).min(samples);
                        start = to_skip;
                        samples_skipped += to_skip;
                    }

                    // Write i16 samples as raw PCM (after preskip)
                    for sample in &pcm[start..samples] {
                        let bytes = sample.to_le_bytes();
                        output_file.write_all(&bytes).map_err(|e| format!("Write error: {}", e))?;
                    }
                    total_samples += samples - start;
                    packet_count += 1;
                    if packet_count <= 5 || packet_count % 50 == 0 {
                        eprintln!("Packet {}: decoded {} samples (wrote {})", packet_count, samples, samples - start);
                    }
                }
                Err(e) => {
                    eprintln!("Decode error on packet {}: {:?}", packet_count, e);
                    packet_count += 1;
                }
            }
        }
    }

    let duration_secs = total_samples as f64 / 48000.0;
    eprintln!(
        "\nDecoded {} packets, {} samples ({:.2}s) to {}",
        packet_count,
        total_samples,
        duration_secs,
        output_path.display()
    );

    Ok(())
}
