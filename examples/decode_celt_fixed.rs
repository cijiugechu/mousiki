/// Example: Decode CELT-only Opus file using fixed-point decoder
///
/// Usage: cargo run --example decode_celt_fixed --features fixed_point <input.opus> <output.pcm>
///
/// This example demonstrates CELT fixed-point decoding by reading an Opus file
/// and writing raw PCM output.

#[cfg(feature = "fixed_point")]
use mousiki::decoder::{Decoder, DecoderError};
#[cfg(feature = "fixed_point")]
use mousiki::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
#[cfg(feature = "fixed_point")]
use std::env;
#[cfg(feature = "fixed_point")]
use std::fs::File;
#[cfg(feature = "fixed_point")]
use std::io::{self, Write};
#[cfg(feature = "fixed_point")]
use std::path::Path;
#[cfg(feature = "fixed_point")]
use std::process;

#[cfg(feature = "fixed_point")]
const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
#[cfg(feature = "fixed_point")]
const PCM_BYTES_PER_FRAME: usize = 1920;

#[cfg(feature = "fixed_point")]
struct FileStream {
    file: File,
}

#[cfg(feature = "fixed_point")]
impl FileStream {
    fn new(file: File) -> Self {
        Self { file }
    }
}

#[cfg(feature = "fixed_point")]
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

#[cfg(feature = "fixed_point")]
fn main() {
    if let Err(err) = run() {
        report_error(err);
        process::exit(1);
    }
}

#[cfg(feature = "fixed_point")]
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

    eprintln!("Opening input file: {}", input_path.display());
    let input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;
    
    let (mut ogg_reader, _) =
        OggReader::new_with(FileStream::new(input_file)).map_err(ExampleError::Ogg)?;

    eprintln!("Creating output file: {}", output_path.display());
    let mut output_file =
        File::create(output_path).map_err(|err| ExampleError::Io("create output", err.kind()))?;
    
    let mut decoder = Decoder::new();
    let mut pcm = [0u8; PCM_BYTES_PER_FRAME];
    let mut frame_count = 0;

    eprintln!("Decoding with CELT fixed-point decoder...");
    
    loop {
        let (segments, _) = match ogg_reader.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(err) => return Err(ExampleError::Ogg(err)),
        };

        if let Some(first) = segments.get(0)
            && first.starts_with(OPUS_TAGS_SIGNATURE)
        {
            eprintln!("Skipping OpusTags header");
            continue;
        }

        for segment in segments.into_iter() {
            if segment.is_empty() {
                continue;
            }

            decoder
                .decode(segment, &mut pcm)
                .map_err(ExampleError::Decoder)?;
            output_file
                .write_all(&pcm)
                .map_err(|err| ExampleError::Io("write output", err.kind()))?;
            
            frame_count += 1;
            if frame_count % 100 == 0 {
                eprintln!("Decoded {} frames...", frame_count);
            }
        }
    }

    eprintln!("Successfully decoded {} frames using CELT fixed-point", frame_count);
    eprintln!("Output written to: {}", output_path.display());
    
    Ok(())
}

#[cfg(feature = "fixed_point")]
fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage => {
            eprintln!("Usage: decode_celt_fixed <input.opus> <output.pcm>");
            eprintln!();
            eprintln!("Decode CELT-only Opus audio using the fixed-point decoder.");
            eprintln!("The input file should be an Ogg Opus file containing CELT frames.");
            eprintln!("The output will be raw PCM data (16-bit signed, little-endian).");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::Ogg(err) => {
            eprintln!("Ogg reader error: {err}");
        }
        ExampleError::Decoder(err) => {
            eprintln!("Decoder error: {err}");
        }
    }
}

#[cfg(feature = "fixed_point")]
enum ExampleError {
    Usage,
    Io(&'static str, io::ErrorKind),
    Ogg(OggReaderError),
    Decoder(DecoderError),
}

#[cfg(feature = "fixed_point")]
impl From<OggReaderError> for ExampleError {
    fn from(value: OggReaderError) -> Self {
        Self::Ogg(value)
    }
}

#[cfg(feature = "fixed_point")]
impl From<DecoderError> for ExampleError {
    fn from(value: DecoderError) -> Self {
        Self::Decoder(value)
    }
}

#[cfg(not(feature = "fixed_point"))]
fn main() {
    eprintln!("This example requires the 'fixed_point' feature to be enabled.");
    eprintln!("Run with: cargo run --example decode_celt_fixed --features fixed_point");
    std::process::exit(1);
}
