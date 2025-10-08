use mousiki::decoder::{Decoder, DecoderError};
use mousiki::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process;

const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
const PCM_BYTES_PER_FRAME: usize = 1920;

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
        report_error(err);
        process::exit(1);
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

    let input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;
    let (mut ogg_reader, _) =
        OggReader::new_with(FileStream::new(input_file)).map_err(ExampleError::Ogg)?;

    let mut output_file =
        File::create(output_path).map_err(|err| ExampleError::Io("create output", err.kind()))?;
    let mut decoder = Decoder::new();
    let mut pcm = [0u8; PCM_BYTES_PER_FRAME];

    loop {
        let (segments, _) = match ogg_reader.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(err) => return Err(ExampleError::Ogg(err)),
        };

        if let Some(first) = segments.get(0) && first.starts_with(OPUS_TAGS_SIGNATURE) {
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
        }
    }

    Ok(())
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage => {
            eprintln!("Usage: decode <in-file> <out-file>");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::Ogg(err) => {
            eprintln!("ogg reader error: {err}");
        }
        ExampleError::Decoder(err) => {
            eprintln!("decoder error: {err}");
        }
    }
}

enum ExampleError {
    Usage,
    Io(&'static str, io::ErrorKind),
    Ogg(OggReaderError),
    Decoder(DecoderError),
}

impl From<OggReaderError> for ExampleError {
    fn from(value: OggReaderError) -> Self {
        Self::Ogg(value)
    }
}

impl From<DecoderError> for ExampleError {
    fn from(value: DecoderError) -> Self {
        Self::Decoder(value)
    }
}
