use mousiki::decoder::Decoder;
use mousiki::oggreader::{OggRead, OggReader, OggReaderError, ReadError};

pub const MAX_PCM_BYTES: usize = 1920;
pub const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
pub const TINY_OGG: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/tiny.ogg"));

pub struct SliceReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
}

impl<'a> OggRead for SliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        if self.position >= self.data.len() {
            return Ok(0);
        }

        let remaining = self.data.len() - self.position;
        let to_copy = remaining.min(buf.len());
        buf[..to_copy].copy_from_slice(&self.data[self.position..self.position + to_copy]);
        self.position += to_copy;
        Ok(to_copy)
    }
}

pub fn fuzz_decoder(data: &[u8]) {
    let _ = decode_stream(data);
}

pub fn decode_stream(data: &[u8]) -> Result<usize, ()> {
    let mut out = [0u8; MAX_PCM_BYTES];

    let (mut ogg, _) = OggReader::new_with(SliceReader::new(data)).map_err(|_| ())?;
    let mut decoder = Decoder::new();
    let mut decoded_frames = 0usize;

    loop {
        let (segments, _) = match ogg.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(_) => return Err(()),
        };

        if let Some(first_segment) = segments.get(0) {
            if first_segment.starts_with(OPUS_TAGS_SIGNATURE) {
                continue;
            }
        }

        for segment in segments.into_iter() {
            if segment.is_empty() {
                continue;
            }

            match decoder.decode(segment, &mut out) {
                Ok(_) => decoded_frames += 1,
                Err(_) => return Err(()),
            }
        }
    }

    Ok(decoded_frames)
}
