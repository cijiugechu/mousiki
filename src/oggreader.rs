use core::fmt;
use log::{debug, trace};
use mousiki_ogg::Page;

const PAGE_HEADER_TYPE_BEGINNING_OF_STREAM: u8 = 0x02;
const PAGE_HEADER_SIGNATURE: [u8; 4] = *b"OggS";
const ID_PAGE_SIGNATURE: [u8; 8] = *b"OpusHead";
const PAGE_HEADER_LEN: usize = 27;
const ID_PAGE_PAYLOAD_LENGTH: usize = 19;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    UnexpectedEof,
    Other,
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("unexpected end of stream"),
            Self::Other => f.write_str("reader error"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OggReaderError {
    BadIdPageSignature,
    BadIdPageType,
    BadIdPageLength,
    BadIdPagePayloadSignature,
    ChecksumMismatch,
    Read(ReadError),
}

impl fmt::Display for OggReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadIdPageSignature => f.write_str("bad header signature"),
            Self::BadIdPageType => f.write_str("wrong header, expected beginning of stream"),
            Self::BadIdPageLength => f.write_str("payload for id page must be 19 bytes"),
            Self::BadIdPagePayloadSignature => f.write_str("bad payload signature"),
            Self::ChecksumMismatch => f.write_str("expected and actual checksum do not match"),
            Self::Read(err) => write!(f, "reader error: {err}"),
        }
    }
}

impl From<ReadError> for OggReaderError {
    fn from(value: ReadError) -> Self {
        Self::Read(value)
    }
}

pub trait OggRead {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OggHeader {
    pub channel_map: u8,
    pub channels: u8,
    pub output_gain: u16,
    pub pre_skip: u16,
    pub sample_rate: u32,
    pub version: u8,
}

pub struct OggReader<R: OggRead> {
    stream: R,
    header: OggHeader,
}

impl<R: OggRead> OggReader<R> {
    pub fn new(stream: R) -> Result<Self, OggReaderError> {
        let mut reader = Self {
            stream,
            header: OggHeader {
                channel_map: 0,
                channels: 0,
                output_gain: 0,
                pre_skip: 0,
                sample_rate: 0,
                version: 0,
            },
        };
        reader.header = reader.read_headers()?;
        debug!(
            "oggreader: initialized with version={}, channels={}, sample_rate={}, pre_skip={}, gain={}, channel_map={}",
            reader.header.version,
            reader.header.channels,
            reader.header.sample_rate,
            reader.header.pre_skip,
            reader.header.output_gain,
            reader.header.channel_map
        );
        Ok(reader)
    }

    #[must_use]
    pub fn header(&self) -> OggHeader {
        self.header
    }

    pub fn next_page(&mut self) -> Result<Page, OggReaderError> {
        self.read_page()
    }

    fn read_headers(&mut self) -> Result<OggHeader, OggReaderError> {
        let page = self.read_page()?;
        if page.header.get(..4) != Some(&PAGE_HEADER_SIGNATURE) {
            return Err(OggReaderError::BadIdPageSignature);
        }
        if page.header.get(5).copied() != Some(PAGE_HEADER_TYPE_BEGINNING_OF_STREAM) {
            return Err(OggReaderError::BadIdPageType);
        }
        let mut segments = page.segments();
        let id_segment = segments.next().ok_or(OggReaderError::BadIdPageLength)?;
        if id_segment.len() != ID_PAGE_PAYLOAD_LENGTH {
            return Err(OggReaderError::BadIdPageLength);
        }
        if id_segment[..8] != ID_PAGE_SIGNATURE {
            return Err(OggReaderError::BadIdPagePayloadSignature);
        }
        Ok(OggHeader {
            version: id_segment[8],
            channels: id_segment[9],
            pre_skip: u16::from_le_bytes([id_segment[10], id_segment[11]]),
            sample_rate: u32::from_le_bytes([
                id_segment[12],
                id_segment[13],
                id_segment[14],
                id_segment[15],
            ]),
            output_gain: u16::from_le_bytes([id_segment[16], id_segment[17]]),
            channel_map: id_segment[18],
        })
    }

    fn read_page(&mut self) -> Result<Page, OggReaderError> {
        let mut header = [0u8; PAGE_HEADER_LEN];
        self.read_exact(&mut header)?;
        let segments_count = header[26] as usize;
        let mut lacing = alloc::vec![0u8; segments_count];
        self.read_exact(&mut lacing)?;
        let total_payload_len: usize = lacing.iter().map(|&value| usize::from(value)).sum();
        let mut body = alloc::vec![0u8; total_payload_len];
        self.read_exact(&mut body)?;

        let mut full_header = header.to_vec();
        full_header.extend_from_slice(&lacing);
        let page = Page::new(full_header, body);
        if !page.checksum_valid() {
            return Err(OggReaderError::ChecksumMismatch);
        }

        trace!(
            "oggreader: page serial={} index={} granule={} segments={} payload_len={}",
            page.serialno(),
            page.pageno(),
            page.granulepos(),
            segments_count,
            page.body_len()
        );

        Ok(page)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), OggReaderError> {
        read_exact_from(&mut self.stream, buf)?;
        Ok(())
    }
}

fn read_exact_from<R: OggRead>(reader: &mut R, mut buf: &mut [u8]) -> Result<(), ReadError> {
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => return Err(ReadError::UnexpectedEof),
            Ok(n) => {
                let (_, rest) = buf.split_at_mut(n);
                buf = rest;
            }
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SliceReader<'a> {
        data: &'a [u8],
        position: usize,
    }

    impl<'a> SliceReader<'a> {
        fn new(data: &'a [u8]) -> Self {
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

    const fn build_ogg_container() -> [u8; 80] {
        [
            0x4f, 0x67, 0x67, 0x53, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x8e, 0x9b, 0x20, 0xaa, 0x00, 0x00, 0x00, 0x00, 0x61, 0xee, 0x61, 0x17, 0x01, 0x13,
            0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64, 0x01, 0x02, 0x00, 0x0f, 0x80, 0xbb,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x4f, 0x67, 0x67, 0x53, 0x00, 0x00, 0xda, 0x93, 0xc2,
            0xd9, 0x00, 0x00, 0x00, 0x00, 0x8e, 0x9b, 0x20, 0xaa, 0x02, 0x00, 0x00, 0x00, 0x49,
            0x97, 0x03, 0x37, 0x01, 0x05, 0x98, 0x36, 0xbe, 0x88, 0x9e,
        ]
    }

    #[test]
    fn parse_valid_header() {
        let container = build_ogg_container();
        let mut reader =
            OggReader::new(SliceReader::new(&container)).expect("reader should initialize");
        let header = reader.header();
        assert_eq!(
            header,
            OggHeader {
                channel_map: 0x0,
                channels: 0x2,
                output_gain: 0x0,
                pre_skip: 0x0f00,
                sample_rate: 0x00bb80,
                version: 0x1,
            }
        );
        let page = reader.next_page().expect("second page should parse");
        let segments = page.segments().collect::<alloc::vec::Vec<_>>();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], &[0x98, 0x36, 0xbe, 0x88, 0x9e]);
    }

    #[test]
    fn parse_next_page_iterates_pages() {
        let container = build_ogg_container();
        let mut reader =
            OggReader::new(SliceReader::new(&container)).expect("reader should initialize");
        let page = reader.next_page().expect("should parse comment page");
        let segments = page.segments().collect::<alloc::vec::Vec<_>>();
        assert_eq!(segments.len(), 1);
        assert!(matches!(
            reader.next_page(),
            Err(OggReaderError::Read(ReadError::UnexpectedEof))
        ));
    }
}
