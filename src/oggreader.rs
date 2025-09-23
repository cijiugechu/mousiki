use core::fmt;

const PAGE_HEADER_TYPE_BEGINNING_OF_STREAM: u8 = 0x02;
const PAGE_HEADER_SIGNATURE: [u8; 4] = *b"OggS";
const ID_PAGE_SIGNATURE: [u8; 8] = *b"OpusHead";
const PAGE_HEADER_LEN: usize = 27;
const ID_PAGE_PAYLOAD_LENGTH: usize = 19;
const MAX_SEGMENT_COUNT: usize = 255;
const MAX_SEGMENT_SIZE: usize = 255;
const MAX_PAGE_PAYLOAD_LENGTH: usize = MAX_SEGMENT_COUNT * MAX_SEGMENT_SIZE;

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
    NilStream,
    BadIdPageSignature,
    BadIdPageType,
    BadIdPageLength,
    BadIdPagePayloadSignature,
    ChecksumMismatch,
    PayloadTooLarge,
    Read(ReadError),
}

impl fmt::Display for OggReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NilStream => f.write_str("stream is nil"),
            Self::BadIdPageSignature => f.write_str("bad header signature"),
            Self::BadIdPageType => f.write_str("wrong header, expected beginning of stream"),
            Self::BadIdPageLength => f.write_str("payload for id page must be 19 bytes"),
            Self::BadIdPagePayloadSignature => f.write_str("bad payload signature"),
            Self::ChecksumMismatch => f.write_str("expected and actual checksum do not match"),
            Self::PayloadTooLarge => f.write_str("page payload exceeds preallocated buffer"),
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

#[derive(Debug, Clone, Copy)]
struct SegmentInfo {
    offset: usize,
    len: usize,
}

impl SegmentInfo {
    const fn new() -> Self {
        Self { offset: 0, len: 0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OggPageSegments<'a> {
    payload: &'a [u8],
    infos: &'a [SegmentInfo],
}

impl<'a> OggPageSegments<'a> {
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        self.infos.get(index).map(|info| {
            let start = info.offset;
            let end = start + info.len;
            &self.payload[start..end]
        })
    }

    pub fn iter(&self) -> OggPageSegmentsIter<'a> {
        OggPageSegmentsIter {
            segments: *self,
            index: 0,
        }
    }
}

pub struct OggPageSegmentsIter<'a> {
    segments: OggPageSegments<'a>,
    index: usize,
}

impl<'a> Iterator for OggPageSegmentsIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.segments.get(self.index);
        if item.is_some() {
            self.index += 1;
        }
        item
    }
}

impl<'a> IntoIterator for OggPageSegments<'a> {
    type Item = &'a [u8];
    type IntoIter = OggPageSegmentsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OggPageHeader {
    pub granule_position: u64,
    pub signature: [u8; 4],
    pub version: u8,
    pub header_type: u8,
    pub serial: u32,
    pub index: u32,
    pub segments_count: u8,
}

pub struct OggReader<R: OggRead> {
    stream: R,
    bytes_read_successfully: i64,
    checksum_table: [u32; 256],
    do_checksum: bool,
    payload_buffer: [u8; MAX_PAGE_PAYLOAD_LENGTH],
    payload_len: usize,
    segment_infos: [SegmentInfo; MAX_SEGMENT_COUNT],
    segments_count: usize,
}

impl<R: OggRead> OggReader<R> {
    pub fn new_with(stream: R) -> Result<(Self, OggHeader), OggReaderError> {
        Self::build(stream, true)
    }

    pub fn new_with_option(stream: Option<R>) -> Result<(Self, OggHeader), OggReaderError> {
        let stream = stream.ok_or(OggReaderError::NilStream)?;
        Self::build(stream, true)
    }

    #[cfg(test)]
    fn new_with_checksum(
        stream: R,
        do_checksum: bool,
    ) -> Result<(Self, OggHeader), OggReaderError> {
        Self::build(stream, do_checksum)
    }

    fn build(stream: R, do_checksum: bool) -> Result<(Self, OggHeader), OggReaderError> {
        let mut reader = OggReader {
            stream,
            bytes_read_successfully: 0,
            checksum_table: generate_checksum_table(),
            do_checksum,
            payload_buffer: [0u8; MAX_PAGE_PAYLOAD_LENGTH],
            payload_len: 0,
            segment_infos: [SegmentInfo::new(); MAX_SEGMENT_COUNT],
            segments_count: 0,
        };

        let header = reader.read_headers()?;

        Ok((reader, header))
    }

    pub fn parse_next_page(
        &mut self,
    ) -> Result<(OggPageSegments<'_>, OggPageHeader), OggReaderError> {
        self.parse_next_page_inner()
    }

    pub fn reset_reader<F>(&mut self, reset: F)
    where
        F: FnOnce(i64) -> R,
    {
        self.stream = reset(self.bytes_read_successfully);
        self.bytes_read_successfully = 0;
        self.payload_len = 0;
        self.segments_count = 0;
    }

    fn read_headers(&mut self) -> Result<OggHeader, OggReaderError> {
        let (segments, page_header) = self.parse_next_page_inner()?;

        if page_header.signature != PAGE_HEADER_SIGNATURE {
            return Err(OggReaderError::BadIdPageSignature);
        }

        if page_header.header_type != PAGE_HEADER_TYPE_BEGINNING_OF_STREAM {
            return Err(OggReaderError::BadIdPageType);
        }

        let id_segment = segments.get(0).ok_or(OggReaderError::BadIdPageLength)?;
        if id_segment.len() != ID_PAGE_PAYLOAD_LENGTH {
            return Err(OggReaderError::BadIdPageLength);
        }

        if &id_segment[..8] != ID_PAGE_SIGNATURE {
            return Err(OggReaderError::BadIdPagePayloadSignature);
        }

        let header = OggHeader {
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
        };

        Ok(header)
    }

    fn parse_next_page_inner(
        &mut self,
    ) -> Result<(OggPageSegments<'_>, OggPageHeader), OggReaderError> {
        let mut header = [0u8; PAGE_HEADER_LEN];
        self.read_exact(&mut header)?;

        let segments_count = header[26] as usize;
        let mut lacing_values = [0u8; MAX_SEGMENT_COUNT];
        let lacing_slice = &mut lacing_values[..segments_count];
        self.read_exact(lacing_slice)?;

        let mut total_payload_len = 0usize;
        for &size in lacing_slice.iter() {
            total_payload_len = total_payload_len
                .checked_add(size as usize)
                .ok_or(OggReaderError::PayloadTooLarge)?;
            if total_payload_len > MAX_PAGE_PAYLOAD_LENGTH {
                return Err(OggReaderError::PayloadTooLarge);
            }
        }

        let mut cursor = 0usize;
        let mut segment_buffer = [0u8; MAX_SEGMENT_SIZE];
        for (index, &size) in lacing_slice.iter().enumerate() {
            let len = size as usize;
            if len > 0 {
                let temp = &mut segment_buffer[..len];
                read_exact_from(&mut self.stream, temp)?;
                self.bytes_read_successfully += len as i64;
                self.payload_buffer[cursor..cursor + len].copy_from_slice(temp);
            }
            self.segment_infos[index] = SegmentInfo {
                offset: cursor,
                len,
            };
            cursor += len;
        }
        for info in self.segment_infos[segments_count..].iter_mut() {
            *info = SegmentInfo::new();
        }
        self.payload_len = cursor;
        self.segments_count = segments_count;

        if self.do_checksum {
            let mut checksum = 0u32;
            for (idx, &byte) in header.iter().enumerate() {
                if idx >= 22 && idx < 26 {
                    update_checksum(&mut checksum, 0, &self.checksum_table);
                    continue;
                }
                update_checksum(&mut checksum, byte, &self.checksum_table);
            }
            for &lace in lacing_slice.iter() {
                update_checksum(&mut checksum, lace, &self.checksum_table);
            }
            for info in self.segment_infos[..segments_count].iter() {
                let start = info.offset;
                let end = start + info.len;
                for &byte in self.payload_buffer[start..end].iter() {
                    update_checksum(&mut checksum, byte, &self.checksum_table);
                }
            }
            let expected = u32::from_le_bytes([header[22], header[23], header[24], header[25]]);
            if checksum != expected {
                return Err(OggReaderError::ChecksumMismatch);
            }
        }

        let page_header = OggPageHeader {
            signature: [header[0], header[1], header[2], header[3]],
            version: header[4],
            header_type: header[5],
            granule_position: u64::from_le_bytes([
                header[6], header[7], header[8], header[9], header[10], header[11], header[12],
                header[13],
            ]),
            serial: u32::from_le_bytes([header[14], header[15], header[16], header[17]]),
            index: u32::from_le_bytes([header[18], header[19], header[20], header[21]]),
            segments_count: header[26],
        };

        let segments = OggPageSegments {
            payload: &self.payload_buffer[..self.payload_len],
            infos: &self.segment_infos[..segments_count],
        };

        Ok((segments, page_header))
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), OggReaderError> {
        read_exact_from(&mut self.stream, buf)?;
        self.bytes_read_successfully += buf.len() as i64;
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

fn update_checksum(checksum: &mut u32, value: u8, table: &[u32; 256]) {
    let index = ((*checksum >> 24) as u8) ^ value;
    *checksum = (*checksum << 8) ^ table[index as usize];
}

fn generate_checksum_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    const POLY: u32 = 0x04c11db7;

    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if (r & 0x8000_0000) != 0 {
                r = (r << 1) ^ POLY;
            } else {
                r <<= 1;
            }
            table[i] = r & 0xffff_ffff;
            j += 1;
        }
        i += 1;
    }

    table
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
        let (mut reader, header) =
            OggReader::new_with(SliceReader::new(&container)).expect("reader should initialize");

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

        let (segments, _) = reader.parse_next_page().expect("second page should parse");
        assert_eq!(segments.len(), 1);
        let expected: &[u8] = &[0x98, 0x36, 0xbe, 0x88, 0x9e];
        assert_eq!(segments.get(0).unwrap(), expected);
    }

    #[test]
    fn parse_next_page_iterates_pages() {
        let container = build_ogg_container();
        let (mut reader, _) =
            OggReader::new_with(SliceReader::new(&container)).expect("reader should initialize");

        let (segments, _) = reader.parse_next_page().expect("should parse comment page");
        assert_eq!(segments.len(), 1);

        let err = reader.parse_next_page();
        assert!(matches!(
            err,
            Err(OggReaderError::Read(ReadError::UnexpectedEof))
        ));
    }

    #[test]
    fn parse_errors() {
        let err = OggReader::<SliceReader<'static>>::new_with_option(None);
        assert!(matches!(err, Err(OggReaderError::NilStream)));

        let mut ogg = build_ogg_container();
        ogg[0] = 0;
        let err = OggReader::new_with_checksum(SliceReader::new(&ogg), false);
        assert!(matches!(err, Err(OggReaderError::BadIdPageSignature)));

        let mut ogg = build_ogg_container();
        ogg[5] = 0;
        let err = OggReader::new_with_checksum(SliceReader::new(&ogg), false);
        assert!(matches!(err, Err(OggReaderError::BadIdPageType)));

        let mut ogg = build_ogg_container();
        ogg[27] = 0;
        let err = OggReader::new_with_checksum(SliceReader::new(&ogg), false);
        assert!(matches!(err, Err(OggReaderError::BadIdPageLength)));

        let mut ogg = build_ogg_container();
        ogg[35] = 0;
        let err = OggReader::new_with_checksum(SliceReader::new(&ogg), false);
        assert!(matches!(
            err,
            Err(OggReaderError::BadIdPagePayloadSignature)
        ));

        let mut ogg = build_ogg_container();
        ogg[22] = 0;
        let err = OggReader::new_with(SliceReader::new(&ogg));
        assert!(matches!(err, Err(OggReaderError::ChecksumMismatch)));
    }
}
