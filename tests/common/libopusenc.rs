use std::fs;
use std::path::Path;
use std::vec::Vec;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TestBuffer {
    pub data: Vec<u8>,
}

impl TestBuffer {
    pub fn reset(&mut self) {
        self.data.clear();
    }

    pub fn append(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    pub fn contains(&self, needle: &[u8]) -> bool {
        self.count(needle) > 0
    }

    pub fn count(&self, needle: &[u8]) -> usize {
        if needle.is_empty() || self.data.len() < needle.len() {
            return 0;
        }
        self.data
            .windows(needle.len())
            .filter(|window| *window == needle)
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehaviorPageInfo {
    pub granulepos: u64,
    pub serialno: u32,
    pub pageno: u32,
    pub body_len: usize,
    pub flags: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehaviorPacketInfo {
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorOpusHead {
    pub valid: bool,
    pub version: u8,
    pub channels: u8,
    pub preskip: u16,
    pub input_sample_rate: u32,
    pub gain_raw: u16,
    pub mapping: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorOpusTags {
    pub valid: bool,
    pub vendor: Vec<u8>,
    pub comments: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorManifest {
    pub encoded: Vec<u8>,
    pub pages: Vec<BehaviorPageInfo>,
    pub packets: Vec<BehaviorPacketInfo>,
    pub head: BehaviorOpusHead,
    pub tags: BehaviorOpusTags,
}

impl BehaviorManifest {
    pub fn build(encoded: &[u8]) -> Result<Self, &'static str> {
        let mut manifest = Self {
            encoded: encoded.to_vec(),
            ..Self::default()
        };

        let mut offset = 0usize;
        let mut open_packet = Vec::new();
        let mut have_open_packet = false;
        while offset < encoded.len() {
            let page = ParsedPage::parse(&encoded[offset..])?;
            manifest.pages.push(BehaviorPageInfo {
                granulepos: page.granulepos,
                serialno: page.serialno,
                pageno: page.pageno,
                body_len: page.body_len,
                flags: page.flags,
            });

            if have_open_packet && page.flags & 0x01 == 0 {
                return Err("continued packet flag mismatch");
            }
            if !have_open_packet && page.flags & 0x01 != 0 {
                return Err("unexpected continued packet");
            }

            let mut body_offset = 0usize;
            for &segment_len in page.lacing {
                let end = body_offset + segment_len as usize;
                if end > page.body.len() {
                    return Err("segment past body");
                }
                open_packet.extend_from_slice(&page.body[body_offset..end]);
                body_offset = end;
                have_open_packet = true;
                if segment_len < 255 {
                    manifest.packets.push(BehaviorPacketInfo {
                        data: core::mem::take(&mut open_packet),
                    });
                    have_open_packet = false;
                }
            }

            offset += page.total_len;
        }

        if have_open_packet || manifest.packets.len() < 2 {
            return Err("incomplete packet stream");
        }

        manifest.head = parse_head(&manifest.packets[0].data)?;
        manifest.tags = parse_tags(&manifest.packets[1].data)?;
        Ok(manifest)
    }

    pub fn build_from_file(path: impl AsRef<Path>) -> Result<Self, &'static str> {
        let data = fs::read(path).map_err(|_| "read file failed")?;
        Self::build(&data)
    }
}

struct ParsedPage<'a> {
    flags: u8,
    granulepos: u64,
    serialno: u32,
    pageno: u32,
    lacing: &'a [u8],
    body: &'a [u8],
    body_len: usize,
    total_len: usize,
}

impl<'a> ParsedPage<'a> {
    fn parse(data: &'a [u8]) -> Result<Self, &'static str> {
        if data.len() < 27 || &data[..4] != b"OggS" {
            return Err("invalid ogg page");
        }
        let header_len = 27 + data[26] as usize;
        if header_len > data.len() {
            return Err("short ogg page");
        }
        let body_len = data[27..header_len]
            .iter()
            .map(|&value| value as usize)
            .sum::<usize>();
        if header_len + body_len > data.len() {
            return Err("page body overflow");
        }
        Ok(Self {
            flags: data[5],
            granulepos: u64::from_le_bytes(data[6..14].try_into().unwrap()),
            serialno: u32::from_le_bytes(data[14..18].try_into().unwrap()),
            pageno: u32::from_le_bytes(data[18..22].try_into().unwrap()),
            lacing: &data[27..header_len],
            body: &data[header_len..header_len + body_len],
            body_len,
            total_len: header_len + body_len,
        })
    }
}

fn parse_head(packet: &[u8]) -> Result<BehaviorOpusHead, &'static str> {
    if packet.len() < 19 || &packet[..8] != b"OpusHead" {
        return Err("invalid opus head");
    }
    Ok(BehaviorOpusHead {
        valid: true,
        version: packet[8],
        channels: packet[9],
        preskip: u16::from_le_bytes(packet[10..12].try_into().unwrap()),
        input_sample_rate: u32::from_le_bytes(packet[12..16].try_into().unwrap()),
        gain_raw: u16::from_le_bytes(packet[16..18].try_into().unwrap()),
        mapping: packet[18],
    })
}

fn parse_tags(packet: &[u8]) -> Result<BehaviorOpusTags, &'static str> {
    if packet.len() < 16 || &packet[..8] != b"OpusTags" {
        return Err("invalid opus tags");
    }
    let vendor_length = u32::from_le_bytes(packet[8..12].try_into().unwrap()) as usize;
    let mut offset = 12usize;
    if offset + vendor_length + 4 > packet.len() {
        return Err("short vendor");
    }
    let vendor = packet[offset..offset + vendor_length].to_vec();
    offset += vendor_length;
    let comment_count = u32::from_le_bytes(packet[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut comments = Vec::with_capacity(comment_count);
    for _ in 0..comment_count {
        if offset + 4 > packet.len() {
            return Err("short comment len");
        }
        let len = u32::from_le_bytes(packet[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > packet.len() {
            return Err("short comment");
        }
        comments.push(packet[offset..offset + len].to_vec());
        offset += len;
    }
    Ok(BehaviorOpusTags {
        valid: true,
        vendor,
        comments,
    })
}
