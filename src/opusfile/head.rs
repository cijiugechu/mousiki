use crate::opusfile::OpusfileError;
use crate::opusfile::util::{add_granule_position, parse_i16_le, parse_u16_le, parse_u32_le};

pub const OPUS_CHANNEL_COUNT_MAX: usize = 255;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    pub version: u8,
    pub channel_count: u8,
    pub pre_skip: u16,
    pub input_sample_rate: u32,
    pub output_gain: i16,
    pub mapping_family: u8,
    pub stream_count: u8,
    pub coupled_count: u8,
    pub mapping: [u8; OPUS_CHANNEL_COUNT_MAX],
}

impl OpusHead {
    pub fn parse(data: &[u8]) -> Result<Self, OpusfileError> {
        if data.len() < 8 {
            return Err(OpusfileError::NotFormat);
        }
        if &data[..8] != b"OpusHead" {
            return Err(OpusfileError::NotFormat);
        }
        if data.len() < 9 {
            return Err(OpusfileError::BadHeader);
        }
        let version = data[8];
        if version > 15 {
            return Err(OpusfileError::Version);
        }
        if data.len() < 19 {
            return Err(OpusfileError::BadHeader);
        }

        let channel_count = data[9];
        let pre_skip = parse_u16_le(&data[10..12]);
        let input_sample_rate = parse_u32_le(&data[12..16]);
        let output_gain = parse_i16_le(&data[16..18]);
        let mapping_family = data[18];
        let mut head = Self {
            version,
            channel_count,
            pre_skip,
            input_sample_rate,
            output_gain,
            mapping_family,
            stream_count: 0,
            coupled_count: 0,
            mapping: [0; OPUS_CHANNEL_COUNT_MAX],
        };

        match mapping_family {
            0 => {
                if !(1..=2).contains(&channel_count) {
                    return Err(OpusfileError::BadHeader);
                }
                if version <= 1 && data.len() > 19 {
                    return Err(OpusfileError::BadHeader);
                }
                head.stream_count = 1;
                head.coupled_count = channel_count - 1;
                head.mapping[0] = 0;
                head.mapping[1] = 1;
            }
            1 => {
                if !(1..=8).contains(&channel_count) {
                    return Err(OpusfileError::BadHeader);
                }
                let size = 21 + usize::from(channel_count);
                if data.len() < size || (version <= 1 && data.len() > size) {
                    return Err(OpusfileError::BadHeader);
                }
                let stream_count = data[19];
                if stream_count < 1 {
                    return Err(OpusfileError::BadHeader);
                }
                let coupled_count = data[20];
                if coupled_count > stream_count {
                    return Err(OpusfileError::BadHeader);
                }
                for &mapping in &data[21..size] {
                    if mapping >= stream_count.saturating_add(coupled_count) && mapping != 255 {
                        return Err(OpusfileError::BadHeader);
                    }
                }
                head.stream_count = stream_count;
                head.coupled_count = coupled_count;
                head.mapping[..usize::from(channel_count)].copy_from_slice(&data[21..size]);
            }
            255 => return Err(OpusfileError::Unimplemented),
            _ => return Err(OpusfileError::BadHeader),
        }

        Ok(head)
    }

    #[must_use]
    pub fn granule_sample(&self, granule_position: i64) -> Option<i64> {
        if granule_position == -1 {
            return None;
        }
        add_granule_position(granule_position, -(i32::from(self.pre_skip)))
    }
}

#[cfg(test)]
mod tests {
    use super::{OPUS_CHANNEL_COUNT_MAX, OpusHead};
    use crate::opusfile::OpusfileError;

    #[test]
    fn parses_mapping_zero_header() {
        let header = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 2, 0x38, 0x01, 0x80, 0xbb, 0, 0,
            0, 0, 0,
        ];
        let parsed = OpusHead::parse(&header).expect("valid header");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.channel_count, 2);
        assert_eq!(parsed.pre_skip, 312);
        assert_eq!(parsed.input_sample_rate, 48_000);
        assert_eq!(parsed.output_gain, 0);
        assert_eq!(parsed.mapping_family, 0);
        assert_eq!(parsed.stream_count, 1);
        assert_eq!(parsed.coupled_count, 1);
        assert_eq!(parsed.mapping[0], 0);
        assert_eq!(parsed.mapping[1], 1);
        assert_eq!(parsed.mapping.len(), OPUS_CHANNEL_COUNT_MAX);
    }

    #[test]
    fn parses_mapping_family_one_header() {
        let header = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 3, 0x80, 0x00, 0x44, 0xac, 0, 0,
            0, 0, 1, 2, 1, 0, 2, 1,
        ];
        let parsed = OpusHead::parse(&header).expect("valid mapped header");
        assert_eq!(parsed.channel_count, 3);
        assert_eq!(parsed.stream_count, 2);
        assert_eq!(parsed.coupled_count, 1);
        assert_eq!(&parsed.mapping[..3], &[0, 2, 1]);
    }

    #[test]
    fn rejects_unsupported_mapping_family_255() {
        let header = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 255,
        ];
        assert_eq!(
            OpusHead::parse(&header),
            Err(OpusfileError::Unimplemented)
        );
    }

    #[test]
    fn rejects_bad_mapping_index() {
        let header = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 3, 1,
        ];
        assert_eq!(OpusHead::parse(&header), Err(OpusfileError::BadHeader));
    }

    #[test]
    fn granule_sample_applies_preskip() {
        let head = OpusHead::parse(&[
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 2, 0x38, 0x01, 0, 0, 0, 0, 0, 0,
            0,
        ])
        .expect("valid header");
        assert_eq!(head.granule_sample(1000), Some(688));
        assert_eq!(head.granule_sample(100), None);
        assert_eq!(head.granule_sample(-1), None);
    }
}
