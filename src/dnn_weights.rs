use alloc::vec::Vec;

const WEIGHT_BLOCK_SIZE: usize = 64;
const WEIGHT_NAME_LEN: usize = 44;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightError {
    InvalidBlob,
    MissingArray(&'static str),
    SizeMismatch(&'static str),
    InvalidIndex(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WeightArray<'a> {
    pub name: &'a str,
    pub size: usize,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub(crate) struct WeightBlob<'a> {
    arrays: Vec<WeightArray<'a>>,
}

impl<'a> WeightBlob<'a> {
    pub(crate) fn parse(mut data: &'a [u8]) -> Result<Self, WeightError> {
        let mut arrays = Vec::new();
        while !data.is_empty() {
            if data.len() < WEIGHT_BLOCK_SIZE {
                return Err(WeightError::InvalidBlob);
            }
            let header = &data[..WEIGHT_BLOCK_SIZE];
            let size = read_i32(header, 12)?;
            let block_size = read_i32(header, 16)?;
            if size < 0 || block_size < size {
                return Err(WeightError::InvalidBlob);
            }
            let size = usize::try_from(size).map_err(|_| WeightError::InvalidBlob)?;
            let block_size = usize::try_from(block_size).map_err(|_| WeightError::InvalidBlob)?;
            if block_size > data.len().saturating_sub(WEIGHT_BLOCK_SIZE) {
                return Err(WeightError::InvalidBlob);
            }

            let name_bytes = &header[20..20 + WEIGHT_NAME_LEN];
            if name_bytes[WEIGHT_NAME_LEN - 1] != 0 {
                return Err(WeightError::InvalidBlob);
            }
            let name_len = name_bytes
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(WEIGHT_NAME_LEN);
            let name = core::str::from_utf8(&name_bytes[..name_len])
                .map_err(|_| WeightError::InvalidBlob)?;

            let data_start = WEIGHT_BLOCK_SIZE;
            let data_end = data_start
                .checked_add(size)
                .ok_or(WeightError::InvalidBlob)?;
            if data_end > WEIGHT_BLOCK_SIZE + block_size {
                return Err(WeightError::InvalidBlob);
            }
            let payload = &data[data_start..data_end];
            arrays.push(WeightArray {
                name,
                size,
                data: payload,
            });

            let advance = WEIGHT_BLOCK_SIZE
                .checked_add(block_size)
                .ok_or(WeightError::InvalidBlob)?;
            data = &data[advance..];
        }

        Ok(Self { arrays })
    }

    pub(crate) fn find(&self, name: &str) -> Option<&WeightArray<'a>> {
        self.arrays.iter().find(|array| array.name == name)
    }
}

pub(crate) fn require_bytes<'a>(
    blob: &'a WeightBlob<'a>,
    name: &'static str,
    expected: usize,
) -> Result<&'a [u8], WeightError> {
    let array = blob
        .find(name)
        .ok_or(WeightError::MissingArray(name))?;
    if array.size != expected {
        return Err(WeightError::SizeMismatch(name));
    }
    Ok(array.data)
}

pub(crate) fn optional_bytes<'a>(
    blob: &'a WeightBlob<'a>,
    name: &'static str,
    expected: usize,
) -> Result<Option<&'a [u8]>, WeightError> {
    let Some(array) = blob.find(name) else {
        return Ok(None);
    };
    if array.size != expected {
        return Err(WeightError::SizeMismatch(name));
    }
    Ok(Some(array.data))
}

fn read_i32(data: &[u8], offset: usize) -> Result<i32, WeightError> {
    let end = offset
        .checked_add(4)
        .ok_or(WeightError::InvalidBlob)?;
    if end > data.len() {
        return Err(WeightError::InvalidBlob);
    }
    let bytes: [u8; 4] = data[offset..end]
        .try_into()
        .map_err(|_| WeightError::InvalidBlob)?;
    Ok(i32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{WeightBlob, WeightError, WEIGHT_BLOCK_SIZE, WEIGHT_NAME_LEN};
    use alloc::vec;
    use alloc::vec::Vec;

    fn build_blob(name: &str, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() > 0);
        let mut blob = vec![0u8; WEIGHT_BLOCK_SIZE + payload.len()];
        let size = payload.len() as i32;
        blob[12..16].copy_from_slice(&size.to_le_bytes());
        blob[16..20].copy_from_slice(&size.to_le_bytes());
        let name_bytes = name.as_bytes();
        assert!(name_bytes.len() < WEIGHT_NAME_LEN);
        blob[20..20 + name_bytes.len()].copy_from_slice(name_bytes);
        blob[WEIGHT_BLOCK_SIZE..].copy_from_slice(payload);
        blob
    }

    #[test]
    fn parse_rejects_short_header() {
        let err = WeightBlob::parse(&[0u8; 12]).unwrap_err();
        assert_eq!(err, WeightError::InvalidBlob);
    }

    #[test]
    fn parse_accepts_single_entry() {
        let payload = [1u8, 2, 3, 4];
        let blob = build_blob("test", &payload);
        let parsed = WeightBlob::parse(&blob).expect("parse");
        let entry = parsed.find("test").expect("entry");
        assert_eq!(entry.size, payload.len());
        assert_eq!(entry.data, payload);
    }
}
