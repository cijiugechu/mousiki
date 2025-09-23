/// Utilities for converting audio sample bit depths without heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepthError {
    /// The provided output buffer is smaller than the bytes required for the conversion.
    OutputBufferTooSmall { required: usize, provided: usize },
    /// The required output length overflows `usize` during calculation.
    RequiredSizeOverflow,
}

/// Converts 32-bit floating point samples (little-endian) to 16-bit signed integers
/// encoded in little-endian byte order.
///
/// The `out` buffer must be large enough to hold `in.len() * resample_count * 2` bytes.
/// The conversion uses `floor(sample * 32767.0)` to match the Go implementation.
pub fn convert_float32_le_to_signed16_le(
    input: &[f32],
    output: &mut [u8],
    resample_count: usize,
) -> Result<(), BitDepthError> {
    const SCALE_FACTOR: f32 = 32_767.0;

    let required_samples = input
        .len()
        .checked_mul(resample_count)
        .ok_or(BitDepthError::RequiredSizeOverflow)?;
    let required_bytes = required_samples
        .checked_mul(2)
        .ok_or(BitDepthError::RequiredSizeOverflow)?;

    if output.len() < required_bytes {
        return Err(BitDepthError::OutputBufferTooSmall {
            required: required_bytes,
            provided: output.len(),
        });
    }

    let mut index = 0;
    for &sample in input {
        let scaled = sample * SCALE_FACTOR;
        let floored = floor_to_i32(scaled);
        let converted = floored as i16;
        let bytes = converted.to_le_bytes();

        for _ in 0..resample_count {
            output[index] = bytes[0];
            output[index + 1] = bytes[1];
            index += 2;
        }
    }

    Ok(())
}

fn floor_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if value >= 0.0 || (truncated as f32) == value {
        truncated
    } else {
        truncated - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_float32_le_to_signed16_le_matches_go_reference() {
        let input = [0.3_f32, 0.0, 0.55, 0.72, -0.05];
        let mut output = [0_u8; 10];

        convert_float32_le_to_signed16_le(&input, &mut output, 1).unwrap();
        assert_eq!(
            output,
            [0x66, 0x26, 0x00, 0x00, 0x65, 0x46, 0x28, 0x5c, 0x99, 0xf9]
        );
    }

    #[test]
    fn convert_float32_le_to_signed16_le_checks_buffer_size() {
        let input = [0.3_f32, 0.0];
        let mut output = [0_u8; 3];

        let err = convert_float32_le_to_signed16_le(&input, &mut output, 1).unwrap_err();
        assert_eq!(
            err,
            BitDepthError::OutputBufferTooSmall {
                required: 4,
                provided: 3,
            }
        );
    }
}
