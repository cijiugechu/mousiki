/// Resampling utilities without dynamic allocation.
#[inline]
pub fn up(input: &[f32], output: &mut [f32], upsample_count: usize) {
    let mut index = 0;
    for &sample in input {
        for _ in 0..upsample_count {
            debug_assert!(index < output.len());
            if index >= output.len() {
                return;
            }
            output[index] = sample;
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::up;

    #[test]
    fn upsample_repeats_samples() {
        let input = [0.1_f32, -0.4, 0.8];
        let mut output = [0.0_f32; 9];
        up(&input, &mut output, 3);
        assert_eq!(output, [0.1, 0.1, 0.1, -0.4, -0.4, -0.4, 0.8, 0.8, 0.8]);
    }
}
