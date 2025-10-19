//! Port of the `silk_sum_sqr_shift` helper from the reference SILK implementation.
//!
//! The routine computes a right-shift count and scaled energy for a slice of
//! 16-bit samples such that the shifted sum of squares fits in a signed 32-bit
//! integer with a couple of guard bits.

/// Computes the energy of `x` while determining how many bits it must be
/// shifted right to fit the accumulator in a 32-bit signed integer.
///
/// Returns `(energy, shift)` just like the original C function writes to its
/// out-parameters.
pub fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    if x.is_empty() {
        return (0, 0);
    }

    let len = x.len() as i32;
    let mut shift = 31 - (len as u32).leading_zeros() as i32;
    let mut energy = len;

    energy = accumulate_energy(energy, x, shift);

    debug_assert!(energy >= 0);

    let clz = if energy == 0 {
        32
    } else {
        (energy as u32).leading_zeros() as i32
    };
    shift = (shift + 3 - clz).max(0);

    let energy = accumulate_energy(0, x, shift);
    debug_assert!(energy >= 0);

    (energy, shift)
}

fn accumulate_energy(initial: i32, x: &[i16], shift: i32) -> i32 {
    let mut acc = initial;
    let mut i = 0;
    while i + 1 < x.len() {
        let pair_energy = pair_square(x[i], x[i + 1]);
        acc = add_rshift(acc, pair_energy, shift);
        i += 2;
    }

    if i < x.len() {
        let tail = square(x[i]);
        acc = add_rshift(acc, tail, shift);
    }

    acc
}

#[inline]
fn pair_square(a: i16, b: i16) -> u32 {
    square(a).wrapping_add(square(b))
}

#[inline]
fn square(a: i16) -> u32 {
    let prod = i32::from(a) * i32::from(a);
    prod as u32
}

fn add_rshift(acc: i32, value: u32, shift: i32) -> i32 {
    if shift <= 0 {
        return acc.wrapping_add(value as i32);
    }

    let shift = shift as u32;
    let shifted = if shift >= 32 { 0 } else { value >> shift };
    acc.wrapping_add(shifted as i32)
}

#[cfg(test)]
mod tests {
    use super::sum_sqr_shift;

    #[test]
    fn matches_reference_for_basic_vectors() {
        let samples = [1, 2, 3, 4];
        assert_eq!(sum_sqr_shift(&samples), (30, 0));

        let samples = [32767, -32768, 12345, -23456];
        assert_eq!(sum_sqr_shift(&samples), (356_250_134, 3));

        let samples = [0i16; 4];
        assert_eq!(sum_sqr_shift(&samples), (0, 0));

        let samples = [30_000i16; 4];
        assert_eq!(sum_sqr_shift(&samples), (450_000_000, 3));

        let samples = [1234i16];
        assert_eq!(sum_sqr_shift(&samples), (1_522_756, 0));

        let mut samples = [0i16; 31];
        for (idx, sample) in samples.iter_mut().enumerate() {
            *sample = if idx % 2 == 0 { -20_000 } else { 20_000 };
        }
        assert_eq!(sum_sqr_shift(&samples), (387_500_000, 5));
    }

    #[test]
    fn empty_slice_returns_zeroes() {
        assert_eq!(sum_sqr_shift(&[]), (0, 0));
    }
}
