//! High-level SILK resampler entry points.
//!
//! Ports `silk/resampler.c`, providing the `silk_resampler_init` and
//! `silk_resampler` functions that pick the appropriate low-level kernels based
//! on the input/output sampling rates, manage the per-call delay buffer, and
//! maintain the fixed-point step sizes used by the hybrid resamplers.

use core::cmp::Ordering;
use core::fmt;

use super::resampler_private_down_fir::{ResamplerStateDownFIR, resampler_private_down_fir};
use super::resampler_private_iir_fir::{ResamplerStateIirFir, resampler_private_iir_fir};
use super::resampler_private_up2_hq::ResamplerStateUp2Hq;
use super::resampler_rom::{
    RESAMPLER_DOWN_ORDER_FIR0, RESAMPLER_DOWN_ORDER_FIR1, RESAMPLER_DOWN_ORDER_FIR2,
    SILK_RESAMPLER_1_2_COEFS, SILK_RESAMPLER_1_3_COEFS, SILK_RESAMPLER_1_4_COEFS,
    SILK_RESAMPLER_1_6_COEFS, SILK_RESAMPLER_2_3_COEFS, SILK_RESAMPLER_3_4_COEFS,
};

const RESAMPLER_MAX_BATCH_SIZE_MS: usize = 10;
const RESAMPLER_MAX_FS_KHZ: usize = 48;
const RESAMPLER_DELAY_BUF_SIZE: usize = RESAMPLER_MAX_FS_KHZ * 2;

/// Errors returned by [`silk_resampler_init`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerInitError {
    /// The requested sampling-rate pair is not supported by the SILK resampler.
    UnsupportedSampleRate,
    /// The combination of input/output rates does not map to a known fractional kernel.
    UnsupportedRatio,
    /// Sampling rates must be integer multiples of 1000 Hz.
    NonIntegralKilohertz,
}

impl fmt::Display for ResamplerInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSampleRate => write!(f, "unsupported sample rate"),
            Self::UnsupportedRatio => write!(f, "unsupported resampling ratio"),
            Self::NonIntegralKilohertz => write!(f, "sampling rate must be a multiple of 1000"),
        }
    }
}

/// High-level SILK resampler state mirroring `silk_resampler_state_struct`.
#[derive(Clone, Debug)]
pub struct Resampler {
    fs_in_khz: usize,
    fs_out_khz: usize,
    batch_size: usize,
    input_delay: usize,
    inv_ratio_q16: i32,
    delay_buf: [i16; RESAMPLER_DELAY_BUF_SIZE],
    kernel: ResamplerKernel,
}

impl Default for Resampler {
    fn default() -> Self {
        Self {
            fs_in_khz: 0,
            fs_out_khz: 0,
            batch_size: 0,
            input_delay: 0,
            inv_ratio_q16: 0,
            delay_buf: [0; RESAMPLER_DELAY_BUF_SIZE],
            kernel: ResamplerKernel::Copy,
        }
    }
}

impl Resampler {
    /// Returns the number of input samples processed per 10 ms batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the configured input sampling rate expressed in kHz.
    pub fn fs_in_khz(&self) -> usize {
        self.fs_in_khz
    }

    /// Returns the configured output sampling rate expressed in kHz.
    pub fn fs_out_khz(&self) -> usize {
        self.fs_out_khz
    }

    /// Returns the number of input samples kept as delay between calls.
    pub fn input_delay(&self) -> usize {
        self.input_delay
    }

    /// Returns the Q16 fixed-point ratio between input and output samples.
    pub fn inv_ratio_q16(&self) -> i32 {
        self.inv_ratio_q16
    }

    /// Returns the active resampler mode for diagnostics and testing.
    pub fn mode(&self) -> ResamplerMode {
        match self.kernel {
            ResamplerKernel::Copy => ResamplerMode::Copy,
            ResamplerKernel::Up2(_) => ResamplerMode::Up2,
            ResamplerKernel::IirFir(_) => ResamplerMode::IirFir,
            ResamplerKernel::DownFIR(_) => ResamplerMode::DownFIR,
        }
    }

    /// Initialise the resampler for the given input and output sampling rates.
    ///
    /// Mirrors `silk_resampler_init` from the reference implementation.
    pub fn silk_resampler_init(
        &mut self,
        fs_hz_in: i32,
        fs_hz_out: i32,
        for_enc: bool,
    ) -> Result<(), ResamplerInitError> {
        if fs_hz_in <= 0
            || fs_hz_out <= 0
            || fs_hz_in > (RESAMPLER_MAX_FS_KHZ as i32) * 1000
            || fs_hz_out > (RESAMPLER_MAX_FS_KHZ as i32) * 1000
        {
            return Err(ResamplerInitError::UnsupportedSampleRate);
        }

        if fs_hz_in % 1000 != 0 || (for_enc && fs_hz_out % 1000 != 0) {
            return Err(ResamplerInitError::NonIntegralKilohertz);
        }

        let input_index = if for_enc {
            ENCODER_INPUT_RATES
                .iter()
                .position(|&rate| rate == fs_hz_in)
        } else {
            DECODER_INPUT_RATES
                .iter()
                .position(|&rate| rate == fs_hz_in)
        }
        .ok_or(ResamplerInitError::UnsupportedSampleRate)?;

        let input_delay = if for_enc {
            let out_idx = ENCODER_OUTPUT_RATES
                .iter()
                .position(|&rate| rate == fs_hz_out)
                .ok_or(ResamplerInitError::UnsupportedSampleRate)?;
            usize::from(DELAY_MATRIX_ENC[input_index][out_idx])
        } else {
            let out_idx = DECODER_OUTPUT_RATES
                .iter()
                .position(|&rate| rate == fs_hz_out);
            out_idx
                .map(|idx| usize::from(DELAY_MATRIX_DEC[input_index][idx]))
                .unwrap_or_else(|| decoder_delay_fallback(fs_hz_in, fs_hz_out))
        };

        let fs_in_khz = (fs_hz_in / 1000) as usize;
        let fs_out_khz = (fs_hz_out / 1000) as usize;
        let batch_size = fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;

        let mode = match fs_hz_out.cmp(&fs_hz_in) {
            Ordering::Greater => {
                if fs_hz_out == fs_hz_in * 2 {
                    ResamplerMode::Up2
                } else {
                    ResamplerMode::IirFir
                }
            }
            Ordering::Less => ResamplerMode::DownFIR,
            Ordering::Equal => ResamplerMode::Copy,
        };

        let up2x = u32::from(matches!(mode, ResamplerMode::IirFir));
        let inv_ratio_q16 = compute_inv_ratio_q16(fs_hz_in, fs_hz_out, up2x)?;

        self.fs_in_khz = fs_in_khz;
        self.fs_out_khz = fs_out_khz;
        self.batch_size = batch_size;
        self.input_delay = input_delay;
        self.inv_ratio_q16 = inv_ratio_q16;
        self.delay_buf = [0; RESAMPLER_DELAY_BUF_SIZE];

        self.kernel = match mode {
            ResamplerMode::Copy => ResamplerKernel::Copy,
            ResamplerMode::Up2 => ResamplerKernel::Up2(ResamplerStateUp2Hq::default()),
            ResamplerMode::IirFir => {
                ResamplerKernel::IirFir(ResamplerStateIirFir::new(batch_size, inv_ratio_q16))
            }
            ResamplerMode::DownFIR => {
                let (fir_fracs, fir_order, coefs) = down_fir_config(fs_hz_in, fs_hz_out)
                    .ok_or(ResamplerInitError::UnsupportedRatio)?;
                ResamplerKernel::DownFIR(ResamplerStateDownFIR::new(
                    batch_size,
                    inv_ratio_q16,
                    fir_order,
                    fir_fracs,
                    coefs,
                ))
            }
        };

        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ResamplerKernel {
    Copy,
    Up2(ResamplerStateUp2Hq),
    IirFir(ResamplerStateIirFir),
    DownFIR(ResamplerStateDownFIR<'static>),
}

/// Logical modes exposed by the high-level resampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerMode {
    Copy,
    Up2,
    IirFir,
    DownFIR,
}

const ENCODER_INPUT_RATES: [i32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];
const ENCODER_OUTPUT_RATES: [i32; 3] = [8_000, 12_000, 16_000];
const DECODER_INPUT_RATES: [i32; 3] = [8_000, 12_000, 16_000];
const DECODER_OUTPUT_RATES: [i32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

const DELAY_MATRIX_ENC: [[u8; ENCODER_OUTPUT_RATES.len()]; ENCODER_INPUT_RATES.len()] =
    [[6, 0, 3], [0, 7, 3], [0, 1, 10], [0, 2, 6], [18, 10, 12]];

const DELAY_MATRIX_DEC: [[u8; DECODER_OUTPUT_RATES.len()]; DECODER_INPUT_RATES.len()] =
    [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];

fn compute_inv_ratio_q16(
    fs_hz_in: i32,
    fs_hz_out: i32,
    up2x: u32,
) -> Result<i32, ResamplerInitError> {
    debug_assert!(fs_hz_in > 0 && fs_hz_out > 0);
    let shift = 16 + up2x;
    let numerator = i64::from(fs_hz_in) << shift;
    let mut inv_ratio_q16 = numerator / i64::from(fs_hz_out);
    while ((inv_ratio_q16 * i64::from(fs_hz_out)) >> 16) < (i64::from(fs_hz_in) << up2x) {
        inv_ratio_q16 += 1;
    }
    i32::try_from(inv_ratio_q16).map_err(|_| ResamplerInitError::UnsupportedRatio)
}

fn down_fir_config(fs_hz_in: i32, fs_hz_out: i32) -> Option<(usize, usize, &'static [i16])> {
    let in64 = i64::from(fs_hz_in);
    let out64 = i64::from(fs_hz_out);
    if out64 * 4 == in64 * 3 {
        Some((3, RESAMPLER_DOWN_ORDER_FIR0, &SILK_RESAMPLER_3_4_COEFS))
    } else if out64 * 3 == in64 * 2 {
        Some((2, RESAMPLER_DOWN_ORDER_FIR0, &SILK_RESAMPLER_2_3_COEFS))
    } else if out64 * 2 == in64 {
        Some((1, RESAMPLER_DOWN_ORDER_FIR1, &SILK_RESAMPLER_1_2_COEFS))
    } else if out64 * 3 == in64 {
        Some((1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_3_COEFS))
    } else if out64 * 4 == in64 {
        Some((1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_4_COEFS))
    } else if out64 * 6 == in64 {
        Some((1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_6_COEFS))
    } else {
        None
    }
}

fn decoder_delay_fallback(fs_hz_in: i32, fs_hz_out: i32) -> usize {
    debug_assert!(fs_hz_in > 0 && fs_hz_out > 0);
    if fs_hz_out >= fs_hz_in {
        0
    } else {
        let fs_in_khz = (fs_hz_in / 1000) as usize;
        fs_in_khz.min(RESAMPLER_DELAY_BUF_SIZE / 2)
    }
}

/// Resample `input` into `output`, returning the number of produced samples.
///
/// Mirrors the behaviour of `silk_resampler` from the C implementation.
pub fn silk_resampler(state: &mut Resampler, output: &mut [i16], input: &[i16]) -> usize {
    assert!(
        state.fs_in_khz > 0 && state.fs_out_khz > 0,
        "resampler not initialised"
    );
    assert!(
        input.len() >= state.fs_in_khz,
        "need at least {} input samples",
        state.fs_in_khz
    );
    assert!(
        state.input_delay <= state.fs_in_khz,
        "input delay {} exceeds {}",
        state.input_delay,
        state.fs_in_khz
    );

    let n_samples = state.fs_in_khz - state.input_delay;
    if n_samples > 0 {
        state.delay_buf[state.input_delay..state.input_delay + n_samples]
            .copy_from_slice(&input[..n_samples]);
    }

    let tail_start = input.len() - state.input_delay;
    let second_input = if tail_start > n_samples {
        &input[n_samples..tail_start]
    } else {
        &[]
    };

    let mut produced = 0usize;
    match &mut state.kernel {
        ResamplerKernel::Copy => {
            assert!(output.len() >= state.fs_in_khz + second_input.len());
            output[..state.fs_in_khz].copy_from_slice(&state.delay_buf[..state.fs_in_khz]);
            produced += state.fs_in_khz;
            if !second_input.is_empty() {
                let len = second_input.len();
                output[produced..produced + len].copy_from_slice(second_input);
                produced += len;
            }
        }
        ResamplerKernel::Up2(inner) => {
            let first_out = state.fs_out_khz;
            assert!(output.len() >= first_out);
            inner.resampler_private_up2_hq_wrapper(
                &mut output[..first_out],
                &state.delay_buf[..state.fs_in_khz],
            );
            produced += first_out;

            if !second_input.is_empty() {
                let required = second_input.len() * 2;
                assert!(output.len() >= produced + required);
                inner.resampler_private_up2_hq_wrapper(
                    &mut output[produced..produced + required],
                    second_input,
                );
                produced += required;
            }
        }
        ResamplerKernel::IirFir(inner) => {
            produced += resampler_private_iir_fir(
                inner,
                &mut output[produced..],
                &state.delay_buf[..state.fs_in_khz],
            );
            if !second_input.is_empty() {
                produced += resampler_private_iir_fir(inner, &mut output[produced..], second_input);
            }
        }
        ResamplerKernel::DownFIR(inner) => {
            produced += resampler_private_down_fir(
                inner,
                &mut output[produced..],
                &state.delay_buf[..state.fs_in_khz],
            );
            if !second_input.is_empty() {
                produced +=
                    resampler_private_down_fir(inner, &mut output[produced..], second_input);
            }
        }
    }

    if state.input_delay > 0 {
        let start = input.len() - state.input_delay;
        state.delay_buf[..state.input_delay].copy_from_slice(&input[start..]);
    }

    produced
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::ResamplerKernel::*;
    use super::*;

    #[test]
    fn rejects_invalid_rates() {
        let mut state = Resampler::default();
        let err = state
            .silk_resampler_init(11_000, 48_000, false)
            .expect_err("11 kHz input should not be accepted");
        assert_eq!(err, ResamplerInitError::UnsupportedSampleRate);
    }

    #[test]
    fn accepts_non_integral_decoder_output_rate() {
        let mut state = Resampler::default();
        state
            .silk_resampler_init(16_000, 44_100, false)
            .expect("decoder path should accept 44.1 kHz");
        assert_eq!(state.mode(), ResamplerMode::IirFir);
        assert_eq!(state.fs_in_khz(), 16);
        assert_eq!(state.fs_out_khz(), 44);
        assert_eq!(state.input_delay(), 0);
    }

    #[test]
    fn encoder_rejects_non_integral_output_rate() {
        let mut state = Resampler::default();
        let err = state
            .silk_resampler_init(16_000, 44_100, true)
            .expect_err("encoder path still requires integral output rate");
        assert_eq!(err, ResamplerInitError::NonIntegralKilohertz);
    }

    #[test]
    fn copy_mode_matches_expected_flow() {
        let mut state = Resampler::default();
        state.silk_resampler_init(16_000, 16_000, false).unwrap();
        assert_eq!(state.mode(), ResamplerMode::Copy);

        let fs_in_khz = state.fs_in_khz();
        let input_delay = state.input_delay();
        let n_samples = fs_in_khz - input_delay;

        let input: Vec<i16> = (0..(fs_in_khz * 5)).map(|x| x as i16).collect();
        let mut output = vec![0i16; input.len()];

        let produced = silk_resampler(&mut state, &mut output, &input);
        assert_eq!(produced, input.len());

        let mut expected = vec![0i16; input.len()];
        expected[..input_delay].fill(0);
        expected[input_delay..fs_in_khz].copy_from_slice(&input[..n_samples]);
        if input.len() > fs_in_khz {
            let tail = input.len() - fs_in_khz;
            expected[fs_in_khz..fs_in_khz + tail]
                .copy_from_slice(&input[n_samples..n_samples + tail]);
        }

        assert_eq!(&output[..produced], &expected[..produced]);
    }

    #[test]
    fn up2_mode_matches_direct_wrapper() {
        let mut state = Resampler::default();
        state.silk_resampler_init(8_000, 16_000, false).unwrap();
        assert_eq!(state.mode(), ResamplerMode::Up2);

        let fs_in_khz = state.fs_in_khz();
        let input_delay = state.input_delay();
        let n_samples = fs_in_khz - input_delay;
        let inv_input_tail = state.input_delay();

        let input: Vec<i16> = (0..(fs_in_khz * 5 * 2)).map(|x| (x as i16) - 20).collect();
        let mut output = vec![0i16; input.len() * 2];

        let mut reference_state = ResamplerStateUp2Hq::default();
        let mut chunk0 = vec![0i16; fs_in_khz];
        if n_samples > 0 {
            chunk0[input_delay..].copy_from_slice(&input[..n_samples]);
        }
        let mut expected = vec![0i16; 0];
        let mut buffer = vec![0i16; fs_in_khz * 2];
        reference_state.resampler_private_up2_hq_wrapper(&mut buffer, &chunk0);
        expected.extend_from_slice(&buffer);

        if input.len() > fs_in_khz {
            let tail_start = input.len() - inv_input_tail;
            let second_input = &input[n_samples..tail_start];
            if !second_input.is_empty() {
                let mut buf = vec![0i16; second_input.len() * 2];
                reference_state.resampler_private_up2_hq_wrapper(&mut buf, second_input);
                expected.extend_from_slice(&buf);
            }
        }

        let produced = silk_resampler(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], &expected[..produced]);
    }

    #[test]
    fn iir_fir_mode_matches_component() {
        let mut state = Resampler::default();
        state.silk_resampler_init(12_000, 48_000, false).unwrap();
        assert_eq!(state.mode(), ResamplerMode::IirFir);

        let fs_in_khz = state.fs_in_khz();
        let input_delay = state.input_delay();
        let n_samples = fs_in_khz - input_delay;
        let tail_len = state.input_delay();

        let input: Vec<i16> = (0..(state.batch_size() + fs_in_khz))
            .map(|x| ((x as i32 * 3) - 200) as i16)
            .collect();
        let mut output = vec![0i16; 4 * input.len()];

        let mut reference = ResamplerStateIirFir::new(state.batch_size(), state.inv_ratio_q16());
        let mut expected = Vec::new();

        let mut chunk0 = vec![0i16; fs_in_khz];
        if n_samples > 0 {
            chunk0[input_delay..].copy_from_slice(&input[..n_samples]);
        }
        let mut buf = vec![0i16; output.len()];
        let produced0 = resampler_private_iir_fir(&mut reference, &mut buf[..], &chunk0);
        expected.extend_from_slice(&buf[..produced0]);

        if input.len() > fs_in_khz {
            let tail_start = input.len() - tail_len;
            let second_input = &input[n_samples..tail_start];
            if !second_input.is_empty() {
                let produced1 =
                    resampler_private_iir_fir(&mut reference, &mut buf[..], second_input);
                expected.extend_from_slice(&buf[..produced1]);
            }
        }

        let produced = silk_resampler(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], &expected[..produced]);
    }

    #[test]
    fn down_fir_mode_matches_component() {
        let mut state = Resampler::default();
        state.silk_resampler_init(16_000, 8_000, false).unwrap();
        assert_eq!(state.mode(), ResamplerMode::DownFIR);

        let fs_in_khz = state.fs_in_khz();
        let input_delay = state.input_delay();
        let n_samples = fs_in_khz - input_delay;
        let tail_len = state.input_delay();

        let input: Vec<i16> = (0..(state.batch_size() + fs_in_khz))
            .map(|x| ((x as i32 * 5) - 512) as i16)
            .collect();
        let mut output = vec![0i16; input.len()];

        let reference = match &state.kernel {
            DownFIR(inner) => ResamplerStateDownFIR::new(
                state.batch_size(),
                state.inv_ratio_q16(),
                inner.fir_order(),
                inner.fir_fracs(),
                inner.coefficients(),
            ),
            _ => unreachable!(),
        };
        let mut reference = reference;
        let mut expected = Vec::new();
        let mut buf = vec![0i16; output.len()];

        let mut chunk0 = vec![0i16; fs_in_khz];
        if n_samples > 0 {
            chunk0[input_delay..].copy_from_slice(&input[..n_samples]);
        }
        let produced0 = resampler_private_down_fir(&mut reference, &mut buf[..], &chunk0);
        expected.extend_from_slice(&buf[..produced0]);

        if input.len() > fs_in_khz {
            let tail_start = input.len() - tail_len;
            let second_input = &input[n_samples..tail_start];
            if !second_input.is_empty() {
                let produced1 =
                    resampler_private_down_fir(&mut reference, &mut buf[..], second_input);
                expected.extend_from_slice(&buf[..produced1]);
            }
        }

        let produced = silk_resampler(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], &expected[..produced]);
    }
}
