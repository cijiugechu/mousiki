#![allow(dead_code)]

use crate::mlp_data::{
    LAYER0_BIAS, LAYER0_WEIGHTS, LAYER1_BIAS, LAYER1_RECUR_WEIGHTS, LAYER1_WEIGHTS, LAYER2_BIAS,
    LAYER2_WEIGHTS,
};
use libm::fmaf;

/// Scaling factor applied to all dense and GRU outputs.
pub(crate) const WEIGHTS_SCALE: f32 = 1.0 / 128.0;

/// Upper bound on the number of neurons handled by the analysis GRU.
pub(crate) const MAX_NEURONS: usize = 32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnalysisDenseLayer {
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub sigmoid: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AnalysisGRULayer {
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub recurrent_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
}

#[inline]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    // Keep fused semantics to match the reference C build's contraction.
    fmaf(a, b, c)
}

#[allow(clippy::excessive_precision)]
#[inline]
fn tansig_approx(x: f32) -> f32 {
    const N0: f32 = 952.528_015_14;
    const N1: f32 = 96.392_356_87;
    const N2: f32 = 0.608_630_42;
    const D0: f32 = 952.723_999_02;
    const D1: f32 = 413.368_011_47;
    const D2: f32 = 11.886_009_22;

    let x2 = x * x;
    let num = fmadd(fmadd(N2, x2, N1), x2, N0);
    let den = fmadd(fmadd(D2, x2, D1), x2, D0);
    let value = num * x / den;
    value.clamp(-1.0, 1.0)
}

#[inline]
fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approx(0.5 * x)
}

fn gemm_accum(
    out: &mut [f32],
    weights: &[i8],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    debug_assert!(out.len() >= rows);
    debug_assert!(x.len() >= cols);
    let required = rows + col_stride.saturating_mul(cols.saturating_sub(1));
    debug_assert!(weights.len() >= required);

    for i in 0..rows {
        let mut acc = out[i];
        for j in 0..cols {
            let weight = f32::from(weights[j * col_stride + i]);
            // Mirror C's multiply-accumulate contraction.
            acc = fmaf(weight, x[j], acc);
        }
        out[i] = acc;
    }
}

pub(crate) fn analysis_compute_dense(
    layer: &AnalysisDenseLayer,
    output: &mut [f32],
    input: &[f32],
) {
    let m = layer.nb_inputs;
    let n = layer.nb_neurons;
    assert!(input.len() >= m, "dense input shorter than expected");
    assert!(output.len() >= n, "dense output shorter than expected");
    assert_eq!(
        layer.bias.len(),
        n,
        "dense bias must contain one entry per neuron"
    );
    assert!(
        layer.input_weights.len() >= n * m,
        "dense weights buffer is too small"
    );

    for (dst, &bias) in output.iter_mut().take(n).zip(layer.bias.iter()) {
        *dst = f32::from(bias);
    }
    gemm_accum(output, layer.input_weights, n, m, n, input);

    if layer.sigmoid {
        for value in output.iter_mut().take(n) {
            *value = sigmoid_approx(*value * WEIGHTS_SCALE);
        }
    } else {
        for value in output.iter_mut().take(n) {
            *value = tansig_approx(*value * WEIGHTS_SCALE);
        }
    }
}

pub(crate) fn analysis_compute_gru(gru: &AnalysisGRULayer, state: &mut [f32], input: &[f32]) {
    let m = gru.nb_inputs;
    let n = gru.nb_neurons;
    assert!(n <= MAX_NEURONS, "GRU exceeds max neuron count");
    assert!(state.len() >= n, "GRU state shorter than expected");
    assert!(input.len() >= m, "GRU input shorter than expected");
    assert_eq!(
        gru.bias.len(),
        3 * n,
        "GRU bias must contain three slices of length `nb_neurons`"
    );
    let stride = 3 * n;
    assert!(
        gru.input_weights.len() >= stride * m,
        "GRU input weights buffer is too small"
    );
    assert!(
        gru.recurrent_weights.len() >= stride * n,
        "GRU recurrent weights buffer is too small"
    );

    let mut tmp = [0.0f32; MAX_NEURONS];
    let mut z = [0.0f32; MAX_NEURONS];
    let mut r = [0.0f32; MAX_NEURONS];
    let mut h = [0.0f32; MAX_NEURONS];

    // Update gate.
    for (dst, &bias) in z.iter_mut().zip(&gru.bias[..n]) {
        *dst = f32::from(bias);
    }
    gemm_accum(&mut z, gru.input_weights, n, m, stride, input);
    gemm_accum(&mut z, gru.recurrent_weights, n, n, stride, state);
    for value in z.iter_mut().take(n) {
        *value = sigmoid_approx(*value * WEIGHTS_SCALE);
    }

    // Reset gate.
    for (dst, &bias) in r.iter_mut().zip(&gru.bias[n..2 * n]) {
        *dst = f32::from(bias);
    }
    gemm_accum(&mut r, &gru.input_weights[n..], n, m, stride, input);
    gemm_accum(&mut r, &gru.recurrent_weights[n..], n, n, stride, state);
    for value in r.iter_mut().take(n) {
        *value = sigmoid_approx(*value * WEIGHTS_SCALE);
    }

    // Candidate output.
    for (dst, &bias) in h.iter_mut().zip(&gru.bias[2 * n..3 * n]) {
        *dst = f32::from(bias);
    }
    for i in 0..n {
        tmp[i] = state[i] * r[i];
    }
    gemm_accum(&mut h, &gru.input_weights[2 * n..], n, m, stride, input);
    gemm_accum(&mut h, &gru.recurrent_weights[2 * n..], n, n, stride, &tmp);
    for i in 0..n {
        h[i] = z[i] * state[i] + (1.0 - z[i]) * tansig_approx(h[i] * WEIGHTS_SCALE);
    }
    state[..n].copy_from_slice(&h[..n]);
}

pub(crate) const LAYER0: AnalysisDenseLayer = AnalysisDenseLayer {
    bias: &LAYER0_BIAS,
    input_weights: &LAYER0_WEIGHTS,
    nb_inputs: 25,
    nb_neurons: 32,
    sigmoid: false,
};

pub(crate) const LAYER1: AnalysisGRULayer = AnalysisGRULayer {
    bias: &LAYER1_BIAS,
    input_weights: &LAYER1_WEIGHTS,
    recurrent_weights: &LAYER1_RECUR_WEIGHTS,
    nb_inputs: 32,
    nb_neurons: 24,
};

pub(crate) const LAYER2: AnalysisDenseLayer = AnalysisDenseLayer {
    bias: &LAYER2_BIAS,
    input_weights: &LAYER2_WEIGHTS,
    nb_inputs: 24,
    nb_neurons: 2,
    sigmoid: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layer_matches_manual_product() {
        let layer = AnalysisDenseLayer {
            bias: &[0, 0],
            input_weights: &[1, -1, 2, -2],
            nb_inputs: 2,
            nb_neurons: 2,
            sigmoid: false,
        };
        let mut output = [0.0f32; 2];
        let input = [1.0f32, -1.0];

        analysis_compute_dense(&layer, &mut output, &input);

        let expected0 = tansig_approx((1.0 - 2.0) * WEIGHTS_SCALE);
        let expected1 = tansig_approx((-1.0 + 2.0) * WEIGHTS_SCALE);
        assert!((output[0] - expected0).abs() < 1e-6);
        assert!((output[1] - expected1).abs() < 1e-6);
    }

    #[test]
    fn gru_updates_state_with_expected_shape() {
        // Two-neuron GRU with a single input; only the candidate gate is biased.
        const BIAS: [i8; 6] = [0, 0, 0, 0, 64, 64];
        const WEIGHTS: [i8; 6] = [1, 2, 0, 0, 0, 0];
        const RECUR: [i8; 12] = [0; 12];
        let gru = AnalysisGRULayer {
            bias: &BIAS,
            input_weights: &WEIGHTS,
            recurrent_weights: &RECUR,
            nb_inputs: 1,
            nb_neurons: 2,
        };
        let mut state = [0.0f32; 2];
        let input = [1.0f32];

        analysis_compute_gru(&gru, &mut state, &input);

        // Larger update weight on the second neuron damps the biased candidate
        // a bit more than the first.
        assert!(state[0] > 0.0);
        assert!(state[1] > 0.0);
        assert!(state[0] > state[1]);
    }

    #[test]
    fn fmadd_uses_fma_semantics() {
        let mut seed: u32 = 0x9e37_79b9;
        let mut found = false;
        for _ in 0..200_000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let a_bits = (seed & 0x007f_ffff) | 0x3f00_0000;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let b_bits = (seed & 0x007f_ffff) | 0x3f80_0000;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let c_bits = (seed & 0x007f_ffff) | 0x3f00_0000;

            let a = f32::from_bits(a_bits);
            let b = f32::from_bits(b_bits);
            let c = f32::from_bits(c_bits);
            let fused = fmaf(a, b, c);
            let unfused = a * b + c;
            if fused.to_bits() != unfused.to_bits() {
                assert_eq!(fmadd(a, b, c).to_bits(), fused.to_bits());
                found = true;
                break;
            }
        }
        assert!(found, "expected to find FMA-sensitive inputs");
    }
}
