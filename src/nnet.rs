//! Neural network helpers for DRED/PLC inference.
//!
//! This is a scalar-only port of the C helpers in `dnn/nnet.c` and `dnn/vec.h`.

use crate::dred_constants::{DRED_MAX_CONV_INPUTS, DRED_MAX_RNN_NEURONS};

pub(crate) const ACTIVATION_LINEAR: i32 = 0;
pub(crate) const ACTIVATION_SIGMOID: i32 = 1;
pub(crate) const ACTIVATION_TANH: i32 = 2;
pub(crate) const ACTIVATION_RELU: i32 = 3;
pub(crate) const ACTIVATION_SOFTMAX: i32 = 4;
pub(crate) const ACTIVATION_SWISH: i32 = 5;

const MAX_INPUTS: usize = 2048;
const SPARSE_BLOCK_SIZE: usize = 32;
const MAX_CONV2D_INPUTS: usize = 8192;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearLayer {
    pub bias: Option<&'static [f32]>,
    #[allow(dead_code)]
    pub subias: Option<&'static [f32]>,
    pub weights: Option<&'static [i8]>,
    pub float_weights: Option<&'static [f32]>,
    pub weights_idx: Option<&'static [i32]>,
    pub diag: Option<&'static [f32]>,
    pub scale: Option<&'static [f32]>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Conv2dLayer {
    pub bias: Option<&'static [f32]>,
    pub float_weights: Option<&'static [f32]>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

#[inline]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[allow(clippy::excessive_precision)]
#[inline]
fn tanh_approx(x: f32) -> f32 {
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
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

fn softmax(output: &mut [f32], input: &[f32]) {
    let mut sum = 0.0f32;
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        let value = libm::expf(src);
        *dst = value;
        sum += value;
    }
    let scale = 1.0 / (sum + 1.0e-30);
    for dst in output.iter_mut() {
        *dst *= scale;
    }
}

fn vec_tanh_inplace(output: &mut [f32]) {
    for value in output.iter_mut() {
        *value = tanh_approx(*value);
    }
}

fn vec_sigmoid_inplace(output: &mut [f32]) {
    for value in output.iter_mut() {
        *value = sigmoid_approx(*value);
    }
}

fn vec_swish_inplace(output: &mut [f32]) {
    let count = output.len();
    let mut tmp = [0.0f32; MAX_INPUTS];
    debug_assert!(count <= tmp.len());
    tmp[..count].copy_from_slice(output);
    vec_sigmoid_inplace(&mut tmp[..count]);
    for idx in 0..count {
        output[idx] = output[idx] * tmp[idx];
    }
}

pub(crate) fn compute_activation(output: &mut [f32], activation: i32) {
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid_inplace(output),
        ACTIVATION_TANH => vec_tanh_inplace(output),
        ACTIVATION_SWISH => vec_swish_inplace(output),
        ACTIVATION_RELU => {
            for dst in output.iter_mut() {
                *dst = (*dst).max(0.0);
            }
        }
        ACTIVATION_SOFTMAX => {
            let count = output.len();
            let mut tmp = [0.0f32; MAX_INPUTS];
            debug_assert!(count <= tmp.len());
            tmp[..count].copy_from_slice(output);
            softmax(output, &tmp[..count]);
        }
        _ => {
            debug_assert_eq!(activation, ACTIVATION_LINEAR);
        }
    }
}

fn conv2d_float(
    output: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        out_row.fill(0.0);
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    let weight_base = ((i * in_channels + m) * ktime + t) * kheight + h;
                    let weight = weights[weight_base];
                    let input_base = (t * in_channels + m) * in_stride + h;
                    for j in 0..height {
                        out_row[j] += weight * input[input_base + j];
                    }
                }
            }
        }
    }
}

fn conv2d_3x3_float(
    output: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let kheight = 3;
    let ktime = 3;
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        out_row.fill(0.0);
        for m in 0..in_channels {
            for j in 0..height {
                let weight_base = (i * in_channels + m) * ktime * kheight;
                let input_base = m * in_stride + j;
                out_row[j] += weights[weight_base + 0] * input[input_base + 0]
                    + weights[weight_base + 1] * input[input_base + 1]
                    + weights[weight_base + 2] * input[input_base + 2]
                    + weights[weight_base + 3] * input[input_base + in_channels * in_stride + 0]
                    + weights[weight_base + 4] * input[input_base + in_channels * in_stride + 1]
                    + weights[weight_base + 5] * input[input_base + in_channels * in_stride + 2]
                    + weights[weight_base + 6] * input[input_base + 2 * in_channels * in_stride + 0]
                    + weights[weight_base + 7] * input[input_base + 2 * in_channels * in_stride + 1]
                    + weights[weight_base + 8] * input[input_base + 2 * in_channels * in_stride + 2];
            }
        }
    }
}

pub(crate) fn compute_conv2d(
    layer: &Conv2dLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
    _arch: i32,
) {
    let Some(weights) = layer.float_weights else {
        output.fill(0.0);
        return;
    };

    let time_stride = layer
        .in_channels
        .checked_mul(height + layer.kheight - 1)
        .expect("conv2d time stride overflow");
    let total_inputs = layer
        .ktime
        .checked_mul(time_stride)
        .expect("conv2d input length overflow");
    debug_assert!(
        total_inputs <= MAX_CONV2D_INPUTS,
        "conv2d input buffer too large"
    );
    let mem_len = (layer.ktime - 1) * time_stride;
    debug_assert!(mem_len <= mem.len());
    debug_assert!(time_stride <= input.len());
    debug_assert!(output.len() >= layer.out_channels * hstride);

    let mut input_buf = [0.0f32; MAX_CONV2D_INPUTS];
    input_buf[..mem_len].copy_from_slice(&mem[..mem_len]);
    input_buf[mem_len..mem_len + time_stride].copy_from_slice(&input[..time_stride]);
    if mem_len > 0 {
        let start = time_stride;
        let end = start + mem_len;
        mem[..mem_len].copy_from_slice(&input_buf[start..end]);
    }

    if layer.kheight == 3 && layer.ktime == 3 {
        conv2d_3x3_float(
            output,
            weights,
            layer.in_channels,
            layer.out_channels,
            &input_buf[..total_inputs],
            height,
            hstride,
        );
    } else {
        conv2d_float(
            output,
            weights,
            layer.in_channels,
            layer.out_channels,
            layer.ktime,
            layer.kheight,
            &input_buf[..total_inputs],
            height,
            hstride,
        );
    }

    if let Some(bias) = layer.bias {
        debug_assert!(bias.len() >= layer.out_channels);
        for i in 0..layer.out_channels {
            let out_row = &mut output[i * hstride..i * hstride + height];
            let value = bias[i];
            for slot in out_row.iter_mut() {
                *slot += value;
            }
        }
    }

    for i in 0..layer.out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        compute_activation(out_row, activation);
    }
}

fn sgemv(out: &mut [f32], weights: &[f32], rows: usize, cols: usize, input: &[f32]) {
    out.fill(0.0);
    for i in 0..rows {
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += weights[j * rows + i] * input[j];
        }
        out[i] = acc;
    }
}

fn sparse_sgemv8x4(
    out: &mut [f32],
    weights: &[f32],
    idx: &[i32],
    rows: usize,
    input: &[f32],
) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);

    let mut w_pos = 0usize;
    let mut idx_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = input[pos];
            let xj1 = input[pos + 1];
            let xj2 = input[pos + 2];
            let xj3 = input[pos + 3];
            let y = &mut out[row..row + 8];
            y[0] += weights[w_pos + 0] * xj0
                + weights[w_pos + 1] * xj1
                + weights[w_pos + 2] * xj2
                + weights[w_pos + 3] * xj3;
            y[1] += weights[w_pos + 4] * xj0
                + weights[w_pos + 5] * xj1
                + weights[w_pos + 6] * xj2
                + weights[w_pos + 7] * xj3;
            y[2] += weights[w_pos + 8] * xj0
                + weights[w_pos + 9] * xj1
                + weights[w_pos + 10] * xj2
                + weights[w_pos + 11] * xj3;
            y[3] += weights[w_pos + 12] * xj0
                + weights[w_pos + 13] * xj1
                + weights[w_pos + 14] * xj2
                + weights[w_pos + 15] * xj3;
            y[4] += weights[w_pos + 16] * xj0
                + weights[w_pos + 17] * xj1
                + weights[w_pos + 18] * xj2
                + weights[w_pos + 19] * xj3;
            y[5] += weights[w_pos + 20] * xj0
                + weights[w_pos + 21] * xj1
                + weights[w_pos + 22] * xj2
                + weights[w_pos + 23] * xj3;
            y[6] += weights[w_pos + 24] * xj0
                + weights[w_pos + 25] * xj1
                + weights[w_pos + 26] * xj2
                + weights[w_pos + 27] * xj3;
            y[7] += weights[w_pos + 28] * xj0
                + weights[w_pos + 29] * xj1
                + weights[w_pos + 30] * xj2
                + weights[w_pos + 31] * xj3;
            w_pos += SPARSE_BLOCK_SIZE;
        }
        row += 8;
    }
}

fn quantize_input(int_buf: &mut [i8], input: &[f32]) {
    debug_assert!(input.len() <= int_buf.len());
    for (dst, &src) in int_buf.iter_mut().zip(input.iter()) {
        let value = libm::floorf(127.0 * src + 0.5);
        let clamped = value.clamp(-128.0, 127.0) as i32;
        *dst = clamped as i8;
    }
}

fn sparse_cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);
    debug_assert!(cols <= MAX_INPUTS);

    let mut x = [0i8; MAX_INPUTS];
    quantize_input(&mut x[..cols], input);

    let mut w_pos = 0usize;
    let mut idx_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = x[pos] as i32;
            let xj1 = x[pos + 1] as i32;
            let xj2 = x[pos + 2] as i32;
            let xj3 = x[pos + 3] as i32;
            let y = &mut out[row..row + 8];
            y[0] += (weights[w_pos + 0] as i32 * xj0
                + weights[w_pos + 1] as i32 * xj1
                + weights[w_pos + 2] as i32 * xj2
                + weights[w_pos + 3] as i32 * xj3) as f32;
            y[1] += (weights[w_pos + 4] as i32 * xj0
                + weights[w_pos + 5] as i32 * xj1
                + weights[w_pos + 6] as i32 * xj2
                + weights[w_pos + 7] as i32 * xj3) as f32;
            y[2] += (weights[w_pos + 8] as i32 * xj0
                + weights[w_pos + 9] as i32 * xj1
                + weights[w_pos + 10] as i32 * xj2
                + weights[w_pos + 11] as i32 * xj3) as f32;
            y[3] += (weights[w_pos + 12] as i32 * xj0
                + weights[w_pos + 13] as i32 * xj1
                + weights[w_pos + 14] as i32 * xj2
                + weights[w_pos + 15] as i32 * xj3) as f32;
            y[4] += (weights[w_pos + 16] as i32 * xj0
                + weights[w_pos + 17] as i32 * xj1
                + weights[w_pos + 18] as i32 * xj2
                + weights[w_pos + 19] as i32 * xj3) as f32;
            y[5] += (weights[w_pos + 20] as i32 * xj0
                + weights[w_pos + 21] as i32 * xj1
                + weights[w_pos + 22] as i32 * xj2
                + weights[w_pos + 23] as i32 * xj3) as f32;
            y[6] += (weights[w_pos + 24] as i32 * xj0
                + weights[w_pos + 25] as i32 * xj1
                + weights[w_pos + 26] as i32 * xj2
                + weights[w_pos + 27] as i32 * xj3) as f32;
            y[7] += (weights[w_pos + 28] as i32 * xj0
                + weights[w_pos + 29] as i32 * xj1
                + weights[w_pos + 30] as i32 * xj2
                + weights[w_pos + 31] as i32 * xj3) as f32;
            w_pos += SPARSE_BLOCK_SIZE;
        }
        row += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

fn cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);
    debug_assert!(cols <= MAX_INPUTS);

    let mut x = [0i8; MAX_INPUTS];
    quantize_input(&mut x[..cols], input);

    let mut w_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let mut col = 0usize;
        while col < cols {
            let xj0 = x[col] as i32;
            let xj1 = x[col + 1] as i32;
            let xj2 = x[col + 2] as i32;
            let xj3 = x[col + 3] as i32;
            let y = &mut out[row..row + 8];
            y[0] += (weights[w_pos + 0] as i32 * xj0
                + weights[w_pos + 1] as i32 * xj1
                + weights[w_pos + 2] as i32 * xj2
                + weights[w_pos + 3] as i32 * xj3) as f32;
            y[1] += (weights[w_pos + 4] as i32 * xj0
                + weights[w_pos + 5] as i32 * xj1
                + weights[w_pos + 6] as i32 * xj2
                + weights[w_pos + 7] as i32 * xj3) as f32;
            y[2] += (weights[w_pos + 8] as i32 * xj0
                + weights[w_pos + 9] as i32 * xj1
                + weights[w_pos + 10] as i32 * xj2
                + weights[w_pos + 11] as i32 * xj3) as f32;
            y[3] += (weights[w_pos + 12] as i32 * xj0
                + weights[w_pos + 13] as i32 * xj1
                + weights[w_pos + 14] as i32 * xj2
                + weights[w_pos + 15] as i32 * xj3) as f32;
            y[4] += (weights[w_pos + 16] as i32 * xj0
                + weights[w_pos + 17] as i32 * xj1
                + weights[w_pos + 18] as i32 * xj2
                + weights[w_pos + 19] as i32 * xj3) as f32;
            y[5] += (weights[w_pos + 20] as i32 * xj0
                + weights[w_pos + 21] as i32 * xj1
                + weights[w_pos + 22] as i32 * xj2
                + weights[w_pos + 23] as i32 * xj3) as f32;
            y[6] += (weights[w_pos + 24] as i32 * xj0
                + weights[w_pos + 25] as i32 * xj1
                + weights[w_pos + 26] as i32 * xj2
                + weights[w_pos + 27] as i32 * xj3) as f32;
            y[7] += (weights[w_pos + 28] as i32 * xj0
                + weights[w_pos + 29] as i32 * xj1
                + weights[w_pos + 30] as i32 * xj2
                + weights[w_pos + 31] as i32 * xj3) as f32;
            w_pos += SPARSE_BLOCK_SIZE;
            col += 4;
        }
        row += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

fn compute_linear(layer: &LinearLayer, out: &mut [f32], input: &[f32]) {
    debug_assert!(input.len() >= layer.nb_inputs);
    debug_assert!(out.len() >= layer.nb_outputs);

    if let Some(float_weights) = layer.float_weights {
        if let Some(weights_idx) = layer.weights_idx {
            sparse_sgemv8x4(out, float_weights, weights_idx, layer.nb_outputs, input);
        } else {
            sgemv(out, float_weights, layer.nb_outputs, layer.nb_inputs, input);
        }
    } else if let Some(weights) = layer.weights {
        let scale = layer
            .scale
            .expect("quantized weights require scale values");
        if let Some(weights_idx) = layer.weights_idx {
            sparse_cgemv8x4(out, weights, weights_idx, scale, layer.nb_outputs, layer.nb_inputs, input);
        } else {
            cgemv8x4(out, weights, scale, layer.nb_outputs, layer.nb_inputs, input);
        }
    } else {
        out.fill(0.0);
    }

    if let Some(bias) = layer.bias {
        for (dst, &value) in out.iter_mut().take(layer.nb_outputs).zip(bias.iter()) {
            *dst += value;
        }
    }

    if let Some(diag) = layer.diag {
        let m = layer.nb_inputs;
        debug_assert_eq!(3 * m, layer.nb_outputs);
        for i in 0..m {
            out[i] += diag[i] * input[i];
            out[i + m] += diag[i + m] * input[i];
            out[i + 2 * m] += diag[i + 2 * m] * input[i];
        }
    }
}

pub(crate) fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
    _arch: i32,
) {
    compute_linear(layer, output, input);
    compute_activation(output, activation);
}

pub(crate) fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
    _arch: i32,
) {
    let n = recurrent_weights.nb_inputs;
    debug_assert_eq!(recurrent_weights.nb_outputs, 3 * n);
    debug_assert_eq!(input_weights.nb_outputs, recurrent_weights.nb_outputs);
    debug_assert!(n <= DRED_MAX_RNN_NEURONS);
    debug_assert!(state.len() >= n);

    let mut zrh = [0.0f32; 3 * DRED_MAX_RNN_NEURONS];
    let mut recur = [0.0f32; 3 * DRED_MAX_RNN_NEURONS];

    compute_linear(input_weights, &mut zrh[..3 * n], input);
    compute_linear(recurrent_weights, &mut recur[..3 * n], state);
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    compute_activation(&mut zrh[..2 * n], ACTIVATION_SIGMOID);
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i];
    }
    compute_activation(&mut zrh[2 * n..2 * n + n], ACTIVATION_TANH);
    for i in 0..n {
        let z = zrh[i];
        let h = zrh[2 * n + i];
        state[i] = z * state[i] + (1.0 - z) * h;
    }
}

pub(crate) fn compute_glu(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    _arch: i32,
) {
    debug_assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let mut act2 = [0.0f32; MAX_INPUTS];
    let count = layer.nb_outputs;
    debug_assert!(count <= act2.len());
    compute_linear(layer, &mut act2[..count], input);
    compute_activation(&mut act2[..count], ACTIVATION_SIGMOID);
    if core::ptr::eq(output.as_ptr(), input.as_ptr()) {
        for i in 0..count {
            output[i] *= act2[i];
        }
    } else {
        for i in 0..count {
            output[i] = input[i] * act2[i];
        }
    }
}

pub(crate) fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: i32,
    _arch: i32,
) {
    let mut tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
    let total_inputs = layer.nb_inputs;
    debug_assert!(total_inputs <= tmp.len());
    debug_assert_eq!(input_size, input.len());

    if total_inputs != input_size {
        let offset = total_inputs - input_size;
        tmp[..offset].copy_from_slice(&mem[..offset]);
        tmp[offset..offset + input_size].copy_from_slice(input);
    } else {
        tmp[..input_size].copy_from_slice(input);
    }

    compute_linear(layer, output, &tmp[..total_inputs]);
    compute_activation(output, activation);

    if total_inputs != input_size {
        let offset = total_inputs - input_size;
        mem[..offset].copy_from_slice(&tmp[input_size..input_size + offset]);
    }
}

pub(crate) fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: i32,
    _arch: i32,
) {
    let mut tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
    let total_inputs = layer.nb_inputs;
    debug_assert!(total_inputs <= tmp.len());
    debug_assert_eq!(input_size, input.len());
    let ksize = total_inputs / input_size;

    if dilation == 1 {
        let offset = total_inputs - input_size;
        tmp[..offset].copy_from_slice(&mem[..offset]);
    } else {
        for i in 0..ksize - 1 {
            let src = i * input_size * dilation;
            tmp[i * input_size..(i + 1) * input_size]
                .copy_from_slice(&mem[src..src + input_size]);
        }
    }

    tmp[total_inputs - input_size..total_inputs].copy_from_slice(input);
    compute_linear(layer, output, &tmp[..total_inputs]);
    compute_activation(output, activation);

    if dilation == 1 {
        let offset = total_inputs - input_size;
        mem[..offset].copy_from_slice(&tmp[input_size..input_size + offset]);
    } else {
        let span = input_size * dilation * (ksize - 1) - input_size;
        mem.copy_within(input_size..input_size + span, 0);
        mem[span..span + input_size].copy_from_slice(input);
    }
}
