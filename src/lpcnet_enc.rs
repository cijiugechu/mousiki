use crate::celt::{celt_fir, celt_inner_prod, celt_log2, celt_pitch_xcorr, KissFftCpx, KissFftState};
use crate::dnn_weights::WeightError;
use crate::pitchdnn::{
    compute_pitchdnn, PitchDnnState, NB_XCORR_FEATURES, PITCH_IF_FEATURES, PITCH_IF_MAX_FREQ,
    PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
};
use core::f32::consts::PI;
use libm::{cosf, expf, floorf, logf, powf, sinf, sqrtf};

const NB_TOTAL_FEATURES: usize = 36;
const LPC_ORDER: usize = 16;
const PREEMPHASIS: f32 = 0.85;

const FRAME_SIZE_5MS: usize = 2;
const OVERLAP_SIZE_5MS: usize = 2;
const TRAINING_OFFSET_5MS: usize = 1;
const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;

const FRAME_SIZE: usize = 80 * FRAME_SIZE_5MS;
const OVERLAP_SIZE: usize = 80 * OVERLAP_SIZE_5MS;
const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;
const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;

const NB_BANDS: usize = 18;

const PITCH_FRAME_SIZE: usize = 320;
const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;

const LPC_COMPENSATION: [f32; NB_BANDS] = [
    0.8,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.666_667,
    0.5,
    0.5,
    0.5,
    0.333_333,
    0.25,
    0.25,
    0.2,
    0.166_667,
    0.173_913,
];

const EBAND_5MS: [i16; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

const LP_B: [f32; 2] = [-0.84946, 1.0];
const LP_A: [f32; 2] = [-1.54220, 0.70781];

#[derive(Clone, Debug)]
pub(crate) struct LpcNetEncState {
    pitchdnn: PitchDnnState,
    analysis_mem: [f32; OVERLAP_SIZE],
    mem_preemph: f32,
    prev_if: [KissFftCpx; PITCH_IF_MAX_FREQ],
    if_features: [f32; PITCH_IF_FEATURES],
    xcorr_features: [f32; NB_XCORR_FEATURES],
    dnn_pitch: f32,
    pitch_mem: [f32; LPC_ORDER],
    pitch_filt: f32,
    exc_buf: [f32; PITCH_BUF_SIZE],
    lp_buf: [f32; PITCH_BUF_SIZE],
    lp_mem: [f32; 2],
    lpc: [f32; LPC_ORDER],
    features: [f32; NB_TOTAL_FEATURES],
    kfft: KissFftState,
    half_window: [f32; OVERLAP_SIZE],
    dct_table: [f32; NB_BANDS * NB_BANDS],
}

impl Default for LpcNetEncState {
    fn default() -> Self {
        let mut state = Self {
            pitchdnn: PitchDnnState::default(),
            analysis_mem: [0.0; OVERLAP_SIZE],
            mem_preemph: 0.0,
            prev_if: [KissFftCpx::default(); PITCH_IF_MAX_FREQ],
            if_features: [0.0; PITCH_IF_FEATURES],
            xcorr_features: [0.0; NB_XCORR_FEATURES],
            dnn_pitch: 0.0,
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            exc_buf: [0.0; PITCH_BUF_SIZE],
            lp_buf: [0.0; PITCH_BUF_SIZE],
            lp_mem: [0.0; 2],
            lpc: [0.0; LPC_ORDER],
            features: [0.0; NB_TOTAL_FEATURES],
            kfft: KissFftState::new(WINDOW_SIZE),
            half_window: [0.0; OVERLAP_SIZE],
            dct_table: [0.0; NB_BANDS * NB_BANDS],
        };
        init_tables(&mut state.half_window, &mut state.dct_table);
        state
    }
}

#[allow(dead_code)]
pub(crate) fn lpcnet_encoder_get_size() -> usize {
    core::mem::size_of::<LpcNetEncState>()
}

pub(crate) fn lpcnet_encoder_init(state: &mut LpcNetEncState) -> i32 {
    *state = LpcNetEncState::default();
    0
}

pub(crate) fn lpcnet_encoder_load_model(
    state: &mut LpcNetEncState,
    data: &[u8],
) -> Result<(), WeightError> {
    state.pitchdnn.load_model(data)
}

#[allow(dead_code)]
pub(crate) fn lpcnet_compute_single_frame_features(
    state: &mut LpcNetEncState,
    pcm: &[i16],
    features: &mut [f32],
    arch: i32,
) -> i32 {
    if pcm.len() < FRAME_SIZE || features.len() < NB_TOTAL_FEATURES {
        return -1;
    }
    let mut x = [0.0f32; FRAME_SIZE];
    for (dst, &src) in x.iter_mut().zip(pcm.iter()) {
        *dst = src as f32;
    }
    lpcnet_compute_single_frame_features_impl(state, &mut x, features, arch)
}

pub(crate) fn lpcnet_compute_single_frame_features_float(
    state: &mut LpcNetEncState,
    pcm: &[f32],
    features: &mut [f32],
    arch: i32,
) -> i32 {
    if pcm.len() < FRAME_SIZE || features.len() < NB_TOTAL_FEATURES {
        return -1;
    }
    let mut x = [0.0f32; FRAME_SIZE];
    x.copy_from_slice(&pcm[..FRAME_SIZE]);
    lpcnet_compute_single_frame_features_impl(state, &mut x, features, arch)
}

fn lpcnet_compute_single_frame_features_impl(
    state: &mut LpcNetEncState,
    x: &mut [f32; FRAME_SIZE],
    features: &mut [f32],
    arch: i32,
) -> i32 {
    preemphasis(x, &mut state.mem_preemph, PREEMPHASIS);
    compute_frame_features(state, x, arch);
    features[..NB_TOTAL_FEATURES].copy_from_slice(&state.features[..NB_TOTAL_FEATURES]);
    0
}

fn init_tables(half_window: &mut [f32; OVERLAP_SIZE], dct_table: &mut [f32; NB_BANDS * NB_BANDS]) {
    let half_pi = 0.5 * PI;
    let overlap = OVERLAP_SIZE as f32;
    for i in 0..OVERLAP_SIZE {
        let x = half_pi * (i as f32 + 0.5) / overlap;
        let s = sinf(x);
        half_window[i] = sinf(half_pi * s * s);
    }

    let nb_bands = NB_BANDS as f32;
    let scale = sqrtf(0.5);
    for i in 0..NB_BANDS {
        for j in 0..NB_BANDS {
            let mut value = cosf((i as f32 + 0.5) * j as f32 * PI / nb_bands);
            if j == 0 {
                value *= scale;
            }
            dct_table[i * NB_BANDS + j] = value;
        }
    }
}

fn celt_log10(x: f32) -> f32 {
    core::f32::consts::LOG10_2 * celt_log2(x)
}

fn frame_analysis(
    state: &mut LpcNetEncState,
    xfreq: &mut [KissFftCpx; FREQ_SIZE],
    band_energy: &mut [f32; NB_BANDS],
    input: &[f32; FRAME_SIZE],
) {
    let mut x = [0.0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&state.analysis_mem);
    x[OVERLAP_SIZE..OVERLAP_SIZE + FRAME_SIZE].copy_from_slice(input);
    state.analysis_mem.copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..]);
    apply_window(&mut x, &state.half_window);
    forward_transform(&state.kfft, xfreq, &x);
    lpcn_compute_band_energy(band_energy, xfreq);
}

fn apply_window(x: &mut [f32; WINDOW_SIZE], half_window: &[f32; OVERLAP_SIZE]) {
    for i in 0..OVERLAP_SIZE {
        let value = half_window[i];
        x[i] *= value;
        let mirror = WINDOW_SIZE - 1 - i;
        x[mirror] *= value;
    }
}

fn forward_transform(
    fft: &KissFftState,
    out: &mut [KissFftCpx; FREQ_SIZE],
    input: &[f32; WINDOW_SIZE],
) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    for (dst, &src) in x.iter_mut().zip(input.iter()) {
        dst.r = src;
        dst.i = 0.0;
    }
    fft.fft(&x, &mut y);
    out.copy_from_slice(&y[..FREQ_SIZE]);
}

fn inverse_transform(fft: &KissFftState, out: &mut [f32; WINDOW_SIZE], input: &[KissFftCpx; FREQ_SIZE]) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    for i in 0..FREQ_SIZE {
        x[i] = input[i];
    }
    for i in FREQ_SIZE..WINDOW_SIZE {
        let mirror = WINDOW_SIZE - i;
        x[i].r = x[mirror].r;
        x[i].i = -x[mirror].i;
    }
    fft.fft(&x, &mut y);
    out[0] = WINDOW_SIZE as f32 * y[0].r;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
    }
}

fn lpcn_compute_band_energy(band_energy: &mut [f32; NB_BANDS], xfreq: &[KissFftCpx; FREQ_SIZE]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size =
            (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize * WINDOW_SIZE_5MS;
        let band_start = EBAND_5MS[i] as usize * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = band_start + j;
            let value = xfreq[idx].r * xfreq[idx].r + xfreq[idx].i * xfreq[idx].i;
            sum[i] += (1.0 - frac) * value;
            sum[i + 1] += frac * value;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_energy.copy_from_slice(&sum);
}

fn dct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS], dct_table: &[f32; NB_BANDS * NB_BANDS]) {
    let scale = sqrtf(2.0 / NB_BANDS as f32);
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * dct_table[j * NB_BANDS + i];
        }
        out[i] = sum * scale;
    }
}

fn idct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS], dct_table: &[f32; NB_BANDS * NB_BANDS]) {
    let scale = sqrtf(2.0 / NB_BANDS as f32);
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * dct_table[i * NB_BANDS + j];
        }
        out[i] = sum * scale;
    }
}

fn interp_band_gain(output: &mut [f32; FREQ_SIZE], band_energy: &[f32; NB_BANDS]) {
    output.fill(0.0);
    for i in 0..NB_BANDS - 1 {
        let band_size =
            (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize * WINDOW_SIZE_5MS;
        let band_start = EBAND_5MS[i] as usize * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = band_start + j;
            output[idx] = (1.0 - frac) * band_energy[i] + frac * band_energy[i + 1];
        }
    }
}

fn lpc_from_bands(
    lpc: &mut [f32; LPC_ORDER],
    band_energy: &[f32; NB_BANDS],
    fft: &KissFftState,
) -> f32 {
    let mut xr = [0.0f32; FREQ_SIZE];
    let mut x_auto = [0.0f32; WINDOW_SIZE];
    let mut ac = [0.0f32; LPC_ORDER + 1];
    let mut x_auto_freq = [KissFftCpx::default(); FREQ_SIZE];

    interp_band_gain(&mut xr, band_energy);
    xr[FREQ_SIZE - 1] = 0.0;
    for i in 0..FREQ_SIZE {
        x_auto_freq[i].r = xr[i];
        x_auto_freq[i].i = 0.0;
    }
    inverse_transform(fft, &mut x_auto, &x_auto_freq);
    ac.copy_from_slice(&x_auto[..LPC_ORDER + 1]);

    ac[0] += ac[0] * 1.0e-4 + 320.0 / 12.0 / 38.0;
    for i in 1..=LPC_ORDER {
        let ii = i as f32;
        ac[i] *= 1.0 - 6.0e-5 * ii * ii;
    }

    lpcn_lpc(lpc, &ac)
}

fn lpcn_lpc(lpc: &mut [f32; LPC_ORDER], ac: &[f32; LPC_ORDER + 1]) -> f32 {
    lpc.fill(0.0);
    if ac[0] == 0.0 {
        return 0.0;
    }

    let mut error = ac[0];
    for i in 0..LPC_ORDER {
        let mut rr = 0.0;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        lpc[i] = r;
        let half = (i + 1) / 2;
        for j in 0..half {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= r * r * error;
        if error < 0.001 * ac[0] {
            break;
        }
    }
    error
}

fn lpc_from_cepstrum(
    lpc: &mut [f32; LPC_ORDER],
    cepstrum: &[f32; NB_BANDS],
    fft: &KissFftState,
    dct_table: &[f32; NB_BANDS * NB_BANDS],
) {
    let mut tmp = *cepstrum;
    let mut ex = [0.0f32; NB_BANDS];
    tmp[0] += 4.0;
    idct(&mut ex, &tmp, dct_table);
    for i in 0..NB_BANDS {
        ex[i] = powf(10.0, ex[i]) * LPC_COMPENSATION[i];
    }
    let _ = lpc_from_bands(lpc, &ex, fft);
}

fn biquad(y: &mut [f32], mem: &mut [f32; 2]) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for yi in y.iter_mut() {
        let xi = *yi;
        let y0 = xi + mem0;
        let mem00 = mem0;
        mem0 = (LP_B[0] - LP_A[0]) * xi + mem1 - LP_A[0] * mem0;
        mem1 = (LP_B[1] - LP_A[1]) * xi + 1.0e-30 - LP_A[1] * mem00;
        *yi = y0;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

fn preemphasis(x: &mut [f32], mem: &mut f32, coef: f32) {
    for value in x.iter_mut() {
        let yi = *value + *mem;
        *mem = -coef * *value;
        *value = yi;
    }
}

fn compute_frame_features(state: &mut LpcNetEncState, input: &[f32; FRAME_SIZE], arch: i32) {
    let mut aligned_in = [0.0f32; FRAME_SIZE];
    let mut ly = [0.0f32; NB_BANDS];
    let mut xfreq = [KissFftCpx::default(); FREQ_SIZE];
    let mut ex = [0.0f32; NB_BANDS];
    let mut xcorr = [0.0f32; NB_XCORR_FEATURES];
    let mut ener_norm = [0.0f32; NB_XCORR_FEATURES];
    let mut x = [0.0f32; FRAME_SIZE + LPC_ORDER];

    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&state.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..]);
    frame_analysis(state, &mut xfreq, &mut ex, input);

    let mag0 = xfreq[0].r * xfreq[0].r;
    state.if_features[0] = ((10.0 * celt_log10(1.0e-15 + mag0) - 6.0) / 64.0)
        .clamp(-1.0, 1.0);
    for i in 1..PITCH_IF_MAX_FREQ {
        let prev = state.prev_if[i];
        let curr = xfreq[i];
        let prod = KissFftCpx::new(
            curr.r * prev.r + curr.i * prev.i,
            curr.i * prev.r - curr.r * prev.i,
        );
        let norm = 1.0 / sqrtf(1.0e-15 + prod.r * prod.r + prod.i * prod.i);
        let prod = KissFftCpx::new(prod.r * norm, prod.i * norm);
        state.if_features[3 * i - 2] = prod.r;
        state.if_features[3 * i - 1] = prod.i;
        let mag = curr.r * curr.r + curr.i * curr.i;
        state.if_features[3 * i] = ((10.0 * celt_log10(1.0e-15 + mag) - 6.0) / 64.0)
            .clamp(-1.0, 1.0);
    }
    state.prev_if.copy_from_slice(&xfreq[..PITCH_IF_MAX_FREQ]);

    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        let mut value = celt_log10(1.0e-2 + ex[i]);
        value = value.max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(value);
        follow = (follow - 2.5).max(value);
        ly[i] = value;
    }

    dct(
        (&mut state.features[..NB_BANDS]).try_into().expect("slice length"),
        &ly,
        &state.dct_table,
    );
    state.features[0] -= 4.0;
    lpc_from_cepstrum(
        &mut state.lpc,
        (&state.features[..NB_BANDS]).try_into().expect("slice length"),
        &state.kfft,
        &state.dct_table,
    );
    for i in 0..LPC_ORDER {
        state.features[NB_BANDS + 2 + i] = state.lpc[i];
    }

    state
        .exc_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);
    state
        .lp_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);

    aligned_in[TRAINING_OFFSET..].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);
    x[..LPC_ORDER].copy_from_slice(&state.pitch_mem);
    x[LPC_ORDER..LPC_ORDER + FRAME_SIZE].copy_from_slice(&aligned_in);
    state.pitch_mem.copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..]);

    let lp_out = &mut state.lp_buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE];
    celt_fir(&x, &state.lpc, lp_out);
    for i in 0..FRAME_SIZE {
        let value = lp_out[i];
        state.exc_buf[PITCH_MAX_PERIOD + i] = value + 0.7 * state.pitch_filt;
        state.pitch_filt = value;
    }
    biquad(lp_out, &mut state.lp_mem);

    let buf = &state.exc_buf;
    let x_slice = &buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE];
    celt_pitch_xcorr(x_slice, buf, FRAME_SIZE, NB_XCORR_FEATURES, &mut xcorr);
    let ener0 = celt_inner_prod(x_slice, x_slice);
    let mut ener1 = celt_inner_prod(&buf[..FRAME_SIZE], &buf[..FRAME_SIZE]);
    for i in 0..NB_XCORR_FEATURES {
        let ener = 1.0 + ener0 + ener1;
        state.xcorr_features[i] = 2.0 * xcorr[i];
        ener_norm[i] = ener;
        let next = i + FRAME_SIZE;
        ener1 += buf[next] * buf[next] - buf[i] * buf[i];
    }
    for i in 0..NB_XCORR_FEATURES {
        state.xcorr_features[i] /= ener_norm[i];
    }

    state.dnn_pitch = compute_pitchdnn(
        &mut state.pitchdnn,
        &state.if_features,
        &state.xcorr_features,
        arch,
    );
    let pitch = floorf(0.5 + 256.0 / powf(2.0, state.dnn_pitch + 1.5)) as i32;
    let pitch = pitch.clamp(PITCH_MIN_PERIOD as i32, PITCH_MAX_PERIOD as i32) as usize;
    let base = PITCH_MAX_PERIOD;
    let pitch_base = PITCH_MAX_PERIOD - pitch;
    let lp_current = &state.lp_buf[base..base + FRAME_SIZE];
    let lp_delayed = &state.lp_buf[pitch_base..pitch_base + FRAME_SIZE];
    let xx = celt_inner_prod(lp_current, lp_current);
    let yy = celt_inner_prod(lp_delayed, lp_delayed);
    let xy = celt_inner_prod(lp_current, lp_delayed);
    let mut frame_corr = xy / sqrtf(1.0 + xx * yy);
    frame_corr = logf(1.0 + expf(5.0 * frame_corr)) / logf(1.0 + expf(5.0));

    state.features[NB_BANDS] = state.dnn_pitch;
    state.features[NB_BANDS + 1] = frame_corr - 0.5;
}

#[cfg(test)]
mod tests {
    use super::{lpcnet_compute_single_frame_features_float, LpcNetEncState, FRAME_SIZE, NB_TOTAL_FEATURES};

    #[test]
    fn lpcnet_features_are_finite_for_silence() {
        let mut state = LpcNetEncState::default();
        let mut features = [0.0f32; NB_TOTAL_FEATURES];
        let pcm = [0.0f32; FRAME_SIZE];
        let ret = lpcnet_compute_single_frame_features_float(&mut state, &pcm, &mut features, 0);
        assert_eq!(ret, 0);
        assert!(features.iter().all(|v| v.is_finite()));
    }
}
