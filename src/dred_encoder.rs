use crate::celt::{ec_tell, float2int16, EcEnc, EcEncSnapshot};
use crate::dred_constants::{
    DRED_DFRAME_SIZE, DRED_FRAME_SIZE, DRED_LATENT_DIM, DRED_MAX_FRAMES, DRED_NUM_FEATURES,
    DRED_NUM_REDUNDANCY_FRAMES, DRED_SILK_ENCODER_DELAY, DRED_STATE_DIM,
};
use crate::dred_rdovae_enc::{dred_rdovae_encode_dframe, RdovaeEnc, RdovaeEncState};
use crate::dred_stats_data::{
    DRED_LATENT_DEAD_ZONE_Q8, DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8, DRED_LATENT_R_Q8,
    DRED_STATE_DEAD_ZONE_Q8, DRED_STATE_P0_Q8, DRED_STATE_QUANT_SCALES_Q8, DRED_STATE_R_Q8,
};
use crate::dnn_weights::WeightError;
use crate::lpcnet_enc::{
    lpcnet_compute_single_frame_features_float, lpcnet_encoder_init, lpcnet_encoder_load_model,
    LpcNetEncState,
};
use crate::nnet::{compute_activation, ACTIVATION_TANH};
use libm::floorf;

const RESAMPLING_ORDER: usize = 8;
const MAX_DOWNMIX_BUFFER: usize = 960 * 2;
const LPCNET_TOTAL_FEATURES: usize = 36;
const DRED_MAX_DIM: usize = if DRED_STATE_DIM > DRED_LATENT_DIM {
    DRED_STATE_DIM
} else {
    DRED_LATENT_DIM
};

#[derive(Debug)]
pub(crate) struct DredEnc {
    pub model: RdovaeEnc,
    pub lpcnet_enc_state: LpcNetEncState,
    pub rdovae_enc: RdovaeEncState,
    pub loaded: bool,
    pub fs: i32,
    pub channels: i32,
    input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    input_buffer_fill: i32,
    dred_offset: i32,
    latent_offset: i32,
    last_extra_dred_offset: i32,
    latents_buffer: [f32; DRED_MAX_FRAMES * DRED_LATENT_DIM],
    pub(crate) latents_buffer_fill: i32,
    state_buffer: [f32; DRED_MAX_FRAMES * DRED_STATE_DIM],
    resample_mem: [f32; RESAMPLING_ORDER + 1],
}

impl Default for DredEnc {
    fn default() -> Self {
        Self {
            model: RdovaeEnc::new(),
            lpcnet_enc_state: LpcNetEncState::default(),
            rdovae_enc: RdovaeEncState::default(),
            loaded: false,
            fs: 0,
            channels: 0,
            input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: 0,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: [0.0; RESAMPLING_ORDER + 1],
        }
    }
}

pub(crate) fn dred_encoder_init(enc: &mut DredEnc, fs: i32, channels: i32) {
    enc.fs = fs;
    enc.channels = channels;
    enc.model = RdovaeEnc::new();
    enc.loaded = true;
    dred_encoder_reset(enc);
}

pub(crate) fn dred_encoder_reset(enc: &mut DredEnc) {
    enc.input_buffer.fill(0.0);
    enc.input_buffer_fill = DRED_SILK_ENCODER_DELAY;
    enc.dred_offset = 0;
    enc.latent_offset = 0;
    enc.last_extra_dred_offset = 0;
    enc.latents_buffer.fill(0.0);
    enc.latents_buffer_fill = 0;
    enc.state_buffer.fill(0.0);
    enc.resample_mem.fill(0.0);
    let _ = lpcnet_encoder_init(&mut enc.lpcnet_enc_state);
    enc.rdovae_enc = RdovaeEncState::default();
}

pub(crate) fn dred_encoder_load_model(enc: &mut DredEnc, data: &[u8]) -> Result<(), WeightError> {
    enc.model = RdovaeEnc::from_weights(data)?;
    lpcnet_encoder_load_model(&mut enc.lpcnet_enc_state, data)?;
    enc.loaded = true;
    Ok(())
}

fn filter_df2t(
    samples: &mut [f32],
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
) {
    for i in 0..samples.len() {
        let xi = samples[i];
        let yi = xi * b0 + mem[0];
        let nyi = -yi;
        for j in 0..order {
            mem[j] = mem[j + 1] + b[j] * xi + a[j] * nyi;
        }
        samples[i] = yi;
    }
}

fn dred_convert_to_16k(
    fs: i32,
    channels: i32,
    resample_mem: &mut [f32; RESAMPLING_ORDER + 1],
    input: &[f32],
    in_len: usize,
    output: &mut [f32],
    out_len: usize,
) {
    let mut downmix = [0.0f32; MAX_DOWNMIX_BUFFER];
    let in_channels = channels as usize;
    debug_assert!(in_channels * in_len <= MAX_DOWNMIX_BUFFER);
    debug_assert_eq!(in_len as i32 * 16_000, out_len as i32 * fs);

    let up = match fs {
        8_000 => 2,
        12_000 => 4,
        16_000 => 1,
        24_000 => 2,
        48_000 => 1,
        _ => {
            debug_assert!(false, "unsupported sample rate");
            1
        }
    };

    let up_len = up * in_len;
    downmix[..up_len].fill(0.0);

    if in_channels == 1 {
        for i in 0..in_len {
            downmix[up * i] = f32::from(float2int16((up as f32) * input[i]));
        }
    } else {
        for i in 0..in_len {
            let mixed = 0.5 * (input[2 * i] + input[2 * i + 1]);
            downmix[up * i] = f32::from(float2int16((up as f32) * mixed));
        }
    }

    if fs == 16_000 {
        output[..out_len].copy_from_slice(&downmix[..out_len]);
    } else if fs == 48_000 || fs == 24_000 {
        const FILTER_B: [f32; 8] = [
            0.005_873_358_047,
            0.012_980_854_831,
            0.014_531_340_042,
            0.014_531_340_042,
            0.012_980_854_831,
            0.005_873_358_047,
            0.004_523_418_224,
            0.0,
        ];
        const FILTER_A: [f32; 8] = [
            -3.878_718_597_768,
            7.748_834_257_468,
            -9.653_651_699_533,
            8.007_342_726_666,
            -4.379_450_178_552,
            1.463_182_111_81,
            -0.231_720_677_804,
            0.0,
        ];
        let b0 = 0.004_523_418_224;
        filter_df2t(
            &mut downmix[..up_len],
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            output[i] = downmix[3 * i];
        }
    } else if fs == 12_000 {
        const FILTER_B: [f32; 8] = [
            -0.001_017_101_081,
            0.003_673_127_243,
            0.001_009_165_267,
            0.001_009_165_267,
            0.003_673_127_243,
            -0.001_017_101_081,
            0.002_033_596_776,
            0.0,
        ];
        const FILTER_A: [f32; 8] = [
            -4.930_414_411_612,
            11.291_643_096_504,
            -15.322_037_343_815,
            13.216_403_930_898,
            -7.220_409_219_553,
            2.310_550_142_771,
            -0.334_338_618_782,
            0.0,
        ];
        let b0 = 0.002_033_596_776;
        filter_df2t(
            &mut downmix[..up_len],
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            output[i] = downmix[3 * i];
        }
    } else if fs == 8_000 {
        const FILTER_B: [f32; 8] = [
            0.081_670_120_929,
            0.180_401_598_565,
            0.259_391_051_971,
            0.259_391_051_971,
            0.180_401_598_565,
            0.081_670_120_929,
            0.020_109_185_709,
            0.0,
        ];
        const FILTER_A: [f32; 8] = [
            -1.393_651_933_659,
            2.609_789_872_676,
            -2.403_541_968_806,
            2.056_814_957_331,
            -1.148_908_574_57,
            0.473_001_413_788,
            -0.110_359_852_412,
            0.0,
        ];
        let b0 = 0.020_109_185_709;
        output[..out_len].copy_from_slice(&downmix[..out_len]);
        filter_df2t(
            &mut output[..out_len],
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
    } else {
        debug_assert!(false, "unsupported sample rate");
    }
}

fn dred_process_frame(enc: &mut DredEnc, arch: i32) {
    let mut feature_buffer = [0.0f32; 2 * LPCNET_TOTAL_FEATURES];
    let mut input_buffer = [0.0f32; 2 * DRED_NUM_FEATURES];

    debug_assert!(enc.loaded);

    let latent_stride = DRED_LATENT_DIM;
    let state_stride = DRED_STATE_DIM;
    let latents_len = DRED_MAX_FRAMES * latent_stride;
    let states_len = DRED_MAX_FRAMES * state_stride;
    enc.latents_buffer
        .copy_within(0..latents_len - latent_stride, latent_stride);
    enc.state_buffer
        .copy_within(0..states_len - state_stride, state_stride);

    let first = &enc.input_buffer[..DRED_FRAME_SIZE];
    let second = &enc.input_buffer[DRED_FRAME_SIZE..DRED_FRAME_SIZE * 2];
    let _ = lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        first,
        &mut feature_buffer[..LPCNET_TOTAL_FEATURES],
        arch,
    );
    let _ = lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        second,
        &mut feature_buffer[LPCNET_TOTAL_FEATURES..],
        arch,
    );

    input_buffer[..DRED_NUM_FEATURES]
        .copy_from_slice(&feature_buffer[..DRED_NUM_FEATURES]);
    input_buffer[DRED_NUM_FEATURES..]
        .copy_from_slice(&feature_buffer[LPCNET_TOTAL_FEATURES..LPCNET_TOTAL_FEATURES + DRED_NUM_FEATURES]);

    dred_rdovae_encode_dframe(
        &mut enc.rdovae_enc,
        &enc.model,
        &mut enc.latents_buffer[..DRED_LATENT_DIM],
        &mut enc.state_buffer[..DRED_STATE_DIM],
        &input_buffer,
        arch,
    );
    enc.latents_buffer_fill =
        (enc.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES as i32);
}

pub(crate) fn dred_compute_latents(
    enc: &mut DredEnc,
    pcm: &[f32],
    frame_size: usize,
    extra_delay: i32,
    arch: i32,
) {
    let frame_size16k = (frame_size as i32 * 16_000) / enc.fs;
    let mut _curr_offset16k =
        40 + extra_delay * 16_000 / enc.fs - enc.input_buffer_fill;
    debug_assert!(enc.loaded);
    enc.dred_offset = floorf((_curr_offset16k as f32 + 20.0) / 40.0) as i32;
    enc.latent_offset = 0;

    let mut frame_size16k = frame_size16k;
    let mut pcm_pos = 0usize;
    while frame_size16k > 0 {
        let process_size16k = (2 * DRED_FRAME_SIZE as i32).min(frame_size16k);
        let process_size = (process_size16k * enc.fs) / 16_000;
        let process_size = process_size as usize;
        let process_size16k_usize = process_size16k as usize;

        let input_fill = enc.input_buffer_fill as usize;
        let output_end = input_fill + process_size16k_usize;
        let fs = enc.fs;
        let channels = enc.channels;
        let (resample_mem, input_buffer) = (&mut enc.resample_mem, &mut enc.input_buffer);
        dred_convert_to_16k(
            fs,
            channels,
            resample_mem,
            &pcm[pcm_pos..],
            process_size,
            &mut input_buffer[input_fill..output_end],
            process_size16k_usize,
        );

        enc.input_buffer_fill += process_size16k;
        if enc.input_buffer_fill >= 2 * DRED_FRAME_SIZE as i32 {
            _curr_offset16k += 320;
            dred_process_frame(enc, arch);
            enc.input_buffer_fill -= 2 * DRED_FRAME_SIZE as i32;
            let keep = enc.input_buffer_fill as usize;
            enc.input_buffer
                .copy_within(2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + keep, 0);
            if enc.dred_offset < 6 {
                enc.dred_offset += 8;
            } else {
                enc.latent_offset += 1;
            }
        }

        pcm_pos = pcm_pos.saturating_add(process_size);
        frame_size16k -= process_size16k;
    }
}

fn dred_encode_latents(
    enc: &mut EcEnc<'_>,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    let mut q = [0i32; DRED_MAX_DIM];
    let mut xq = [0.0f32; DRED_MAX_DIM];
    let mut delta = [0.0f32; DRED_MAX_DIM];
    let mut deadzone = [0.0f32; DRED_MAX_DIM];
    let eps = 0.1f32;

    for i in 0..dim {
        delta[i] = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    compute_activation(&mut deadzone[..dim], ACTIVATION_TANH);
    for i in 0..dim {
        xq[i] -= delta[i] * deadzone[i];
        q[i] = floorf(0.5 + xq[i]) as i32;
    }
    for i in 0..dim {
        if r[i] == 0 || p0[i] == 255 {
            q[i] = 0;
        } else {
            ec_laplace_encode_p0_ec(enc, q[i], (p0[i] as u16) << 7, (r[i] as u16) << 7);
        }
    }
}

fn ec_laplace_encode_p0_ec(enc: &mut EcEnc<'_>, value: i32, p0: u16, decay: u16) {
    let mut sign_icdf = [0u16; 3];
    sign_icdf[0] = 32768 - p0;
    sign_icdf[1] = sign_icdf[0] / 2;
    sign_icdf[2] = 0;

    let sign_symbol = match value.cmp(&0) {
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Less => 2,
    };
    enc.enc_icdf16(sign_symbol as usize, &sign_icdf, 15);

    let mut remaining = value.abs();
    if remaining != 0 {
        let mut icdf = [0u16; 8];
        icdf[0] = decay.max(7);
        for i in 1..7 {
            let baseline = (7i32 - i as i32).max(0) as u16;
            let decayed = ((icdf[i - 1] as u32 * decay as u32) >> 15) as u16;
            icdf[i] = baseline.max(decayed);
        }
        icdf[7] = 0;

        remaining -= 1;
        loop {
            let symbol = remaining.min(7) as usize;
            enc.enc_icdf16(symbol, &icdf, 15);
            remaining -= 7;
            if remaining < 0 {
                break;
            }
        }
    }
}

fn dred_voice_active(activity_mem: &[u8], offset: i32) -> bool {
    let base = 8 * offset as usize;
    for i in 0..16 {
        if activity_mem[base + i] == 1 {
            return true;
        }
    }
    false
}

pub(crate) fn dred_encode_silk_frame(
    enc: &mut DredEnc,
    buf: &mut [u8],
    max_chunks: i32,
    max_bytes: i32,
    q0: i32,
    d_q: i32,
    qmax: i32,
    activity_mem: &[u8],
) -> i32 {
    let max_bytes_i32 = max_bytes;
    if max_bytes_i32 <= 0 {
        return 0;
    }
    let max_bytes = max_bytes_i32 as usize;
    let buf_len = buf.len();
    let buf = &mut buf[..max_bytes.min(buf_len)];
    let mut ec_encoder = EcEnc::new(buf);

    let mut latent_offset = enc.latent_offset;
    let mut extra_dred_offset = 0;
    let mut dred_encoded = 0;
    let mut delayed_dred = 0;
    let mut prev_active = 0;

    if activity_mem.first().copied().unwrap_or(0) != 0 && enc.last_extra_dred_offset > 0 {
        latent_offset = enc.last_extra_dred_offset;
        delayed_dred = 1;
        enc.last_extra_dred_offset = 0;
    }
    while latent_offset < enc.latents_buffer_fill
        && !dred_voice_active(activity_mem, latent_offset)
    {
        latent_offset += 1;
        extra_dred_offset += 1;
    }
    if delayed_dred == 0 {
        enc.last_extra_dred_offset = extra_dred_offset;
    }

    ec_encoder.enc_uint(q0 as u32, 16);
    ec_encoder.enc_uint(d_q as u32, 8);

    let total_offset = 16 - (enc.dred_offset - extra_dred_offset * 8);
    debug_assert!(total_offset >= 0);
    if total_offset > 31 {
        ec_encoder.enc_uint(1, 2);
        ec_encoder.enc_uint((total_offset >> 5) as u32, 256);
        ec_encoder.enc_uint((total_offset & 31) as u32, 32);
    } else {
        ec_encoder.enc_uint(0, 2);
        ec_encoder.enc_uint(total_offset as u32, 32);
    }

    debug_assert!(qmax >= q0);
    if q0 < 14 && d_q > 0 {
        let nvals = 15 - (q0 + 1);
        debug_assert!(qmax > q0);
        let low = if qmax >= 15 {
            0
        } else {
            nvals + qmax - (q0 + 1)
        };
        let high = if qmax >= 15 {
            nvals
        } else {
            nvals + qmax - q0
        };
        ec_encoder.encode(low as u32, high as u32, (2 * nvals) as u32);
    }

    let state_qoffset = (q0 * DRED_STATE_DIM as i32) as usize;
    dred_encode_latents(
        &mut ec_encoder,
        &enc.state_buffer[latent_offset as usize * DRED_STATE_DIM
            ..latent_offset as usize * DRED_STATE_DIM + DRED_STATE_DIM],
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        &DRED_STATE_DEAD_ZONE_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        &DRED_STATE_R_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        &DRED_STATE_P0_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        DRED_STATE_DIM,
    );
    if ec_tell(ec_encoder.ctx()) > 8 * max_bytes_i32 {
        return 0;
    }

    let mut ec_bak = EcEncSnapshot::capture(&ec_encoder);
    let max_iters = (2 * max_chunks).min(enc.latents_buffer_fill - latent_offset - 1);
    for i in (0..max_iters).step_by(2) {
        let q_level = compute_quantizer(q0, d_q, qmax, (i / 2) as i32);
        let offset = (q_level * DRED_LATENT_DIM as i32) as usize;

        dred_encode_latents(
            &mut ec_encoder,
            &enc.latents_buffer[(i + latent_offset) as usize * DRED_LATENT_DIM
                ..(i + latent_offset) as usize * DRED_LATENT_DIM + DRED_LATENT_DIM],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..offset + DRED_LATENT_DIM],
            &DRED_LATENT_DEAD_ZONE_Q8[offset..offset + DRED_LATENT_DIM],
            &DRED_LATENT_R_Q8[offset..offset + DRED_LATENT_DIM],
            &DRED_LATENT_P0_Q8[offset..offset + DRED_LATENT_DIM],
            DRED_LATENT_DIM,
        );
        if ec_tell(ec_encoder.ctx()) > 8 * max_bytes_i32 {
            if i == 0 {
                return 0;
            }
            break;
        }
        let active = dred_voice_active(activity_mem, i + latent_offset);
        if active || prev_active != 0 {
            ec_bak = EcEncSnapshot::capture(&ec_encoder);
            dred_encoded = i + 2;
        }
        prev_active = i32::from(active);
    }

    if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset > 0) {
        return 0;
    }
    ec_bak.restore(&mut ec_encoder);

    let ec_buffer_fill = (ec_tell(ec_encoder.ctx()) + 7) as u32 / 8;
    ec_encoder.enc_shrink(ec_buffer_fill);
    ec_encoder.enc_done();
    ec_buffer_fill as i32
}

fn compute_quantizer(q0: i32, d_q: i32, qmax: i32, index: i32) -> i32 {
    const D_Q_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    debug_assert!(
        (0..D_Q_TABLE.len() as i32).contains(&d_q),
        "dQ index out of range"
    );
    let quant = q0 + (D_Q_TABLE[d_q as usize] * index + 8) / 16;
    quant.min(qmax)
}

#[cfg(test)]
mod tests {
    use super::{dred_encode_silk_frame, DredEnc};
    use crate::dred_constants::{DRED_LATENT_DIM, DRED_STATE_DIM};

    #[test]
    fn dred_encode_returns_zero_without_latents() {
        let mut enc = DredEnc::default();
        enc.latents_buffer_fill = 0;
        enc.loaded = true;
        let mut buf = [0u8; 128];
        let activity = [1u8; 64];
        let bytes = dred_encode_silk_frame(&mut enc, &mut buf, 1, 64, 6, 0, 15, &activity);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn dred_encode_produces_payload_with_active_latents() {
        let mut enc = DredEnc::default();
        enc.loaded = true;
        enc.latents_buffer_fill = 2;
        enc.latents_buffer[..DRED_LATENT_DIM].fill(0.1);
        enc.state_buffer[..DRED_STATE_DIM].fill(0.2);
        let mut buf = [0u8; 512];
        let activity = [1u8; 512];
        let bytes = dred_encode_silk_frame(&mut enc, &mut buf, 1, 256, 6, 0, 15, &activity);
        assert!(bytes > 0);
    }
}
