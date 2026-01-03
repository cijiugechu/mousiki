use mousiki::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decoder_create, opus_decoder_ctl,
};
use mousiki::opus_encoder::{
    OpusEncodeError, OpusEncoderCtlError, OpusEncoderCtlRequest, opus_encode,
    opus_encoder_create, opus_encoder_ctl,
};
use mousiki::opus_multistream::{
    OpusMultistreamDecoderCtlRequest, OpusMultistreamEncoderCtlRequest, OpusMultistreamEncoderError,
    opus_multistream_decode, opus_multistream_decoder_create, opus_multistream_decoder_ctl,
    opus_multistream_encode, opus_multistream_encoder_create, opus_multistream_encoder_ctl,
    opus_multistream_encoder_get_encoder_state,
};
use mousiki::packet::{
    opus_packet_get_mode, opus_packet_get_nb_frames, opus_packet_get_samples_per_frame,
    opus_packet_parse,
};
use mousiki::repacketizer::{
    opus_multistream_packet_pad, opus_multistream_packet_unpad, opus_packet_pad, opus_packet_unpad,
};

const OPUS_APPLICATION_VOIP: i32 = 2048;
const OPUS_APPLICATION_AUDIO: i32 = 2049;
const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

const OPUS_AUTO: i32 = -1000;
const OPUS_BITRATE_MAX: i32 = -1;
const OPUS_UNIMPLEMENTED: i32 = -5;

const MODE_SILK_ONLY: i32 = 1000;
const MODE_HYBRID: i32 = 1001;
const MODE_CELT_ONLY: i32 = 1002;

const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
const OPUS_FRAMESIZE_5_MS: i32 = 5002;
const OPUS_FRAMESIZE_10_MS: i32 = 5003;
const OPUS_FRAMESIZE_20_MS: i32 = 5004;
const OPUS_FRAMESIZE_40_MS: i32 = 5005;
const OPUS_FRAMESIZE_60_MS: i32 = 5006;
const OPUS_FRAMESIZE_80_MS: i32 = 5007;
const OPUS_FRAMESIZE_100_MS: i32 = 5008;
const OPUS_FRAMESIZE_120_MS: i32 = 5009;

const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

const OPUS_SIGNAL_VOICE: i32 = 3001;
const OPUS_SIGNAL_MUSIC: i32 = 3002;

const MAX_PACKET: usize = 1500;
const MAX_FRAME_SAMP: usize = 5760;
const SAMPLES: usize = 48_000 * 30;
const SSAMPLES: usize = SAMPLES / 3;

#[derive(Clone)]
struct FastRand {
    rz: u32,
    rw: u32,
}

impl FastRand {
    fn new(seed: u32) -> Self {
        Self { rz: seed, rw: seed }
    }

    fn next(&mut self) -> u32 {
        self.rz = 36969u32
            .wrapping_mul(self.rz & 0xFFFF)
            .wrapping_add(self.rz >> 16);
        self.rw = 18000u32
            .wrapping_mul(self.rw & 0xFFFF)
            .wrapping_add(self.rw >> 16);
        (self.rz << 16).wrapping_add(self.rw)
    }
}

fn rand_sample<T: Copy>(rng: &mut FastRand, values: &[T]) -> T {
    let idx = rng.next() as usize % values.len();
    values[idx]
}

fn deb2_impl(t: &mut [u8], out: &mut [u8], p: &mut usize, k: usize, x: usize, y: usize) {
    if x > 2 {
        if y < 3 {
            for i in 0..y {
                *p = p.saturating_sub(1);
                out[*p] = t[i + 1];
            }
        }
        return;
    }

    t[x] = t[x - y];
    deb2_impl(t, out, p, k, x + 1, y);
    for i in (t[x - y] + 1)..(k as u8) {
        t[x] = i;
        deb2_impl(t, out, p, k, x + 1, x);
    }
}

fn debruijn2(k: usize) -> Vec<u8> {
    let mut out = vec![0u8; k * k];
    let mut t = vec![0u8; k * 2];
    let mut p = k * k;
    deb2_impl(&mut t, &mut out, &mut p, k, 1, 1);
    out
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn generate_music(buf: &mut [i16], len: usize, rng: &mut FastRand) {
    let samples = len.min(buf.len() / 2);
    if samples == 0 {
        return;
    }

    let mut a1 = 0i32;
    let mut b1 = 0i32;
    let mut a2 = 0i32;
    let mut b2 = 0i32;
    let mut c1 = 0i32;
    let mut c2 = 0i32;
    let mut d1 = 0i32;
    let mut d2 = 0i32;
    let mut j = 0i32;

    let silence = samples.min(2880);
    for i in 0..silence {
        let idx = i * 2;
        buf[idx] = 0;
        buf[idx + 1] = 0;
    }

    for i in silence..samples {
        let mut v1 = (((j * ((j >> 12) ^ ((j >> 10 | j >> 12) & 26 & (j >> 7)))) & 128) + 128)
            << 15;
        let mut v2 = v1;
        let r = rng.next();
        v1 += (r & 65_535) as i32;
        v1 -= (r >> 16) as i32;
        let r = rng.next();
        v2 += (r & 65_535) as i32;
        v2 -= (r >> 16) as i32;
        b1 = v1 - a1 + ((b1 * 61 + 32) >> 6);
        a1 = v1;
        b2 = v2 - a2 + ((b2 * 61 + 32) >> 6);
        a2 = v2;
        c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
        d1 = b1;
        c2 = (30 * (c2 + b2 + d2) + 32) >> 6;
        d2 = b2;
        v1 = (c1 + 128) >> 8;
        v2 = (c2 + 128) >> 8;
        let idx = i * 2;
        buf[idx] = clamp_i16(v1);
        buf[idx + 1] = clamp_i16(v2);
        if i % 6 == 0 {
            j += 1;
        }
    }
}

fn get_frame_size_enum(frame_size: usize, sampling_rate: usize) -> i32 {
    if frame_size == sampling_rate / 400 {
        OPUS_FRAMESIZE_2_5_MS
    } else if frame_size == sampling_rate / 200 {
        OPUS_FRAMESIZE_5_MS
    } else if frame_size == sampling_rate / 100 {
        OPUS_FRAMESIZE_10_MS
    } else if frame_size == sampling_rate / 50 {
        OPUS_FRAMESIZE_20_MS
    } else if frame_size == sampling_rate / 25 {
        OPUS_FRAMESIZE_40_MS
    } else if frame_size == 3 * sampling_rate / 50 {
        OPUS_FRAMESIZE_60_MS
    } else if frame_size == 4 * sampling_rate / 50 {
        OPUS_FRAMESIZE_80_MS
    } else if frame_size == 5 * sampling_rate / 50 {
        OPUS_FRAMESIZE_100_MS
    } else if frame_size == 6 * sampling_rate / 50 {
        OPUS_FRAMESIZE_120_MS
    } else {
        panic!("unsupported frame size: {frame_size}");
    }
}

fn seed_from_env() -> u32 {
    std::env::var("SEED")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0xC0DEC0DE)
}

fn no_fuzz() -> bool {
    std::env::var_os("TEST_OPUS_NOFUZZ").is_some()
}

fn is_bad_arg_multistream_create(err: &OpusMultistreamEncoderError) -> bool {
    matches!(
        err,
        OpusMultistreamEncoderError::BadArgument
            | OpusMultistreamEncoderError::EncoderInit(
                mousiki::opus_encoder::OpusEncoderInitError::BadArgument
            )
    )
}

fn test_encode(
    enc: &mut mousiki::opus_encoder::OpusEncoder<'_>,
    channels: usize,
    frame_size: usize,
    dec: &mut mousiki::opus_decoder::OpusDecoder<'_>,
    rng: &mut FastRand,
) -> Result<(), String> {
    let mut samp_count = 0usize;
    let mut inbuf = vec![0i16; SSAMPLES];
    generate_music(&mut inbuf, SSAMPLES / 2, rng);

    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * 3];
    let mut packet = vec![0u8; MAX_PACKET + 257];

    let limit = SSAMPLES / 2;
    if limit <= MAX_FRAME_SAMP {
        return Ok(());
    }

    while samp_count < limit - MAX_FRAME_SAMP {
        let start = samp_count
            .checked_mul(channels)
            .ok_or("input offset overflow")?;
        let end = start
            .checked_add(frame_size * channels)
            .ok_or("input end overflow")?;
        if end > inbuf.len() {
            return Err("input buffer too small".to_string());
        }

        let len = opus_encode(enc, &inbuf[start..end], frame_size, &mut packet[..MAX_PACKET])
            .map_err(|err| format!("opus_encode failed: {err:?}"))?;
        if len > MAX_PACKET {
            return Err("opus_encode returned oversized packet".to_string());
        }

        let out_samples = match opus_decode(
            dec,
            Some(&packet[..len]),
            len,
            &mut outbuf,
            MAX_FRAME_SAMP,
            false,
        ) {
            Ok(samples) => samples,
            Err(err) => {
                let mut fs = 0i32;
                let _ = opus_decoder_ctl(dec, OpusDecoderCtlRequest::GetSampleRate(&mut fs));
                let fs_u32 = fs.max(0) as u32;
                let frame_samples = opus_packet_get_samples_per_frame(&packet[..len], fs_u32)
                    .unwrap_or_default();
                let nb_frames = opus_packet_get_nb_frames(&packet[..len], len).unwrap_or_default();
                let mode = opus_packet_get_mode(&packet[..len]).ok();
                return Err(format!(
                    "opus_decode failed: {err:?}, len={len}, packet_samples={frame_samples}, \
                     nb_frames={nb_frames}, mode={mode:?}"
                ));
            }
        };
        if out_samples != frame_size {
            let mut fs = 0i32;
            let _ = opus_decoder_ctl(dec, OpusDecoderCtlRequest::GetSampleRate(&mut fs));
            let fs_u32 = fs.max(0) as u32;
            let frame_samples = opus_packet_get_samples_per_frame(&packet[..len], fs_u32)
                .unwrap_or_default();
            let nb_frames = opus_packet_get_nb_frames(&packet[..len], len).unwrap_or_default();
            let mode = opus_packet_get_mode(&packet[..len]).ok();
            return Err(format!(
                "decoded sample count mismatch: expected {frame_size}, got {out_samples}, \
                 packet_samples={frame_samples}, nb_frames={nb_frames}, mode={mode:?}"
            ));
        }

        samp_count += frame_size;
    }

    Ok(())
}

fn fuzz_encoder_settings(
    num_encoders: usize,
    num_setting_changes: usize,
    rng: &mut FastRand,
) -> Result<(), String> {
    let sampling_rates = [8000, 12_000, 16_000, 24_000, 48_000];
    let channels = [1, 2];
    let applications = [
        OPUS_APPLICATION_AUDIO,
        OPUS_APPLICATION_VOIP,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    ];
    let bitrates = [
        6000,
        12_000,
        16_000,
        24_000,
        32_000,
        48_000,
        64_000,
        96_000,
        510_000,
        OPUS_AUTO,
        OPUS_BITRATE_MAX,
    ];
    let force_channels = [OPUS_AUTO, OPUS_AUTO, 1, 2];
    let use_vbr = [0, 1, 1];
    let vbr_constraints = [0, 1, 1];
    let complexities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max_bandwidths = [
        OPUS_BANDWIDTH_NARROWBAND,
        OPUS_BANDWIDTH_MEDIUMBAND,
        OPUS_BANDWIDTH_WIDEBAND,
        OPUS_BANDWIDTH_SUPERWIDEBAND,
        OPUS_BANDWIDTH_FULLBAND,
        OPUS_BANDWIDTH_FULLBAND,
    ];
    let signals = [OPUS_AUTO, OPUS_AUTO, OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC];
    let inband_fecs = [0, 0, 1];
    let packet_loss_perc = [0, 1, 2, 5];
    let lsb_depths = [8, 24];
    let prediction_disabled = [0, 0, 1];
    let use_dtx = [0, 1];
    let frame_sizes_ms_x2 = [5, 10, 20, 40, 80, 120, 160, 200, 240];

    for _ in 0..num_encoders {
        let sampling_rate = rand_sample(rng, &sampling_rates);
        let num_channels = rand_sample(rng, &channels);
        let application = rand_sample(rng, &applications);

        let mut dec = opus_decoder_create(sampling_rate, num_channels)
            .map_err(|err| format!("decoder create failed: {err:?}"))?;
        let mut enc = opus_encoder_create(sampling_rate, num_channels, application)
            .map_err(|err| format!("encoder create failed: {err:?}"))?;

        for _ in 0..num_setting_changes {
            let bitrate = rand_sample(rng, &bitrates);
            let mut force_channel = rand_sample(rng, &force_channels);
            let vbr = rand_sample(rng, &use_vbr) != 0;
            let vbr_constraint = rand_sample(rng, &vbr_constraints) != 0;
            let complexity = rand_sample(rng, &complexities);
            let max_bw = rand_sample(rng, &max_bandwidths);
            let sig = rand_sample(rng, &signals);
            let inband_fec = rand_sample(rng, &inband_fecs) != 0;
            let pkt_loss = rand_sample(rng, &packet_loss_perc);
            let lsb_depth = rand_sample(rng, &lsb_depths);
            let pred_disabled = rand_sample(rng, &prediction_disabled) != 0;
            let dtx = rand_sample(rng, &use_dtx) != 0;
            let frame_size_ms_x2 = rand_sample(rng, &frame_sizes_ms_x2);
            let frame_size = frame_size_ms_x2 as usize * sampling_rate as usize / 2000;
            let frame_size_enum = get_frame_size_enum(frame_size, sampling_rate as usize);
            if force_channel != OPUS_AUTO {
                force_channel = force_channel.min(num_channels);
            }

            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(bitrate))
                .map_err(|err| format!("set bitrate failed: {err:?}"))?;
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetForceChannels(force_channel),
            )
            .map_err(|err| format!("set force channels failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(vbr))
                .map_err(|err| format!("set vbr failed: {err:?}"))?;
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetVbrConstraint(vbr_constraint),
            )
            .map_err(|err| format!("set vbr constraint failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(complexity))
                .map_err(|err| format!("set complexity failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetMaxBandwidth(max_bw))
                .map_err(|err| format!("set max bandwidth failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(sig))
                .map_err(|err| format!("set signal failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetInbandFec(inband_fec))
                .map_err(|err| format!("set inband fec failed: {err:?}"))?;
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetPacketLossPerc(pkt_loss),
            )
            .map_err(|err| format!("set packet loss failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLsbDepth(lsb_depth))
                .map_err(|err| format!("set lsb depth failed: {err:?}"))?;
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetPredictionDisabled(pred_disabled),
            )
            .map_err(|err| format!("set prediction disabled failed: {err:?}"))?;
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDtx(dtx))
                .map_err(|err| format!("set dtx failed: {err:?}"))?;
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetExpertFrameDuration(frame_size_enum),
            )
            .map_err(|err| format!("set frame duration failed: {err:?}"))?;

            if let Err(msg) = test_encode(&mut enc, num_channels as usize, frame_size, &mut dec, rng)
            {
                return Err(format!(
                    "fuzz_encoder_settings failed: {msg}, fs={}k, ch={}, app={}, bitrate={}, \
                     force_ch={}, vbr={}, vbr_constraint={}, complexity={}, max_bw={}, signal={}, \
                     inband_fec={}, pkt_loss={}, lsb_depth={}, pred_disabled={}, dtx={}, frame_ms_x2={}",
                    sampling_rate / 1000,
                    num_channels,
                    application,
                    bitrate,
                    force_channel,
                    vbr as i32,
                    vbr_constraint as i32,
                    complexity,
                    max_bw,
                    sig,
                    inband_fec as i32,
                    pkt_loss,
                    lsb_depth,
                    pred_disabled as i32,
                    dtx as i32,
                    frame_size_ms_x2
                ));
            }
        }
    }

    Ok(())
}

fn run_test1(no_fuzz: bool, rng: &mut FastRand) -> Result<(), String> {
    let fsizes = [960 * 3, 960 * 2, 120, 240, 480, 960];
    let mut mapping = [0u8; 256];
    mapping[0] = 0;
    mapping[1] = 1;
    mapping[2] = 255;

    let mut enc = opus_encoder_create(48_000, 2, OPUS_APPLICATION_VOIP)
        .map_err(|err| format!("encoder create failed: {err:?}"))?;

    let invalid = opus_multistream_encoder_create(8000, 2, 2, 0, &mapping, OPUS_UNIMPLEMENTED);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }
    let invalid = opus_multistream_encoder_create(8000, 0, 1, 0, &mapping, OPUS_APPLICATION_VOIP);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }
    let invalid = opus_multistream_encoder_create(44_100, 2, 2, 0, &mapping, OPUS_APPLICATION_VOIP);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }
    let invalid = opus_multistream_encoder_create(8000, 2, 2, 3, &mapping, OPUS_APPLICATION_VOIP);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }
    let invalid = opus_multistream_encoder_create(8000, 2, usize::MAX, 0, &mapping, OPUS_APPLICATION_VOIP);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }
    let invalid = opus_multistream_encoder_create(8000, 256, 2, 0, &mapping, OPUS_APPLICATION_VOIP);
    if !matches!(invalid, Err(ref err) if is_bad_arg_multistream_create(err)) {
        return Err("invalid multistream create should fail".to_string());
    }

    let mut ms_enc = opus_multistream_encoder_create(8000, 2, 2, 0, &mapping, OPUS_APPLICATION_AUDIO)
        .map_err(|err| format!("multistream encoder create failed: {err:?}"))?;

    let mut lsb = 0;
    opus_multistream_encoder_ctl(&mut ms_enc, OpusMultistreamEncoderCtlRequest::GetBitrate(&mut lsb))
        .map_err(|err| format!("get bitrate failed: {err:?}"))?;
    opus_multistream_encoder_ctl(&mut ms_enc, OpusMultistreamEncoderCtlRequest::GetLsbDepth(&mut lsb))
        .map_err(|err| format!("get lsb depth failed: {err:?}"))?;
    if lsb < 16 {
        return Err("lsb depth below expected minimum".to_string());
    }

    let stream_encoder = opus_multistream_encoder_get_encoder_state(&mut ms_enc, 1)
        .map_err(|err| format!("get encoder state failed: {err:?}"))?;
    let mut stream_lsb = 0;
    opus_encoder_ctl(stream_encoder, OpusEncoderCtlRequest::GetLsbDepth(&mut stream_lsb))
        .map_err(|err| format!("get stream lsb depth failed: {err:?}"))?;
    if lsb != stream_lsb {
        return Err("lsb depth mismatch between multistream and stream".to_string());
    }
    let invalid_state = opus_multistream_encoder_get_encoder_state(&mut ms_enc, 2);
    if !matches!(invalid_state, Err(OpusMultistreamEncoderError::BadArgument)) {
        return Err("invalid stream id should fail".to_string());
    }

    let mut dec = opus_decoder_create(48_000, 2)
        .map_err(|err| format!("decoder create failed: {err:?}"))?;
    let mut ms_dec = opus_multistream_decoder_create(48_000, 2, 2, 0, &mapping)
        .map_err(|err| format!("multistream decoder create failed: {err:?}"))?;
    let mut ms_dec_err = opus_multistream_decoder_create(48_000, 3, 2, 0, &mapping)
        .map_err(|err| format!("multistream decoder create failed: {err:?}"))?;

    let mut dec_err = Vec::with_capacity(10);
    dec_err.push(opus_decoder_create(48_000, 2).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(48_000, 1).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(24_000, 2).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(24_000, 1).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(16_000, 2).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(16_000, 1).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(12_000, 2).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(12_000, 1).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(8000, 2).map_err(|err| format!("decoder create failed: {err:?}"))?);
    dec_err.push(opus_decoder_create(8000, 1).map_err(|err| format!("decoder create failed: {err:?}"))?);

    let mut inbuf = vec![0i16; SAMPLES * 2];
    let mut outbuf = vec![0i16; SAMPLES * 2];
    let mut out2buf = vec![0i16; MAX_FRAME_SAMP * 3];
    generate_music(&mut inbuf, SAMPLES, rng);

    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBandwidth(OPUS_AUTO))
        .map_err(|err| format!("set bandwidth failed: {err:?}"))?;
    let invalid = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(-2));
    if !matches!(invalid, Err(OpusEncoderCtlError::BadArgument)) {
        return Err("invalid force mode should fail".to_string());
    }
    let mut packet = vec![0u8; MAX_PACKET + 257];
    let invalid = opus_encode(&mut enc, &inbuf, 500, &mut packet[..MAX_PACKET]);
    if !matches!(invalid, Err(OpusEncodeError::BadArgument)) {
        return Err("encode with invalid frame size should fail".to_string());
    }

    let allow_silk_final_range = std::env::var_os("ALLOW_SILK_FINAL_RANGE").is_some();
    let silk_range_limit = 48_000 / 50;

    for rc in 0..3 {
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(rc < 2))
            .map_err(|err| format!("set vbr failed: {err:?}"))?;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbrConstraint(rc == 1))
            .map_err(|err| format!("set vbr constraint failed: {err:?}"))?;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbrConstraint(rc == 1))
            .map_err(|err| format!("set vbr constraint failed: {err:?}"))?;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetInbandFec(rc == 0))
            .map_err(|err| format!("set inband fec failed: {err:?}"))?;

        for j in 0..13 {
            let modes = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
            let rates = [
                6000, 12_000, 48_000, 16_000, 32_000, 48_000, 64_000, 512_000, 13_000,
                24_000, 48_000, 64_000, 96_000,
            ];
            let frame = [960 * 2, 960, 480, 960, 960, 960, 480, 960 * 3, 960 * 3, 960, 480, 240, 120];

            let rate = rates[j] + (rng.next() % rates[j] as u32) as i32;
            let mut i = 0usize;
            let mut count = 0i32;
            while i < SSAMPLES - MAX_FRAME_SAMP {
                let frame_size = frame[j];
                if rng.next() & 255 == 0 {
                    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset encoder failed: {err:?}"))?;
                    opus_decoder_ctl(&mut dec, OpusDecoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                    if rng.next() & 1 != 0 {
                        opus_decoder_ctl(&mut dec_err[(rng.next() & 1) as usize], OpusDecoderCtlRequest::ResetState)
                            .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                    }
                }
                if rng.next() & 127 == 0 {
                    opus_decoder_ctl(&mut dec_err[(rng.next() & 1) as usize], OpusDecoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                }
                if rng.next() % 10 == 0 {
                    let complex = (rng.next() % 11) as i32;
                    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(complex))
                        .map_err(|err| format!("set complexity failed: {err:?}"))?;
                }
                if rng.next() % 50 == 0 {
                    opus_decoder_ctl(&mut dec, OpusDecoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                }
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetInbandFec(rc == 0))
                    .map_err(|err| format!("set inband fec failed: {err:?}"))?;
                let mode = MODE_SILK_ONLY + modes[j];
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(mode))
                .map_err(|err| format!("set force mode failed: {err:?}"))?;
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDtx(rng.next() & 1 != 0))
                    .map_err(|err| format!("set dtx failed: {err:?}"))?;
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(rate))
                    .map_err(|err| format!("set bitrate failed: {err:?}"))?;
                let forced_channels = if rates[j] >= 64_000 { 2 } else { 1 };
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(forced_channels))
                    .map_err(|err| format!("set force channels failed: {err:?}"))?;
                opus_encoder_ctl(
                    &mut enc,
                    OpusEncoderCtlRequest::SetComplexity((count >> 2) % 11),
                )
                .map_err(|err| format!("set complexity failed: {err:?}"))?;
                let pkt_loss = (rng.next() & 15) & (rng.next() % 15);
                opus_encoder_ctl(
                    &mut enc,
                    OpusEncoderCtlRequest::SetPacketLossPerc(pkt_loss as i32),
                )
                .map_err(|err| format!("set packet loss failed: {err:?}"))?;
                let mut bw = if modes[j] == 0 {
                    OPUS_BANDWIDTH_NARROWBAND + (rng.next() % 3) as i32
                } else if modes[j] == 1 {
                    OPUS_BANDWIDTH_SUPERWIDEBAND + (rng.next() & 1) as i32
                } else {
                    OPUS_BANDWIDTH_NARROWBAND + (rng.next() % 5) as i32
                };
                if modes[j] == 2 && bw == OPUS_BANDWIDTH_MEDIUMBAND {
                    bw += 3;
                }
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBandwidth(bw))
                    .map_err(|err| format!("set bandwidth failed: {err:?}"))?;

                let offset = i * 2;
                let len = opus_encode(
                    &mut enc,
                    &inbuf[offset..],
                    frame_size,
                    &mut packet[..MAX_PACKET],
                )
                .map_err(|err| {
                    format!(
                        "encode failed: {err:?}, mode={mode}, rate={rate}, frame_size={frame_size}, \
                         bw={bw}, rc={rc}, pkt_loss={pkt_loss}, forced_channels={forced_channels}, \
                         count={count}, offset={offset}"
                    )
                })?;
                if len > MAX_PACKET {
                    return Err("encoded packet too large".to_string());
                }
                let mut enc_final_range = 0u32;
                opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetFinalRange(&mut enc_final_range))
                    .map_err(|err| format!("get final range failed: {err:?}"))?;

                let mut len = len;
                if rng.next() & 3 == 0 {
                    opus_packet_pad(&mut packet, len, len + 1)
                        .map_err(|err| format!("packet pad failed: {err:?}"))?;
                    len += 1;
                }
                if rng.next() & 7 == 0 {
                    opus_packet_pad(&mut packet, len, len + 256)
                        .map_err(|err| format!("packet pad failed: {err:?}"))?;
                    len += 256;
                }
                if rng.next() & 3 == 0 {
                    len = opus_packet_unpad(&mut packet, len)
                        .map_err(|err| format!("packet unpad failed: {err:?}"))?;
                }

                let out_samples = opus_decode(
                    &mut dec,
                    Some(&packet[..len]),
                    len,
                    &mut outbuf[offset..],
                    MAX_FRAME_SAMP,
                    false,
                )
                .map_err(|err| {
                    format!(
                        "decode failed: {err:?}, stage=main, mode={}, frame_size={}, len={}, \
                         offset={}, rc={}, bw={}, rate={}",
                        MODE_SILK_ONLY + modes[j],
                        frame_size,
                        len,
                        offset,
                        rc,
                        bw,
                        rate
                    )
                })?;
                if out_samples != frame_size {
                    return Err("decoded sample count mismatch".to_string());
                }

                let mut dec_final_range = 0u32;
                opus_decoder_ctl(&mut dec, OpusDecoderCtlRequest::GetFinalRange(&mut dec_final_range))
                    .map_err(|err| format!("get final range failed: {err:?}"))?;
                if allow_silk_final_range
                    && modes[j] == 0
                    && frame_size <= silk_range_limit
                    && enc_final_range != dec_final_range
                {
                    return Err(format!(
                        "final range mismatch: mode={}, frame_size={}, enc={}, dec={}, \
                         rate={}, bw={}, rc={}, pkt_loss={}, forced_channels={}, count={}",
                        MODE_SILK_ONLY + modes[j],
                        frame_size,
                        enc_final_range,
                        dec_final_range,
                        rate,
                        bw,
                        rc,
                        pkt_loss,
                        forced_channels,
                        count
                    ));
                }

                let out_samples = opus_decode(
                    &mut dec_err[0],
                    Some(&packet[..len]),
                    len,
                    &mut out2buf,
                    frame_size,
                    rng.next() & 3 != 0,
                )
                .map_err(|err| {
                    format!(
                        "decode failed: {err:?}, stage=lbr, mode={}, frame_size={}, len={}, \
                         offset={}, rc={}, bw={}, rate={}",
                        MODE_SILK_ONLY + modes[j],
                        frame_size,
                        len,
                        offset,
                        rc,
                        bw,
                        rate
                    )
                })?;
                if out_samples != frame_size {
                    return Err("lbr decode sample count mismatch".to_string());
                }

                let use_packet = rng.next() & 3 != 0;
                let packet_len = if use_packet { len } else { 0 };
                let packet_opt = if use_packet {
                    Some(&packet[..len])
                } else {
                    None
                };
                let out_samples = opus_decode(
                    &mut dec_err[1],
                    packet_opt,
                    packet_len,
                    &mut out2buf,
                    MAX_FRAME_SAMP,
                    rng.next() & 7 != 0,
                )
                .map_err(|err| {
                    format!(
                        "decode failed: {err:?}, stage=lbr_plc, mode={}, frame_size={}, len={}, \
                         offset={}, rc={}, bw={}, rate={}, use_packet={}",
                        MODE_SILK_ONLY + modes[j],
                        frame_size,
                        packet_len,
                        offset,
                        rc,
                        bw,
                        rate,
                        use_packet as i32
                    )
                })?;
                if out_samples < 120 {
                    return Err("lbr decode produced too few samples".to_string());
                }

                i += frame_size;
                count += 1;
            }
        }
    }

    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(OPUS_AUTO))
        .map_err(|err| format!("set force mode failed: {err:?}"))?;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(OPUS_AUTO))
        .map_err(|err| format!("set force channels failed: {err:?}"))?;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetInbandFec(false))
        .map_err(|err| format!("set inband fec failed: {err:?}"))?;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDtx(false))
        .map_err(|err| format!("set dtx failed: {err:?}"))?;

    for rc in 0..3 {
        opus_multistream_encoder_ctl(&mut ms_enc, OpusMultistreamEncoderCtlRequest::SetVbr(rc < 2))
            .map_err(|err| format!("set vbr failed: {err:?}"))?;
        opus_multistream_encoder_ctl(
            &mut ms_enc,
            OpusMultistreamEncoderCtlRequest::SetVbrConstraint(rc == 1),
        )
        .map_err(|err| format!("set vbr constraint failed: {err:?}"))?;
        opus_multistream_encoder_ctl(
            &mut ms_enc,
            OpusMultistreamEncoderCtlRequest::SetVbrConstraint(rc == 1),
        )
        .map_err(|err| format!("set vbr constraint failed: {err:?}"))?;
        opus_multistream_encoder_ctl(
            &mut ms_enc,
            OpusMultistreamEncoderCtlRequest::SetInbandFec(rc == 0),
        )
        .map_err(|err| format!("set inband fec failed: {err:?}"))?;

        for j in 0..16 {
            let modes = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2];
            let rates = [
                4000, 12_000, 32_000, 8000, 16_000, 32_000, 48_000, 88_000, 4000, 12_000,
                32_000, 8000, 16_000, 32_000, 48_000, 88_000,
            ];
            let frame = [160, 160, 80, 160, 160, 80, 40, 20, 160, 160, 80, 160, 160, 80, 40, 20];
            opus_multistream_encoder_ctl(
                &mut ms_enc,
                OpusMultistreamEncoderCtlRequest::SetInbandFec(rc == 0 && j == 1),
            )
            .map_err(|err| format!("set inband fec failed: {err:?}"))?;
            opus_multistream_encoder_ctl(
                &mut ms_enc,
                OpusMultistreamEncoderCtlRequest::SetForceMode(MODE_SILK_ONLY + modes[j]),
            )
            .map_err(|err| format!("set force mode failed: {err:?}"))?;
            let rate = rates[j] + (rng.next() % rates[j] as u32) as i32;
            opus_multistream_encoder_ctl(
                &mut ms_enc,
                OpusMultistreamEncoderCtlRequest::SetDtx(rng.next() & 1 != 0),
            )
            .map_err(|err| format!("set dtx failed: {err:?}"))?;
            opus_multistream_encoder_ctl(
                &mut ms_enc,
                OpusMultistreamEncoderCtlRequest::SetBitrate(rate),
            )
            .map_err(|err| format!("set bitrate failed: {err:?}"))?;
            let mut i = 0usize;
            let mut count = 0i32;
            while i < (SSAMPLES / 12) - MAX_FRAME_SAMP {
                let mut pred = false;
                opus_multistream_encoder_ctl(
                    &mut ms_enc,
                    OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(&mut pred),
                )
                .map_err(|err| format!("get prediction disabled failed: {err:?}"))?;
                let new_pred = (rng.next() & 15) < if pred { 11 } else { 4 };
                opus_multistream_encoder_ctl(
                    &mut ms_enc,
                    OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(new_pred),
                )
                .map_err(|err| format!("set prediction disabled failed: {err:?}"))?;
                let frame_size = frame[j];
                opus_multistream_encoder_ctl(
                    &mut ms_enc,
                    OpusMultistreamEncoderCtlRequest::SetComplexity((count >> 2) % 11),
                )
                .map_err(|err| format!("set complexity failed: {err:?}"))?;
                let pkt_loss = (rng.next() & 15) & (rng.next() % 15);
                opus_multistream_encoder_ctl(
                    &mut ms_enc,
                    OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(pkt_loss as i32),
                )
                .map_err(|err| format!("set packet loss failed: {err:?}"))?;
                if rng.next() & 255 == 0 {
                    opus_multistream_encoder_ctl(&mut ms_enc, OpusMultistreamEncoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset encoder failed: {err:?}"))?;
                    opus_multistream_decoder_ctl(&mut ms_dec, OpusMultistreamDecoderCtlRequest::ResetState)
                        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                    if rng.next() & 3 != 0 {
                        opus_multistream_decoder_ctl(
                            &mut ms_dec_err,
                            OpusMultistreamDecoderCtlRequest::ResetState,
                        )
                        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                    }
                }
                if rng.next() & 255 == 0 {
                    opus_multistream_decoder_ctl(
                        &mut ms_dec_err,
                        OpusMultistreamDecoderCtlRequest::ResetState,
                    )
                    .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                }

                let offset = i * 2;
                let len = opus_multistream_encode(
                    &mut ms_enc,
                    &inbuf[offset..],
                    frame_size,
                    &mut packet,
                )
                .map_err(|err| format!("multistream encode failed: {err:?}"))?;
                if len > MAX_PACKET {
                    return Err("multistream packet too large".to_string());
                }

                let mut enc_final_range = 0u32;
                opus_multistream_encoder_ctl(
                    &mut ms_enc,
                    OpusMultistreamEncoderCtlRequest::GetFinalRange(&mut enc_final_range),
                )
                .map_err(|err| format!("get final range failed: {err:?}"))?;

                let mut len = len;
                if rng.next() & 3 == 0 {
                    opus_multistream_packet_pad(&mut packet, len, len + 1, 2)
                        .map_err(|err| format!("packet pad failed: {err:?}"))?;
                    len += 1;
                }
                if rng.next() & 7 == 0 {
                    opus_multistream_packet_pad(&mut packet, len, len + 256, 2)
                        .map_err(|err| format!("packet pad failed: {err:?}"))?;
                    len += 256;
                }
                if rng.next() & 3 == 0 {
                    len = opus_multistream_packet_unpad(&mut packet, len, 2)
                        .map_err(|err| format!("packet unpad failed: {err:?}"))?;
                }

                let out_samples = opus_multistream_decode(
                    &mut ms_dec,
                    &packet[..len],
                    len,
                    &mut out2buf,
                    MAX_FRAME_SAMP,
                    false,
                )
                .map_err(|err| format!("multistream decode failed: {err:?}"))?;
                if out_samples != frame_size * 6 {
                    return Err("multistream decoded sample count mismatch".to_string());
                }

                let mut dec_final_range = 0u32;
                opus_multistream_decoder_ctl(
                    &mut ms_dec,
                    OpusMultistreamDecoderCtlRequest::GetFinalRange(&mut dec_final_range),
                )
                .map_err(|err| format!("get final range failed: {err:?}"))?;
                if allow_silk_final_range && modes[j] == 0 && enc_final_range != dec_final_range {
                    return Err(format!(
                        "multistream final range mismatch: frame_size={}, enc={}, dec={}",
                        frame_size, enc_final_range, dec_final_range
                    ));
                }

                let loss = rng.next() & 63 == 0;
                let loss_len = if loss { 0 } else { len };
                let loss_packet = if loss { &packet[..0] } else { &packet[..len] };
                let out_samples = opus_multistream_decode(
                    &mut ms_dec_err,
                    loss_packet,
                    loss_len,
                    &mut out2buf,
                    frame_size * 6,
                    rng.next() & 3 != 0,
                )
                .map_err(|err| format!("multistream decode failed: {err:?}"))?;
                if out_samples != frame_size * 6 {
                    return Err("multistream lbr decoded sample count mismatch".to_string());
                }

                i += frame_size;
                count += 1;
            }
        }
    }

    let mut bitrate_bps = 512_000i32;
    let mut fsize = (rng.next() % 31) as usize;
    let mut fswitch = 100i32;

    let db62 = debruijn2(6);
    let mut i = 0usize;
    while i < SAMPLES * 4 {
        let frame_size = fsizes[db62[fsize] as usize];
        let offset = i % (SAMPLES - MAX_FRAME_SAMP);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(bitrate_bps))
            .map_err(|err| format!("set bitrate failed: {err:?}"))?;

        let len = opus_encode(
            &mut enc,
            &inbuf[offset * 2..],
            frame_size,
            &mut packet[..MAX_PACKET],
        )
        .map_err(|err| format!("encode failed: {err:?}"))?;
        if len > MAX_PACKET {
            return Err("encoded packet too large".to_string());
        }

        let mut enc_final_range = 0u32;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetFinalRange(&mut enc_final_range))
            .map_err(|err| format!("get final range failed: {err:?}"))?;

        let out_samples = opus_decode(
            &mut dec,
            Some(&packet[..len]),
            len,
            &mut outbuf[offset * 2..],
            MAX_FRAME_SAMP,
            false,
        )
        .map_err(|err| format!("decode failed: {err:?}"))?;
        if out_samples != frame_size {
            return Err("decoded sample count mismatch".to_string());
        }

        let mut dec_final_range = 0u32;
        opus_decoder_ctl(&mut dec, OpusDecoderCtlRequest::GetFinalRange(&mut dec_final_range))
            .map_err(|err| format!("get final range failed: {err:?}"))?;
        let compare_range = allow_silk_final_range
            && matches!(
                opus_packet_get_mode(&packet[..len]),
                Ok(mousiki::packet::Mode::SILK)
            );
        if compare_range && frame_size <= silk_range_limit && dec_final_range != enc_final_range {
            return Err(format!(
                "final range mismatch: frame_size={}, enc={}, dec={}, bitrate={}",
                frame_size, enc_final_range, dec_final_range, bitrate_bps
            ));
        }

        let parsed = opus_packet_parse(&packet[..len], len)
            .map_err(|err| format!("packet parse failed: {err:?}"))?;

        let mut len = len;
        if rng.next() & 1023 == 0 {
            len = 0;
        }
        for byte_idx in parsed.payload_offset..len {
            for bit in 0..8 {
                if !no_fuzz && rng.next() & 1023 == 0 {
                    packet[byte_idx] ^= 1u8 << bit;
                }
            }
        }

        let out_samples = if len > 0 {
            opus_decode(
                &mut dec_err[0],
                Some(&packet[..len]),
                len,
                &mut out2buf,
                MAX_FRAME_SAMP,
                false,
            )
            .map_err(|err| format!("decode failed: {err:?}"))?
        } else {
            opus_decode(&mut dec_err[0], None, 0, &mut out2buf, MAX_FRAME_SAMP, false)
                .map_err(|err| format!("decode failed: {err:?}"))?
        };
        if len > 0 && out_samples != frame_size {
            return Err("decoded sample count mismatch".to_string());
        }

        let mut dec_final_range = 0u32;
        opus_decoder_ctl(&mut dec_err[0], OpusDecoderCtlRequest::GetFinalRange(&mut dec_final_range))
            .map_err(|err| format!("get final range failed: {err:?}"))?;

        let dec2 = (rng.next() % 9 + 1) as usize;
        let out_samples = if len > 0 {
            opus_decode(
                &mut dec_err[dec2],
                Some(&packet[..len]),
                len,
                &mut out2buf,
                MAX_FRAME_SAMP,
                false,
            )
            .map_err(|err| format!("decode failed: {err:?}"))?
        } else {
            opus_decode(
                &mut dec_err[dec2],
                None,
                0,
                &mut out2buf,
                MAX_FRAME_SAMP,
                false,
            )
            .map_err(|err| format!("decode failed: {err:?}"))?
        };
        if out_samples > MAX_FRAME_SAMP {
            return Err("decoded sample count exceeds max".to_string());
        }

        let mut dec_final_range2 = 0u32;
        opus_decoder_ctl(
            &mut dec_err[dec2],
            OpusDecoderCtlRequest::GetFinalRange(&mut dec_final_range2),
        )
        .map_err(|err| format!("get final range failed: {err:?}"))?;
        if len > 0 && dec_final_range != dec_final_range2 {
            return Err(format!(
                "final range mismatch: frame_size={}, dec1={}, dec2={}",
                frame_size, dec_final_range, dec_final_range2
            ));
        }

        fswitch -= 1;
        if fswitch < 1 {
            let new_size;
            fsize = (fsize + 1) % 36;
            new_size = fsizes[db62[fsize] as usize];
            if new_size == 960 || new_size == 480 {
                fswitch = (2880 / new_size) as i32 * ((rng.next() % 19) as i32 + 1);
            } else {
                fswitch = (rng.next() % (2880 / new_size) as u32) as i32 + 1;
            }
        }
        bitrate_bps = (((rng.next() % 508_000) as i32 + 4000) + bitrate_bps) >> 1;
        i += frame_size;
    }

    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::ResetState)
        .map_err(|err| format!("reset encoder failed: {err:?}"))?;
    opus_multistream_encoder_ctl(&mut ms_enc, OpusMultistreamEncoderCtlRequest::ResetState)
        .map_err(|err| format!("reset encoder failed: {err:?}"))?;
    opus_decoder_ctl(&mut dec, OpusDecoderCtlRequest::ResetState)
        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
    opus_multistream_decoder_ctl(&mut ms_dec, OpusMultistreamDecoderCtlRequest::ResetState)
        .map_err(|err| format!("reset decoder failed: {err:?}"))?;
    opus_multistream_decoder_ctl(&mut ms_dec_err, OpusMultistreamDecoderCtlRequest::ResetState)
        .map_err(|err| format!("reset decoder failed: {err:?}"))?;

    Ok(())
}

#[test]
fn opus_encode_decode_smoke() {
    let sampling_rate = 48_000usize;
    let channels = 2usize;
    let frame_size = sampling_rate / 50;
    let frame_enum = get_frame_size_enum(frame_size, sampling_rate);

    let mut enc = opus_encoder_create(sampling_rate as i32, channels as i32, OPUS_APPLICATION_AUDIO)
        .expect("encoder");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetExpertFrameDuration(frame_enum),
    )
    .expect("set frame duration");

    let mut dec = opus_decoder_create(sampling_rate as i32, channels as i32).expect("decoder");

    let total_frames = 10usize;
    let total_samples = frame_size * total_frames;
    let mut inbuf = vec![0i16; total_samples * channels];
    let mut rng = FastRand::new(0xC0DEC0DE);
    generate_music(&mut inbuf, total_samples, &mut rng);

    let mut packet = vec![0u8; MAX_PACKET];
    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * channels];

    for frame_idx in 0..total_frames {
        let start = frame_idx * frame_size * channels;
        let end = start + frame_size * channels;
        let len = opus_encode(&mut enc, &inbuf[start..end], frame_size, &mut packet)
            .expect("encode");
        assert!(len <= MAX_PACKET);

        let decoded = opus_decode(
            &mut dec,
            Some(&packet[..len]),
            len,
            &mut outbuf,
            MAX_FRAME_SAMP,
            false,
        )
        .expect("decode");
        assert_eq!(decoded, frame_size);
    }
}

#[test]
fn opus_encode_silk_10ms_round_trip() {
    let sampling_rate = 48_000usize;
    let channels = 2usize;
    let frame_size = sampling_rate / 100;

    let mut enc = opus_encoder_create(sampling_rate as i32, channels as i32, OPUS_APPLICATION_AUDIO)
        .expect("encoder");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_SILK_ONLY))
        .expect("force silk");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_10_MS),
    )
    .expect("set frame duration");

    let mut dec = opus_decoder_create(sampling_rate as i32, channels as i32).expect("decoder");

    let mut inbuf = vec![0i16; frame_size * channels];
    let mut rng = FastRand::new(0xDEC0DE01);
    generate_music(&mut inbuf, frame_size, &mut rng);

    let mut packet = vec![0u8; MAX_PACKET];
    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * channels];
    let len = opus_encode(&mut enc, &inbuf, frame_size, &mut packet).expect("encode");
    let decoded = opus_decode(
        &mut dec,
        Some(&packet[..len]),
        len,
        &mut outbuf,
        MAX_FRAME_SAMP,
        false,
    )
    .expect("decode");
    assert_eq!(decoded, frame_size);
}

#[test]
fn opus_encode_celt_60ms_multiframe_round_trip() {
    let sampling_rate = 48_000usize;
    let channels = 2usize;
    let frame_size = 3 * sampling_rate / 50;

    let mut enc = opus_encoder_create(sampling_rate as i32, channels as i32, OPUS_APPLICATION_AUDIO)
        .expect("encoder");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY))
        .expect("force celt");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_60_MS),
    )
    .expect("set frame duration");

    let mut dec = opus_decoder_create(sampling_rate as i32, channels as i32).expect("decoder");

    let mut inbuf = vec![0i16; frame_size * channels];
    let mut rng = FastRand::new(0xDEC0DE02);
    generate_music(&mut inbuf, frame_size, &mut rng);

    let mut packet = vec![0u8; MAX_PACKET];
    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * channels];
    let len = opus_encode(&mut enc, &inbuf, frame_size, &mut packet).expect("encode");
    let decoded = opus_decode(
        &mut dec,
        Some(&packet[..len]),
        len,
        &mut outbuf,
        MAX_FRAME_SAMP,
        false,
    )
    .expect("decode");
    assert_eq!(decoded, frame_size);
}

#[test]
fn opus_encode_hybrid_round_trip() {
    let sampling_rate = 48_000usize;
    let channels = 2usize;
    let frame_size = sampling_rate / 50;

    let mut enc = opus_encoder_create(sampling_rate as i32, channels as i32, OPUS_APPLICATION_AUDIO)
        .expect("encoder");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID))
        .expect("force hybrid");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_20_MS),
    )
    .expect("set frame duration");

    let mut dec = opus_decoder_create(sampling_rate as i32, channels as i32).expect("decoder");

    let mut inbuf = vec![0i16; frame_size * channels];
    let mut rng = FastRand::new(0xDEC0DE03);
    generate_music(&mut inbuf, frame_size, &mut rng);

    let mut packet = vec![0u8; MAX_PACKET];
    let mut outbuf = vec![0i16; MAX_FRAME_SAMP * channels];
    let len = opus_encode(&mut enc, &inbuf, frame_size, &mut packet).expect("encode");
    let decoded = opus_decode(
        &mut dec,
        Some(&packet[..len]),
        len,
        &mut outbuf,
        MAX_FRAME_SAMP,
        false,
    )
    .expect("decode");
    assert_eq!(decoded, frame_size);
}

#[test]
fn opus_encode_run_test1() {
    let mut rng = FastRand::new(seed_from_env());
    run_test1(no_fuzz(), &mut rng).expect("run_test1 should succeed");
}

#[test]
fn opus_encode_fuzz_encoder_settings() {
    if no_fuzz() {
        return;
    }
    let mut rng = FastRand::new(seed_from_env().wrapping_add(1));
    fuzz_encoder_settings(5, 40, &mut rng).expect("fuzz settings should succeed");
}
