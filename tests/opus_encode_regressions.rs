//! Regression vectors ported from `opus-c/tests/opus_encode_regressions.c`.

mod opus_encode_regressions_data;

use mousiki::opus_encoder::{OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl};
use mousiki::opus_multistream::{
    OpusMultistreamEncoderCtlRequest, opus_multistream_encode, opus_multistream_encoder_create,
    opus_multistream_encoder_ctl, opus_multistream_surround_encoder_create,
};
use opus_encode_regressions_data::*;

const OPUS_APPLICATION_VOIP: i32 = 2048;
const OPUS_APPLICATION_AUDIO: i32 = 2049;
const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

const OPUS_AUTO: i32 = -1000;
const OPUS_SIGNAL_VOICE: i32 = 3001;
const OPUS_SIGNAL_MUSIC: i32 = 3002;

const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

fn celt_ec_internal_error() {
    let (mut enc, _layout) =
        opus_multistream_surround_encoder_create(16_000, 1, 1, OPUS_APPLICATION_VOIP)
            .expect("surround encoder");
    let mut data = vec![0u8; 2460];

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(false));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(0));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_AUTO));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(8));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(0),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBitrate(OPUS_AUTO),
    );
    let len = opus_multistream_encode(&mut enc, &CELT_EC_INTERNAL_ERROR_PCM0, 320, &mut data)
        .expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(10));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(18));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(90),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBitrate(280_130),
    );
    let len = opus_multistream_encode(&mut enc, &CELT_EC_INTERNAL_ERROR_PCM1, 160, &mut data)
        .expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(10));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(18));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(90),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBitrate(280_130),
    );
    let len = opus_multistream_encode(&mut enc, &CELT_EC_INTERNAL_ERROR_PCM2, 160, &mut data)
        .expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(10));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(18));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(90),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBitrate(280_130),
    );
    let len = opus_multistream_encode(&mut enc, &CELT_EC_INTERNAL_ERROR_PCM3, 160, &mut data)
        .expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(10));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(18));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(90),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBitrate(280_130),
    );
    let mut pcm = [0i16; CELT_EC_INTERNAL_ERROR_PCM4_LEN];
    pcm[..CELT_EC_INTERNAL_ERROR_PCM4.len()].copy_from_slice(&CELT_EC_INTERNAL_ERROR_PCM4);
    let len = opus_multistream_encode(&mut enc, &pcm, 160, &mut data).expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(0));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_AUTO));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(12));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(41),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBitrate(21_425));
    let len = opus_multistream_encode(&mut enc, &CELT_EC_INTERNAL_ERROR_PCM5, 40, &mut data)
        .expect("encode");
    assert!(len > 0);
}

fn mscbr_encode_fail10() {
    let mapping: [u8; 255] = core::array::from_fn(|i| i as u8);
    let mut enc = opus_multistream_encoder_create(
        8_000,
        255,
        254,
        1,
        &mapping,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("ms encoder");

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(false));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetForceChannels(2),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(2));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_AUTO));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(14));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(57),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBitrate(3_642_675));

    let pcm = [0i16; 20 * 255];
    let mut data = vec![0u8; 627_300];
    let len = opus_multistream_encode(&mut enc, &pcm, 20, &mut data).expect("encode");
    assert!(len > 0);
}

fn mscbr_encode_fail() {
    let mapping: [u8; 192] = core::array::from_fn(|i| i as u8);
    let mut enc = opus_multistream_encoder_create(
        8_000,
        192,
        189,
        3,
        &mapping,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("ms encoder");

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(false));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetForceChannels(OPUS_AUTO),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(0));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_MEDIUMBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_AUTO));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(8));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(0),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBitrate(15_360));

    let pcm = [0i16; 20 * 192];
    let mut data = vec![0u8; 472_320];
    let len = opus_multistream_encode(&mut enc, &pcm, 20, &mut data).expect("encode");
    assert!(len > 0);
}

fn surround_analysis_uninit() {
    let (mut enc, _layout) =
        opus_multistream_surround_encoder_create(24_000, 3, 1, OPUS_APPLICATION_AUDIO)
            .expect("surround encoder");
    let mut data = vec![0u8; 7_380];

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbrConstraint(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetForceChannels(OPUS_AUTO),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(false),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(false));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(0));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(8));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBitrate(84_315));
    let len = opus_multistream_encode(&mut enc, &SURROUND_ANALYSIS_UNINIT_PCM0, 960, &mut data)
        .expect("encode");
    assert!(len > 0);

    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetVbr(true));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetVbrConstraint(false),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetForceChannels(OPUS_AUTO),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(true),
    );
    let _ = opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetDtx(true));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetComplexity(6));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBandwidth(OPUS_AUTO));
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetLsbDepth(9));
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetInbandFec(true),
    );
    let _ = opus_multistream_encoder_ctl(
        &mut enc,
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(5),
    );
    let _ =
        opus_multistream_encoder_ctl(&mut enc, OpusMultistreamEncoderCtlRequest::SetBitrate(775_410));
    let mut pcm = [0i16; SURROUND_ANALYSIS_UNINIT_PCM1_LEN];
    pcm[..SURROUND_ANALYSIS_UNINIT_PCM1.len()].copy_from_slice(&SURROUND_ANALYSIS_UNINIT_PCM1);
    let len = opus_multistream_encode(&mut enc, &pcm, 1440, &mut data).expect("encode");
    assert!(len > 0);
}

fn ec_enc_shrink_assert() {
    let mut enc = opus_encoder_create(48_000, 1, OPUS_APPLICATION_AUDIO).expect("encoder");
    let mut data = [0u8; 2000];

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(10));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPacketLossPerc(6));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(6_000));

    let mut pcm1 = [0i16; 960];
    pcm1[0] = 5140;
    let len = opus_encode(&mut enc, &pcm1, 960, &mut data).expect("encode");
    assert!(len > 0);

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPredictionDisabled(true));
    let _ = opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND),
    );
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetInbandFec(true));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(15_600));

    let mut pcm2 = [0i16; EC_ENC_SHRINK_ASSERT_PCM20_LEN];
    pcm2[..EC_ENC_SHRINK_ASSERT_PCM20.len()].copy_from_slice(&EC_ENC_SHRINK_ASSERT_PCM20);
    let len = opus_encode(&mut enc, &pcm2, 2880, &mut data[..122]).expect("encode");
    assert!(len > 0);

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(27_000));

    let pcm3 = [0i16; 2880];
    let len = opus_encode(&mut enc, &pcm3, 2880, &mut data[..122]).expect("encode");
    assert!(len > 0);
}

fn ec_enc_shrink_assert2() {
    let mut enc = opus_encoder_create(48_000, 1, OPUS_APPLICATION_AUDIO).expect("encoder");
    let mut data = [0u8; 2000];

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(6));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE));
    let _ =
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPacketLossPerc(26));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(27_000));

    let pcm = [0i16; 960];
    let len = opus_encode(&mut enc, &pcm, 960, &mut data).expect("encode");
    assert!(len > 0);

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC));

    let mut pcm = [0i16; EC_ENC_SHRINK_ASSERT2_PCM0_LEN];
    pcm[..EC_ENC_SHRINK_ASSERT2_PCM0.len()].copy_from_slice(&EC_ENC_SHRINK_ASSERT2_PCM0);
    let len = opus_encode(&mut enc, &pcm, 480, &mut data[..19]).expect("encode");
    assert!(len > 0);
}

fn silk_gain_assert() {
    let mut enc = opus_encoder_create(8_000, 1, OPUS_APPLICATION_AUDIO).expect("encoder");
    let mut data = [0u8; 1000];

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(3));
    let _ = opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_NARROWBAND),
    );
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(6_000));

    let pcm1 = [0i16; 160];
    let len = opus_encode(&mut enc, &pcm1, 160, &mut data).expect("encode");
    assert!(len > 0);

    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(false));
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetComplexity(0));
    let _ = opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_MEDIUMBAND),
    );
    let _ = opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(2_867));

    let mut pcm2 = [0i16; SILK_GAIN_ASSERT_PCM20_LEN];
    pcm2[..SILK_GAIN_ASSERT_PCM20.len()].copy_from_slice(&SILK_GAIN_ASSERT_PCM20);
    let len = opus_encode(&mut enc, &pcm2, 960, &mut data).expect("encode");
    assert!(len > 0);
}

#[test]
fn opus_encode_regressions() {
    celt_ec_internal_error();
    mscbr_encode_fail10();
    mscbr_encode_fail();
    surround_analysis_uninit();
    ec_enc_shrink_assert();
    ec_enc_shrink_assert2();
    silk_gain_assert();
}
