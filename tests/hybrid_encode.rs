#![cfg(not(feature = "fixed_point"))]

use mousiki::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};
use mousiki::packet::{Mode, opus_packet_get_mode};

const OPUS_APPLICATION_VOIP: i32 = 2048;
const OPUS_APPLICATION_AUDIO: i32 = 2049;

const MODE_HYBRID: i32 = 1001;

const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

const OPUS_SIGNAL_VOICE: i32 = 3001;
const OPUS_SIGNAL_MUSIC: i32 = 3002;

const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: usize = 2;
const FRAME_SIZE: usize = 960;
const MAX_PACKET: usize = 1500;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/hybrid_encode_vectors.rs"
));

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn fill_test_pcm(buf: &mut [i16], channels: usize, seed: i32) {
    let frames = buf.len() / channels;
    for frame in 0..frames {
        for ch in 0..channels {
            let idx = frame * channels + ch;
            let base = 2000 + seed;
            let wave = ((frame as i32 * 37 + ch as i32 * 13 + seed) % 400) - 200;
            let value = base + wave * 10;
            buf[idx] = clamp_i16(value);
        }
    }
}

#[test]
fn hybrid_encode_hp_filter_and_delay_vectors_match_reference() {
    let mut enc =
        opus_encoder_create(SAMPLE_RATE, CHANNELS as i32, OPUS_APPLICATION_VOIP).expect("encoder");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID))
        .expect("force hybrid");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    )
    .expect("set bandwidth");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(20_000))
        .expect("set bitrate");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(false)).expect("disable vbr");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDtx(false)).expect("disable dtx");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_VOICE))
        .expect("set signal");

    let mut pcm = vec![0i16; FRAME_SIZE * CHANNELS * 2];
    fill_test_pcm(&mut pcm, CHANNELS, 0);
    let mut packet = vec![0u8; MAX_PACKET];

    let len0 = opus_encode(&mut enc, &pcm[..FRAME_SIZE * CHANNELS], FRAME_SIZE, &mut packet)
        .expect("encode frame 0");
    assert_eq!(&packet[..len0], &HP_DELAY_PACKET0[..]);
    assert_eq!(opus_packet_get_mode(&packet[..len0]), Ok(Mode::HYBRID));
    let mut range0 = 0u32;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetFinalRange(&mut range0))
        .expect("get final range 0");
    assert_eq!(range0, HP_DELAY_RANGE0);

    let len1 = opus_encode(
        &mut enc,
        &pcm[FRAME_SIZE * CHANNELS..],
        FRAME_SIZE,
        &mut packet,
    )
    .expect("encode frame 1");
    assert_eq!(&packet[..len1], &HP_DELAY_PACKET1[..]);
    assert_eq!(opus_packet_get_mode(&packet[..len1]), Ok(Mode::HYBRID));
    let mut range1 = 0u32;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetFinalRange(&mut range1))
        .expect("get final range 1");
    assert_eq!(range1, HP_DELAY_RANGE1);
}

#[test]
fn hybrid_encode_stereo_width_vectors_match_reference() {
    let mut enc =
        opus_encoder_create(SAMPLE_RATE, CHANNELS as i32, OPUS_APPLICATION_AUDIO).expect("encoder");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID))
        .expect("force hybrid");
    opus_encoder_ctl(
        &mut enc,
        OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_FULLBAND),
    )
    .expect("set bandwidth");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(12_000))
        .expect("set bitrate");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(false)).expect("disable vbr");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDtx(false)).expect("disable dtx");
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC))
        .expect("set signal");

    let mut pcm = vec![0i16; FRAME_SIZE * CHANNELS];
    fill_test_pcm(&mut pcm, CHANNELS, 42);
    let mut packet = vec![0u8; MAX_PACKET];

    let len = opus_encode(&mut enc, &pcm, FRAME_SIZE, &mut packet).expect("encode frame");
    assert_eq!(&packet[..len], &STEREO_WIDTH_PACKET[..]);
    assert_eq!(opus_packet_get_mode(&packet[..len]), Ok(Mode::HYBRID));
    let mut range = 0u32;
    opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetFinalRange(&mut range))
        .expect("get final range");
    assert_eq!(range, STEREO_WIDTH_RANGE);
}
