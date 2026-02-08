use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
use mousiki::opus_encoder::{
    opus_encode, opus_encoder_create, opus_encoder_ctl, OpusEncoderCtlRequest,
};

const FRAME_SIZE: usize = 960;
const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: i32 = 2;
const APPLICATION: i32 = 2049; // OPUS_APPLICATION_AUDIO
const BITRATE: i32 = 64_000;
const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

#[test]
fn trivial_example_round_trip() {
    let mut encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION).expect("encoder init");
    opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(BITRATE))
        .expect("set bitrate");
    let mut decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS).expect("decoder init");

    let channels = CHANNELS as usize;
    let mut input = vec![0i16; FRAME_SIZE * channels];
    for (idx, sample) in input.iter_mut().enumerate() {
        *sample = ((idx as i32 * 31) % i16::MAX as i32) as i16;
    }

    let mut packet = vec![0u8; MAX_PACKET_SIZE];
    let packet_len = opus_encode(&mut encoder, &input, FRAME_SIZE, &mut packet).expect("encode");
    assert!(packet_len > 0);

    let mut output = vec![0i16; MAX_FRAME_SIZE * channels];
    let decoded = opus_decode(
        &mut decoder,
        Some(&packet[..packet_len]),
        packet_len,
        &mut output,
        MAX_FRAME_SIZE,
        false,
    )
    .expect("decode");
    assert!(decoded > 0);
    assert!(decoded <= MAX_FRAME_SIZE);
}
