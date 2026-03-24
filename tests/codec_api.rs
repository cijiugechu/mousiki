use mousiki::{
    Application, Bitrate, Channels, Decoder, Encoder, FrameDuration, OpusDecodeError,
    OpusEncodeError, Signal,
};

const FRAME_SIZE: usize = 960;
const SAMPLE_RATE: u32 = 48_000;
const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

#[test]
fn high_level_round_trip_and_configuration() {
    let mut encoder = Encoder::builder(SAMPLE_RATE, Channels::Stereo, Application::Audio)
        .bitrate(Bitrate::Bits(64_000))
        .complexity(7)
        .vbr(false)
        .signal(Signal::Music)
        .frame_duration(FrameDuration::Ms20)
        .build()
        .expect("encoder");

    assert_eq!(encoder.bitrate().expect("bitrate"), Bitrate::Bits(64_000));
    assert_eq!(encoder.complexity().expect("complexity"), 7);
    assert!(!encoder.vbr().expect("vbr"));
    assert_eq!(encoder.signal().expect("signal"), Signal::Music);
    assert_eq!(
        encoder.frame_duration().expect("frame duration"),
        FrameDuration::Ms20
    );

    let mut decoder = Decoder::builder(SAMPLE_RATE, Channels::Stereo)
        .gain(256)
        .complexity(6)
        .build()
        .expect("decoder");

    assert_eq!(decoder.gain().expect("gain"), 256);
    assert_eq!(decoder.complexity().expect("complexity"), 6);

    let channels = Channels::Stereo.count();
    let mut input = vec![0i16; FRAME_SIZE * channels];
    for (idx, sample) in input.iter_mut().enumerate() {
        *sample = ((idx as i32 * 31) % i16::MAX as i32) as i16;
    }

    let mut packet = vec![0u8; MAX_PACKET_SIZE];
    let packet_len = encoder.encode(&input, &mut packet).expect("encode");
    assert!(packet_len > 0);
    assert_eq!(
        decoder
            .packet_samples(&packet[..packet_len])
            .expect("packet samples"),
        FRAME_SIZE
    );

    let mut output = vec![0i16; MAX_FRAME_SIZE * channels];
    let decoded = decoder
        .decode(&packet[..packet_len], &mut output, false)
        .expect("decode");
    assert_eq!(decoded, FRAME_SIZE);
    assert!(
        decoder
            .last_packet_duration()
            .expect("last packet duration")
            > 0
    );
    assert!(decoder.final_range().expect("final range") > 0);
}

#[test]
fn high_level_encode_rejects_partial_frame_slices() {
    let mut encoder =
        Encoder::new(SAMPLE_RATE, Channels::Stereo, Application::Audio).expect("encoder");
    let pcm = [0i16; FRAME_SIZE * 2 - 1];
    let mut packet = [0u8; MAX_PACKET_SIZE];

    let err = encoder
        .encode(&pcm, &mut packet)
        .expect_err("partial stereo frame should fail");
    assert_eq!(err, OpusEncodeError::BadArgument);
}

#[test]
fn high_level_decode_rejects_partial_frame_buffers() {
    let mut decoder = Decoder::new(SAMPLE_RATE, Channels::Stereo).expect("decoder");
    let packet = [0u8; 1];
    let mut pcm = [0i16; MAX_FRAME_SIZE * 2 - 1];

    let err = decoder
        .decode(&packet, &mut pcm, false)
        .expect_err("partial stereo buffer should fail");
    assert_eq!(err, OpusDecodeError::BadArgument);
}
