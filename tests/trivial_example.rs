use mousiki::c_style_api::opus_decoder::{opus_decode, opus_decoder_create};
use mousiki::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};
#[cfg(not(feature = "fixed_point"))]
use sha2::{Digest, Sha256};

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

#[cfg(not(feature = "fixed_point"))]
#[test]
fn trivial_example_default_build_golden_hash() {
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    const INPUT_PATH: &str = "testdata/ehren-paper_lights-96.pcm";
    const EXPECTED_PCM_SHA256: &str =
        "c7e5724d06fbcb41e94998e553811d0732ba35b3660d975be9f8747c726757bd";

    let mut encoder =
        opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION).expect("encoder init");
    opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(BITRATE))
        .expect("set bitrate");
    let mut decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS).expect("decoder init");

    let input_path = Path::new(INPUT_PATH);
    let mut input_file = File::open(input_path).expect("open trivial_example input");

    let channels = CHANNELS as usize;
    let mut input_bytes = vec![0u8; FRAME_SIZE * channels * 2];
    let mut input_pcm = vec![0i16; FRAME_SIZE * channels];
    let mut output_pcm = vec![0i16; MAX_FRAME_SIZE * channels];
    let mut packet = vec![0u8; MAX_PACKET_SIZE];
    let mut hash = Sha256::new();

    loop {
        match input_file.read_exact(&mut input_bytes) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => panic!("read trivial_example input: {err}"),
        }

        for (sample, chunk) in input_pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
            *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let packet_len =
            opus_encode(&mut encoder, &input_pcm, FRAME_SIZE, &mut packet).expect("encode");
        let decoded = opus_decode(
            &mut decoder,
            Some(&packet[..packet_len]),
            packet_len,
            &mut output_pcm,
            MAX_FRAME_SIZE,
            false,
        )
        .expect("decode");

        let total_samples = decoded * channels;
        for &sample in output_pcm.iter().take(total_samples) {
            hash.update(sample.to_le_bytes());
        }
    }

    let digest = format!("{:x}", hash.finalize());
    assert_eq!(digest, EXPECTED_PCM_SHA256);
}
