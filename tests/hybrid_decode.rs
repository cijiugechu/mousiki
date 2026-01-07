use mousiki::opus_decoder::{
    opus_decode_float, opus_decoder_create, opus_decoder_ctl, OpusDecoderCtlRequest,
};

const FRAME_SIZE: usize = 960;
#[cfg(not(feature = "fixed_point"))]
const PCM_TOLERANCE: f32 = 1.0e-4;

// Vectors generated from the opus-c reference encoder/decoder.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/hybrid_decode_vectors.rs"
));

#[test]
fn hybrid_transition_final_range_matches_reference() {
    let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
    let mut pcm = vec![0.0f32; FRAME_SIZE];

    let decoded = opus_decode_float(
        &mut decoder,
        Some(&TRANSITION_SILK_PACKET),
        TRANSITION_SILK_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("silk decode should succeed");
    assert_eq!(decoded, FRAME_SIZE);

    let decoded = opus_decode_float(
        &mut decoder,
        Some(&TRANSITION_HYBRID_PACKET),
        TRANSITION_HYBRID_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("hybrid decode should succeed");
    assert_eq!(decoded, FRAME_SIZE);

    let mut range = 0;
    opus_decoder_ctl(
        &mut decoder,
        OpusDecoderCtlRequest::GetFinalRange(&mut range),
    )
    .expect("get final range");
    assert_eq!(range, TRANSITION_HYBRID_RANGE);
}

#[cfg(not(feature = "fixed_point"))]
#[test]
fn hybrid_transition_pcm_and_plc_match_reference() {
    let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
    let mut pcm = vec![0.0f32; FRAME_SIZE];

    opus_decode_float(
        &mut decoder,
        Some(&TRANSITION_SILK_PACKET),
        TRANSITION_SILK_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("silk decode should succeed");

    opus_decode_float(
        &mut decoder,
        Some(&TRANSITION_HYBRID_PACKET),
        TRANSITION_HYBRID_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("hybrid decode should succeed");

    assert_pcm_matches(&pcm, &TRANSITION_HYBRID_PCM, PCM_TOLERANCE);

    let decoded = opus_decode_float(&mut decoder, None, 0, &mut pcm, FRAME_SIZE, false)
        .expect("hybrid PLC should succeed");
    assert_eq!(decoded, FRAME_SIZE);
    assert_pcm_matches(&pcm, &HYBRID_PLC_PCM, PCM_TOLERANCE);
}

#[test]
fn hybrid_fec_final_range_matches_reference() {
    let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
    let mut pcm = vec![0.0f32; FRAME_SIZE];

    let decoded = opus_decode_float(
        &mut decoder,
        Some(&FEC_PREV_PACKET),
        FEC_PREV_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("hybrid decode should succeed");
    assert_eq!(decoded, FRAME_SIZE);

    let decoded = opus_decode_float(
        &mut decoder,
        Some(&FEC_PACKET),
        FEC_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        true,
    )
    .expect("hybrid FEC decode should succeed");
    assert_eq!(decoded, FRAME_SIZE);

    let mut range = 0;
    opus_decoder_ctl(
        &mut decoder,
        OpusDecoderCtlRequest::GetFinalRange(&mut range),
    )
    .expect("get final range");
    assert_eq!(range, FEC_RANGE);
}

#[cfg(not(feature = "fixed_point"))]
#[test]
fn hybrid_fec_pcm_matches_reference() {
    let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
    let mut pcm = vec![0.0f32; FRAME_SIZE];

    opus_decode_float(
        &mut decoder,
        Some(&FEC_PREV_PACKET),
        FEC_PREV_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        false,
    )
    .expect("hybrid decode should succeed");

    opus_decode_float(
        &mut decoder,
        Some(&FEC_PACKET),
        FEC_PACKET.len(),
        &mut pcm,
        FRAME_SIZE,
        true,
    )
    .expect("hybrid FEC decode should succeed");

    assert_pcm_matches(&pcm, &FEC_PCM, PCM_TOLERANCE);
}

#[cfg(not(feature = "fixed_point"))]
fn assert_pcm_matches(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual_sample, &expected_sample)) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        let delta = (actual_sample - expected_sample).abs();
        assert!(
            delta <= tol,
            "sample {index} mismatch: {actual_sample} vs {expected_sample} (delta {delta})"
        );
    }
}
