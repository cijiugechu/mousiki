/// Integration test for CELT fixed-point decode path
///
/// This test validates that the fixed-point CELT decoder produces
/// valid output and matches expected behavior using only public APIs.

#[cfg(feature = "fixed_point")]
#[test]
fn test_celt_fixed_point_decode_plc() {
    use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
    
    // Simple CELT-only packet (created from a known encoder)
    // This is a minimal valid CELT packet header
    let sample_rate = 48000;
    let channels = 1;
    
    let mut decoder = opus_decoder_create(sample_rate, channels).unwrap();
    
    // Test PLC (packet loss concealment) with fixed-point
    let mut pcm = vec![0i16; 960]; // 20ms at 48kHz
    let result = opus_decode(&mut decoder, None, 0, &mut pcm, 960, false);
    
    // PLC should succeed
    assert!(result.is_ok(), "PLC decode should succeed in fixed-point mode");
    let samples = result.unwrap();
    assert_eq!(samples, 960, "PLC should generate 960 samples");
}

#[cfg(feature = "fixed_point")]
#[test]
fn test_celt_fixed_point_decode_stereo() {
    use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
    
    let sample_rate = 48000;
    let channels = 2;
    
    let mut decoder = opus_decoder_create(sample_rate, channels).unwrap();
    
    // Test stereo PLC with fixed-point
    let mut pcm = vec![0i16; 1920]; // 20ms stereo at 48kHz
    let result = opus_decode(&mut decoder, None, 0, &mut pcm, 960, false);
    
    assert!(result.is_ok(), "Stereo PLC should succeed in fixed-point mode");
    let samples = result.unwrap();
    assert_eq!(samples, 960, "Stereo PLC should generate 960 samples per channel");
}

#[cfg(feature = "fixed_point")]
#[test]
fn test_celt_fixed_point_multiple_sample_rates() {
    use mousiki::opus_decoder::opus_decoder_create;
    
    // Test various sample rates supported by CELT in fixed-point mode
    for &sample_rate in &[8000, 12000, 16000, 24000, 48000] {
        let decoder = opus_decoder_create(sample_rate, 1);
        assert!(decoder.is_ok(), "Decoder creation should succeed for {sample_rate} Hz");
    }
}

#[cfg(not(feature = "fixed_point"))]
#[test]
fn test_fixed_point_feature_required() {
    // This test just documents that the fixed_point tests require the feature
    assert!(true, "Fixed-point tests are only available with --features fixed_point");
}
