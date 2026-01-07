//! Tests ported from `opus-c/tests/test_opus_projection.c`
//!
//! Covers:
//! - Creation argument validation for all channel counts
//! - Full encode/decode pipeline with generated audio

use mousiki::projection::{
    opus_projection_ambisonics_encoder_create, opus_projection_decode, opus_projection_decoder_create,
    opus_projection_encode, opus_projection_encoder_ctl, OpusProjectionEncoderCtlRequest,
};
use mousiki::opus_multistream::OpusMultistreamEncoderCtlRequest;

const BUFFER_SIZE: usize = 960;
const MAX_DATA_BYTES: usize = 32768;
const OPUS_APPLICATION_AUDIO: i32 = 2049;

/// Simple LCG for deterministic pseudo-random numbers.
struct FastRand {
    seed: u32,
}

impl FastRand {
    fn new(seed: u32) -> Self {
        Self { seed }
    }

    fn next(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }
}

/// Tests that the projection encoder/decoder creation rejects invalid channel
/// counts and accepts valid Ambisonics configurations.
///
/// Valid configurations for mapping family 3:
/// - channels = (order+1)^2 + 2j where order ∈ [1, 5] and j ∈ {0, 1}
/// - order+1 must be in [2, 6] for precomputed matrices
#[test]
fn creation_arguments_validate_channel_counts() {
    for channels in 0..255 {
        let result = opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3, // mapping family
            OPUS_APPLICATION_AUDIO,
        );

        // Compute expected validity
        let order_plus_one = (channels as f64).sqrt().floor() as usize;
        let acn_channels = order_plus_one * order_plus_one;
        let nondiegetic_channels = channels.saturating_sub(acn_channels);
        let is_channels_valid = (2..=6).contains(&order_plus_one)
            && (nondiegetic_channels == 0 || nondiegetic_channels == 2);

        match result {
            Ok((mut encoder, streams, coupled_streams)) => {
                assert!(
                    is_channels_valid,
                    "Encoder succeeded for invalid channels: {} (order+1: {}, nondiegetic: {})",
                    channels, order_plus_one, nondiegetic_channels
                );

                // Verify demixing matrix can be retrieved
                let mut matrix_size = 0usize;
                opus_projection_encoder_ctl(
                    &mut encoder,
                    OpusProjectionEncoderCtlRequest::GetDemixingMatrixSize(&mut matrix_size),
                )
                .expect("GetDemixingMatrixSize");
                assert!(matrix_size > 0);

                let mut matrix = vec![0u8; matrix_size];
                opus_projection_encoder_ctl(
                    &mut encoder,
                    OpusProjectionEncoderCtlRequest::GetDemixingMatrix(&mut matrix),
                )
                .expect("GetDemixingMatrix");

                // Verify decoder can be created
                let decoder_result = opus_projection_decoder_create(
                    48000,
                    channels,
                    streams,
                    coupled_streams,
                    &matrix,
                );
                assert!(
                    decoder_result.is_ok(),
                    "Decoder creation failed for valid channels: {}",
                    channels
                );
            }
            Err(_) => {
                assert!(
                    !is_channels_valid,
                    "Encoder failed for valid channels: {} (order+1: {}, nondiegetic: {})",
                    channels, order_plus_one, nondiegetic_channels
                );
            }
        }
    }
}

/// Generates deterministic pseudo-random music for testing.
fn generate_music(buf: &mut [i16], channels: usize, rng: &mut FastRand) {
    let frame_count = buf.len() / channels;
    let mut a = vec![0i32; channels];
    let mut b = vec![0i32; channels];
    let mut c = vec![0i32; channels];
    let mut d = vec![0i32; channels];
    let mut j: u32 = 0;

    for i in 0..frame_count {
        for k in 0..channels {
            let v_base = (((j.wrapping_mul(
                (j >> 12) ^ ((j >> 10 | j >> 12) & 26 & j >> 7),
            )) & 128)
                .wrapping_add(128) as i32)
                << 15;
            let r = rng.next();
            let mut v = v_base.wrapping_add((r & 65535) as i32);
            v = v.wrapping_sub((r >> 16) as i32);

            b[k] = v.wrapping_sub(a[k]).wrapping_add((b[k].wrapping_mul(61).wrapping_add(32)) >> 6);
            a[k] = v;
            c[k] = (30i32
                .wrapping_mul(c[k].wrapping_add(b[k]).wrapping_add(d[k]))
                .wrapping_add(32))
                >> 6;
            d[k] = b[k];
            let sample = (c[k].wrapping_add(128)) >> 8;
            buf[i * channels + k] = sample.clamp(-32768, 32767) as i16;

            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }
}

/// Tests the full encode/decode pipeline with generated audio.
#[test]
fn encode_decode_pipeline() {
    let channels = 18; // 4th order + 2 non-diegetic
    let bitrate_per_stream = 64 * 1000;

    let (mut encoder, streams, coupled) =
        opus_projection_ambisonics_encoder_create(48000, channels, 3, OPUS_APPLICATION_AUDIO)
            .expect("encoder creation");

    // Set bitrate
    let total_bitrate = bitrate_per_stream * (streams + coupled) as i32;
    opus_projection_encoder_ctl(
        &mut encoder,
        OpusProjectionEncoderCtlRequest::Multistream(OpusMultistreamEncoderCtlRequest::SetBitrate(
            total_bitrate,
        )),
    )
    .expect("set bitrate");

    // Get demixing matrix
    let mut matrix_size = 0usize;
    opus_projection_encoder_ctl(
        &mut encoder,
        OpusProjectionEncoderCtlRequest::GetDemixingMatrixSize(&mut matrix_size),
    )
    .expect("GetDemixingMatrixSize");

    let mut matrix = vec![0u8; matrix_size];
    opus_projection_encoder_ctl(
        &mut encoder,
        OpusProjectionEncoderCtlRequest::GetDemixingMatrix(&mut matrix),
    )
    .expect("GetDemixingMatrix");

    // Create decoder
    let mut decoder =
        opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
            .expect("decoder creation");

    // Generate test audio
    let mut buffer_in = vec![0i16; BUFFER_SIZE * channels];
    let mut rng = FastRand::new(12345);
    generate_music(&mut buffer_in, channels, &mut rng);

    // Encode
    let mut data = vec![0u8; MAX_DATA_BYTES];
    let len = opus_projection_encode(&mut encoder, &buffer_in, BUFFER_SIZE, &mut data)
        .expect("encode");
    assert!(len > 0 && len <= MAX_DATA_BYTES);

    // Decode
    let mut buffer_out = vec![0i16; BUFFER_SIZE * channels];
    let out_samples =
        opus_projection_decode(&mut decoder, &data, len, &mut buffer_out, BUFFER_SIZE, false)
            .expect("decode");
    assert_eq!(out_samples, BUFFER_SIZE);
}

/// Tests various valid Ambisonics configurations.
#[test]
fn encode_decode_various_orders() {
    // Test configurations: (channels, order_plus_one, description)
    let configs = [
        (4, 2, "FOA (1st order)"),
        (6, 2, "FOA + stereo"),
        (9, 3, "SOA (2nd order)"),
        (11, 3, "SOA + stereo"),
        (16, 4, "TOA (3rd order)"),
        (18, 4, "TOA + stereo"),
        (25, 5, "4th order"),
        (27, 5, "4th order + stereo"),
        (36, 6, "5th order"),
        (38, 6, "5th order + stereo"),
    ];

    for (channels, expected_order, desc) in configs {
        let result =
            opus_projection_ambisonics_encoder_create(48000, channels, 3, OPUS_APPLICATION_AUDIO);

        let (mut encoder, streams, coupled) = result.unwrap_or_else(|e| {
            panic!("Failed to create encoder for {}: {:?}", desc, e);
        });

        // Verify order
        let order_plus_one = (channels as f64).sqrt().floor() as usize;
        assert_eq!(
            order_plus_one, expected_order,
            "Unexpected order for {}",
            desc
        );

        // Get demixing matrix
        let mut matrix_size = 0usize;
        opus_projection_encoder_ctl(
            &mut encoder,
            OpusProjectionEncoderCtlRequest::GetDemixingMatrixSize(&mut matrix_size),
        )
        .expect("GetDemixingMatrixSize");

        let mut matrix = vec![0u8; matrix_size];
        opus_projection_encoder_ctl(
            &mut encoder,
            OpusProjectionEncoderCtlRequest::GetDemixingMatrix(&mut matrix),
        )
        .expect("GetDemixingMatrix");

        // Create decoder
        let mut decoder =
            opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
                .expect("decoder creation");

        // Encode/decode silence
        let buffer_in = vec![0i16; BUFFER_SIZE * channels];
        let mut data = vec![0u8; MAX_DATA_BYTES];
        let len = opus_projection_encode(&mut encoder, &buffer_in, BUFFER_SIZE, &mut data)
            .unwrap_or_else(|e| panic!("Failed to encode for {}: {:?}", desc, e));
        assert!(len > 0, "Encode returned 0 bytes for {}", desc);

        let mut buffer_out = vec![0i16; BUFFER_SIZE * channels];
        let out_samples =
            opus_projection_decode(&mut decoder, &data, len, &mut buffer_out, BUFFER_SIZE, false)
                .unwrap_or_else(|e| panic!("Failed to decode for {}: {:?}", desc, e));
        assert_eq!(out_samples, BUFFER_SIZE, "Wrong output size for {}", desc);
    }
}

