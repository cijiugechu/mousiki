#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::resample::{ResamplerError, SpeexResampler};

const FLOAT_TOL: f32 = 1e-4;
const COMPARE_TOL: f32 = 1e-5;
const SENTINEL_FLOAT: f32 = -1234.5;
const SENTINEL_INT: i16 = 0x5a5a;

fn assert_close(expected: f32, actual: f32, tol: f32) {
    let diff = (expected - actual).abs();
    assert!(
        diff <= tol,
        "expected {expected:.9} got {actual:.9} (tol {tol:.9})"
    );
}

fn assert_all_close(expected: &[f32], actual: &[f32], tol: f32) {
    assert_eq!(expected.len(), actual.len());
    for (index, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tol,
            "index {index}: expected {lhs:.9} got {rhs:.9} (tol {tol:.9})"
        );
    }
}

fn fill_impulse(buffer: &mut [f32], at: usize, amplitude: f32) {
    for (i, sample) in buffer.iter_mut().enumerate() {
        *sample = if i == at { amplitude } else { 0.0 };
    }
}

fn fill_sine(buffer: &mut [f32], scale: f32) {
    for (i, sample) in buffer.iter_mut().enumerate() {
        *sample = scale * libm::sinf(i as f32 * 0.17);
    }
}

fn fill_stereo_asymmetric(buffer: &mut [f32], frames: usize) {
    for i in 0..frames {
        buffer[2 * i] = if i == 0 { 1.0 } else { 0.0 };
        buffer[2 * i + 1] = if i < 24 { 0.5 } else { -0.5 };
    }
}

fn first_nonzero_index(buffer: &[f32], threshold: f32) -> Option<usize> {
    buffer.iter().position(|sample| sample.abs() > threshold)
}

fn peak_abs(buffer: &[f32]) -> f32 {
    buffer.iter().map(|sample| sample.abs()).fold(0.0, f32::max)
}

#[test]
fn init_and_configuration_contracts_match_ctest() {
    assert_eq!(
        ResamplerError::InvalidArg,
        SpeexResampler::new(0, 44_100, 48_000, 5).unwrap_err()
    );
    assert_eq!(
        ResamplerError::InvalidArg,
        SpeexResampler::new_frac(1, 0, 1, 0, 1, 5).unwrap_err()
    );
    assert_eq!(
        ResamplerError::InvalidArg,
        SpeexResampler::new_frac(1, 1, 0, 1, 0, 5).unwrap_err()
    );
    assert_eq!(
        ResamplerError::InvalidArg,
        SpeexResampler::new(1, 44_100, 48_000, -1).unwrap_err()
    );
    assert_eq!(
        ResamplerError::InvalidArg,
        SpeexResampler::new(1, 44_100, 48_000, 11).unwrap_err()
    );

    let mut st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
    assert_eq!((44_100, 48_000), st.rates());
    assert_eq!((147, 160), st.ratio());

    st.set_quality(10).expect("set quality");
    assert_eq!(10, st.quality());

    assert_eq!(Err(ResamplerError::InvalidArg), st.set_quality(-1));
    assert_eq!(Err(ResamplerError::InvalidArg), st.set_quality(11));
    assert_eq!(10, st.quality());

    assert_eq!(
        Err(ResamplerError::InvalidArg),
        st.set_rate_frac(0, 1, 0, 1)
    );
    assert_eq!(
        Err(ResamplerError::InvalidArg),
        st.set_rate_frac(1, 0, 1, 0)
    );

    st.set_rate(48_000, 32_000).expect("set rate");
    assert_eq!((48_000, 32_000), st.rates());
    assert_eq!((3, 2), st.ratio());
    st.set_rate_frac(48_000, 32_000, 48_000, 32_000)
        .expect("set rate frac");

    st.set_input_stride(7);
    assert_eq!(7, st.input_stride());
    st.set_output_stride(9);
    assert_eq!(9, st.output_stride());
}

#[test]
fn strerror_and_latency_match_ctest() {
    assert_eq!("Success.", ResamplerError::Success.strerror());
    assert_eq!("Invalid argument.", ResamplerError::InvalidArg.strerror());
    assert_eq!(
        "Input and output buffers overlap.",
        ResamplerError::PtrOverlap.strerror()
    );
    assert_eq!(
        "Unknown error. Bad error code or strange version mismatch.",
        ResamplerError::Overflow.strerror()
    );

    let st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
    let input_latency = st.input_latency();
    let output_latency = st.output_latency();
    assert!(input_latency > 0);
    assert!(output_latency > 0);

    let (ratio_num, ratio_den) = st.ratio();
    let expected_output_latency = (input_latency * ratio_den + (ratio_num >> 1)) / ratio_num;
    assert_eq!(expected_output_latency, output_latency);
}

#[test]
fn mono_baseline_numeric_matches_ctest() {
    let indices = [0usize, 1, 2, 17, 63, 127, 235];
    let expected = [
        0.002053946,
        0.120288044,
        0.249981761,
        0.373679519,
        -0.322577417,
        0.667169452,
        -0.671027243,
    ];

    let mut st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
    let mut input = [0.0f32; 256];
    let mut output = [0.0f32; 512];
    fill_sine(&mut input, 0.8);
    st.skip_zeros().expect("skip zeros");

    let mut in_len = 256;
    let mut out_len = 512;
    st.process_float(0, Some(&input), &mut in_len, &mut output, &mut out_len)
        .expect("process");
    assert_eq!(256, in_len);
    assert_eq!(236, out_len);

    for (index, expected_value) in indices.into_iter().zip(expected) {
        assert_close(expected_value, output[index], FLOAT_TOL);
    }
}

#[test]
fn partial_output_and_resume_match_ctest() {
    let mut full_state = SpeexResampler::new(1, 44_100, 48_000, 5).expect("full");
    let mut split_state = SpeexResampler::new(1, 44_100, 48_000, 5).expect("split");
    let mut input = [0.0f32; 256];
    let mut full_out = [0.0f32; 512];
    let mut part1_out = [SENTINEL_FLOAT; 16];
    let mut part2_out = [SENTINEL_FLOAT; 512];
    let mut combined = [0.0f32; 236];
    fill_sine(&mut input, 0.8);
    full_state.skip_zeros().expect("skip zeros");
    split_state.skip_zeros().expect("skip zeros");

    let mut full_in_len = 256;
    let mut full_out_len = 512;
    full_state
        .process_float(
            0,
            Some(&input),
            &mut full_in_len,
            &mut full_out,
            &mut full_out_len,
        )
        .expect("full process");
    assert_eq!(256, full_in_len);
    assert_eq!(236, full_out_len);

    let mut first_in_len = 256;
    let mut first_out_len = 10;
    split_state
        .process_float(
            0,
            Some(&input),
            &mut first_in_len,
            &mut part1_out,
            &mut first_out_len,
        )
        .expect("part1");
    assert_eq!(49, first_in_len);
    assert_eq!(10, first_out_len);
    for sample in &part1_out[10..] {
        assert_close(SENTINEL_FLOAT, *sample, 0.0);
    }

    let consumed_first = first_in_len as usize;
    let mut second_in_len = 256 - consumed_first as u32;
    let mut second_out_len = 512;
    split_state
        .process_float(
            0,
            Some(&input[consumed_first..]),
            &mut second_in_len,
            &mut part2_out,
            &mut second_out_len,
        )
        .expect("part2");
    assert_eq!(207, second_in_len);
    assert_eq!(226, second_out_len);

    combined[..10].copy_from_slice(&part1_out[..10]);
    combined[10..].copy_from_slice(&part2_out[..226]);
    assert_all_close(&full_out[..236], &combined, COMPARE_TOL);
}

#[test]
fn skip_zeros_and_reset_mem_match_ctest() {
    let mut no_skip = SpeexResampler::new(1, 44_100, 48_000, 5).expect("no_skip");
    let mut skip = SpeexResampler::new(1, 44_100, 48_000, 5).expect("skip");
    let mut reset = SpeexResampler::new(1, 44_100, 48_000, 5).expect("reset");
    let mut fresh = SpeexResampler::new(1, 44_100, 48_000, 5).expect("fresh");
    let mut input = [0.0f32; 256];
    fill_sine(&mut input, 0.8);

    let mut out_no_skip = [0.0f32; 512];
    let mut in_len = 256;
    let mut out_len_no_skip = 512;
    no_skip
        .process_float(
            0,
            Some(&input),
            &mut in_len,
            &mut out_no_skip,
            &mut out_len_no_skip,
        )
        .expect("no skip process");
    assert_eq!(256, in_len);
    assert_eq!(279, out_len_no_skip);
    assert!(first_nonzero_index(&out_no_skip[..out_len_no_skip as usize], 1e-5).unwrap() > 0);
    assert!(peak_abs(&out_no_skip[..out_len_no_skip as usize]) > 0.1);

    skip.skip_zeros().expect("skip zeros");
    let mut out_skip = [0.0f32; 512];
    let mut in_len = 256;
    let mut out_len_skip = 512;
    skip.process_float(0, Some(&input), &mut in_len, &mut out_skip, &mut out_len_skip)
        .expect("skip process");
    assert_eq!(256, in_len);
    assert_eq!(236, out_len_skip);
    assert_eq!(
        Some(0),
        first_nonzero_index(&out_skip[..out_len_skip as usize], 1e-5)
    );

    let mut out_after_reset = [0.0f32; 512];
    let mut in_len = 256;
    let mut out_len_reset = 512;
    reset
        .process_float(
            0,
            Some(&input),
            &mut in_len,
            &mut out_after_reset,
            &mut out_len_reset,
        )
        .expect("reset warmup");
    reset.reset_mem().expect("reset mem");

    out_after_reset.fill(0.0);
    in_len = 256;
    out_len_reset = 512;
    reset
        .process_float(
            0,
            Some(&input),
            &mut in_len,
            &mut out_after_reset,
            &mut out_len_reset,
        )
        .expect("after reset");

    let mut out_fresh = [0.0f32; 512];
    let mut in_len_fresh = 256;
    let mut out_len_fresh = 512;
    fresh
        .process_float(
            0,
            Some(&input),
            &mut in_len_fresh,
            &mut out_fresh,
            &mut out_len_fresh,
        )
        .expect("fresh process");

    assert_eq!(out_len_fresh, out_len_reset);
    assert_all_close(
        &out_fresh[..out_len_fresh as usize],
        &out_after_reset[..out_len_reset as usize],
        COMPARE_TOL,
    );
}

#[test]
fn stateful_rate_change_matches_ctest() {
    let mut st = SpeexResampler::new(1, 48_000, 48_000, 5).expect("resampler");
    st.skip_zeros().expect("skip zeros");

    let mut impulse = [0.0f32; 128];
    let mut out = [0.0f32; 512];
    fill_impulse(&mut impulse, 0, 1.0);
    let mut in_len = 128;
    let mut out_len = 512;
    st.process_float(0, Some(&impulse), &mut in_len, &mut out, &mut out_len)
        .expect("48k process");
    assert!(out_len > 0);

    st.set_rate_frac(48_000, 32_000, 48_000, 32_000)
        .expect("set rate frac");
    assert_eq!((3, 2), st.ratio());
    assert_eq!(60, st.input_latency());
    assert_eq!(40, st.output_latency());

    fill_impulse(&mut impulse, 0, 1.0);
    out.fill(0.0);
    in_len = 128;
    out_len = 512;
    st.process_float(0, Some(&impulse), &mut in_len, &mut out, &mut out_len)
        .expect("32k process");
    assert_eq!(128, in_len);
    assert_eq!(72, out_len);
    assert_close(0.000699097, out[0], FLOAT_TOL);
    assert_close(-0.000801697, out[1], FLOAT_TOL);
    assert_close(0.000838438, out[2], FLOAT_TOL);
}

#[test]
fn stride_processing_matches_ctest() {
    let mut reference = SpeexResampler::new(1, 16_000, 48_000, 5).expect("reference");
    let mut stride = SpeexResampler::new(1, 16_000, 48_000, 5).expect("stride");

    let mut compact_in = [0.0f32; 96];
    let mut sparse_in = [SENTINEL_FLOAT; 192];
    let mut reference_out = [0.0f32; 512];
    let mut sparse_out = [SENTINEL_FLOAT; 1536];
    fill_sine(&mut compact_in, 0.6);
    for (i, sample) in compact_in.iter().enumerate() {
        sparse_in[2 * i] = *sample;
    }

    reference.skip_zeros().expect("reference skip");
    stride.skip_zeros().expect("stride skip");
    stride.set_input_stride(2);
    stride.set_output_stride(3);
    assert_eq!(2, stride.input_stride());
    assert_eq!(3, stride.output_stride());

    let mut in_len_ref = 96;
    let mut out_len_ref = 512;
    reference
        .process_float(
            0,
            Some(&compact_in),
            &mut in_len_ref,
            &mut reference_out,
            &mut out_len_ref,
        )
        .expect("reference process");

    let mut in_len_stride = 96;
    let mut out_len_stride = 512;
    stride
        .process_float(
            0,
            Some(&sparse_in),
            &mut in_len_stride,
            &mut sparse_out,
            &mut out_len_stride,
        )
        .expect("stride process");
    assert_eq!(in_len_ref, in_len_stride);
    assert_eq!(out_len_ref, out_len_stride);

    for i in 0..out_len_stride as usize {
        assert_close(reference_out[i], sparse_out[3 * i], COMPARE_TOL);
        assert_close(SENTINEL_FLOAT, sparse_out[3 * i + 1], 0.0);
        assert_close(SENTINEL_FLOAT, sparse_out[3 * i + 2], 0.0);
    }
    for sample in &sparse_out[out_len_stride as usize * 3..] {
        assert_close(SENTINEL_FLOAT, *sample, 0.0);
    }
}

#[test]
fn interleaved_float_channel_isolation_matches_ctest() {
    let mut stereo = SpeexResampler::new(2, 44_100, 48_000, 5).expect("stereo");
    let mut left = SpeexResampler::new(1, 44_100, 48_000, 5).expect("left");
    let mut right = SpeexResampler::new(1, 44_100, 48_000, 5).expect("right");
    let mut interleaved_in = [0.0f32; 96];
    let mut left_in = [0.0f32; 48];
    let mut right_in = [0.0f32; 48];
    let mut stereo_out = [0.0f32; 512];
    let mut left_out = [0.0f32; 256];
    let mut right_out = [0.0f32; 256];
    fill_stereo_asymmetric(&mut interleaved_in, 48);
    for i in 0..48 {
        left_in[i] = interleaved_in[2 * i];
        right_in[i] = interleaved_in[2 * i + 1];
    }

    stereo.skip_zeros().expect("stereo skip");
    left.skip_zeros().expect("left skip");
    right.skip_zeros().expect("right skip");

    let mut stereo_in_len = 48;
    let mut stereo_out_len = 256;
    stereo
        .process_interleaved_float(
            Some(&interleaved_in),
            &mut stereo_in_len,
            &mut stereo_out,
            &mut stereo_out_len,
        )
        .expect("stereo process");

    let mut mono_in_len = 48;
    let mut left_out_len = 256;
    left.process_float(0, Some(&left_in), &mut mono_in_len, &mut left_out, &mut left_out_len)
        .expect("left process");
    assert_eq!(48, mono_in_len);

    mono_in_len = 48;
    let mut right_out_len = 256;
    right
        .process_float(
            0,
            Some(&right_in),
            &mut mono_in_len,
            &mut right_out,
            &mut right_out_len,
        )
        .expect("right process");
    assert_eq!(48, mono_in_len);

    assert_eq!(left_out_len, stereo_out_len);
    assert_eq!(right_out_len, stereo_out_len);
    for i in 0..stereo_out_len as usize {
        assert_close(left_out[i], stereo_out[2 * i], 1e-6);
        assert_close(right_out[i], stereo_out[2 * i + 1], 1e-6);
    }
}

#[test]
fn interleaved_int_smoke_matches_ctest() {
    let mut st = SpeexResampler::new(2, 16_000, 48_000, 5).expect("resampler");
    let mut input = [0i16; 256];
    let mut output = [SENTINEL_INT; 1032];
    for i in 0..128 {
        input[2 * i] = if i < 64 { 12_000 } else { -12_000 };
        input[2 * i + 1] = 0;
    }
    st.skip_zeros().expect("skip zeros");

    let mut in_len = 128;
    let mut out_len = 512;
    st.process_interleaved_int(Some(&input), &mut in_len, &mut output, &mut out_len)
        .expect("process");
    assert_eq!(128, in_len);
    assert!(out_len > 0);
    for i in 0..out_len as usize {
        assert_eq!(0, output[2 * i + 1]);
    }
    assert!(output[0] != 0 || output[2] != 0);
    for sample in &output[2 * out_len as usize..] {
        assert_eq!(SENTINEL_INT, *sample);
    }
}
