extern crate alloc;

use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;
use std::borrow::Cow;

use libm::{cosf, powf, sqrtf};
use mousiki::dred::DredVectorDecoder;
use mousiki::fargan::{FarganState, FARGAN_CONT_SAMPLES, FARGAN_FRAME_SIZE};
use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
#[cfg(feature = "deep_plc_weights")]
use mousiki_deep_plc_weights::DNN_BLOB;

#[path = "../celt/mini_kfft.rs"]
mod mini_kfft;

use mini_kfft::{KissFftCpx, MiniKissFftr};

const NB_FEATURES: usize = 20;
const NBANDS: usize = 17;
const NFREQS: usize = 320;
const TEST_WIN_SIZE: usize = 640;
const TEST_WIN_STEP: usize = 160;
const BANDS: [usize; NBANDS + 1] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 272, 320,
];
const PITCH_MIN: usize = 32;
const PITCH_MAX: usize = 256;
const PITCH_FRAME: usize = 320;
const LOUDNESS: f32 = 0.2;

const DRED_DECODE_THRESHOLDS: (f32, f32, f32) = (0.5, 0.15, 0.02);
const FARGAN_THRESHOLDS: (f32, f32, f32) = (0.25, 1.0, 0.15);
const OPUS_THRESHOLDS: (f32, f32, f32) = (0.5, 1.5, 0.25);

#[cfg(feature = "deep_plc_weights")]
const USAGE: &str = "usage: dred_vectors [--dnn-blob <path>] <vector path>\n\
       dred_vectors [--dnn-blob <path>] <exec path> <vector path>\n\
       (defaults to embedded weights; or set DRED_VECTORS_PATH)";
#[cfg(not(feature = "deep_plc_weights"))]
const USAGE: &str = "usage: dred_vectors --dnn-blob <path> <vector path>\n\
       dred_vectors --dnn-blob <path> <exec path> <vector path>\n\
       (set DRED_VECTORS_PATH to skip positional args)";

#[cfg(not(feature = "deep_plc"))]
fn main() {
    eprintln!("dred_vectors requires the deep_plc feature");
    process::exit(1);
}

#[cfg(feature = "deep_plc")]
fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}

#[cfg(feature = "deep_plc")]
fn run() -> Result<(), String> {
    let args = parse_args()?;
    if !args.vector_path.is_dir() {
        println!("No test vectors found");
        return Ok(());
    }

    println!("Test vectors found in {}", args.vector_path.display());

    #[cfg(feature = "deep_plc_weights")]
    let blob = match args.dnn_blob {
        Some(ref path) => Cow::Owned(fs::read(path).map_err(|err| {
            format!("Error opening DNN blob {}: {err}", path.display())
        })?),
        None => Cow::Borrowed(DNN_BLOB),
    };
    #[cfg(not(feature = "deep_plc_weights"))]
    let blob = {
        let path = args
            .dnn_blob
            .as_ref()
            .expect("dnn_blob required when deep_plc_weights is disabled");
        Cow::Owned(
            fs::read(path)
                .map_err(|err| format!("Error opening DNN blob {}: {err}", path.display()))?,
        )
    };

    let mut fargan = FarganState::new();
    fargan
        .load_model(&blob)
        .map_err(|err| format!("Failed to load FARGAN model: {err:?}"))?;

    let dred_decoder = DredVectorDecoder::new();

    println!("==============");
    println!("Testing DRED decoding");
    println!("==============");
    println!();
    run_dred_decode_tests(&args.vector_path, &dred_decoder)?;

    println!("==============");
    println!("Testing DRED synthesis");
    println!("==============");
    println!();
    run_fargan_tests(&args.vector_path, &mut fargan)?;

    println!("==============");
    println!("Testing Opus decoding");
    println!("==============");
    println!();
    run_opus_tests(&args.vector_path)?;

    Ok(())
}

#[cfg(feature = "deep_plc")]
struct Args {
    vector_path: PathBuf,
    dnn_blob: Option<PathBuf>,
}

#[cfg(feature = "deep_plc")]
fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut dnn_blob = None;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "--dnn-blob" {
            let Some(value) = iter.next() else {
                return Err(USAGE.to_string());
            };
            dnn_blob = Some(PathBuf::from(value));
        } else if arg == "--help" || arg == "-h" {
            return Err(USAGE.to_string());
        } else {
            positional.push(arg);
        }
    }

    let vector_path = match positional.len() {
        0 => env::var_os("DRED_VECTORS_PATH").map(PathBuf::from),
        1 => Some(PathBuf::from(&positional[0])),
        2 => Some(PathBuf::from(&positional[1])),
        _ => return Err(USAGE.to_string()),
    }
    .ok_or_else(|| USAGE.to_string())?;

    let dnn_blob = dnn_blob.or_else(|| env::var_os("DNN_BLOB").map(PathBuf::from));
    if !cfg!(feature = "deep_plc_weights") && dnn_blob.is_none() {
        return Err("Missing DNN blob (set --dnn-blob or DNN_BLOB).".to_string());
    }

    Ok(Args {
        vector_path,
        dnn_blob,
    })
}

#[cfg(feature = "deep_plc")]
fn run_dred_decode_tests(vector_path: &Path, decoder: &DredVectorDecoder) -> Result<(), String> {
    for i in 1..=8 {
        let dred_bit = vector_path.join(format!("vector{i}_dred.bit"));
        if !dred_bit.exists() {
            return Err(format!("Bitstream file not found: vector{i}_dred.bit"));
        }
        println!("Testing vector{i}_dred.bit");

        let decoded = decode_dred_file(&dred_bit, decoder)?;
        println!("successfully decoded");

        let reference =
            read_f32_file(&vector_path.join(format!("vector{i}_dred_dec.f32")))?;
        compare_features(&reference, &decoded, DRED_DECODE_THRESHOLDS)?;
        println!("output matches reference");
        println!();
    }
    Ok(())
}

#[cfg(feature = "deep_plc")]
fn run_fargan_tests(vector_path: &Path, fargan: &mut FarganState) -> Result<(), String> {
    for i in 1..=8 {
        let feature_path = vector_path.join(format!("vector{i}_features.f32"));
        if !feature_path.exists() {
            return Err(format!("Bitstream file not found: vector{i}_features.f32"));
        }
        println!("Testing vector{i}_features.f32");

        let features = read_f32_file(&feature_path)?;
        fargan.reset();
        let pcm = synthesize_fargan_audio(fargan, &features)?;
        println!("successfully decoded");

        let reference = read_i16_file(&vector_path.join(format!("vector{i}_orig.sw")))?;
        compare_audio_i16(&reference, &pcm, FARGAN_THRESHOLDS)?;
        println!("output matches reference");
        println!();
    }
    Ok(())
}

#[cfg(feature = "deep_plc")]
fn run_opus_tests(vector_path: &Path) -> Result<(), String> {
    for i in 1..=8 {
        let opus_bit = vector_path.join(format!("vector{i}_opus.bit"));
        if !opus_bit.exists() {
            return Err(format!("Bitstream file not found: vector{i}_opus.bit"));
        }
        println!("Testing vector{i}_opus.bit");

        let pcm = decode_opus_file(&opus_bit, 16_000, 1)?;
        println!("successfully decoded");

        let reference = read_i16_file(&vector_path.join(format!("vector{i}_orig.sw")))?;
        compare_audio_i16(&reference, &pcm, OPUS_THRESHOLDS)?;
        println!("output matches reference");
        println!();
    }
    Ok(())
}

#[cfg(feature = "deep_plc")]
fn decode_dred_file(path: &Path, decoder: &DredVectorDecoder) -> Result<Vec<f32>, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("Error opening {}: {err}", path.display()))?;
    let mut output = Vec::new();

    loop {
        let Some(q0) = read_u32_be(&mut file)? else {
            break;
        };
        let nb_chunks = read_u32_be(&mut file)?
            .ok_or_else(|| "Truncated DRED header".to_string())?;
        let nb_bytes = read_u32_be(&mut file)?
            .ok_or_else(|| "Truncated DRED header".to_string())?;

        let mut payload = vec![0u8; nb_bytes as usize];
        if let Err(err) = file.read_exact(&mut payload) {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(format!("Failed reading {}: {err}", path.display()));
        }

        let mut features = vec![0.0f32; nb_chunks as usize * 2 * NB_FEATURES];
        let frames = decoder
            .decode_packet(q0, nb_chunks as usize, &payload, &mut features)
            .map_err(|err| format!("DRED decode failed: {err:?}"))?;
        features.truncate(frames * NB_FEATURES);
        output.extend_from_slice(&features);
    }

    Ok(output)
}

#[cfg(feature = "deep_plc")]
fn read_u32_be(reader: &mut impl Read) -> Result<Option<u32>, String> {
    let mut buf = [0u8; 4];
    match reader.read_exact(&mut buf) {
        Ok(()) => Ok(Some(u32::from_be_bytes(buf))),
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(format!("Failed reading header: {err}")),
    }
}

#[cfg(feature = "deep_plc")]
fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let data =
        fs::read(path).map_err(|err| format!("Error opening {}: {err}", path.display()))?;
    if data.len() % 4 != 0 {
        return Err(format!("Invalid float data length: {}", path.display()));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_bits(u32::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ])));
    }
    Ok(out)
}

#[cfg(feature = "deep_plc")]
fn read_i16_file(path: &Path) -> Result<Vec<i16>, String> {
    let data =
        fs::read(path).map_err(|err| format!("Error opening {}: {err}", path.display()))?;
    if data.len() % 2 != 0 {
        return Err(format!("Invalid PCM data length: {}", path.display()));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

#[cfg(feature = "deep_plc")]
fn synthesize_fargan_audio(
    fargan: &mut FarganState,
    features: &[f32],
) -> Result<Vec<i16>, String> {
    if !features.len().is_multiple_of(NB_FEATURES) {
        return Err("Feature vector length is not a multiple of 20".to_string());
    }

    let mut iter = features.chunks_exact(NB_FEATURES);
    let first = iter.next().ok_or("Empty feature stream")?;
    let mut init = vec![0.0f32; NB_FEATURES * 5];
    for idx in 0..5 {
        init[idx * NB_FEATURES..(idx + 1) * NB_FEATURES].copy_from_slice(first);
    }
    let zeros = vec![0.0f32; FARGAN_CONT_SAMPLES];
    fargan.fargan_cont(&zeros, &init);

    let mut last = first.to_vec();
    let mut output = Vec::new();
    let mut stop = 0;
    let mut skip = FARGAN_FRAME_SIZE / 2;
    loop {
        if let Some(frame) = iter.next() {
            last.copy_from_slice(frame);
        } else {
            stop += 1;
        }

        let mut pcm = vec![0i16; FARGAN_FRAME_SIZE];
        fargan.fargan_synthesize_int(&mut pcm, &last);
        if stop == 2 {
            output.extend_from_slice(&pcm[skip..skip + FARGAN_FRAME_SIZE / 2]);
            break;
        }
        output.extend_from_slice(&pcm[skip..]);
        skip = 0;
    }
    Ok(output)
}

#[cfg(feature = "deep_plc")]
fn decode_opus_file(path: &Path, sampling_rate: i32, channels: i32) -> Result<Vec<i16>, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("Error opening {}: {err}", path.display()))?;
    let mut decoder = opus_decoder_create(sampling_rate, channels)
        .map_err(|err| format!("opus_decoder_create failed: {err:?}"))?;
    let max_frame_size = (6 * sampling_rate / 50) as usize;
    let channels = channels as usize;
    let mut output = Vec::new();

    loop {
        let Some(len) = read_u32_be(&mut file)? else {
            break;
        };
        let _range = read_u32_be(&mut file)?
            .ok_or_else(|| "Truncated Opus packet header".to_string())?;
        let mut payload = vec![0u8; len as usize];
        if len > 0 && let Err(err) = file.read_exact(&mut payload) {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(format!("Failed reading {}: {err}", path.display()));
        }

        let mut pcm = vec![0i16; max_frame_size * channels];
        let decoded = if len == 0 {
            opus_decode(&mut decoder, None, 0, &mut pcm, max_frame_size, false)
        } else {
            opus_decode(
                &mut decoder,
                Some(&payload),
                payload.len(),
                &mut pcm,
                max_frame_size,
                false,
            )
        }
        .map_err(|err| format!("opus_decode failed: {err:?}"))?;

        output.extend_from_slice(&pcm[..decoded * channels]);
    }

    Ok(output)
}

#[cfg(feature = "deep_plc")]
fn compare_features(
    reference: &[f32],
    actual: &[f32],
    thresholds: (f32, f32, f32),
) -> Result<(), String> {
    if !reference.len().is_multiple_of(NB_FEATURES)
        || !actual.len().is_multiple_of(NB_FEATURES)
    {
        return Err("Feature buffer length must be a multiple of 20".to_string());
    }
    if reference.len() != actual.len() {
        return Err(format!(
            "Feature lengths do not match ({} != {})",
            reference.len(),
            actual.len()
        ));
    }

    let frames = reference.len() / NB_FEATURES;
    if frames == 0 {
        return Err("Empty feature comparison".to_string());
    }

    let mut mse = [0.0f64; NB_FEATURES];
    let mut pitch_error = 0.0f64;
    let mut pitch_count = 0u64;

    for frame in 0..frames {
        let base = frame * NB_FEATURES;
        for i in 0..NB_FEATURES {
            let e = f64::from(reference[base + i] - actual[base + i]);
            mse[i] += e * e;
        }
        if reference[base + NB_FEATURES - 1] > 0.2 {
            pitch_error += f64::from(
                (reference[base + NB_FEATURES - 2] - actual[base + NB_FEATURES - 2]).abs(),
            );
            pitch_count += 1;
        }
    }

    if pitch_count > 0 {
        pitch_error /= pitch_count as f64;
    }

    let mut tot_error = 0.0f64;
    let mut max_error = 0.0f64;
    for (i, entry) in mse.iter_mut().enumerate() {
        *entry /= frames as f64;
        if i != NB_FEATURES - 2 {
            tot_error += *entry;
            if *entry > max_error {
                max_error = *entry;
            }
        }
    }

    tot_error = tot_error.sqrt();
    max_error = max_error.sqrt();
    eprintln!(
        "total = {tot_error}, max = {max_error}, pitch = {pitch_error}"
    );

    let (tot_threshold, max_threshold, pitch_threshold) = thresholds;
    if tot_error <= f64::from(tot_threshold)
        && max_error <= f64::from(max_threshold)
        && pitch_error <= f64::from(pitch_threshold)
    {
        eprintln!("Comparison PASSED");
        Ok(())
    } else {
        Err(format!(
            "*** Comparison FAILED *** (thresholds were {tot_threshold} {max_threshold} {pitch_threshold})"
        ))
    }
}

#[cfg(feature = "deep_plc")]
fn compare_audio_i16(
    reference: &[i16],
    actual: &[i16],
    thresholds: (f32, f32, f32),
) -> Result<(), String> {
    let x = biquad_filter(&pcm_i16_to_f32(reference));
    let y = biquad_filter(&pcm_i16_to_f32(actual));
    compare_audio_float(&x, &y, thresholds)
}

#[cfg(feature = "deep_plc")]
fn pcm_i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| f32::from(s)).collect()
}

#[cfg(feature = "deep_plc")]
fn biquad_filter(samples: &[f32]) -> Vec<f32> {
    let a = [-1.97354_f64, 0.97417_f64];
    let b = [-2.0_f64, 1.0_f64];
    let mut mem = [0.0f64; 2];
    let mut out = Vec::with_capacity(samples.len());
    for &xi in samples {
        let xi64 = f64::from(xi);
        let yi = xi64 + mem[0];
        mem[0] = mem[1] + (b[0] * xi64 - a[0] * yi);
        mem[1] = b[1] * xi64 - a[1] * yi;
        out.push(yi as f32);
    }
    out
}

#[cfg(feature = "deep_plc")]
fn compare_audio_float(
    reference: &[f32],
    actual: &[f32],
    thresholds: (f32, f32, f32),
) -> Result<(), String> {
    let xlength = reference.len();
    let mut ylength = actual.len();
    if ylength > xlength {
        ylength = xlength;
    }
    if xlength != ylength {
        return Err(format!(
            "Sample counts do not match ({} != {})",
            xlength,
            ylength
        ));
    }
    if xlength < TEST_WIN_SIZE {
        return Err(format!(
            "Insufficient sample data ({} < {})",
            xlength, TEST_WIN_SIZE
        ));
    }

    let nframes = (xlength - TEST_WIN_SIZE + TEST_WIN_STEP) / TEST_WIN_STEP;
    let mut pitch_error = 0.0f32;
    let mut pitch_count = 0u32;
    for frame in 2..nframes.saturating_sub(2) {
        let offset = frame * TEST_WIN_STEP;
        let xcorr = compute_xcorr(reference, offset);
        let ycorr = compute_xcorr(actual, offset);
        let mut maxcorr = -1.0f32;
        let mut pitch = 0usize;
        for (i, &value) in xcorr
            .iter()
            .enumerate()
            .take(PITCH_MAX + 1)
            .skip(PITCH_MIN)
        {
            if value > maxcorr {
                maxcorr = value;
                pitch = i;
            }
        }
        if xcorr[pitch] > 0.7 {
            pitch_error += (xcorr[pitch] - ycorr[pitch]).abs();
            pitch_count += 1;
        }
    }
    if pitch_count > 0 {
        pitch_error /= pitch_count as f32;
    }

    let mut decay_l = [0.0f32; NFREQS];
    let mut decay_r = [0.0f32; NFREQS];
    psydecay_init(&mut decay_l, &mut decay_r, NFREQS, 16_000);

    let mut x = vec![0.0f32; nframes * NFREQS];
    let mut y = vec![0.0f32; nframes * NFREQS];
    spectrum(&mut x, reference, nframes, TEST_WIN_SIZE, TEST_WIN_STEP);
    spectrum(&mut y, actual, nframes, TEST_WIN_SIZE, TEST_WIN_STEP);

    let mut norm = [0.0f32; NFREQS];
    norm[0] = 1.0;
    for i in 1..NFREQS {
        norm[i] = 1.0 + decay_r[i] * norm[i - 1];
    }
    for i in (0..NFREQS - 1).rev() {
        norm[i] += decay_l[i] * norm[i + 1];
    }
    for value in &mut norm {
        *value = 1.0 / *value;
    }

    for frame in 0..nframes {
        let base = frame * NFREQS;
        for i in 1..NFREQS {
            x[base + i] += decay_r[i] * x[base + i - 1];
            y[base + i] += decay_r[i] * y[base + i - 1];
        }
        for i in (0..NFREQS - 1).rev() {
            x[base + i] += decay_l[i] * x[base + i + 1];
            y[base + i] += decay_l[i] * y[base + i + 1];
        }
        for i in 0..NFREQS {
            x[base + i] *= norm[i];
            y[base + i] *= norm[i];
        }
    }

    for frame in 0..nframes {
        let base = frame * NFREQS;
        let mut max_e = 0.0f32;
        for i in 0..NFREQS {
            max_e = max_e.max(x[base + i]);
        }
        for i in 0..NFREQS {
            let floor = 1.0e-8 * max_e;
            if x[base + i] < floor {
                x[base + i] = floor;
            }
            if y[base + i] < floor {
                y[base + i] = floor;
            }
        }
        if frame > 0 {
            let prev = (frame - 1) * NFREQS;
            for i in 0..NFREQS {
                x[base + i] += 0.5 * x[prev + i];
                y[base + i] += 0.5 * y[prev + i];
            }
        }
    }

    for frame in (0..nframes - 1).rev() {
        let base = frame * NFREQS;
        let next = (frame + 1) * NFREQS;
        for i in 0..NFREQS {
            x[base + i] += 0.1 * x[next + i];
            y[base + i] += 0.1 * y[next + i];
        }
    }

    let mut err4 = 0.0f64;
    let mut err16 = 0.0f64;
    let mut t2 = 0.0f64;
    for frame in 0..nframes {
        let base = frame * NFREQS;
        let mut ef2 = 0.0f64;
        let mut ef4 = 0.0f64;
        let mut tf2 = 0.0f64;
        for band in 0..NBANDS {
            let mut eb2 = 0.0f64;
            let mut eb4 = 0.0f64;
            let mut tb2 = 0.0f64;
            let band_len = (BANDS[band + 1] - BANDS[band]) as f64;
            let w = 1.0f64 / band_len;
            for bin in BANDS[band]..BANDS[band + 1] {
                let f = bin as f32 * core::f32::consts::PI / 960.0;
                let thresh = 0.1 / (0.15 * 0.15 + f * f);
                let re = powf(y[base + bin] + thresh, LOUDNESS)
                    - powf(x[base + bin] + thresh, LOUDNESS);
                let im = re * re;
                tb2 += w * f64::from(powf(x[base + bin] + thresh, 2.0 * LOUDNESS));
                eb2 += w * f64::from(im);

                let re = powf(y[base + bin] + 10.0 * thresh, LOUDNESS)
                    - powf(x[base + bin] + 10.0 * thresh, LOUDNESS);
                let im = re * re;
                eb4 += w * f64::from(im);
            }
            eb2 /= band_len;
            eb4 /= band_len;
            tb2 /= band_len;
            ef2 += eb2;
            ef4 += eb4 * eb4;
            tf2 += tb2;
        }
        ef2 /= NBANDS as f64;
        ef4 /= NBANDS as f64;
        ef4 *= ef4;
        tf2 /= NBANDS as f64;
        err4 += ef2 * ef2;
        err16 += ef4 * ef4;
        t2 += tf2;
    }

    let nframes_f = nframes as f64;
    let err4 = 100.0 * (err4 / nframes_f).powf(0.25) / t2.sqrt();
    let err16 = 100.0 * (err16 / nframes_f).powf(1.0 / 16.0) / t2.sqrt();
    eprintln!("err4 = {err4}, err16 = {err16}, pitch = {pitch_error}");

    let (err4_threshold, err16_threshold, pitch_threshold) = thresholds;
    if err4 <= f64::from(err4_threshold)
        && err16 <= f64::from(err16_threshold)
        && pitch_error <= pitch_threshold
    {
        eprintln!("Comparison PASSED");
        Ok(())
    } else {
        Err(format!(
            "*** Comparison FAILED *** (thresholds were {err4_threshold} {err16_threshold} {pitch_threshold})"
        ))
    }
}

#[cfg(feature = "deep_plc")]
fn psydecay_init(decay_l: &mut [f32], decay_r: &mut [f32], len: usize, fs: i32) {
    for i in 0..len {
        let f = fs as f32 * i as f32 * (1.0 / (2.0 * len as f32));
        let deriv = (8.288e-8 * f) / (3.4225e-16 * f * f * f * f + 1.0)
            + 0.009694 / (5.476e-7 * f * f + 1.0)
            + 1.0e-4;
        let deriv = deriv * fs as f32 * (1.0 / (2.0 * len as f32));
        decay_r[i] = powf(0.1, deriv);
        decay_l[i] = powf(0.0031623, deriv);
    }
}

#[cfg(feature = "deep_plc")]
fn compute_xcorr(x: &[f32], offset: usize) -> [f32; PITCH_MAX + 1] {
    let mut filtered = [0.0f32; PITCH_FRAME + PITCH_MAX];
    for (i, slot) in filtered.iter_mut().enumerate() {
        let idx = offset + i - PITCH_MAX;
        let xi = x[idx];
        let xi_prev = x[idx - 1];
        *slot = xi - 0.8 * xi_prev;
    }

    let frame = &filtered[PITCH_MAX..PITCH_MAX + PITCH_FRAME];
    let xx = inner_prod(frame, frame);
    let mut xcorr = [0.0f32; PITCH_MAX + 1];
    for i in 0..=PITCH_MAX {
        let xy = inner_prod(frame, &filtered[PITCH_MAX - i..PITCH_MAX - i + PITCH_FRAME]);
        let yy = inner_prod(
            &filtered[PITCH_MAX - i..PITCH_MAX - i + PITCH_FRAME],
            &filtered[PITCH_MAX - i..PITCH_MAX - i + PITCH_FRAME],
        );
        xcorr[i] = xy / sqrtf(xx * yy + PITCH_FRAME as f32);
    }
    xcorr
}

#[cfg(feature = "deep_plc")]
fn inner_prod(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

#[cfg(feature = "deep_plc")]
fn spectrum(
    ps: &mut [f32],
    input: &[f32],
    nframes: usize,
    window_size: usize,
    step: usize,
) {
    let ps_sz = window_size / 2;
    let mut window = vec![0.0f32; window_size];
    for (i, slot) in window.iter_mut().enumerate().take(window_size) {
        let n = (i as f32 + 0.5) / window_size as f32;
        *slot = 0.35875
            - 0.48829 * cosf(2.0 * core::f32::consts::PI * n)
            + 0.14128 * cosf(4.0 * core::f32::consts::PI * n)
            - 0.01168 * cosf(6.0 * core::f32::consts::PI * n);
    }

    let mut fft = MiniKissFftr::new(window_size, false);
    let mut frame = vec![0.0f32; window_size];
    let mut freq = vec![KissFftCpx::default(); ps_sz + 1];

    for frame_idx in 0..nframes {
        let base = frame_idx * step;
        for i in 0..window_size {
            frame[i] = window[i] * input[base + i];
        }
        fft.process(&frame, &mut freq);
        for bin in 0..ps_sz {
            let re = freq[bin].r;
            let im = freq[bin].i;
            ps[frame_idx * ps_sz + bin] = re * re + im * im + 0.1;
        }
    }
}
