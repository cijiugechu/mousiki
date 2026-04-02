use mousiki::opusfile::{OpusFile, OpusfileError, OpusfileOpenError};
use std::env;
use std::path::{Path, PathBuf};
use std::process;

const SEEK_TESTS: usize = 128;
const VERIFY_SAMPLES: usize = 960;

fn main() {
    if let Err(err) = run() {
        report_error(err);
        process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let input = args
        .next()
        .ok_or_else(|| ExampleError::Usage(program.clone()))?;
    if args.next().is_some() {
        return Err(ExampleError::Usage(program));
    }

    let input_path = PathBuf::from(input);
    let mut file = OpusFile::open_file(&input_path).map_err(ExampleError::Open)?;
    if !file.is_seekable() {
        return Err(ExampleError::NotSeekable);
    }

    let total_pcm = file.pcm_total(None).map_err(ExampleError::Decode)?;
    let total_raw = file.raw_total(None).map_err(ExampleError::Decode)?;
    let link_starts = compute_link_starts(&file).map_err(ExampleError::Decode)?;
    let reference = decode_reference_stereo(&input_path)?;

    eprintln!(
        "loaded {} links, {} PCM samples, {} raw bytes",
        file.link_count(),
        total_pcm,
        total_raw
    );

    let mut rng = Lcg::new(0xC0DEC0DEu64);
    let mut raw_probe = vec![0.0f32; VERIFY_SAMPLES * 2];
    let mut pcm_probe = vec![0.0f32; VERIFY_SAMPLES * 2];

    for index in 0..SEEK_TESTS {
        let raw_target = random_bounded(&mut rng, total_raw.saturating_add(1));
        file.raw_seek(raw_target).map_err(ExampleError::Decode)?;
        verify_raw_seek(index, raw_target, &mut file, &link_starts, &reference, &mut raw_probe)?;
    }

    for index in 0..SEEK_TESTS {
        let pcm_target = random_bounded(&mut rng, total_pcm.saturating_add(1));
        file.pcm_seek(pcm_target).map_err(ExampleError::Decode)?;
        verify_pcm_seek(index, pcm_target, &mut file, &link_starts, &reference, &mut pcm_probe)?;
    }

    eprintln!("all {SEEK_TESTS} raw seeks and {SEEK_TESTS} exact PCM seeks matched reference output");
    Ok(())
}

fn decode_reference_stereo(path: &Path) -> Result<Vec<f32>, ExampleError> {
    let mut opus = OpusFile::open_file(path).map_err(ExampleError::Open)?;
    let total_pcm = opus.pcm_total(None).map_err(ExampleError::Decode)?;
    let mut reference = vec![0.0f32; total_pcm * 2];
    let mut cursor = 0usize;
    while cursor < total_pcm {
        let read = opus
            .read_float_stereo(&mut reference[cursor * 2..])
            .map_err(ExampleError::Decode)?;
        if read.samples_per_channel == 0 {
            break;
        }
        cursor += read.samples_per_channel;
    }
    reference.truncate(cursor * 2);
    Ok(reference)
}

fn compute_link_starts(opus: &OpusFile) -> Result<Vec<usize>, OpusfileError> {
    let mut starts = Vec::with_capacity(opus.link_count() + 1);
    let mut pcm_start = 0usize;
    starts.push(0);
    for link_index in 0..opus.link_count() {
        pcm_start = pcm_start.saturating_add(opus.pcm_total(Some(link_index))?);
        starts.push(pcm_start);
    }
    Ok(starts)
}

fn verify_raw_seek(
    iteration: usize,
    requested_raw: usize,
    file: &mut OpusFile,
    link_starts: &[usize],
    reference: &[f32],
    probe: &mut [f32],
) -> Result<(), ExampleError> {
    let actual_raw = file.raw_tell();
    let actual_pcm = file.pcm_tell();
    let actual_link = file.current_link().map_err(ExampleError::Decode)?;
    if actual_raw > requested_raw {
        return Err(ExampleError::SeekMismatch {
            kind: "raw",
            iteration,
            requested: requested_raw,
            actual: actual_raw,
            detail: "raw_tell moved past requested byte offset",
        });
    }
    verify_link_position(actual_pcm, actual_link, link_starts, iteration, "raw")?;
    verify_probe_matches_reference(iteration, "raw", actual_pcm, file, probe, reference)
}

fn verify_pcm_seek(
    iteration: usize,
    requested_pcm: usize,
    file: &mut OpusFile,
    link_starts: &[usize],
    reference: &[f32],
    probe: &mut [f32],
) -> Result<(), ExampleError> {
    let actual_pcm = file.pcm_tell();
    if actual_pcm != requested_pcm {
        return Err(ExampleError::SeekMismatch {
            kind: "pcm",
            iteration,
            requested: requested_pcm,
            actual: actual_pcm,
            detail: "pcm_tell did not land on the requested sample",
        });
    }
    let actual_link = file.current_link().map_err(ExampleError::Decode)?;
    verify_link_position(actual_pcm, actual_link, link_starts, iteration, "pcm")?;
    verify_probe_matches_reference(iteration, "pcm", actual_pcm, file, probe, reference)
}

fn verify_link_position(
    pcm_offset: usize,
    actual_link: usize,
    link_starts: &[usize],
    iteration: usize,
    kind: &'static str,
) -> Result<(), ExampleError> {
    let expected_link = link_starts
        .windows(2)
        .position(|window| pcm_offset >= window[0] && pcm_offset < window[1])
        .unwrap_or_else(|| link_starts.len().saturating_sub(2));
    if actual_link != expected_link {
        return Err(ExampleError::SeekLinkMismatch {
            kind,
            iteration,
            pcm_offset,
            expected_link,
            actual_link,
        });
    }
    Ok(())
}

fn verify_probe_matches_reference(
    iteration: usize,
    kind: &'static str,
    pcm_offset: usize,
    file: &mut OpusFile,
    probe: &mut [f32],
    reference: &[f32],
) -> Result<(), ExampleError> {
    probe.fill(0.0);
    let read = file
        .read_float_stereo(probe)
        .map_err(ExampleError::Decode)?;
    let samples = read.samples_per_channel;
    let reference_start = pcm_offset.saturating_mul(2);
    let reference_end = reference_start.saturating_add(samples.saturating_mul(2));
    let expected = reference.get(reference_start..reference_end).ok_or(ExampleError::SeekRange {
        kind,
        iteration,
        pcm_offset,
        samples,
    })?;
    if probe[..samples * 2] != *expected {
        let mismatch = probe[..samples * 2]
            .iter()
            .zip(expected.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Err(ExampleError::SampleMismatch {
            kind,
            iteration,
            pcm_offset,
            sample_index: mismatch / 2,
            channel: mismatch % 2,
            actual: probe[mismatch],
            expected: expected[mismatch],
        });
    }
    if let Ok(bitrate) = file.bitrate_instant() {
        eprintln!(
            "{kind} seek #{iteration:03}: pcm={} bitrate={} bps",
            pcm_offset, bitrate
        );
    }
    Ok(())
}

fn random_bounded(rng: &mut Lcg, upper_bound: usize) -> usize {
    if upper_bound <= 1 {
        return 0;
    }
    (rng.next_u64() % upper_bound as u64) as usize
}

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1442695040888963407;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        self.state
    }
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage(program) => {
            let name = Path::new(&program)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("opusfile_seeking_example");
            eprintln!("usage: {name} <input.opus>");
        }
        ExampleError::Open(err) => {
            eprintln!("open failed: {err}");
        }
        ExampleError::Decode(err) => {
            eprintln!("decode failed: {err}");
        }
        ExampleError::NotSeekable => {
            eprintln!("input is not seekable");
        }
        ExampleError::SeekMismatch {
            kind,
            iteration,
            requested,
            actual,
            detail,
        } => {
            eprintln!("{kind} seek #{iteration:03} mismatch: requested={requested} actual={actual} ({detail})");
        }
        ExampleError::SeekLinkMismatch {
            kind,
            iteration,
            pcm_offset,
            expected_link,
            actual_link,
        } => {
            eprintln!(
                "{kind} seek #{iteration:03} landed in wrong link: pcm={pcm_offset} expected={expected_link} actual={actual_link}"
            );
        }
        ExampleError::SeekRange {
            kind,
            iteration,
            pcm_offset,
            samples,
        } => {
            eprintln!("{kind} seek #{iteration:03} ran past reference buffer: pcm={pcm_offset} samples={samples}");
        }
        ExampleError::SampleMismatch {
            kind,
            iteration,
            pcm_offset,
            sample_index,
            channel,
            actual,
            expected,
        } => {
            eprintln!(
                "{kind} seek #{iteration:03} sample mismatch at pcm={} sample={} ch={}: actual={} expected={}",
                pcm_offset, sample_index, channel, actual, expected
            );
        }
    }
}

enum ExampleError {
    Usage(std::ffi::OsString),
    Open(OpusfileOpenError),
    Decode(OpusfileError),
    NotSeekable,
    SeekMismatch {
        kind: &'static str,
        iteration: usize,
        requested: usize,
        actual: usize,
        detail: &'static str,
    },
    SeekLinkMismatch {
        kind: &'static str,
        iteration: usize,
        pcm_offset: usize,
        expected_link: usize,
        actual_link: usize,
    },
    SeekRange {
        kind: &'static str,
        iteration: usize,
        pcm_offset: usize,
        samples: usize,
    },
    SampleMismatch {
        kind: &'static str,
        iteration: usize,
        pcm_offset: usize,
        sample_index: usize,
        channel: usize,
        actual: f32,
        expected: f32,
    },
}
