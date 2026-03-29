#![cfg(feature = "libopusenc")]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod common;

use crate::common::libopusenc::{BehaviorManifest, TestBuffer};
use mousiki::libopusenc::{MappingFamily, OggOpusComments, OggOpusEncoderBuilder};

const RESAMPLE_SERIALNO: i32 = 4242;
const RESAMPLE_RATE: usize = 44_100;
const RESAMPLE_CHANNELS: usize = 2;
const RESAMPLE_TOTAL_FRAMES: usize = 4_410;
const RESAMPLE_SHORT_FRAMES: usize = 441;
const RESAMPLE_EXPECTED_GRANULE: usize = 5_112;

#[derive(Clone, Copy)]
struct ResampleScenario {
    flush_header: bool,
    chunks: [usize; 3],
    chunk_count: usize,
}

fn fill_resample_pcm_int(pcm: &mut [i16], frames: usize, channels: usize) {
    for i in 0..frames {
        for c in 0..channels {
            let sign = if ((i / 23) + c) & 1 == 0 { -1 } else { 1 };
            let value = ((i * 137) + (c * 761)) % 14_000;
            pcm[i * channels + c] = (sign * (value as i32 + 1000)) as i16;
        }
    }
}

fn fill_resample_pcm_float(pcm: &mut [f32], pcm_int: &[i16]) {
    for (dst, &src) in pcm.iter_mut().zip(pcm_int.iter()) {
        *dst = src as f32 / 32768.0;
    }
}

fn create_resample_comments() -> OggOpusComments {
    let mut comments = OggOpusComments::new().expect("comments");
    comments.add("ARTIST", "Smoke").expect("artist");
    comments.add_string("TITLE=Resample").expect("title");
    comments
}

static UNIQUE_SUFFIX: AtomicU64 = AtomicU64::new(0);

fn create_temp_output_path() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let suffix = UNIQUE_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "mousiki-libopusenc-resample-{unique}-{suffix}.opus"
    ))
}

fn collect_pull(enc: &mut mousiki::libopusenc::OggOpusPullEncoder) -> Vec<u8> {
    let mut encoded = TestBuffer::default();
    while let Some(page) = enc.next_page().expect("next page") {
        encoded.append(&page);
    }
    encoded.data
}

fn encode_pull_write(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
) -> BehaviorManifest {
    let comments = create_resample_comments();
    let mut enc = OggOpusEncoderBuilder::new(
        comments,
        RESAMPLE_RATE as u32,
        RESAMPLE_CHANNELS as u8,
        MappingFamily::MonoStereo,
    )
    .expect("builder")
    .serialno(RESAMPLE_SERIALNO)
    .build_pull()
    .expect("pull encoder");
    if scenario.flush_header {
        enc.flush_headers().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    enc.finish().expect("finish");
    BehaviorManifest::build(&collect_pull(&mut enc)).expect("pull manifest")
}

fn encode_writer_write(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
) -> BehaviorManifest {
    let comments = create_resample_comments();
    let mut enc = OggOpusEncoderBuilder::new(
        comments,
        RESAMPLE_RATE as u32,
        RESAMPLE_CHANNELS as u8,
        MappingFamily::MonoStereo,
    )
    .expect("builder")
    .serialno(RESAMPLE_SERIALNO)
    .build_writer(TestBuffer::default())
    .expect("writer encoder");
    if scenario.flush_header {
        enc.flush_headers().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    let sink = enc.finish().expect("finish");
    BehaviorManifest::build(&sink.data).expect("writer manifest")
}

fn encode_file_write(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
) -> BehaviorManifest {
    let path = create_temp_output_path();
    let comments = create_resample_comments();
    let mut enc = OggOpusEncoderBuilder::new(
        comments,
        RESAMPLE_RATE as u32,
        RESAMPLE_CHANNELS as u8,
        MappingFamily::MonoStereo,
    )
    .expect("builder")
    .serialno(RESAMPLE_SERIALNO)
    .build_file(&path)
    .expect("file encoder");
    if scenario.flush_header {
        enc.flush_headers().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    let _ = enc.finish().expect("finish");

    let manifest = BehaviorManifest::build_from_file(&path).expect("file manifest");
    fs::remove_file(path).expect("remove file");
    manifest
}

fn encode_pull_write_float(pcm: &[f32], total_frames: usize) -> BehaviorManifest {
    let comments = create_resample_comments();
    let mut enc = OggOpusEncoderBuilder::new(
        comments,
        RESAMPLE_RATE as u32,
        RESAMPLE_CHANNELS as u8,
        MappingFamily::MonoStereo,
    )
    .expect("builder")
    .serialno(RESAMPLE_SERIALNO)
    .build_pull()
    .expect("pull encoder");
    enc.flush_headers().expect("flush header");
    enc.write_float(pcm, total_frames).expect("write float");
    enc.finish().expect("finish");
    BehaviorManifest::build(&collect_pull(&mut enc)).expect("float manifest")
}

fn assert_resample_manifest(manifest: &BehaviorManifest, expected_granule: usize, exact: bool) {
    assert!(manifest.head.valid);
    assert!(manifest.tags.valid);
    assert!(manifest.pages.len() >= 3);
    assert!(manifest.packets.len() >= 3);
    assert_eq!(RESAMPLE_CHANNELS as u8, manifest.head.channels);
    assert_eq!(0, manifest.head.mapping);
    assert_eq!(RESAMPLE_RATE as u32, manifest.head.input_sample_rate);
    assert_eq!(2, manifest.tags.comments.len());
    assert_eq!(b"ARTIST=Smoke", manifest.tags.comments[0].as_slice());
    assert_eq!(b"TITLE=Resample", manifest.tags.comments[1].as_slice());
    assert_ne!(0, manifest.pages[0].flags & 0x02);
    assert_ne!(0, manifest.pages[manifest.pages.len() - 1].flags & 0x04);
    assert_eq!(RESAMPLE_SERIALNO, manifest.pages[0].serialno as i32);
    assert_eq!(expected_granule, manifest.head.preskip as usize + 4800);
    assert!(manifest.pages[manifest.pages.len() - 1].granulepos > manifest.head.preskip as u64);
    if exact {
        assert_eq!(
            RESAMPLE_EXPECTED_GRANULE as u64,
            manifest.pages[manifest.pages.len() - 1].granulepos
        );
    } else {
        assert_eq!(
            expected_granule as u64,
            manifest.pages[manifest.pages.len() - 1].granulepos
        );
    }
}

fn assert_resample_three_way_parity(
    pull_manifest: &BehaviorManifest,
    writer_manifest: &BehaviorManifest,
    file_manifest: &BehaviorManifest,
    expected_granule: usize,
    exact_granule: bool,
) {
    assert_eq!(pull_manifest, writer_manifest);
    assert_eq!(pull_manifest, file_manifest);
    assert_resample_manifest(pull_manifest, expected_granule, exact_granule);
    assert_resample_manifest(writer_manifest, expected_granule, exact_granule);
    assert_resample_manifest(file_manifest, expected_granule, exact_granule);
}

fn run_resample_three_path_scenario(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
    expected_granule: usize,
    exact_granule: bool,
) {
    let pull_manifest = encode_pull_write(scenario, pcm, total_frames);
    let writer_manifest = encode_writer_write(scenario, pcm, total_frames);
    let file_manifest = encode_file_write(scenario, pcm, total_frames);
    assert_resample_three_way_parity(
        &pull_manifest,
        &writer_manifest,
        &file_manifest,
        expected_granule,
        exact_granule,
    );
}

#[test]
fn explicit_header_resample_outputs_match_ctest() {
    let mut pcm = vec![0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        ResampleScenario {
            flush_header: true,
            chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
            chunk_count: 1,
        },
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn implicit_header_resample_outputs_match_ctest() {
    let mut pcm = vec![0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        ResampleScenario {
            flush_header: false,
            chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
            chunk_count: 1,
        },
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn chunked_resample_outputs_match_ctest() {
    let mut pcm = vec![0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        ResampleScenario {
            flush_header: true,
            chunks: [882, 1_764, 1_764],
            chunk_count: 3,
        },
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn write_and_write_float_match_for_44k1_ctest() {
    let mut pcm_int = vec![0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm_int, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    let mut pcm_float = vec![0f32; pcm_int.len()];
    fill_resample_pcm_float(&mut pcm_float, &pcm_int);

    let int_manifest = encode_pull_write(
        ResampleScenario {
            flush_header: true,
            chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
            chunk_count: 1,
        },
        &pcm_int,
        RESAMPLE_TOTAL_FRAMES,
    );
    let float_manifest = encode_pull_write_float(&pcm_float, RESAMPLE_TOTAL_FRAMES);

    assert_eq!(int_manifest, float_manifest);
    assert_resample_manifest(&int_manifest, RESAMPLE_EXPECTED_GRANULE, true);
}

#[test]
fn short_input_resample_outputs_match_ctest() {
    let mut pcm = vec![0i16; RESAMPLE_SHORT_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_SHORT_FRAMES, RESAMPLE_CHANNELS);

    let pull_manifest = encode_pull_write(
        ResampleScenario {
            flush_header: true,
            chunks: [RESAMPLE_SHORT_FRAMES, 0, 0],
            chunk_count: 1,
        },
        &pcm,
        RESAMPLE_SHORT_FRAMES,
    );
    assert!(pull_manifest.head.valid);
    assert!(pull_manifest.tags.valid);
    assert!(pull_manifest.packets.len() >= 3);
    assert_ne!(
        0,
        pull_manifest.pages[pull_manifest.pages.len() - 1].flags & 0x04
    );
    assert!(
        pull_manifest.pages[pull_manifest.pages.len() - 1].granulepos
            > pull_manifest.head.preskip as u64
    );
}
