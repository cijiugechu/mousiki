#![cfg(feature = "libopusenc")]

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod common;

use crate::common::libopusenc::{BehaviorManifest, TestBuffer};
use mousiki::libopusenc::{OggOpusComments, OggOpusEnc, OpeError, OpusEncCallbacks};

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

#[derive(Default)]
struct CallbackSinkState {
    encoded: TestBuffer,
    close_calls: usize,
}

struct SharedCallbackSink(Rc<RefCell<CallbackSinkState>>);

impl OpusEncCallbacks for SharedCallbackSink {
    fn write(&mut self, data: &[u8]) -> Result<(), OpeError> {
        self.0.borrow_mut().encoded.append(data);
        Ok(())
    }

    fn close(&mut self) -> Result<(), OpeError> {
        self.0.borrow_mut().close_calls += 1;
        Ok(())
    }
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
    let mut comments = OggOpusComments::create().expect("comments");
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

fn collect_pull(enc: &mut OggOpusEnc) -> Vec<u8> {
    let mut encoded = TestBuffer::default();
    while let Some(page) = enc.get_page(true).expect("get page") {
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
    let mut enc =
        OggOpusEnc::create_pull(&comments, RESAMPLE_RATE as i32, RESAMPLE_CHANNELS as i32, 0)
            .expect("pull encoder");
    enc.set_serialno(RESAMPLE_SERIALNO).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    enc.drain().expect("drain");
    BehaviorManifest::build(&collect_pull(&mut enc)).expect("pull manifest")
}

fn encode_callbacks_write(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
) -> (BehaviorManifest, usize) {
    let comments = create_resample_comments();
    let sink = Rc::new(RefCell::new(CallbackSinkState::default()));
    let callbacks = Box::new(SharedCallbackSink(sink.clone()));
    let mut enc = OggOpusEnc::create_callbacks(
        callbacks,
        &comments,
        RESAMPLE_RATE as i32,
        RESAMPLE_CHANNELS as i32,
        0,
    )
    .expect("callbacks encoder");
    enc.set_serialno(RESAMPLE_SERIALNO).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    enc.drain().expect("drain");

    let sink = sink.borrow();
    (
        BehaviorManifest::build(&sink.encoded.data).expect("callback manifest"),
        sink.close_calls,
    )
}

fn encode_file_write(
    scenario: ResampleScenario,
    pcm: &[i16],
    total_frames: usize,
) -> BehaviorManifest {
    let path = create_temp_output_path();
    let comments = create_resample_comments();
    let mut enc = OggOpusEnc::create_file(
        path.to_str().unwrap(),
        &comments,
        RESAMPLE_RATE as i32,
        RESAMPLE_CHANNELS as i32,
        0,
    )
    .expect("file encoder");
    enc.set_serialno(RESAMPLE_SERIALNO).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * RESAMPLE_CHANNELS..], chunk)
            .expect("write chunk");
        offset += chunk;
    }
    assert_eq!(total_frames, offset);
    enc.drain().expect("drain");

    let manifest = BehaviorManifest::build_from_file(&path).expect("file manifest");
    fs::remove_file(path).expect("remove file");
    manifest
}

fn encode_pull_write_float(pcm: &[f32], total_frames: usize) -> BehaviorManifest {
    let comments = create_resample_comments();
    let mut enc =
        OggOpusEnc::create_pull(&comments, RESAMPLE_RATE as i32, RESAMPLE_CHANNELS as i32, 0)
            .expect("pull encoder");
    enc.set_serialno(RESAMPLE_SERIALNO).expect("serial");
    enc.flush_header().expect("flush header");
    enc.write_float(pcm, total_frames).expect("write float");
    enc.drain().expect("drain");
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
    callback_manifest: &BehaviorManifest,
    file_manifest: &BehaviorManifest,
    expected_granule: usize,
    exact_granule: bool,
) {
    assert_eq!(pull_manifest, callback_manifest);
    assert_eq!(pull_manifest, file_manifest);
    assert_resample_manifest(pull_manifest, expected_granule, exact_granule);
    assert_resample_manifest(callback_manifest, expected_granule, exact_granule);
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
    let (callback_manifest, close_calls) = encode_callbacks_write(scenario, pcm, total_frames);
    let file_manifest = encode_file_write(scenario, pcm, total_frames);
    assert_eq!(1, close_calls);
    assert_resample_three_way_parity(
        &pull_manifest,
        &callback_manifest,
        &file_manifest,
        expected_granule,
        exact_granule,
    );
}

#[test]
fn explicit_header_resample_outputs_match_ctest() {
    let scenario = ResampleScenario {
        flush_header: true,
        chunk_count: 1,
        chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
    };
    let mut pcm = [0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        scenario,
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn implicit_header_resample_outputs_match_ctest() {
    let scenario = ResampleScenario {
        flush_header: false,
        chunk_count: 1,
        chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
    };
    let mut pcm = [0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        scenario,
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn chunked_resample_outputs_match_ctest() {
    let scenario = ResampleScenario {
        flush_header: true,
        chunk_count: 3,
        chunks: [1470, 1470, 1470],
    };
    let mut pcm = [0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    run_resample_three_path_scenario(
        scenario,
        &pcm,
        RESAMPLE_TOTAL_FRAMES,
        RESAMPLE_EXPECTED_GRANULE,
        true,
    );
}

#[test]
fn short_input_resample_outputs_match_ctest() {
    let scenario = ResampleScenario {
        flush_header: true,
        chunk_count: 2,
        chunks: [221, 220, 0],
    };
    let mut pcm = [0i16; RESAMPLE_SHORT_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm, RESAMPLE_SHORT_FRAMES, RESAMPLE_CHANNELS);

    let pull_manifest = encode_pull_write(scenario, &pcm, RESAMPLE_SHORT_FRAMES);
    let (callback_manifest, _) = encode_callbacks_write(scenario, &pcm, RESAMPLE_SHORT_FRAMES);
    let file_manifest = encode_file_write(scenario, &pcm, RESAMPLE_SHORT_FRAMES);

    assert_eq!(pull_manifest, callback_manifest);
    assert_eq!(pull_manifest, file_manifest);
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

#[test]
fn write_and_write_float_match_for_44k1_ctest() {
    let mut pcm_int = [0i16; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    let mut pcm_float = [0.0f32; RESAMPLE_TOTAL_FRAMES * RESAMPLE_CHANNELS];
    fill_resample_pcm_int(&mut pcm_int, RESAMPLE_TOTAL_FRAMES, RESAMPLE_CHANNELS);
    fill_resample_pcm_float(&mut pcm_float, &pcm_int);

    let scenario = ResampleScenario {
        flush_header: true,
        chunk_count: 1,
        chunks: [RESAMPLE_TOTAL_FRAMES, 0, 0],
    };
    let write_manifest = encode_pull_write(scenario, &pcm_int, RESAMPLE_TOTAL_FRAMES);
    let write_float_manifest = encode_pull_write_float(&pcm_float, RESAMPLE_TOTAL_FRAMES);

    assert_eq!(write_manifest, write_float_manifest);
    assert_resample_manifest(&write_manifest, RESAMPLE_EXPECTED_GRANULE, true);
    assert_resample_manifest(&write_float_manifest, RESAMPLE_EXPECTED_GRANULE, true);
}
