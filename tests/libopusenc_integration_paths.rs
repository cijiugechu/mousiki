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

#[derive(Clone, Copy)]
struct IntegrationScenario {
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

fn fill_pcm_pattern(pcm: &mut [i16], frames: usize, channels: usize) {
    for i in 0..frames {
        for c in 0..channels {
            let sign = if ((i / 29) + c) & 1 == 0 { -1 } else { 1 };
            let value = ((i * 211) + (c * 593)) % 12_000;
            pcm[i * channels + c] = (sign * (value as i32 + 700)) as i16;
        }
    }
}

fn create_shared_comments() -> OggOpusComments {
    let mut comments = OggOpusComments::create().expect("comments");
    comments.add("ARTIST", "Smoke").expect("artist");
    comments.add_string("TITLE=Parity").expect("title");
    comments
}

static UNIQUE_SUFFIX: AtomicU64 = AtomicU64::new(0);

fn create_temp_output_path() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let suffix = UNIQUE_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mousiki-libopusenc-{unique}-{suffix}.opus"))
}

fn collect_pull(enc: &mut OggOpusEnc) -> Vec<u8> {
    let mut encoded = TestBuffer::default();
    while let Some(page) = enc.get_page(true).expect("get page") {
        encoded.append(&page);
    }
    encoded.data
}

fn encode_with_pull(scenario: IntegrationScenario, pcm: &[i16]) -> BehaviorManifest {
    let comments = create_shared_comments();
    let mut enc = OggOpusEnc::create_pull(&comments, 48_000, 2, 0).expect("pull encoder");
    enc.set_serialno(4242).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * 2..], chunk).expect("write chunk");
        offset += chunk;
    }
    assert_eq!(960, offset);
    enc.drain().expect("drain");
    BehaviorManifest::build(&collect_pull(&mut enc)).expect("pull manifest")
}

fn encode_with_callbacks(scenario: IntegrationScenario, pcm: &[i16]) -> (BehaviorManifest, usize) {
    let comments = create_shared_comments();
    let sink = Rc::new(RefCell::new(CallbackSinkState::default()));
    let callbacks = Box::new(SharedCallbackSink(sink.clone()));
    let mut enc =
        OggOpusEnc::create_callbacks(callbacks, &comments, 48_000, 2, 0).expect("callback encoder");
    enc.set_serialno(4242).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * 2..], chunk).expect("write chunk");
        offset += chunk;
    }
    assert_eq!(960, offset);
    enc.drain().expect("drain");

    let sink = sink.borrow();
    (
        BehaviorManifest::build(&sink.encoded.data).expect("callback manifest"),
        sink.close_calls,
    )
}

fn encode_with_file(scenario: IntegrationScenario, pcm: &[i16]) -> BehaviorManifest {
    let path = create_temp_output_path();
    let comments = create_shared_comments();
    let mut enc = OggOpusEnc::create_file(path.to_str().unwrap(), &comments, 48_000, 2, 0)
        .expect("file encoder");
    enc.set_serialno(4242).expect("serial");
    if scenario.flush_header {
        enc.flush_header().expect("flush header");
    }

    let mut offset = 0usize;
    for &chunk in &scenario.chunks[..scenario.chunk_count] {
        enc.write(&pcm[offset * 2..], chunk).expect("write chunk");
        offset += chunk;
    }
    assert_eq!(960, offset);
    enc.drain().expect("drain");

    let manifest = BehaviorManifest::build_from_file(&path).expect("file manifest");
    fs::remove_file(path).expect("remove temp file");
    manifest
}

fn assert_three_way_parity(
    pull_manifest: &BehaviorManifest,
    callback_manifest: &BehaviorManifest,
    file_manifest: &BehaviorManifest,
) {
    assert_eq!(pull_manifest, callback_manifest);
    assert_eq!(pull_manifest, file_manifest);
    assert!(pull_manifest.pages.len() >= 3);
    assert!(pull_manifest.packets.len() >= 3);
    assert!(pull_manifest.head.valid);
    assert!(pull_manifest.tags.valid);
    assert_eq!(2, pull_manifest.head.channels);
    assert_eq!(0, pull_manifest.head.mapping);
    assert_eq!(48_000, pull_manifest.head.input_sample_rate);
    assert_eq!(2, pull_manifest.tags.comments.len());
    assert_eq!(b"ARTIST=Smoke", pull_manifest.tags.comments[0].as_slice());
    assert_eq!(b"TITLE=Parity", pull_manifest.tags.comments[1].as_slice());
    assert_ne!(0, pull_manifest.pages[0].flags & 0x02);
    assert_ne!(
        0,
        pull_manifest.pages[pull_manifest.pages.len() - 1].flags & 0x04
    );
    assert_eq!(4242, pull_manifest.pages[0].serialno as i32);
}

fn run_three_path_scenario(scenario: IntegrationScenario) {
    let mut pcm = [0i16; 960 * 2];
    fill_pcm_pattern(&mut pcm, 960, 2);

    let pull_manifest = encode_with_pull(scenario, &pcm);
    let (callback_manifest, close_calls) = encode_with_callbacks(scenario, &pcm);
    let file_manifest = encode_with_file(scenario, &pcm);

    assert_eq!(1, close_calls);
    assert_three_way_parity(&pull_manifest, &callback_manifest, &file_manifest);
}

#[test]
fn explicit_header_flush_three_way_parity_matches_ctest() {
    run_three_path_scenario(IntegrationScenario {
        flush_header: true,
        chunks: [960, 0, 0],
        chunk_count: 1,
    });
}

#[test]
fn implicit_header_init_three_way_parity_matches_ctest() {
    run_three_path_scenario(IntegrationScenario {
        flush_header: false,
        chunks: [960, 0, 0],
        chunk_count: 1,
    });
}

#[test]
fn chunked_writes_three_way_parity_matches_ctest() {
    run_three_path_scenario(IntegrationScenario {
        flush_header: true,
        chunks: [120, 240, 600],
        chunk_count: 3,
    });
}

#[test]
fn unwritable_path_fails_cleanly_matches_ctest() {
    let comments = create_shared_comments();
    let enc = OggOpusEnc::create_file("ctests/no-such-dir/out.opus", &comments, 48_000, 2, 0);
    assert!(matches!(enc, Err(OpeError::CannotOpen)));
}
