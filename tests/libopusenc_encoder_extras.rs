#![cfg(feature = "libopusenc")]

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mousiki::libopusenc::encoder::{OggOpusComments, OggOpusEnc, OpusEncCallbacks, OpeError};

#[derive(Default)]
struct MemorySink {
    encoded: Vec<u8>,
    close_calls: usize,
}

struct SharedSink(Rc<RefCell<MemorySink>>);

impl OpusEncCallbacks for SharedSink {
    fn write(&mut self, data: &[u8]) -> Result<(), OpeError> {
        self.0.borrow_mut().encoded.extend_from_slice(data);
        Ok(())
    }

    fn close(&mut self) -> Result<(), OpeError> {
        self.0.borrow_mut().close_calls += 1;
        Ok(())
    }
}

fn count_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    haystack
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

fn silence(frames: usize, channels: usize) -> Vec<i16> {
    vec![0; frames * channels]
}

static UNIQUE_SUFFIX: AtomicU64 = AtomicU64::new(0);

fn temp_path(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let suffix = UNIQUE_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{unique}-{suffix}.opus"))
}

#[test]
fn deferred_init_with_mapping_enables_explicit_family_255_encoder() {
    let comments = OggOpusComments::create().expect("comments");
    let mut enc = OggOpusEnc::create_pull(&comments, 48_000, 2, -1).expect("encoder");
    let pcm = silence(960, 2);

    assert_eq!(Err(OpeError::TooLate), enc.flush_header());
    enc.deferred_init_with_mapping(255, 2, 0, &[0, 1])
        .expect("deferred init");
    assert_eq!(2, enc.nb_streams());
    assert_eq!(0, enc.nb_coupled_streams());

    enc.flush_header().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.drain().expect("drain");

    let mut encoded = Vec::new();
    while let Some(page) = enc.get_page(true).expect("page") {
        encoded.extend_from_slice(&page);
    }
    assert_eq!(1, count_subslice(&encoded, b"OpusHead"));
    assert_eq!(1, count_subslice(&encoded, b"OpusTags"));
}

#[test]
fn packet_callback_receives_header_comment_and_audio_packets() {
    let mut comments = OggOpusComments::create().expect("comments");
    comments.add_string("TITLE=Packets").expect("title");
    let mut enc = OggOpusEnc::create_pull(&comments, 48_000, 2, 0).expect("encoder");
    let pcm = silence(960, 2);
    let packets = Rc::new(RefCell::new(Vec::<Vec<u8>>::new()));
    let seen = packets.clone();

    enc.set_packet_callback(Some(Box::new(move |packet, _flags| {
        seen.borrow_mut().push(packet.to_vec());
    })))
    .expect("set packet callback");
    enc.flush_header().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.drain().expect("drain");

    let packets = packets.borrow();
    assert!(packets.len() >= 3);
    assert_eq!(b"OpusHead", &packets[0][..8]);
    assert_eq!(b"OpusTags", &packets[1][..8]);
}

#[test]
fn chain_current_reuses_output_and_writes_two_logical_streams() {
    let mut first = OggOpusComments::create().expect("comments");
    first.add_string("TITLE=First").expect("title");
    let mut second = OggOpusComments::create().expect("comments");
    second.add_string("TITLE=Second").expect("title");

    let sink = Rc::new(RefCell::new(MemorySink::default()));
    let callbacks = Box::new(SharedSink(sink.clone()));
    let mut enc = OggOpusEnc::create_callbacks(callbacks, &first, 48_000, 2, 0).expect("encoder");
    let pcm = silence(960, 2);

    enc.write(&pcm, 960).expect("first write");
    enc.chain_current(&second).expect("chain");
    enc.write(&pcm, 960).expect("second write");
    enc.drain().expect("drain");

    let sink = sink.borrow();
    assert_eq!(1, sink.close_calls);
    assert_eq!(2, count_subslice(&sink.encoded, b"OpusHead"));
    assert_eq!(2, count_subslice(&sink.encoded, b"OpusTags"));
    assert_eq!(1, count_subslice(&sink.encoded, b"TITLE=First"));
    assert_eq!(1, count_subslice(&sink.encoded, b"TITLE=Second"));
}

#[test]
fn continue_new_callbacks_switches_to_new_sink() {
    let mut first = OggOpusComments::create().expect("comments");
    first.add_string("TITLE=First").expect("title");
    let mut second = OggOpusComments::create().expect("comments");
    second.add_string("TITLE=Second").expect("title");

    let sink1 = Rc::new(RefCell::new(MemorySink::default()));
    let sink2 = Rc::new(RefCell::new(MemorySink::default()));
    let mut enc = OggOpusEnc::create_callbacks(
        Box::new(SharedSink(sink1.clone())),
        &first,
        48_000,
        2,
        0,
    )
    .expect("encoder");
    let pcm = silence(960, 2);

    enc.write(&pcm, 960).expect("first write");
    enc.continue_new_callbacks(Box::new(SharedSink(sink2.clone())), &second)
        .expect("continue");
    enc.write(&pcm, 960).expect("second write");
    enc.drain().expect("drain");

    let sink1 = sink1.borrow();
    let sink2 = sink2.borrow();
    assert_eq!(1, sink1.close_calls);
    assert_eq!(1, sink2.close_calls);
    assert_eq!(1, count_subslice(&sink1.encoded, b"OpusHead"));
    assert_eq!(1, count_subslice(&sink2.encoded, b"OpusHead"));
    assert_eq!(1, count_subslice(&sink1.encoded, b"TITLE=First"));
    assert_eq!(1, count_subslice(&sink2.encoded, b"TITLE=Second"));
}

#[test]
fn continue_new_file_switches_to_new_output_file() {
    let mut first = OggOpusComments::create().expect("comments");
    first.add_string("TITLE=First").expect("title");
    let mut second = OggOpusComments::create().expect("comments");
    second.add_string("TITLE=Second").expect("title");

    let path1 = temp_path("mousiki-libopusenc-first");
    let path2 = temp_path("mousiki-libopusenc-second");
    let mut enc = OggOpusEnc::create_file(path1.to_str().unwrap(), &first, 48_000, 2, 0)
        .expect("encoder");
    let pcm = silence(960, 2);

    enc.write(&pcm, 960).expect("first write");
    enc.continue_new_file(path2.to_str().unwrap(), &second)
        .expect("continue");
    enc.write(&pcm, 960).expect("second write");
    enc.drain().expect("drain");

    let first_data = fs::read(&path1).expect("first file");
    let second_data = fs::read(&path2).expect("second file");
    fs::remove_file(path1).expect("remove first");
    fs::remove_file(path2).expect("remove second");

    assert_eq!(1, count_subslice(&first_data, b"OpusHead"));
    assert_eq!(1, count_subslice(&second_data, b"OpusHead"));
    assert_eq!(1, count_subslice(&first_data, b"TITLE=First"));
    assert_eq!(1, count_subslice(&second_data, b"TITLE=Second"));
}
