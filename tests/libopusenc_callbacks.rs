#![cfg(feature = "libopusenc")]

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;

mod common;

use crate::common::libopusenc::TestBuffer;
use mousiki::libopusenc::{MappingFamily, OggOpusComments, OggOpusEncoder, OggOpusEncoderBuilder};

#[derive(Default)]
struct MemorySink {
    encoded: TestBuffer,
    flush_calls: usize,
}

struct SharedSink(Rc<RefCell<MemorySink>>);

impl Write for SharedSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().encoded.append(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.borrow_mut().flush_calls += 1;
        Ok(())
    }
}

fn build_encoder(
    comments: OggOpusComments,
    sink: Rc<RefCell<MemorySink>>,
) -> OggOpusEncoder<SharedSink> {
    OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_writer(SharedSink(sink))
        .expect("encoder")
}

#[test]
fn writer_encoding_smoke_matches_ctest() {
    let mut comments = OggOpusComments::new().expect("comments");
    comments
        .add_string("TITLE=Writer path")
        .expect("title comment");

    let sink = Rc::new(RefCell::new(MemorySink::default()));
    let mut enc = build_encoder(comments, sink.clone());
    let pcm = [0i16; 960 * 2];

    enc.flush_headers().expect("flush");
    enc.write(&pcm, 960).expect("write");
    let _ = enc.finish().expect("finish");

    let sink = sink.borrow();
    assert!(sink.encoded.contains(b"OggS"));
    assert!(sink.encoded.contains(b"OpusHead"));
    assert!(sink.encoded.contains(b"OpusTags"));
    assert_eq!(1, sink.flush_calls);
}

#[test]
fn writer_path_emits_audio_before_finish_when_delay_zero() {
    let comments = OggOpusComments::new().expect("comments");
    let sink = Rc::new(RefCell::new(MemorySink::default()));
    let mut enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .decision_delay(0)
        .build_writer(SharedSink(sink.clone()))
        .expect("encoder");
    let pcm = [0i16; 960 * 2];

    enc.write(&pcm, 960).expect("first write");
    let after_first_write = sink.borrow().encoded.data.len();
    assert!(after_first_write > 0);

    let mut grew_before_finish = false;
    for _ in 0..128 {
        enc.write(&pcm, 960).expect("streaming write");
        if sink.borrow().encoded.data.len() > after_first_write {
            grew_before_finish = true;
            break;
        }
    }
    assert!(grew_before_finish);

    let _ = enc.finish().expect("finish");
}
