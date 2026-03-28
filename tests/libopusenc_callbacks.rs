#![cfg(feature = "libopusenc")]

use std::cell::RefCell;
use std::rc::Rc;

mod common;

use crate::common::libopusenc::TestBuffer;
use mousiki::libopusenc::{OggOpusComments, OggOpusEnc, OpeError, OpusEncCallbacks};

#[derive(Default)]
struct MemorySink {
    encoded: TestBuffer,
    close_calls: usize,
}

struct SharedSink(Rc<RefCell<MemorySink>>);

impl OpusEncCallbacks for SharedSink {
    fn write(&mut self, data: &[u8]) -> Result<(), OpeError> {
        self.0.borrow_mut().encoded.append(data);
        Ok(())
    }

    fn close(&mut self) -> Result<(), OpeError> {
        self.0.borrow_mut().close_calls += 1;
        Ok(())
    }
}

#[test]
fn callback_encoding_smoke_matches_ctest() {
    let mut comments = OggOpusComments::create().expect("comments");
    comments
        .add_string("TITLE=Callback path")
        .expect("title comment");

    let sink = Rc::new(RefCell::new(MemorySink::default()));
    let callbacks = Box::new(SharedSink(sink.clone()));
    let mut enc =
        OggOpusEnc::create_callbacks(callbacks, &comments, 48_000, 2, 0).expect("encoder");
    let pcm = [0i16; 960 * 2];

    enc.flush_header().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.drain().expect("drain");

    let sink = sink.borrow();
    assert!(sink.encoded.contains(b"OggS"));
    assert!(sink.encoded.contains(b"OpusHead"));
    assert!(sink.encoded.contains(b"OpusTags"));
    assert_eq!(1, sink.close_calls);
}

#[test]
fn callback_path_emits_audio_before_drain_when_delay_zero() {
    let comments = OggOpusComments::create().expect("comments");
    let sink = Rc::new(RefCell::new(MemorySink::default()));
    let callbacks = Box::new(SharedSink(sink.clone()));
    let mut enc =
        OggOpusEnc::create_callbacks(callbacks, &comments, 48_000, 2, 0).expect("encoder");
    let pcm = [0i16; 960 * 2];

    enc.set_decision_delay(0).expect("delay");
    enc.write(&pcm, 960).expect("first write");
    let after_first_write = sink.borrow().encoded.data.len();
    assert!(after_first_write > 0);

    let mut grew_before_drain = false;
    for _ in 0..128 {
        enc.write(&pcm, 960).expect("streaming write");
        if sink.borrow().encoded.data.len() > after_first_write {
            grew_before_drain = true;
            break;
        }
    }
    assert!(grew_before_drain);

    enc.drain().expect("drain");
}
