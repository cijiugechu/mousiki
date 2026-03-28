#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::{OggOpusComments, OggOpusEnc, OpeError};

mod common;

use crate::common::libopusenc::TestBuffer;

fn collect_pull_pages(enc: &mut OggOpusEnc, buffer: &mut TestBuffer) {
    while let Some(page) = enc.get_page(true).expect("get page") {
        buffer.append(&page);
    }
}

#[test]
fn pull_encoding_smoke_matches_ctest() {
    let mut comments = OggOpusComments::create().expect("comments");
    comments.add("ARTIST", "Smoke").expect("artist");
    comments.add_string("TITLE=Pull path").expect("title");

    let mut enc = OggOpusEnc::create_pull(&comments, 48_000, 2, 0).expect("encoder");
    let mut encoded = TestBuffer::default();
    let pcm = [0i16; 960 * 2];

    enc.set_serialno(4242).expect("serial");
    enc.flush_header().expect("flush");
    collect_pull_pages(&mut enc, &mut encoded);

    assert!(encoded.contains(b"OggS"));
    assert!(encoded.contains(b"OpusHead"));
    assert!(encoded.contains(b"OpusTags"));

    enc.write(&pcm, 960).expect("write");
    enc.drain().expect("drain");
    collect_pull_pages(&mut enc, &mut encoded);

    assert!(encoded.count(b"OggS") >= 2);
    assert!(encoded.data.len() > 64);
    assert_eq!(Err(OpeError::TooLate), enc.drain());
}
