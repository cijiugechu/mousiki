#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::{
    LibopusencError, MappingFamily, OggOpusComments, OggOpusEncoderBuilder,
};

mod common;

use crate::common::libopusenc::TestBuffer;

fn collect_pull_pages(enc: &mut mousiki::libopusenc::OggOpusPullEncoder, buffer: &mut TestBuffer) {
    while let Some(page) = enc.next_page().expect("next page") {
        buffer.append(&page);
    }
}

#[test]
fn pull_encoding_smoke_matches_ctest() {
    let mut comments = OggOpusComments::new().expect("comments");
    comments
        .add_string("TITLE=Pull path")
        .expect("title comment");

    let mut enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_pull()
        .expect("encoder");
    let pcm = [0i16; 960 * 2];

    enc.flush_headers().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.finish().expect("finish");

    let mut encoded = TestBuffer::default();
    collect_pull_pages(&mut enc, &mut encoded);
    assert!(encoded.contains(b"OggS"));
    assert!(encoded.contains(b"OpusHead"));
    assert!(encoded.contains(b"OpusTags"));
    assert_eq!(Err(LibopusencError::InvalidState), enc.finish());
}
