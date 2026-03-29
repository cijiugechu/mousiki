#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::{
    LibopusencError, MappingFamily, MuxingDelaySamples, OggOpusComments, OggOpusEncoderBuilder,
    get_version_string,
};

mod common;

use crate::common::libopusenc::{BehaviorManifest, TestBuffer};

#[test]
fn comments_api_matches_ctest() {
    let mut comments = OggOpusComments::new().expect("comments");
    comments.add("ARTIST", "Someone").expect("artist");
    comments.add_string("TITLE=Track").expect("title");

    let mut copy = comments.clone();
    copy.add("ALBUM", "Record").expect("album");
}

#[test]
fn invalid_arguments_match_ctest() {
    let mut comments = OggOpusComments::new().expect("comments");

    assert_eq!(
        Err(LibopusencError::InvalidArgument),
        comments.add("BAD=TAG", "value")
    );
    assert_eq!(
        Err(LibopusencError::InvalidArgument),
        comments.add_string("MISSING_SEPARATOR")
    );

    assert!(OggOpusEncoderBuilder::new(comments.clone(), 0, 2, MappingFamily::MonoStereo).is_err());
    assert!(
        OggOpusEncoderBuilder::new(comments.clone(), 48_000, 0, MappingFamily::MonoStereo).is_err()
    );

    let enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_file("ctests/no-such-dir/out.opus");
    assert!(matches!(
        enc,
        Err(LibopusencError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn builder_options_apply_to_output() {
    let comments = OggOpusComments::new().expect("comments");
    let mut enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .decision_delay(48_000)
        .comment_padding(64)
        .serialno(12_345)
        .header_gain(17)
        .muxing_delay(MuxingDelaySamples(960))
        .build_pull()
        .expect("encoder");

    enc.flush_headers().expect("flush");
    enc.write(&[0i16; 2], 0).expect("write zero");
    enc.finish().expect("finish");

    let mut encoded = TestBuffer::default();
    while let Some(page) = enc.next_page().expect("page") {
        encoded.append(&page);
    }
    let manifest = BehaviorManifest::build(&encoded.data).expect("manifest");
    assert_eq!(12_345, manifest.pages[0].serialno as i32);
    assert_eq!(17i16 as u16, manifest.head.gain_raw);
}

#[test]
fn version_exports_match_ctest() {
    let version = get_version_string();
    assert!(version.starts_with("libopusenc "));
}
