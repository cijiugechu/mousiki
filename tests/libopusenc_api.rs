#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::encoder::{
    OggOpusComments, OggOpusEnc, OpeError, get_abi_version, get_version_string, strerror,
};

#[test]
fn comments_api_matches_ctest() {
    let mut comments = OggOpusComments::create().expect("comments");
    comments.add("ARTIST", "Someone").expect("artist");
    comments.add_string("TITLE=Track").expect("title");

    let mut copy = comments.copy();
    copy.add("ALBUM", "Record").expect("album");
}

#[test]
fn invalid_arguments_match_ctest() {
    let mut comments = OggOpusComments::create().expect("comments");

    assert_eq!(Err(OpeError::BadArg), comments.add("BAD=TAG", "value"));
    assert_eq!(Err(OpeError::BadArg), comments.add_string("MISSING_SEPARATOR"));

    assert!(OggOpusEnc::create_pull(&comments, 0, 2, 0).is_err());
    assert!(OggOpusEnc::create_pull(&comments, 48_000, 2, -2).is_err());

    let enc = OggOpusEnc::create_file("ctests/no-such-dir/out.opus", &comments, 48_000, 2, 0);
    assert!(matches!(enc, Err(OpeError::CannotOpen)));
}

#[test]
fn ctl_contract_matches_ctest() {
    let comments = OggOpusComments::create().expect("comments");
    let mut enc = OggOpusEnc::create_pull(&comments, 48_000, 2, 0).expect("encoder");
    let pcm = [0i16; 2];

    enc.set_decision_delay(48_000).expect("decision delay");
    assert_eq!(48_000, enc.decision_delay());

    enc.set_comment_padding(64).expect("comment padding");
    assert_eq!(64, enc.comment_padding());

    enc.set_serialno(12_345).expect("serial");
    enc.set_header_gain(17).expect("gain");

    enc.flush_header().expect("flush header");

    enc.set_serialno(54_321).expect("serial after flush");
    enc.set_header_gain(9).expect("gain after flush");
    enc.write(&pcm, 0).expect("write zero");
    assert_eq!(Err(OpeError::TooLate), enc.set_serialno(777));
    assert_eq!(Err(OpeError::TooLate), enc.set_header_gain(11));

    assert_eq!(54_321, enc.serialno().expect("serial"));
    assert_eq!(9, enc.header_gain());
}

#[test]
fn version_exports_match_ctest() {
    let version = get_version_string();
    assert!(version.starts_with("libopusenc "));
    assert_eq!(0, get_abi_version());
    assert_eq!("success", strerror(OpeError::Ok));
}
