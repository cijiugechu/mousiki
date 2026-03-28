#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::{
    LibopusencError, MappingFamily, OggOpusComments, OggOpusEncoderBuilder,
};

use crate::common::libopusenc::{BehaviorManifest, TestBuffer};

mod common;

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

#[test]
fn explicit_mapping_configures_independent_encoder() {
    let comments = OggOpusComments::new().expect("comments");
    let pcm = silence(960, 2);

    let mut enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::Independent)
        .expect("builder")
        .mapping(2, 0, &[0, 1])
        .expect("mapping")
        .build_pull()
        .expect("encoder");

    enc.flush_headers().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.finish().expect("finish");

    let mut encoded = Vec::new();
    while let Some(page) = enc.next_page().expect("page") {
        encoded.extend_from_slice(&page);
    }
    let manifest = BehaviorManifest::build(&encoded).expect("manifest");
    assert_eq!(255, manifest.head.mapping);
    assert_eq!(1, count_subslice(&encoded, b"OpusHead"));
    assert_eq!(1, count_subslice(&encoded, b"OpusTags"));
}

#[test]
fn packet_callback_receives_header_comment_and_audio_packets() {
    let mut comments = OggOpusComments::new().expect("comments");
    comments.add_string("TITLE=Packets").expect("title");
    let pcm = silence(960, 2);
    let packets = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Vec<u8>>::new()));
    let seen = packets.clone();

    let mut enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .packet_callback(move |packet: &[u8], _flags: u32| {
            seen.borrow_mut().push(packet.to_vec());
        })
        .build_pull()
        .expect("encoder");

    enc.flush_headers().expect("flush");
    enc.write(&pcm, 960).expect("write");
    enc.finish().expect("finish");

    let packets = packets.borrow();
    assert!(packets.len() >= 3);
    assert_eq!(b"OpusHead", &packets[0][..8]);
    assert_eq!(b"OpusTags", &packets[1][..8]);
}

#[test]
fn start_next_stream_reuses_same_writer_output() {
    let mut first = OggOpusComments::new().expect("comments");
    first.add_string("TITLE=First").expect("title");
    let mut second = OggOpusComments::new().expect("comments");
    second.add_string("TITLE=Second").expect("title");

    let pcm = silence(960, 2);
    let mut enc = OggOpusEncoderBuilder::new(first, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_writer(TestBuffer::default())
        .expect("encoder");

    enc.write(&pcm, 960).expect("first write");
    enc.start_next_stream(second).expect("chain");
    enc.write(&pcm, 960).expect("second write");
    let sink = enc.finish().expect("finish");

    assert_eq!(2, count_subslice(&sink.data, b"OpusHead"));
    assert_eq!(2, count_subslice(&sink.data, b"OpusTags"));
    assert_eq!(1, count_subslice(&sink.data, b"TITLE=First"));
    assert_eq!(1, count_subslice(&sink.data, b"TITLE=Second"));
}

#[test]
fn start_next_stream_on_pull_encoder_reuses_same_page_queue() {
    let mut first = OggOpusComments::new().expect("comments");
    first.add_string("TITLE=First").expect("title");
    let mut second = OggOpusComments::new().expect("comments");
    second.add_string("TITLE=Second").expect("title");

    let pcm = silence(960, 2);
    let mut enc = OggOpusEncoderBuilder::new(first, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_pull()
        .expect("encoder");

    enc.write(&pcm, 960).expect("first write");
    enc.start_next_stream(second).expect("chain");
    enc.write(&pcm, 960).expect("second write");
    enc.finish().expect("finish");

    let mut encoded = Vec::new();
    while let Some(page) = enc.next_page().expect("page") {
        encoded.extend_from_slice(&page);
    }
    assert_eq!(2, count_subslice(&encoded, b"OpusHead"));
    assert_eq!(2, count_subslice(&encoded, b"TITLE="));
}

#[test]
fn file_builder_returns_io_error_for_unwritable_path() {
    let comments = OggOpusComments::new().expect("comments");
    let enc = OggOpusEncoderBuilder::new(comments, 48_000, 2, MappingFamily::MonoStereo)
        .expect("builder")
        .build_file("ctests/no-such-dir/out.opus");
    assert!(matches!(
        enc,
        Err(LibopusencError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound
    ));
}
