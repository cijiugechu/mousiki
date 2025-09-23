#[path = "common/mod.rs"]
mod common;

#[test]
fn fuzz_decoder_seed_inputs() {
    common::fuzz_decoder(&[]);
    common::fuzz_decoder(common::TINY_OGG);
}
