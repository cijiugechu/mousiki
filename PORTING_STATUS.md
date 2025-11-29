# Porting Status

## Small unported modules with ready dependencies
- Self-delimited parsing branch in `opus_decode_native` (`opus-c/src/opus_decoder.c`): the `self_delimited`/`opus_packet_parse_impl` path and frame-size bookkeeping are unported; they sit ahead of the main CELT/SILK decode glue.
- 24-bit output wrapper `opus_decode24` (`opus-c/src/opus_decoder.c`): conversion from the shared `opus_decode_native` result to 24-bit PCM is still missing; it only allocates a temporary buffer and applies `RES2INT24`.

## Recently ported
- Decode gain tracking and top-level decoder CTLs (`OPUS_SET/GET_GAIN`, `OPUS_RESET_STATE`, `OPUS_GET_LAST_PACKET_DURATION`, phase inversion) from `opus_decoder.c`.
- Packet helpers from `opus_decoder.c`: public bandwidth/channel/frame/sample count queries now live in `src/packet.rs` with the decoder wrapper in `src/opus_decoder.rs`, matching the reference return codes.
- `opus_pcm_soft_clip_impl` (`src/opus.rs`): float PCM soft-clip helper plus public wrapper `opus_pcm_soft_clip`.
