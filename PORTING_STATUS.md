# Porting Status

## Small unported modules with ready dependencies
- 24-bit output wrapper `opus_decode24` (`opus-c/src/opus_decoder.c`): conversion from the shared `opus_decode_native` result to 24-bit PCM is still missing; it only allocates a temporary buffer and applies `RES2INT24`.

## Recently ported
- Self-delimited parsing and packet-offset bookkeeping in the `opus_decode_native` front-end. The shared `opus_packet_parse_impl` helper now lives in `src/packet.rs` with support for self-delimited framing and padding offsets ahead of the CELT/SILK decode glue.
- Decode gain tracking and top-level decoder CTLs (`OPUS_SET/GET_GAIN`, `OPUS_RESET_STATE`, `OPUS_GET_LAST_PACKET_DURATION`, phase inversion) from `opus_decoder.c`.
- Packet helpers from `opus_decoder.c`: public bandwidth/channel/frame/sample count queries now live in `src/packet.rs` with the decoder wrapper in `src/opus_decoder.rs`, matching the reference return codes.
- `opus_pcm_soft_clip_impl` (`src/opus.rs`): float PCM soft-clip helper plus public wrapper `opus_pcm_soft_clip`.
