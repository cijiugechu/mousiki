# Porting Status

## Small unported modules with ready dependencies
- Decode gain and lightweight CTL wiring in `opus-c/src/opus_decoder.c`: the `decode_gain` field plus `OPUS_SET/GET_GAIN`, `OPUS_RESET_STATE`, `OPUS_GET_LAST_PACKET_DURATION`, and phase inversion CTLs are still unported but only read/write decoder fields or forward to `celt_decoder_ctl`.
- Packet helpers in `opus-c/src/opus_decoder.c`: `opus_packet_get_bandwidth`, `opus_packet_get_nb_channels`, `opus_packet_get_nb_frames`, and `opus_packet_get_nb_samples` (including `opus_decoder_get_nb_samples`) remain to be ported; they are pure TOC/length parsing without DSP dependencies.
- Self-delimited parsing branch in `opus_decode_native` (`opus-c/src/opus_decoder.c`): the `self_delimited`/`opus_packet_parse_impl` path and frame-size bookkeeping are unported; they sit ahead of the main CELT/SILK decode glue.
- 24-bit output wrapper `opus_decode24` (`opus-c/src/opus_decoder.c`): conversion from the shared `opus_decode_native` result to 24-bit PCM is still missing; it only allocates a temporary buffer and applies `RES2INT24`.

## Recently ported
- `opus_pcm_soft_clip_impl` (`src/opus.rs`): float PCM soft-clip helper plus public wrapper `opus_pcm_soft_clip`.
