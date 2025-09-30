# CELT Codec Notes

This document summarizes the CELT (Constrained Energy Lapped Transform) component of the Opus codec, combining the architectural overview from [DeepWiki's CELT Codec page](https://deepwiki.com/xiph/opus/3.2-celt-codec) with the public C implementation in [xiph/opus](https://github.com/xiph/opus).

## Architectural Summary (DeepWiki)

- **Purpose within Opus.** CELT provides the low-delay, music-optimized branch of Opus, complementing SILK's speech focus. It is intended for high bitrate or hybrid operation where very low algorithmic latency is required.
- **Transform backbone.** CELT uses an MDCT-based analysis/synthesis pipeline with tunable time/frequency resolution. The codec adjusts between long and short MDCTs to track transients while preserving steady-state tonal material.
- **Energy conservation.** Instead of coding absolute spectral samples, CELT allocates bits per perceptual band and preserves band energy, trading detail within the band for overall fidelity.
- **Prediction and enhancement.** Pitch prediction, vector quantization, and spreading control tables (`trim_icdf`, `spread_icdf`, `tapset_icdf`) guide the quantizers and transient handling. Packet loss concealment (PLC) blends pitch extrapolation, noise filling, and smoothing to recover gracefully after loss bursts.
- **Fixed- and floating-point parity.** Implementations support both arithmetic paths and include platform-specific optimizations (x86 SSE, ARM NEON) to meet realtime targets across devices.

## Key Data Structures (xiph/opus)

### `OpusCustomMode` / `CELTMode`

```c
struct OpusCustomMode {
   opus_int32 Fs;
   int         overlap;
   int         nbEBands;
   int         effEBands;
   opus_val16 preemph[4];
   int         maxLM;
   int         nbShortMdcts;
   int         shortMdctSize;
   opus_val16 *eBands;
   opus_val16 *normalization;
   int         windowLow;
   int         windowHigh;
   mdct_lookup mdct;
   mdct_lookup imdct;
   int *logN;
   int         maxLpc;
   int         LM;
   struct OpusCustomMode *prev;
   struct OpusCustomMode *next;
   const int *cache;
   struct CELTModeCache cache;
};
```
_Source: [`celt/celt.c`](https://github.com/xiph/opus/blob/master/celt/celt.c) and [`celt/celt.h`](https://github.com/xiph/opus/blob/master/celt/celt.h), as referenced by DeepWiki._

This mode structure defines the transform geometry, overlap, critical band layout, and cached MDCT lookups. Both encoders and decoders must be initialized with the same `OpusCustomMode`, created through `opus_custom_mode_create()` (`include/opus_custom.h`).

### `OpusCustomEncoder` (`CELTEncoder` alias)

```c
struct OpusCustomEncoder {
   const OpusCustomMode *mode;
   int channels;
   int stream_channels;
   int force_intra;
   int clip;
   int disable_pf;
   int complexity;
   int upsample;
   int start, end;
   opus_int32 bitrate;
   int vbr;
   int signalling;
   int constrained_vbr;
   int loss_rate;
   int lsb_depth;
   int lfe;
   int disable_inv;
   int arch;
   opus_uint32 rng;
   int spread_decision;
   opus_val32 delayedIntra;
   int tonal_average;
   int lastCodedBands;
   int hf_average;
   int tapset_decision;
   int prefilter_period;
   opus_val16 prefilter_gain;
   int prefilter_tapset;
   int consec_transient;
   AnalysisInfo analysis;
   SILKInfo silk_info;
   opus_val32 preemph_memE[2];
   opus_val32 preemph_memD[2];
   opus_int32 vbr_reservoir;
   opus_int32 vbr_drift;
   opus_int32 vbr_offset;
   opus_int32 vbr_count;
   opus_val32 overlap_max;
   opus_val16 stereo_saving;
   int intensity;
   celt_glog *energy_mask;
   celt_glog spec_avg;
   celt_sig in_mem[1];
   /* Additional channel/band work buffers follow in allocated storage. */
};
```
_Source: [`celt/celt_encoder.c`](https://github.com/xiph/opus/blob/master/celt/celt_encoder.c)._ This state object gathers rate control, prediction decisions, and scratch buffers (prefilter, band energy histories) described qualitatively in the DeepWiki article.

### `OpusCustomDecoder` (`CELTDecoder` alias)

```c
struct OpusCustomDecoder {
   const OpusCustomMode *mode;
   int overlap;
   int channels;
   int stream_channels;
   int downsample;
   int start, end;
   int signalling;
   int disable_inv;
   int complexity;
   int arch;
   opus_uint32 rng;
   int error;
   int last_pitch_index;
   int loss_duration;
   int skip_plc;
   int postfilter_period;
   int postfilter_period_old;
   opus_val16 postfilter_gain;
   opus_val16 postfilter_gain_old;
   int postfilter_tapset;
   int postfilter_tapset_old;
   int prefilter_and_fold;
   celt_sig preemph_memD[2];
   celt_sig _decode_mem[1];
   /* LPC history and band energy buffers follow in allocated storage. */
};
```
_Source: [`celt/celt_decoder.c`](https://github.com/xiph/opus/blob/master/celt/celt_decoder.c)._ Decoder PLC and post-filter controls map directly to DeepWiki's packet-loss recovery discussion.

## Public CELT Entry Points (from `celt/celt.h`)

- `int celt_encoder_get_size(int channels);`
- `int celt_encoder_init(CELTEncoder *st, opus_int32 sampling_rate, int channels, int arch);`
- `int celt_encode_with_ec(CELTEncoder *st, const opus_res *pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc);`
- `int celt_decoder_get_size(int channels);`
- `int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels);`
- `int celt_decode_with_ec(CELTDecoder *st, const unsigned char *data, int len, opus_res *pcm, int frame_size, ec_dec *dec, int accum);`

Compile-time aliases map these to the public `opus_custom_*` API so CELT integrates seamlessly with the broader Opus interface. Control (CTL) requests such as `CELT_SET_PREDICTION`, `CELT_SET_ANALYSIS`, and `CELT_SET_SILK_INFO` expose configuration hooks for prediction, external analysis hints, and hybrid SILK/CELT signaling (`celt/celt.h`).

## Processing Highlights

| Stage | Notes | Key Sources |
| --- | --- | --- |
| Time-frequency analysis | MDCT-based analysis, selectable short/long blocks, window overlap per `OpusCustomMode`. | DeepWiki (MDCT section); [`celt/mdct.c`](https://github.com/xiph/opus/blob/master/celt/mdct.c) |
| Band energy quantization | Critical-band layout from mode, bit allocation guided by `quant_bands.c` tables and spread control. | DeepWiki (Energy/Quantization); [`celt/quant_bands.c`](https://github.com/xiph/opus/blob/master/celt/quant_bands.c) |
| Pitch prediction & postfilter | Search/pruned cross-correlation for tonal content (`pitch.c`), comb-filter postfilter parameters in encoder/decoder state. | DeepWiki (Pitch Analysis); [`celt/pitch.c`](https://github.com/xiph/opus/blob/master/celt/pitch.c) |
| Packet loss concealment | Pitch extrapolation, noise fill, smooth transitions implemented in decoder PLC routines. | DeepWiki (Error Resilience); [`celt/celt_decoder.c`](https://github.com/xiph/opus/blob/master/celt/celt_decoder.c) |
| SIMD acceleration | Architecture-specific kernels in `celt/x86` and `celt/arm`. | DeepWiki (Performance); [`celt/x86/` and `celt/arm/`] |

## Practical Usage Flow

1. **Create or reuse a `OpusCustomMode`.** Use `opus_custom_mode_create(sample_rate, frame_size, &err)` and share across streams.
2. **Allocate encoder/decoder states.** Call `opus_custom_encoder_create(mode, channels, &err)` / `opus_custom_decoder_create(mode, channels, &err)`; internally these wrap the CELT-specific size and init helpers above.
3. **Encode frames.** Feed PCM blocks (frame size samples per channel) to `opus_custom_encode_float()` or `opus_custom_encode()`; these reach `celt_encode_with_ec()` after SILK/hybrid routing.
4. **Decode frames.** Pass packet payloads into `opus_custom_decode()`/`opus_custom_decode_float()`, invoking `celt_decode_with_ec()` with PLC and postfilter enabled.
5. **Adjust behavior via CTLs.** Tweak prediction, channel layout, low-frequency effects, or provide external analysis hints using the `CELT_*` request macros if bypassing the higher-level Opus API.

## Cross-reference Checklist

- DeepWiki sections: *Core Principles and Architecture*, *Encoder Structure*, *Decoder Structure*, *Core Processing Elements*, *Error Resilience and Packet Loss Concealment*, *Performance Considerations*.
- xiph/opus files: `celt/celt.h`, `celt/celt_encoder.c`, `celt/celt_decoder.c`, `celt/mdct.c`, `celt/quant_bands.c`, `celt/pitch.c`, `celt/x86/*`, `celt/arm/*`, `include/opus_custom.h`.

These references align DeepWiki's prose with the concrete types and functions exposed by the upstream C implementation.
