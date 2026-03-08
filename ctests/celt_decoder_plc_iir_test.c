#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arch.h"
#include "celt.h"
#include "celt_lpc.h"
#include "opus_custom.h"
#include "opus_defines.h"

#ifdef FIXED_POINT

enum {
  kSampleRate = 48000,
  kFrameSize = 960,
  kMaxPacketSize = 1276,
  kPrimeFrames = 16,
  kMonoChannels = 1,
  kStereoChannels = 2,
  kStartBandNoiseOnly = 17,
  kPitchLossFrames = 2,
  kPcmHashCount = 4
};

#define DECODE_BUFFER_SIZE 2048
#define PLC_UPDATE_FRAMES 4
#define PLC_UPDATE_SAMPLES (PLC_UPDATE_FRAMES * FRAME_SIZE)

typedef struct {
  const OpusCustomMode *mode;
  int overlap;
  int channels;
  int stream_channels;
  int downsample;
  int start;
  int end;
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
#ifdef ENABLE_DEEP_PLC
  opus_int16 plc_pcm[PLC_UPDATE_SAMPLES];
  int plc_fill;
  float plc_preemphasis_mem;
#endif
  celt_sig decode_mem[1];
} DecoderMirror;

typedef struct {
  uint32_t lpc_hash[kPitchLossFrames];
  uint32_t tail_hash[kPitchLossFrames];
  uint32_t pcm_hash[kPitchLossFrames];
  uint32_t noise_lpc_hash;
  uint32_t noise_tail_hash;
} ExpectedInternals;

static uint32_t fnv1a_update(uint32_t hash, unsigned char byte) {
  hash ^= (uint32_t)byte;
  return hash * 16777619u;
}

static uint32_t hash_i16_le(const opus_val16 *values, int count) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < count; ++i) {
    uint16_t v = (uint16_t)values[i];
    hash = fnv1a_update(hash, (unsigned char)(v & 0xFF));
    hash = fnv1a_update(hash, (unsigned char)((v >> 8) & 0xFF));
  }
  return hash;
}

static uint32_t hash_sig_le(const celt_sig *values, int count) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < count; ++i) {
    uint32_t v = (uint32_t)values[i];
    hash = fnv1a_update(hash, (unsigned char)(v & 0xFF));
    hash = fnv1a_update(hash, (unsigned char)((v >> 8) & 0xFF));
    hash = fnv1a_update(hash, (unsigned char)((v >> 16) & 0xFF));
    hash = fnv1a_update(hash, (unsigned char)((v >> 24) & 0xFF));
  }
  return hash;
}

static uint32_t hash_pcm_le(const opus_int16 *pcm, int count) {
  return hash_i16_le((const opus_val16 *)pcm, count);
}

static int64_t sum_abs_i16(const opus_val16 *values, int count) {
  int64_t sum = 0;
  for (int i = 0; i < count; ++i) {
    int v = values[i];
    sum += v < 0 ? -v : v;
  }
  return sum;
}

static int has_nonzero_pcm(const opus_int16 *pcm, int count) {
  for (int i = 0; i < count; ++i) {
    if (pcm[i] != 0) return 1;
  }
  return 0;
}

static void fill_periodic_frame(opus_int16 *pcm, int frame_size, int channels,
                                int phase) {
  const int period = 96;
  const int half = period / 2;
  const int amp = 14000;
  for (int i = 0; i < frame_size; ++i) {
    int pos = (i + phase) % period;
    int tri = pos < half ? pos : (period - pos);
    int centered = (tri * 2) - half;
    int sample_l = (centered * amp) / half;
    if (channels == 1) {
      pcm[i] = (opus_int16)sample_l;
    } else {
      int pos_r = (i + phase + 23) % period;
      int tri_r = pos_r < half ? pos_r : (period - pos_r);
      int centered_r = (tri_r * 2) - half;
      int sample_r = -((centered_r * (amp - 2500)) / half);
      pcm[2 * i] = (opus_int16)sample_l;
      pcm[2 * i + 1] = (opus_int16)sample_r;
    }
  }
}

static int check_eq_int(const char *label, int got, int expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, got, expected);
    return 1;
  }
  return 0;
}

static int check_eq_u32(const char *label, uint32_t got, uint32_t expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got 0x%08x expected 0x%08x\n", label, got,
            expected);
    return 1;
  }
  return 0;
}

static int check_true(const char *label, int cond) {
  if (!cond) {
    fprintf(stderr, "%s failed\n", label);
    return 1;
  }
  return 0;
}

static int configure_encoder(OpusCustomEncoder *enc, int bitrate) {
  int failures = 0;
  failures += check_eq_int("set_bitrate",
                           opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate)),
                           OPUS_OK);
  failures += check_eq_int("set_vbr",
                           opus_custom_encoder_ctl(enc, OPUS_SET_VBR(0)),
                           OPUS_OK);
  failures += check_eq_int("set_complexity",
                           opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10)),
                           OPUS_OK);
  failures += check_eq_int("set_lsb_depth",
                           opus_custom_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(16)),
                           OPUS_OK);
  return failures;
}

static int prime_decoder(OpusCustomEncoder *enc, OpusCustomDecoder *dec,
                         opus_int16 *work_pcm, unsigned char *packet,
                         int channels, int *pitch_out) {
  int failures = 0;
  *pitch_out = 0;
  for (int frame = 0; frame < kPrimeFrames; ++frame) {
    int packet_len;
    int decoded_len;
    fill_periodic_frame(work_pcm, kFrameSize, channels, frame * 7);
    packet_len =
        opus_custom_encode(enc, work_pcm, kFrameSize, packet, kMaxPacketSize);
    if (packet_len <= 0) {
      fprintf(stderr, "encode failed while priming on frame %d: %d\n", frame,
              packet_len);
      return failures + 1;
    }
    decoded_len =
        opus_custom_decode(dec, packet, packet_len, work_pcm, kFrameSize);
    failures += check_eq_int("prime_decode_len", decoded_len, kFrameSize);
    failures += check_eq_int("prime_get_pitch",
                             opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(pitch_out)),
                             OPUS_OK);
    if (*pitch_out > 0) return failures;
  }
  fprintf(stderr, "failed to prime decoder pitch after %d frames\n", kPrimeFrames);
  return failures + 1;
}

static DecoderMirror *decoder_mirror(OpusCustomDecoder *dec) {
  return (DecoderMirror *)dec;
}

static celt_sig *decoder_channel_mem(DecoderMirror *dec, int channel) {
  return dec->decode_mem + channel * (DECODE_BUFFER_SIZE + dec->overlap);
}

static opus_val16 *decoder_lpc(DecoderMirror *dec) {
  return (opus_val16 *)(dec->decode_mem +
                        (DECODE_BUFFER_SIZE + dec->overlap) * dec->channels);
}

static uint32_t decoder_lpc_hash(OpusCustomDecoder *dec) {
  DecoderMirror *mirror = decoder_mirror(dec);
  return hash_i16_le(decoder_lpc(mirror), mirror->channels * CELT_LPC_ORDER);
}

static uint32_t decoder_tail_hash(OpusCustomDecoder *dec) {
  DecoderMirror *mirror = decoder_mirror(dec);
  uint32_t hash = 2166136261u;
  const int tail_len = kFrameSize + mirror->overlap;
  for (int ch = 0; ch < mirror->channels; ++ch) {
    const celt_sig *tail =
        decoder_channel_mem(mirror, ch) + DECODE_BUFFER_SIZE - kFrameSize;
    hash ^= hash_sig_le(tail, tail_len);
    hash *= 16777619u;
  }
  return hash;
}

static int assert_lpc_invariants(const char *label, OpusCustomDecoder *dec) {
  DecoderMirror *mirror = decoder_mirror(dec);
  opus_val16 *lpc = decoder_lpc(mirror);
  int failures = 0;
  for (int ch = 0; ch < mirror->channels; ++ch) {
    int64_t abs_sum = sum_abs_i16(lpc + ch * CELT_LPC_ORDER, CELT_LPC_ORDER);
    char sum_label[96];
    snprintf(sum_label, sizeof(sum_label), "%s_ch%d_abs_sum_nonzero", label, ch);
    failures += check_true(sum_label, abs_sum > 0);
    snprintf(sum_label, sizeof(sum_label), "%s_ch%d_abs_sum_bounded", label, ch);
    failures += check_true(sum_label, abs_sum < 65535);
  }
  return failures;
}

static int dump_expected_enabled(void) {
  const char *env = getenv("CELT_DECODER_PLC_IIR_DUMP");
  return env != NULL && env[0] != '\0' && env[0] != '0';
}

static void dump_case(const char *name, const uint32_t *lpc_hash,
                      const uint32_t *tail_hash, const uint32_t *pcm_hash,
                      int count) {
  printf("%s_lpc_hashes:", name);
  for (int i = 0; i < count; ++i) printf(" 0x%08x", lpc_hash[i]);
  printf("\n%s_tail_hashes:", name);
  for (int i = 0; i < count; ++i) printf(" 0x%08x", tail_hash[i]);
  printf("\n%s_pcm_hashes:", name);
  for (int i = 0; i < count; ++i) printf(" 0x%08x", pcm_hash[i]);
  printf("\n");
}

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  const int dump_expected = dump_expected_enabled();
  const ExpectedInternals mono_expected = {
      {0x51597095u, 0x51597095u},
      {0xdfaa14e9u, 0xefd80164u},
      {0xc1b6b08du, 0xc2a465edu},
      0xc655ff85u,
      0x05e4dd21u,
  };
  const ExpectedInternals stereo_expected = {
      {0xd31a6449u, 0xd31a6449u},
      {0x7bf7ff22u, 0xa5d203d2u},
      {0x03cb2a69u, 0x16d39455u},
      0,
      0,
  };
  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  {
    unsigned char packet[kMaxPacketSize];
    opus_int16 pcm[kFrameSize * kMonoChannels];
    uint32_t lpc_hashes[kPcmHashCount] = {0};
    uint32_t tail_hashes[kPcmHashCount] = {0};
    uint32_t pcm_hashes[kPcmHashCount] = {0};
    int primed_pitch = 0;

    OpusCustomEncoder *enc =
        opus_custom_encoder_create(mode, kMonoChannels, &err);
    OpusCustomDecoder *dec =
        opus_custom_decoder_create(mode, kMonoChannels, &err);
    if (enc == NULL || dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create mono codec pair: %d\n", err);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc, 14000);
    failures += check_eq_int("mono_set_complexity_bad",
                             opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(11)),
                             OPUS_BAD_ARG);
    {
      opus_int16 *null_pcm = NULL;
      failures += check_eq_int("mono_decode_null_pcm",
                               opus_custom_decode(dec, NULL, 0, null_pcm,
                                                  kFrameSize),
                               OPUS_BAD_ARG);
    }
    failures += prime_decoder(enc, dec, pcm, packet, kMonoChannels, &primed_pitch);
    failures += check_true("mono_primed_pitch_nonzero", primed_pitch > 0);

    lpc_hashes[0] = decoder_lpc_hash(dec);
    tail_hashes[0] = decoder_tail_hash(dec);

    packet[0] = 0;
    failures += check_true("mono_tiny_packet_fails",
                           opus_custom_decode(dec, packet, 1, pcm, kFrameSize) < 0);
    lpc_hashes[1] = decoder_lpc_hash(dec);
    tail_hashes[1] = decoder_tail_hash(dec);
    failures += check_eq_u32("mono_lpc_unchanged_after_tiny_packet", lpc_hashes[1],
                             lpc_hashes[0]);
    failures += check_eq_u32("mono_tail_unchanged_after_tiny_packet",
                             tail_hashes[1], tail_hashes[0]);

    for (int loss = 0; loss < kPitchLossFrames; ++loss) {
      int decoded_len = opus_custom_decode(dec, NULL, 0, pcm, kFrameSize);
      char label[96];
      snprintf(label, sizeof(label), "mono_pitch_loss_len_%d", loss);
      failures += check_eq_int(label, decoded_len, kFrameSize);
      lpc_hashes[loss + 2] = decoder_lpc_hash(dec);
      tail_hashes[loss + 2] = decoder_tail_hash(dec);
      pcm_hashes[loss] = hash_pcm_le(pcm, kFrameSize * kMonoChannels);
      snprintf(label, sizeof(label), "mono_pitch_loss_nonzero_%d", loss);
      failures += check_true(label, has_nonzero_pcm(pcm, kFrameSize * kMonoChannels));
      failures += assert_lpc_invariants("mono_pitch_loss", dec);
    }

    failures += check_true("mono_lpc_changes_on_first_pitch_loss",
                           lpc_hashes[2] != lpc_hashes[0]);
    failures += check_eq_u32("mono_lpc_stable_across_pitch_losses", lpc_hashes[3],
                             lpc_hashes[2]);
    failures += check_true("mono_tail_changes_between_pitch_losses",
                           tail_hashes[3] != tail_hashes[2]);

    failures += check_eq_int("mono_reset_state",
                             opus_custom_decoder_ctl(dec, OPUS_RESET_STATE),
                             OPUS_OK);
    failures += check_eq_int("mono_force_noise_start_band",
                             opus_custom_decoder_ctl(dec,
                                                     CELT_SET_START_BAND(
                                                         kStartBandNoiseOnly)),
                             OPUS_OK);
    failures += check_eq_int("mono_force_noise_loss_len",
                             opus_custom_decode(dec, NULL, 0, pcm, kFrameSize),
                             kFrameSize);
    failures += check_true("mono_force_noise_nonzero",
                           has_nonzero_pcm(pcm, kFrameSize * kMonoChannels));
    failures += check_eq_u32("mono_force_noise_lpc_hash", decoder_lpc_hash(dec),
                             mono_expected.noise_lpc_hash);
    failures += check_eq_u32("mono_force_noise_tail_hash", decoder_tail_hash(dec),
                             mono_expected.noise_tail_hash);

    if (mono_expected.lpc_hash[0] != 0) {
      failures += check_eq_u32("mono_pitch_loss_lpc_hash_0", lpc_hashes[2],
                               mono_expected.lpc_hash[0]);
      failures += check_eq_u32("mono_pitch_loss_lpc_hash_1", lpc_hashes[3],
                               mono_expected.lpc_hash[1]);
      failures += check_eq_u32("mono_pitch_loss_tail_hash_0", tail_hashes[2],
                               mono_expected.tail_hash[0]);
      failures += check_eq_u32("mono_pitch_loss_tail_hash_1", tail_hashes[3],
                               mono_expected.tail_hash[1]);
      failures += check_eq_u32("mono_pitch_loss_pcm_hash_0", pcm_hashes[0],
                               mono_expected.pcm_hash[0]);
      failures += check_eq_u32("mono_pitch_loss_pcm_hash_1", pcm_hashes[1],
                               mono_expected.pcm_hash[1]);
    }

    if (dump_expected) {
      dump_case("mono_pitch_loss", &lpc_hashes[2], &tail_hashes[2], pcm_hashes,
                kPitchLossFrames);
      printf("mono_force_noise_lpc_hash: 0x%08x\n", decoder_lpc_hash(dec));
      printf("mono_force_noise_tail_hash: 0x%08x\n", decoder_tail_hash(dec));
    }

    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
  }

  {
    unsigned char packet[kMaxPacketSize];
    opus_int16 pcm[kFrameSize * kStereoChannels];
    uint32_t lpc_hashes[kPitchLossFrames] = {0};
    uint32_t tail_hashes[kPitchLossFrames] = {0};
    uint32_t pcm_hashes[kPitchLossFrames] = {0};
    int primed_pitch = 0;

    OpusCustomEncoder *enc =
        opus_custom_encoder_create(mode, kStereoChannels, &err);
    OpusCustomDecoder *dec =
        opus_custom_decoder_create(mode, kStereoChannels, &err);
    if (enc == NULL || dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create stereo codec pair: %d\n", err);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc, 64000);
    failures += check_eq_int("stereo_decode_negative_len",
                             opus_custom_decode(dec, packet, -1, pcm, kFrameSize),
                             OPUS_INVALID_PACKET);
    failures += prime_decoder(enc, dec, pcm, packet, kStereoChannels, &primed_pitch);
    failures += check_true("stereo_primed_pitch_nonzero", primed_pitch > 0);

    for (int loss = 0; loss < kPitchLossFrames; ++loss) {
      int decoded_len = opus_custom_decode(dec, NULL, 0, pcm, kFrameSize);
      char label[96];
      snprintf(label, sizeof(label), "stereo_pitch_loss_len_%d", loss);
      failures += check_eq_int(label, decoded_len, kFrameSize);
      lpc_hashes[loss] = decoder_lpc_hash(dec);
      tail_hashes[loss] = decoder_tail_hash(dec);
      pcm_hashes[loss] = hash_pcm_le(pcm, kFrameSize * kStereoChannels);
      snprintf(label, sizeof(label), "stereo_pitch_loss_nonzero_%d", loss);
      failures += check_true(label,
                             has_nonzero_pcm(pcm, kFrameSize * kStereoChannels));
      failures += assert_lpc_invariants("stereo_pitch_loss", dec);
    }

    failures += check_eq_u32("stereo_lpc_stable_across_pitch_losses",
                             lpc_hashes[1], lpc_hashes[0]);
    failures += check_true("stereo_tail_changes_between_pitch_losses",
                           tail_hashes[1] != tail_hashes[0]);
    failures += check_true("stereo_pcm_changes_between_pitch_losses",
                           pcm_hashes[1] != pcm_hashes[0]);

    if (stereo_expected.lpc_hash[0] != 0) {
      failures += check_eq_u32("stereo_pitch_loss_lpc_hash_0", lpc_hashes[0],
                               stereo_expected.lpc_hash[0]);
      failures += check_eq_u32("stereo_pitch_loss_lpc_hash_1", lpc_hashes[1],
                               stereo_expected.lpc_hash[1]);
      failures += check_eq_u32("stereo_pitch_loss_tail_hash_0", tail_hashes[0],
                               stereo_expected.tail_hash[0]);
      failures += check_eq_u32("stereo_pitch_loss_tail_hash_1", tail_hashes[1],
                               stereo_expected.tail_hash[1]);
      failures += check_eq_u32("stereo_pitch_loss_pcm_hash_0", pcm_hashes[0],
                               stereo_expected.pcm_hash[0]);
      failures += check_eq_u32("stereo_pitch_loss_pcm_hash_1", pcm_hashes[1],
                               stereo_expected.pcm_hash[1]);
    }

    if (dump_expected) {
      dump_case("stereo_pitch_loss", lpc_hashes, tail_hashes, pcm_hashes,
                kPitchLossFrames);
    }

    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
  }

  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr,
          "celt_decoder_plc_iir_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
