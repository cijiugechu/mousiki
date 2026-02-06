#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "opus_custom.h"
#include "opus_defines.h"

#ifdef FIXED_POINT

enum {
  kSampleRate = 48000,
  kFrameSize = 960,
  kMonoChannels = 1,
  kStereoChannels = 2,
  kMaxPacketSize = 1276,
  kPrimeFrames = 16,
  kMonoLossFrames = 6,
  kStereoLossFrames = 48
};

typedef struct {
  uint32_t plc_hashes[kMonoLossFrames];
  uint32_t reset_plc_hash;
} ExpectedHashes;

static uint32_t fnv1a_update(uint32_t hash, unsigned char byte) {
  hash ^= (uint32_t)byte;
  return hash * 16777619u;
}

static uint32_t fnv1a_pcm_le(const opus_int16 *pcm, int count) {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < count; ++i) {
    unsigned char b0 = (unsigned char)(pcm[i] & 0xFF);
    unsigned char b1 = (unsigned char)((pcm[i] >> 8) & 0xFF);
    hash = fnv1a_update(hash, b0);
    hash = fnv1a_update(hash, b1);
  }
  return hash;
}

static int64_t pcm_energy(const opus_int16 *pcm, int count) {
  int64_t energy = 0;
  for (int i = 0; i < count; ++i) {
    int32_t sample = pcm[i];
    energy += (int64_t)sample * (int64_t)sample;
  }
  return energy;
}

static int has_nonzero(const opus_int16 *pcm, int count) {
  for (int i = 0; i < count; ++i) {
    if (pcm[i] != 0) {
      return 1;
    }
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

static int check_lt_i64(const char *label, int64_t a, int64_t b) {
  if (!(a < b)) {
    fprintf(stderr, "%s failed: %" PRId64 " !< %" PRId64 "\n", label, a, b);
    return 1;
  }
  return 0;
}

static int count_hash_changes(const uint32_t *hashes, int count) {
  int changes = 0;
  for (int i = 1; i < count; ++i) {
    if (hashes[i] != hashes[i - 1]) {
      changes++;
    }
  }
  return changes;
}

static void min_max_i64(const int64_t *values, int count, int64_t *min_out,
                        int64_t *max_out) {
  int64_t min_v = values[0];
  int64_t max_v = values[0];
  for (int i = 1; i < count; ++i) {
    if (values[i] < min_v) min_v = values[i];
    if (values[i] > max_v) max_v = values[i];
  }
  *min_out = min_v;
  *max_out = max_v;
}

static int count_energy_rises(const int64_t *energies, int count) {
  int rises = 0;
  for (int i = 1; i < count; ++i) {
    if (energies[i] > energies[i - 1]) rises++;
  }
  return rises;
}

static int count_energy_drops(const int64_t *energies, int count) {
  int drops = 0;
  for (int i = 1; i < count; ++i) {
    if (energies[i] < energies[i - 1]) drops++;
  }
  return drops;
}

static int prime_decoder_with_postfilter(OpusCustomEncoder *enc,
                                         OpusCustomDecoder *dec,
                                         opus_int16 *work_pcm,
                                         unsigned char *packet,
                                         int channels,
                                         int *primed_pitch) {
  int failures = 0;
  *primed_pitch = 0;
  for (int f = 0; f < kPrimeFrames; ++f) {
    int packet_len;
    int decoded_len;
    fill_periodic_frame(work_pcm, kFrameSize, channels, f * 7);
    packet_len =
        opus_custom_encode(enc, work_pcm, kFrameSize, packet, kMaxPacketSize);
    if (packet_len <= 0) {
      fprintf(stderr, "encode failed while priming on frame %d: %d\n", f,
              packet_len);
      return 1;
    }
    decoded_len =
        opus_custom_decode(dec, packet, packet_len, work_pcm, kFrameSize);
    failures += check_eq_int("prime_decode_len", decoded_len, kFrameSize);
    failures += check_eq_int(
        "prime_get_pitch_ctl",
        opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(primed_pitch)),
        OPUS_OK);
    if (*primed_pitch > 0) {
      return failures;
    }
  }
  fprintf(stderr, "failed to prime decoder postfilter pitch after %d frames\n",
          kPrimeFrames);
  return failures + 1;
}

static int run_loss_sequence(OpusCustomDecoder *dec, opus_int16 *pcm_out,
                             int channels, int loss_frames, uint32_t *hashes,
                             int64_t *energies,
                             int *nonzero_first) {
  int failures = 0;
  for (int i = 0; i < loss_frames; ++i) {
    int decoded_len = opus_custom_decode(dec, NULL, 0, pcm_out, kFrameSize);
    failures += check_eq_int("loss_decode_len", decoded_len, kFrameSize);
    hashes[i] = fnv1a_pcm_le(pcm_out, kFrameSize * channels);
    energies[i] = pcm_energy(pcm_out, kFrameSize * channels);
    if (i == 0) {
      *nonzero_first = has_nonzero(pcm_out, kFrameSize * channels);
    }
  }
  return failures;
}

static int configure_encoder(OpusCustomEncoder *enc, int bitrate) {
  int failures = 0;
  failures += check_eq_int("set_bitrate",
                           opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate)),
                           OPUS_OK);
  failures += check_eq_int("set_vbr",
                           opus_custom_encoder_ctl(enc, OPUS_SET_VBR(0)),
                           OPUS_OK);
  failures += check_eq_int(
      "set_complexity", opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10)),
      OPUS_OK);
  failures += check_eq_int("set_lsb_depth",
                           opus_custom_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(16)),
                           OPUS_OK);
  return failures;
}

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  const int dump_expected = getenv("CELT_DECODER_PLC_DUMP") != NULL &&
                            getenv("CELT_DECODER_PLC_DUMP")[0] != '\0' &&
                            getenv("CELT_DECODER_PLC_DUMP")[0] != '0';
  const ExpectedHashes expected = {
      {0xc1b6b08d, 0xc2a465ed, 0xfc0d6b40,
       0x705d118b, 0x8bb2fba2, 0x9cf92c14},
      0x9811584d,
  };

  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  {
    int pitch = 0;
    int prime_pitch = 0;
    unsigned char packet[kMaxPacketSize];
    opus_int16 pcm[kFrameSize * kMonoChannels];
    uint32_t plc_hashes[kMonoLossFrames] = {0};
    int64_t plc_energies[kMonoLossFrames] = {0};
    uint32_t reset_plc_hash = 0;
    int loss_nonzero_first = 0;

    OpusCustomEncoder *enc =
        opus_custom_encoder_create(mode, kMonoChannels, &err);
    if (enc == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create mono encoder: %d\n", err);
      opus_custom_mode_destroy(mode);
      return 1;
    }
    OpusCustomDecoder *dec =
        opus_custom_decoder_create(mode, kMonoChannels, &err);
    if (dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create mono decoder: %d\n", err);
      opus_custom_encoder_destroy(enc);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc, 14000);

    /* Non-happy paths. */
    failures += check_eq_int("mono_decode_null_pcm",
                             opus_custom_decode(dec, NULL, 0, NULL, kFrameSize),
                             OPUS_BAD_ARG);
    failures += check_eq_int("mono_decode_negative_len",
                             opus_custom_decode(dec, packet, -1, pcm, kFrameSize),
                             OPUS_INVALID_PACKET);
    failures += check_true(
        "mono_decode_bad_frame_size",
        opus_custom_decode(dec, NULL, 0, pcm, kFrameSize - 1) < 0);
    failures += check_eq_int(
        "mono_decoder_set_complexity_bad",
        opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(11)),
        OPUS_BAD_ARG);

    failures += prime_decoder_with_postfilter(enc, dec, pcm, packet, kMonoChannels,
                                              &prime_pitch);
    failures += check_true("mono_primed_pitch_nonzero", prime_pitch > 0);
    failures += check_eq_int(
        "mono_primed_get_pitch_ctl",
        opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(&pitch)),
        OPUS_OK);
    failures += check_true("mono_pitch_after_prime_nonzero", pitch > 0);

    failures += run_loss_sequence(dec, pcm, kMonoChannels, kMonoLossFrames,
                                  plc_hashes, plc_energies, &loss_nonzero_first);
    failures += check_true("mono_loss_frame0_nonzero", loss_nonzero_first);
    failures += check_true("mono_loss_energy0_positive", plc_energies[0] > 0);
    failures +=
        check_true("mono_loss_energy5_positive", plc_energies[kMonoLossFrames - 1] > 0);
    failures += check_lt_i64("mono_loss_energy_decay_prefix", plc_energies[4],
                             plc_energies[0]);
    failures += check_true("mono_loss_energy_noise_rebound",
                           plc_energies[5] > plc_energies[4]);

    if (expected.plc_hashes[0] != 0) {
      for (int i = 0; i < kMonoLossFrames; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "mono_plc_hash_%d", i);
        failures += check_eq_u32(label, plc_hashes[i], expected.plc_hashes[i]);
      }
    }

    failures += check_eq_int("mono_decoder_reset",
                             opus_custom_decoder_ctl(dec, OPUS_RESET_STATE),
                             OPUS_OK);
    failures += check_eq_int("mono_decode_after_reset_len",
                             opus_custom_decode(dec, NULL, 0, pcm, kFrameSize),
                             kFrameSize);
    reset_plc_hash = fnv1a_pcm_le(pcm, kFrameSize * kMonoChannels);
    packet[0] = 0;
    failures += check_true(
        "mono_decode_tiny_packet_fails",
        opus_custom_decode(dec, packet, 1, pcm, kFrameSize) < 0);

    if (expected.reset_plc_hash != 0) {
      failures += check_eq_u32("mono_reset_plc_hash", reset_plc_hash,
                               expected.reset_plc_hash);
    }

    if (dump_expected) {
      printf("mono_plc_hashes:");
      for (int i = 0; i < kMonoLossFrames; ++i) {
        printf(" 0x%08x", plc_hashes[i]);
      }
      printf("\nmono_reset_plc_hash: 0x%08x\n", reset_plc_hash);
      printf("mono_plc_energies:");
      for (int i = 0; i < kMonoLossFrames; ++i) {
        printf(" %" PRId64, plc_energies[i]);
      }
      printf("\n");
    }

    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
  }

  {
    int pitch = 0;
    int prime_pitch = 0;
    int64_t min_energy;
    int64_t max_energy;
    int rises;
    int drops;
    int hash_changes;
    const int64_t max_possible_energy =
        (int64_t)kStereoChannels * kFrameSize * 32767 * 32767;
    unsigned char packet[kMaxPacketSize];
    opus_int16 pcm[kFrameSize * kStereoChannels];
    uint32_t plc_hashes[kStereoLossFrames] = {0};
    int64_t plc_energies[kStereoLossFrames] = {0};
    int loss_nonzero_first = 0;

    OpusCustomEncoder *enc =
        opus_custom_encoder_create(mode, kStereoChannels, &err);
    if (enc == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create stereo encoder: %d\n", err);
      opus_custom_mode_destroy(mode);
      return 1;
    }
    OpusCustomDecoder *dec =
        opus_custom_decoder_create(mode, kStereoChannels, &err);
    if (dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create stereo decoder: %d\n", err);
      opus_custom_encoder_destroy(enc);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc, 64000);

    /* Non-happy paths. */
    failures += check_eq_int(
        "stereo_decoder_set_complexity_bad",
        opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(123)),
        OPUS_BAD_ARG);
    failures += check_eq_int("stereo_decode_negative_len",
                             opus_custom_decode(dec, packet, -1, pcm, kFrameSize),
                             OPUS_INVALID_PACKET);
    failures +=
        check_true("stereo_decode_bad_frame_size",
                   opus_custom_decode(dec, NULL, 0, pcm, kFrameSize + 120) < 0);

    failures += prime_decoder_with_postfilter(enc, dec, pcm, packet, kStereoChannels,
                                              &prime_pitch);
    failures += check_true("stereo_primed_pitch_nonzero", prime_pitch > 0);
    failures += check_eq_int(
        "stereo_get_pitch_ctl",
        opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(&pitch)),
        OPUS_OK);
    failures += check_true("stereo_pitch_in_range", pitch > 0 && pitch <= 720);

    failures += run_loss_sequence(dec, pcm, kStereoChannels, kStereoLossFrames,
                                  plc_hashes, plc_energies, &loss_nonzero_first);
    failures += check_true("stereo_loss_frame0_nonzero", loss_nonzero_first);
    failures += check_true(
        "stereo_loss_last_nonzero",
        has_nonzero(pcm, kFrameSize * kStereoChannels));

    min_max_i64(plc_energies, kStereoLossFrames, &min_energy, &max_energy);
    rises = count_energy_rises(plc_energies, kStereoLossFrames);
    drops = count_energy_drops(plc_energies, kStereoLossFrames);
    hash_changes = count_hash_changes(plc_hashes, kStereoLossFrames);

    failures += check_true("stereo_energy_min_positive", min_energy > 0);
    failures += check_true("stereo_energy_max_bounded",
                           max_energy < max_possible_energy);
    failures += check_true("stereo_energy_has_rises", rises > 0);
    failures += check_true("stereo_energy_has_drops", drops > 0);
    failures += check_true("stereo_hash_changes_many",
                           hash_changes > (kStereoLossFrames / 3));

    failures += check_eq_int("stereo_decoder_reset",
                             opus_custom_decoder_ctl(dec, OPUS_RESET_STATE),
                             OPUS_OK);
    failures += check_eq_int("stereo_decode_after_reset_len",
                             opus_custom_decode(dec, NULL, 0, pcm, kFrameSize),
                             kFrameSize);
    failures += check_true(
        "stereo_decode_after_reset_nonzero",
        has_nonzero(pcm, kFrameSize * kStereoChannels));

    packet[0] = 0;
    failures += check_true(
        "stereo_decode_tiny_packet_fails",
        opus_custom_decode(dec, packet, 1, pcm, kFrameSize) < 0);

    if (dump_expected) {
      printf("stereo_plc_hashes:");
      for (int i = 0; i < kStereoLossFrames; ++i) {
        if (i < 10 || i >= kStereoLossFrames - 4) {
          printf(" [%d]=0x%08x", i, plc_hashes[i]);
        }
      }
      printf("\nstereo_energy_stats: min=%" PRId64 " max=%" PRId64
             " rises=%d drops=%d hash_changes=%d\n",
             min_energy, max_energy, rises, drops, hash_changes);
    }

    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
  }

  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr, "celt_decoder_plc_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
