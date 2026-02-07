#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "celt.h"
#include "opus_custom.h"
#include "opus_defines.h"

#ifdef FIXED_POINT

enum {
  kSampleRate = 48000,
  kFrameSize = 960,
  kChannels = 1,
  kWarmupStartBand = 17,
  kMaxPacketSize = 1276,
  kPrimeFrames = 16,
  kLossFrames = 12
};

typedef struct {
  uint32_t hashes[kLossFrames];
} ExpectedNoiseHashes;

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

static void fill_periodic_frame(opus_int16 *pcm, int frame_size, int phase) {
  const int period = 96;
  const int half = period / 2;
  const int amp = 14000;
  for (int i = 0; i < frame_size; ++i) {
    int pos = (i + phase) % period;
    int tri = pos < half ? pos : (period - pos);
    int centered = (tri * 2) - half;
    pcm[i] = (opus_int16)((centered * amp) / half);
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

static int count_hash_changes_range(const uint32_t *hashes, int begin, int end) {
  int changes = 0;
  for (int i = begin + 1; i < end; ++i) {
    if (hashes[i] != hashes[i - 1]) {
      changes++;
    }
  }
  return changes;
}

static int count_energy_drops_range(const int64_t *energies, int begin, int end) {
  int drops = 0;
  for (int i = begin + 1; i < end; ++i) {
    if (energies[i] < energies[i - 1]) {
      drops++;
    }
  }
  return drops;
}

static int configure_encoder(OpusCustomEncoder *enc) {
  int failures = 0;
  failures += check_eq_int("set_bitrate",
                           opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(14000)),
                           OPUS_OK);
  failures +=
      check_eq_int("set_vbr", opus_custom_encoder_ctl(enc, OPUS_SET_VBR(0)), OPUS_OK);
  failures += check_eq_int(
      "set_complexity", opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10)), OPUS_OK);
  failures += check_eq_int("set_lsb_depth",
                           opus_custom_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(16)),
                           OPUS_OK);
  return failures;
}

static int prime_decoder(OpusCustomEncoder *enc, OpusCustomDecoder *dec,
                         opus_int16 *pcm, unsigned char *packet) {
  int failures = 0;
  for (int f = 0; f < kPrimeFrames; ++f) {
    int packet_len;
    int decoded_len;
    fill_periodic_frame(pcm, kFrameSize, f * 7);
    packet_len = opus_custom_encode(enc, pcm, kFrameSize, packet, kMaxPacketSize);
    if (packet_len <= 0) {
      fprintf(stderr, "encode failed while priming on frame %d: %d\n", f, packet_len);
      return 1;
    }
    decoded_len = opus_custom_decode(dec, packet, packet_len, pcm, kFrameSize);
    failures += check_eq_int("prime_decode_len", decoded_len, kFrameSize);
  }
  return failures;
}

static int run_loss_sequence(OpusCustomDecoder *dec, opus_int16 *pcm,
                             uint32_t *hashes, int64_t *energies,
                             int *nonzero_flags) {
  int failures = 0;
  for (int i = 0; i < kLossFrames; ++i) {
    int decoded_len = opus_custom_decode(dec, NULL, 0, pcm, kFrameSize);
    failures += check_eq_int("loss_decode_len", decoded_len, kFrameSize);
    hashes[i] = fnv1a_pcm_le(pcm, kFrameSize);
    energies[i] = pcm_energy(pcm, kFrameSize);
    nonzero_flags[i] = has_nonzero(pcm, kFrameSize);
  }
  return failures;
}

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  const int dump_expected = getenv("CELT_DECODER_NOISE_RENORM_DUMP") != NULL &&
                            getenv("CELT_DECODER_NOISE_RENORM_DUMP")[0] != '\0' &&
                            getenv("CELT_DECODER_NOISE_RENORM_DUMP")[0] != '0';
  const ExpectedNoiseHashes expected = {
      {0xc5aab7e1, 0x506af494, 0xf6eb9f39, 0x45472a07, 0xf5be2afa, 0x54d339e5,
       0xe2ec6eca, 0xb4ec6a95, 0x85689942, 0xb03963ad, 0x231d1a1a, 0x9e3c6c58}};

  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  {
    unsigned char packet[kMaxPacketSize] = {0};
    opus_int16 pcm[kFrameSize * kChannels] = {0};
    uint32_t hashes[kLossFrames] = {0};
    int64_t energies[kLossFrames] = {0};
    int nonzero_flags[kLossFrames] = {0};

    OpusCustomEncoder *enc = opus_custom_encoder_create(mode, kChannels, &err);
    OpusCustomDecoder *dec = opus_custom_decoder_create(mode, kChannels, &err);
    if (enc == NULL || dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create encoder/decoder: %d\n", err);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc);
    failures += check_eq_int("decoder_set_complexity_10",
                             opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(10)),
                             OPUS_OK);

    /* Non-happy paths. */
    failures += check_eq_int("decode_null_pcm",
                             opus_custom_decode(dec, NULL, 0, NULL, kFrameSize),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decode_negative_len",
                             opus_custom_decode(dec, packet, -1, pcm, kFrameSize),
                             OPUS_INVALID_PACKET);
    failures += check_true("decode_bad_frame_size",
                           opus_custom_decode(dec, NULL, 0, pcm, kFrameSize - 1) < 0);

    failures += prime_decoder(enc, dec, pcm, packet);
    failures += check_eq_int("decoder_set_start_band_warmup",
                             opus_custom_decoder_ctl(dec, CELT_SET_START_BAND(kWarmupStartBand)),
                             OPUS_OK);
    failures += check_eq_int("warmup_loss_decode_len",
                             opus_custom_decode(dec, NULL, 0, pcm, kFrameSize), kFrameSize);
    failures += check_eq_int("decoder_set_start_band_zero",
                             opus_custom_decoder_ctl(dec, CELT_SET_START_BAND(0)), OPUS_OK);
    failures += run_loss_sequence(dec, pcm, hashes, energies, nonzero_flags);

    failures += check_true("loss_frame0_nonzero", nonzero_flags[0]);
    failures += check_true("loss_frame11_nonzero", nonzero_flags[kLossFrames - 1]);
    failures += check_true("loss_energy0_positive", energies[0] > 0);
    failures += check_true("loss_energy11_positive", energies[kLossFrames - 1] > 0);
    failures += check_lt_i64("loss_energy_prefix_decay", energies[4], energies[0]);
    failures += check_true("loss_energy_continues_decay", energies[5] < energies[4]);
    failures += check_true("loss_tail_has_drops",
                           count_energy_drops_range(energies, 6, kLossFrames) >= 2);
    failures += check_true("loss_tail_hash_changes",
                           count_hash_changes_range(hashes, 6, kLossFrames) >= 2);

    for (int i = 0; i < kLossFrames; ++i) {
      char label[64];
      snprintf(label, sizeof(label), "loss_hash_%d", i);
      failures += check_eq_u32(label, hashes[i], expected.hashes[i]);
    }

    failures += check_eq_int("decoder_reset",
                             opus_custom_decoder_ctl(dec, OPUS_RESET_STATE), OPUS_OK);
    failures += check_eq_int("decode_after_reset_len",
                             opus_custom_decode(dec, NULL, 0, pcm, kFrameSize),
                             kFrameSize);
    failures += check_true("decode_after_reset_nonzero", has_nonzero(pcm, kFrameSize));

    if (dump_expected) {
      printf("noise_hashes:");
      for (int i = 0; i < kLossFrames; ++i) {
        printf(" 0x%08x", hashes[i]);
      }
      printf("\nnoise_energies:");
      for (int i = 0; i < kLossFrames; ++i) {
        printf(" %" PRId64, energies[i]);
      }
      printf("\n");
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
          "celt_decoder_noise_renorm_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
