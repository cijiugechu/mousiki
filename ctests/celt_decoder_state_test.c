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
  kMaxPacketSize = 1276,
  kWarmFrames = 8,
  kLossFrames = 6
};

typedef struct {
  uint32_t warm_packet_hashes[kWarmFrames];
  uint32_t warm_pcm_hashes[kWarmFrames];
  uint32_t loss_hashes[kLossFrames];
  uint32_t recover_hash_after_loss;
  uint32_t recover_hash_without_loss;
  uint32_t reset_loss_hash;
} ExpectedStateHashes;

static uint32_t fnv1a_update(uint32_t hash, unsigned char byte) {
  hash ^= (uint32_t)byte;
  return hash * 16777619u;
}

static uint32_t fnv1a_bytes(const unsigned char *data, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; ++i) {
    hash = fnv1a_update(hash, data[i]);
  }
  return hash;
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

static int count_hash_changes(const uint32_t *hashes, int count) {
  int changes = 0;
  for (int i = 1; i < count; ++i) {
    if (hashes[i] != hashes[i - 1]) {
      changes++;
    }
  }
  return changes;
}

static int count_energy_drops(const int64_t *energies, int count) {
  int drops = 0;
  for (int i = 1; i < count; ++i) {
    if (energies[i] < energies[i - 1]) {
      drops++;
    }
  }
  return drops;
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

static void fill_frame(opus_int16 *pcm, int frame_size, int phase) {
  const int period = 96;
  const int half = period / 2;
  const int amp = 13000;
  for (int i = 0; i < frame_size; ++i) {
    int pos = (i + phase) % period;
    int tri = pos < half ? pos : (period - pos);
    int centered = (tri * 2) - half;
    int shaped = (centered * amp) / half;
    shaped += ((i % 9) == 0 ? 1200 : -300);
    pcm[i] = (opus_int16)shaped;
  }
}

static int configure_encoder(OpusCustomEncoder *enc) {
  int failures = 0;
  failures +=
      check_eq_int("set_bitrate", opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(18000)),
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

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  const int dump_expected = getenv("CELT_DECODER_STATE_DUMP") != NULL &&
                            getenv("CELT_DECODER_STATE_DUMP")[0] != '\0' &&
                            getenv("CELT_DECODER_STATE_DUMP")[0] != '0';
  const ExpectedStateHashes expected = {
      {0x12040bb9, 0xfa16704e, 0xc40f7772, 0x7c726ef3,
       0x3ec01bf0, 0x108e4248, 0x0341cfd7, 0x841987b6},
      {0x702593e3, 0xe891961f, 0xb87640ee, 0xd9abf25c,
       0x0d59b40f, 0xa45323c0, 0x387eb7e3, 0xde85f2a0},
      {0x88e915a8, 0xca42f30e, 0x657caa87, 0x746d415b, 0x2e3e688f, 0xdd030e3b},
      0x948ff204,
      0x74561161,
      0x9811584d,
  };

  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  {
    OpusCustomEncoder *enc = opus_custom_encoder_create(mode, kChannels, &err);
    OpusCustomDecoder *dec_a = opus_custom_decoder_create(mode, kChannels, &err);
    OpusCustomDecoder *dec_b = opus_custom_decoder_create(mode, kChannels, &err);
    unsigned char packet[kMaxPacketSize] = {0};
    opus_int16 pcm[kFrameSize * kChannels] = {0};
    opus_int16 out_a[kFrameSize * kChannels] = {0};
    opus_int16 out_b[kFrameSize * kChannels] = {0};
    uint32_t warm_packet_hashes[kWarmFrames] = {0};
    uint32_t warm_pcm_hashes[kWarmFrames] = {0};
    uint32_t loss_hashes[kLossFrames] = {0};
    int64_t loss_energies[kLossFrames] = {0};
    uint32_t recover_hash_after_loss = 0;
    uint32_t recover_hash_without_loss = 0;
    uint32_t reset_loss_hash_a = 0;
    uint32_t reset_loss_hash_b = 0;

    if (enc == NULL || dec_a == NULL || dec_b == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create encoder/decoders: %d\n", err);
      opus_custom_encoder_destroy(enc);
      opus_custom_decoder_destroy(dec_a);
      opus_custom_decoder_destroy(dec_b);
      opus_custom_mode_destroy(mode);
      return 1;
    }

    failures += configure_encoder(enc);
    failures +=
        check_eq_int("decoder_a_set_complexity_10",
                     opus_custom_decoder_ctl(dec_a, OPUS_SET_COMPLEXITY(10)), OPUS_OK);
    failures +=
        check_eq_int("decoder_b_set_complexity_10",
                     opus_custom_decoder_ctl(dec_b, OPUS_SET_COMPLEXITY(10)), OPUS_OK);

    /* Non-happy paths. */
    failures += check_eq_int("decode_null_pcm",
                             opus_custom_decode(dec_a, NULL, 0, NULL, kFrameSize),
                             OPUS_BAD_ARG);
    failures +=
        check_eq_int("decode_negative_len",
                     opus_custom_decode(dec_a, packet, -1, out_a, kFrameSize),
                     OPUS_INVALID_PACKET);
    failures += check_true("decode_bad_frame_size",
                           opus_custom_decode(dec_a, NULL, 0, out_a, kFrameSize - 1) <
                               0);
    failures += check_eq_int("decoder_set_complexity_bad",
                             opus_custom_decoder_ctl(dec_a, OPUS_SET_COMPLEXITY(11)),
                             OPUS_BAD_ARG);

    /* Warm both decoders with identical packets and verify state parity. */
    for (int f = 0; f < kWarmFrames; ++f) {
      int packet_len;
      int decode_a;
      int decode_b;
      uint32_t packet_hash;
      uint32_t hash_a;
      uint32_t hash_b;

      fill_frame(pcm, kFrameSize, f * 7);
      packet_len = opus_custom_encode(enc, pcm, kFrameSize, packet, kMaxPacketSize);
      if (packet_len <= 0) {
        fprintf(stderr, "encode failed in warm frame %d: %d\n", f, packet_len);
        failures++;
        continue;
      }

      decode_a = opus_custom_decode(dec_a, packet, packet_len, out_a, kFrameSize);
      decode_b = opus_custom_decode(dec_b, packet, packet_len, out_b, kFrameSize);
      failures += check_eq_int("warm_decode_len_a", decode_a, kFrameSize);
      failures += check_eq_int("warm_decode_len_b", decode_b, kFrameSize);

      packet_hash = fnv1a_bytes(packet, (size_t)packet_len);
      hash_a = fnv1a_pcm_le(out_a, kFrameSize * kChannels);
      hash_b = fnv1a_pcm_le(out_b, kFrameSize * kChannels);
      warm_packet_hashes[f] = packet_hash;
      warm_pcm_hashes[f] = hash_a;

      failures += check_eq_u32("warm_decoders_match", hash_a, hash_b);

      if (expected.warm_packet_hashes[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "warm_packet_hash_%d", f);
        failures += check_eq_u32(label, packet_hash, expected.warm_packet_hashes[f]);
      }
      if (expected.warm_pcm_hashes[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "warm_pcm_hash_%d", f);
        failures += check_eq_u32(label, hash_a, expected.warm_pcm_hashes[f]);
      }
    }

    /* Drive packet-loss path only on decoder A to intentionally diverge state. */
    for (int i = 0; i < kLossFrames; ++i) {
      int decoded_len = opus_custom_decode(dec_a, NULL, 0, out_a, kFrameSize);
      failures += check_eq_int("loss_decode_len", decoded_len, kFrameSize);
      loss_hashes[i] = fnv1a_pcm_le(out_a, kFrameSize * kChannels);
      loss_energies[i] = pcm_energy(out_a, kFrameSize * kChannels);
      if (expected.loss_hashes[i] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "loss_hash_%d", i);
        failures += check_eq_u32(label, loss_hashes[i], expected.loss_hashes[i]);
      }
    }
    failures += check_true("loss_first_nonzero", has_nonzero(out_a, kFrameSize));
    failures +=
        check_true("loss_hash_changes", count_hash_changes(loss_hashes, kLossFrames) > 0);
    failures += check_true("loss_energy_drops",
                           count_energy_drops(loss_energies, kLossFrames) > 0);

    /* Recovery frame: decoders should now differ due to different internal history. */
    fill_frame(pcm, kFrameSize, 93);
    {
      int packet_len =
          opus_custom_encode(enc, pcm, kFrameSize, packet, kMaxPacketSize);
      if (packet_len <= 0) {
        fprintf(stderr, "encode failed in recovery frame: %d\n", packet_len);
        failures++;
      } else {
        failures += check_eq_int(
            "recover_decode_len_a",
            opus_custom_decode(dec_a, packet, packet_len, out_a, kFrameSize),
            kFrameSize);
        failures += check_eq_int(
            "recover_decode_len_b",
            opus_custom_decode(dec_b, packet, packet_len, out_b, kFrameSize),
            kFrameSize);
        recover_hash_after_loss = fnv1a_pcm_le(out_a, kFrameSize * kChannels);
        recover_hash_without_loss = fnv1a_pcm_le(out_b, kFrameSize * kChannels);
        failures += check_true("recover_hashes_differ_after_state_divergence",
                               recover_hash_after_loss != recover_hash_without_loss);
        if (expected.recover_hash_after_loss != 0) {
          failures += check_eq_u32("recover_hash_after_loss", recover_hash_after_loss,
                                   expected.recover_hash_after_loss);
        }
        if (expected.recover_hash_without_loss != 0) {
          failures +=
              check_eq_u32("recover_hash_without_loss", recover_hash_without_loss,
                           expected.recover_hash_without_loss);
        }
      }
    }

    /* Reset should re-synchronise core decode state. */
    failures += check_eq_int("decoder_a_reset",
                             opus_custom_decoder_ctl(dec_a, OPUS_RESET_STATE), OPUS_OK);
    failures += check_eq_int("decoder_b_reset",
                             opus_custom_decoder_ctl(dec_b, OPUS_RESET_STATE), OPUS_OK);
    failures += check_eq_int(
        "post_reset_loss_len_a", opus_custom_decode(dec_a, NULL, 0, out_a, kFrameSize),
        kFrameSize);
    failures += check_eq_int(
        "post_reset_loss_len_b", opus_custom_decode(dec_b, NULL, 0, out_b, kFrameSize),
        kFrameSize);
    reset_loss_hash_a = fnv1a_pcm_le(out_a, kFrameSize * kChannels);
    reset_loss_hash_b = fnv1a_pcm_le(out_b, kFrameSize * kChannels);
    failures += check_eq_u32("reset_loss_hash_match", reset_loss_hash_a,
                             reset_loss_hash_b);
    if (expected.reset_loss_hash != 0) {
      failures += check_eq_u32("reset_loss_hash", reset_loss_hash_a,
                               expected.reset_loss_hash);
    }

    packet[0] = 0;
    failures +=
        check_true("decode_tiny_packet_fails",
                   opus_custom_decode(dec_a, packet, 1, out_a, kFrameSize) < 0);

    if (dump_expected) {
      printf("warm_packet_hashes:");
      for (int i = 0; i < kWarmFrames; ++i) {
        printf(" 0x%08x", warm_packet_hashes[i]);
      }
      printf("\nwarm_pcm_hashes:");
      for (int i = 0; i < kWarmFrames; ++i) {
        printf(" 0x%08x", warm_pcm_hashes[i]);
      }
      printf("\nloss_hashes:");
      for (int i = 0; i < kLossFrames; ++i) {
        printf(" 0x%08x", loss_hashes[i]);
      }
      printf("\nrecover_hash_after_loss: 0x%08x\n", recover_hash_after_loss);
      printf("recover_hash_without_loss: 0x%08x\n", recover_hash_without_loss);
      printf("reset_loss_hash: 0x%08x\n", reset_loss_hash_a);
    }

    opus_custom_decoder_destroy(dec_b);
    opus_custom_decoder_destroy(dec_a);
    opus_custom_encoder_destroy(enc);
  }

  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr, "celt_decoder_state_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
