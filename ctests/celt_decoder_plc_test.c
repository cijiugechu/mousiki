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
  kChannels = 1,
  kMaxPacketSize = 1276,
  kPrimeFrames = 16,
  kLossFrames = 6
};

typedef struct {
  uint32_t plc_hashes[kLossFrames];
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

static int prime_decoder_with_postfilter(OpusCustomEncoder *enc,
                                         OpusCustomDecoder *dec,
                                         opus_int16 *work_pcm,
                                         unsigned char *packet,
                                         int *primed_pitch) {
  int failures = 0;
  *primed_pitch = 0;
  for (int f = 0; f < kPrimeFrames; ++f) {
    int packet_len;
    int decoded_len;
    fill_periodic_frame(work_pcm, kFrameSize, f * 7);
    packet_len =
        opus_custom_encode(enc, work_pcm, kFrameSize, packet, kMaxPacketSize);
    if (packet_len <= 0) {
      fprintf(stderr, "encode failed while priming on frame %d: %d\n", f,
              packet_len);
      return 1;
    }
    decoded_len = opus_custom_decode(dec, packet, packet_len, work_pcm, kFrameSize);
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
                             uint32_t *hashes, int64_t *energies,
                             int *nonzero_first) {
  int failures = 0;
  for (int i = 0; i < kLossFrames; ++i) {
    int decoded_len = opus_custom_decode(dec, NULL, 0, pcm_out, kFrameSize);
    failures += check_eq_int("loss_decode_len", decoded_len, kFrameSize);
    hashes[i] = fnv1a_pcm_le(pcm_out, kFrameSize * kChannels);
    energies[i] = pcm_energy(pcm_out, kFrameSize * kChannels);
    if (i == 0) {
      *nonzero_first = has_nonzero(pcm_out, kFrameSize * kChannels);
    }
  }
  return failures;
}

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  int pitch = 0;
  int prime_pitch = 0;
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

  OpusCustomEncoder *enc = opus_custom_encoder_create(mode, kChannels, &err);
  if (enc == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create encoder: %d\n", err);
    opus_custom_mode_destroy(mode);
    return 1;
  }

  OpusCustomDecoder *dec = opus_custom_decoder_create(mode, kChannels, &err);
  if (dec == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create decoder: %d\n", err);
    opus_custom_encoder_destroy(enc);
    opus_custom_mode_destroy(mode);
    return 1;
  }

  failures += check_eq_int("set_bitrate",
                           opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(14000)),
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

  unsigned char packet[kMaxPacketSize];
  opus_int16 pcm[kFrameSize * kChannels];
  uint32_t plc_hashes[kLossFrames] = {0};
  int64_t plc_energies[kLossFrames] = {0};
  uint32_t reset_plc_hash = 0;
  int loss_nonzero_first = 0;

  /* Non-happy paths: NULL output pointer and negative packet length. */
  failures += check_eq_int("decode_null_pcm",
                           opus_custom_decode(dec, NULL, 0, NULL, kFrameSize),
                           OPUS_BAD_ARG);
  failures += check_eq_int("decode_negative_len",
                           opus_custom_decode(dec, packet, -1, pcm, kFrameSize),
                           OPUS_INVALID_PACKET);

  failures += prime_decoder_with_postfilter(enc, dec, pcm, packet, &prime_pitch);
  failures += check_true("primed_pitch_nonzero", prime_pitch > 0);
  failures += check_eq_int(
      "primed_get_pitch_ctl",
      opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(&pitch)),
      OPUS_OK);
  failures += check_true("pitch_after_prime_nonzero", pitch > 0);

  failures += run_loss_sequence(dec, pcm, plc_hashes, plc_energies,
                                &loss_nonzero_first);
  failures += check_true("loss_frame0_nonzero", loss_nonzero_first);
  failures += check_true("loss_energy0_positive", plc_energies[0] > 0);
  failures += check_true("loss_energy5_positive", plc_energies[kLossFrames - 1] > 0);
  failures += check_lt_i64("loss_energy_decay_prefix", plc_energies[4],
                           plc_energies[0]);
  failures += check_true("loss_energy_noise_rebound",
                         plc_energies[5] > plc_energies[4]);

  if (expected.plc_hashes[0] != 0) {
    for (int i = 0; i < kLossFrames; ++i) {
      char label[64];
      snprintf(label, sizeof(label), "plc_hash_%d", i);
      failures += check_eq_u32(label, plc_hashes[i], expected.plc_hashes[i]);
    }
  }

  /* Reset and run PLC once without priming to cover cold-start branch. */
  failures += check_eq_int("decoder_reset",
                           opus_custom_decoder_ctl(dec, OPUS_RESET_STATE),
                           OPUS_OK);
  failures += check_eq_int("decode_after_reset_len",
                           opus_custom_decode(dec, NULL, 0, pcm, kFrameSize),
                           kFrameSize);
  reset_plc_hash = fnv1a_pcm_le(pcm, kFrameSize * kChannels);

  /* Non-happy path: invalid tiny packet should fail. */
  packet[0] = 0;
  failures += check_true("decode_tiny_packet_fails",
                         opus_custom_decode(dec, packet, 1, pcm, kFrameSize) < 0);

  if (expected.reset_plc_hash != 0) {
    failures += check_eq_u32("reset_plc_hash", reset_plc_hash,
                             expected.reset_plc_hash);
  }

  if (dump_expected) {
    printf("plc_hashes:");
    for (int i = 0; i < kLossFrames; ++i) {
      printf(" 0x%08x", plc_hashes[i]);
    }
    printf("\nreset_plc_hash: 0x%08x\n", reset_plc_hash);
    printf("plc_energies:");
    for (int i = 0; i < kLossFrames; ++i) {
      printf(" %" PRId64, plc_energies[i]);
    }
    printf("\n");
  }

  opus_custom_decoder_destroy(dec);
  opus_custom_encoder_destroy(enc);
  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr, "celt_decoder_plc_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
