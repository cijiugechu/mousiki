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
  kChannels = 2,
  kMaxPacketSize = 1276,
  kFrames = 13,
  kNbEBands = 21
};

typedef struct {
  uint32_t packet_hashes[kFrames];
  uint32_t pcm_hashes[kFrames];
  uint32_t final_ranges[kFrames];
} ExpectedDataflowHashes;

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

static void fill_stereo_frame(opus_int16 *pcm, int frame_size, int phase) {
  const int period = 120;
  const int half = period / 2;
  const int amp_l = 15000;
  const int amp_r = 11200;
  for (int i = 0; i < frame_size; ++i) {
    int pos_l = (i + phase) % period;
    int pos_r = (i + phase + 29) % period;
    int tri_l = pos_l < half ? pos_l : (period - pos_l);
    int tri_r = pos_r < half ? pos_r : (period - pos_r);
    int centered_l = (tri_l * 2) - half;
    int centered_r = (tri_r * 2) - half;
    int shaped_l = (centered_l * amp_l) / half;
    int shaped_r = -((centered_r * amp_r) / half);
    shaped_l += ((i % 11) == 0 ? 900 : -350);
    shaped_r += ((i % 7) == 0 ? -700 : 220);
    pcm[2 * i] = (opus_int16)shaped_l;
    pcm[2 * i + 1] = (opus_int16)shaped_r;
  }
}

static int configure_encoder(OpusCustomEncoder *enc) {
  int failures = 0;
  failures += check_eq_int("set_bitrate",
                           opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(64000)),
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

static int set_decoder_layout(OpusCustomDecoder *dec, int start_band, int end_band,
                              int stream_channels) {
  int failures = 0;
  failures += check_eq_int("set_start_band",
                           opus_custom_decoder_ctl(dec, CELT_SET_START_BAND(start_band)),
                           OPUS_OK);
  failures += check_eq_int("set_end_band",
                           opus_custom_decoder_ctl(dec, CELT_SET_END_BAND(end_band)),
                           OPUS_OK);
  failures += check_eq_int("set_stream_channels",
                           opus_custom_decoder_ctl(dec, CELT_SET_CHANNELS(stream_channels)),
                           OPUS_OK);
  return failures;
}

int main(void) {
  int failures = 0;
  int err = OPUS_OK;
  const int dump_expected = getenv("CELT_DECODER_DATAFLOW_DUMP") != NULL &&
                            getenv("CELT_DECODER_DATAFLOW_DUMP")[0] != '\0' &&
                            getenv("CELT_DECODER_DATAFLOW_DUMP")[0] != '0';
  const ExpectedDataflowHashes expected = {
      {0x1e1e1269, 0x2dde7528, 0x8e000bce, 0x679092e8, 0x143e3a1e,
       0x69864649, 0xe63d689d, 0x9af6d244, 0x6c7acb7a, 0x6c5bcdd9,
       0xbeadc667, 0x3e08723e, 0x834e54b0},
      {0x72d2ff52, 0xfdbe8b96, 0xf4422f17, 0x15d7fb47, 0x4e10ba6e,
       0xbb8337f9, 0x0942b6c6, 0x8bee73ad, 0xdd70c41d, 0x630c948a,
       0xf86a2a05, 0xf047b0c5, 0x5502c350},
      {0x212ddc00, 0x26b1e200, 0x0392e900, 0x10732300, 0x63285f00,
       0x19466300, 0x0b72b000, 0x04626100, 0x69b16100, 0x1bd0aa00,
       0x067ac700, 0x08315200, 0x3611a800},
  };

  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  {
    OpusCustomEncoder *enc = opus_custom_encoder_create(mode, kChannels, &err);
    OpusCustomDecoder *dec = opus_custom_decoder_create(mode, kChannels, &err);
    unsigned char packet[kMaxPacketSize] = {0};
    opus_int16 pcm[kFrameSize * kChannels] = {0};
    opus_int16 decoded[kFrameSize * kChannels] = {0};
    uint32_t packet_hashes[kFrames] = {0};
    uint32_t pcm_hashes[kFrames] = {0};
    uint32_t final_ranges[kFrames] = {0};

    if (enc == NULL || dec == NULL || err != OPUS_OK) {
      fprintf(stderr, "failed to create encoder/decoder: %d\n", err);
      opus_custom_encoder_destroy(enc);
      opus_custom_decoder_destroy(dec);
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
    failures += check_true("decode_negative_len",
                           opus_custom_decode(dec, packet, -1, decoded, kFrameSize) < 0);
    failures += check_true("decode_bad_frame_size",
                           opus_custom_decode(dec, NULL, 0, decoded, kFrameSize - 1) < 0);
    failures += check_eq_int("decoder_set_complexity_bad",
                             opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(11)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_start_band_bad_neg",
                             opus_custom_decoder_ctl(dec, CELT_SET_START_BAND(-1)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_start_band_bad_large",
                             opus_custom_decoder_ctl(dec, CELT_SET_START_BAND(kNbEBands)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_end_band_bad_zero",
                             opus_custom_decoder_ctl(dec, CELT_SET_END_BAND(0)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_end_band_bad_large",
                             opus_custom_decoder_ctl(dec, CELT_SET_END_BAND(kNbEBands + 1)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_channels_bad_zero",
                             opus_custom_decoder_ctl(dec, CELT_SET_CHANNELS(0)),
                             OPUS_BAD_ARG);
    failures += check_eq_int("decoder_set_channels_bad_three",
                             opus_custom_decoder_ctl(dec, CELT_SET_CHANNELS(3)),
                             OPUS_BAD_ARG);

    failures += set_decoder_layout(dec, 0, kNbEBands, 2);

    for (int f = 0; f < kFrames; ++f) {
      int packet_len;
      int decoded_len;
      opus_uint32 final_range = 0;

      if (f == 4) {
        failures += set_decoder_layout(dec, 3, 18, 2);
      } else if (f == 8) {
        failures += set_decoder_layout(dec, 2, 19, 1);
      } else if (f == 12) {
        failures += check_eq_int("decoder_reset",
                                 opus_custom_decoder_ctl(dec, OPUS_RESET_STATE), OPUS_OK);
        failures += set_decoder_layout(dec, 0, kNbEBands, 2);
      }

      fill_stereo_frame(pcm, kFrameSize, f * 13);
      packet_len = opus_custom_encode(enc, pcm, kFrameSize, packet, kMaxPacketSize);
      if (packet_len <= 0) {
        fprintf(stderr, "encode failed on frame %d: %d\n", f, packet_len);
        failures++;
        continue;
      }

      decoded_len = opus_custom_decode(dec, packet, packet_len, decoded, kFrameSize);
      failures += check_eq_int("decode_len", decoded_len, kFrameSize);
      failures += check_eq_int("get_final_range",
                               opus_custom_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&final_range)),
                               OPUS_OK);
      failures += check_true("decoded_nonzero", has_nonzero(decoded, kFrameSize * kChannels));

      packet_hashes[f] = fnv1a_bytes(packet, (size_t)packet_len);
      pcm_hashes[f] = fnv1a_pcm_le(decoded, kFrameSize * kChannels);
      final_ranges[f] = (uint32_t)final_range;

      if (expected.packet_hashes[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "packet_hash_%d", f);
        failures += check_eq_u32(label, packet_hashes[f], expected.packet_hashes[f]);
      }
      if (expected.pcm_hashes[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "pcm_hash_%d", f);
        failures += check_eq_u32(label, pcm_hashes[f], expected.pcm_hashes[f]);
      }
      if (expected.final_ranges[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "final_range_%d", f);
        failures += check_eq_u32(label, final_ranges[f], expected.final_ranges[f]);
      }
    }

    failures += check_true("packet_hashes_vary", count_hash_changes(packet_hashes, kFrames) >= 9);
    failures += check_true("pcm_hashes_vary", count_hash_changes(pcm_hashes, kFrames) >= 9);
    failures += check_true("range_hashes_vary", count_hash_changes(final_ranges, kFrames) >= 9);

    packet[0] = 0;
    failures +=
        check_true("decode_tiny_packet_fails",
                   opus_custom_decode(dec, packet, 1, decoded, kFrameSize) < 0);
    failures += check_true("decode_oversize_len",
                           opus_custom_decode(dec, packet, kMaxPacketSize, decoded,
                                              kFrameSize) < 0);

    if (dump_expected) {
      printf("packet_hashes:");
      for (int i = 0; i < kFrames; ++i) {
        printf(" 0x%08x", packet_hashes[i]);
      }
      printf("\npcm_hashes:");
      for (int i = 0; i < kFrames; ++i) {
        printf(" 0x%08x", pcm_hashes[i]);
      }
      printf("\nfinal_ranges:");
      for (int i = 0; i < kFrames; ++i) {
        printf(" 0x%08x", final_ranges[i]);
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
          "celt_decoder_dataflow_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
