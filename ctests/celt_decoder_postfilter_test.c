#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opus_custom.h"
#include "opus_defines.h"

#ifdef FIXED_POINT

enum {
  kSampleRate = 48000,
  kFrameSize = 960,
  kMaxPacketSize = 1276,
  kMaxPitchValue = 1024,
  kFrames = 6
};

typedef struct {
  const char *name;
  int channels;
  int bitrate;
  int max_bytes;
  int min_nonzero_pitch_frames;
  int expected_pitch[kFrames];
  uint32_t expected_packet_hash[kFrames];
  uint32_t expected_pcm_hash[kFrames];
} PostfilterCase;

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

static void fill_case_pcm(opus_int16 *pcm, int frame_size, int channels,
                          int frame_idx) {
  for (int i = 0; i < frame_size; ++i) {
    int s0 = ((i * 7 + frame_idx * 31) % 3000) - 1500;
    int s1 = ((i * 11 + frame_idx * 19) % 2600) - 1300;
    int shaped0 = s0 + ((i & 7) == 0 ? 900 : -300);
    int shaped1 = s1 + ((i % 5) == 0 ? -700 : 250);
    if (channels == 1) {
      pcm[i] = (opus_int16)(shaped0 * 8);
    } else {
      pcm[2 * i] = (opus_int16)(shaped0 * 7);
      pcm[2 * i + 1] = (opus_int16)(shaped1 * 7);
    }
  }
}

static int run_case(const OpusCustomMode *mode, const PostfilterCase *test) {
  int failures = 0;
  int err = OPUS_OK;
  int nonzero_pitch_count = 0;
  int pitch_changes = 0;
  int prev_pitch = -1;
  int dump = getenv("CELT_DECODER_POSTFILTER_DUMP") != NULL &&
             getenv("CELT_DECODER_POSTFILTER_DUMP")[0] != '\0' &&
             getenv("CELT_DECODER_POSTFILTER_DUMP")[0] != '0';

  OpusCustomEncoder *enc =
      opus_custom_encoder_create(mode, test->channels, &err);
  if (enc == NULL || err != OPUS_OK) {
    fprintf(stderr, "%s: encoder create failed: %d\n", test->name, err);
    return 1;
  }
  OpusCustomDecoder *dec =
      opus_custom_decoder_create(mode, test->channels, &err);
  if (dec == NULL || err != OPUS_OK) {
    fprintf(stderr, "%s: decoder create failed: %d\n", test->name, err);
    opus_custom_encoder_destroy(enc);
    return 1;
  }

  failures += check_eq_int(
      "enc_set_bitrate", opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(test->bitrate)),
      OPUS_OK);
  failures += check_eq_int("enc_set_vbr",
                           opus_custom_encoder_ctl(enc, OPUS_SET_VBR(0)),
                           OPUS_OK);
  failures += check_eq_int("enc_set_complexity",
                           opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10)),
                           OPUS_OK);
  failures += check_eq_int("enc_set_lsb_depth",
                           opus_custom_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(16)),
                           OPUS_OK);

  {
    unsigned char dummy[2] = {0};
    failures += check_eq_int("decode_null_pcm",
                             opus_custom_decode(dec, NULL, 0, NULL, kFrameSize),
                             OPUS_BAD_ARG);
    failures += check_true(
        "decode_bad_frame_size_fails",
        opus_custom_decode(dec, dummy, 1, (opus_int16 *)dummy, 123) < 0);
    failures += check_eq_int(
        "decoder_set_complexity_bad",
        opus_custom_decoder_ctl(dec, OPUS_SET_COMPLEXITY(99)),
        OPUS_BAD_ARG);
  }

  {
    unsigned char packet[kMaxPacketSize];
    opus_int16 pcm[kFrameSize * 2];
    opus_int16 decoded[kFrameSize * 2];

    for (int f = 0; f < kFrames; ++f) {
      int packet_len;
      int decoded_len;
      int pitch = 0;
      uint32_t packet_hash;
      uint32_t pcm_hash;

      fill_case_pcm(pcm, kFrameSize, test->channels, f);
      packet_len = opus_custom_encode(enc, pcm, kFrameSize, packet, test->max_bytes);
      if (packet_len <= 0) {
        fprintf(stderr, "%s frame %d: encode failed: %d\n", test->name, f,
                packet_len);
        failures++;
        continue;
      }

      decoded_len = opus_custom_decode(dec, packet, packet_len, decoded, kFrameSize);
      failures += check_eq_int("decode_len", decoded_len, kFrameSize);

      failures += check_eq_int("get_pitch_ctl",
                               opus_custom_decoder_ctl(dec, OPUS_GET_PITCH(&pitch)),
                               OPUS_OK);
      failures += check_true("pitch_in_range",
                             pitch >= 0 && pitch <= kMaxPitchValue);

      if (pitch > 0) nonzero_pitch_count++;
      if (prev_pitch >= 0 && pitch != prev_pitch) pitch_changes++;
      prev_pitch = pitch;

      packet_hash = fnv1a_bytes(packet, (size_t)packet_len);
      pcm_hash = fnv1a_pcm_le(decoded, kFrameSize * test->channels);

      if (dump) {
        printf("%s frame %d pitch=%d packet_hash=0x%08x pcm_hash=0x%08x\n",
               test->name, f, pitch, packet_hash, pcm_hash);
      }

      if (test->expected_pitch[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "%s_pitch_%d", test->name, f);
        failures += check_eq_int(label, pitch, test->expected_pitch[f]);
      }
      if (test->expected_packet_hash[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "%s_packet_hash_%d", test->name, f);
        failures +=
            check_eq_u32(label, packet_hash, test->expected_packet_hash[f]);
      }
      if (test->expected_pcm_hash[f] != 0) {
        char label[64];
        snprintf(label, sizeof(label), "%s_pcm_hash_%d", test->name, f);
        failures += check_eq_u32(label, pcm_hash, test->expected_pcm_hash[f]);
      }
    }

    failures += check_true("nonzero_pitch_frames",
                           nonzero_pitch_count >= test->min_nonzero_pitch_frames);
    failures += check_true("pitch_changes_seen", pitch_changes > 0);

    packet[0] = 0;
    failures += check_true("decode_tiny_packet_fails",
                           opus_custom_decode(dec, packet, 1, decoded, kFrameSize) < 0);
  }

  opus_custom_decoder_destroy(dec);
  opus_custom_encoder_destroy(enc);
  return failures;
}

int main(void) {
  int err = OPUS_OK;
  int failures = 0;
  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create mode: %d\n", err);
    return 1;
  }

  PostfilterCase cases[] = {
      {
          "mono_postfilter",
          1,
          24000,
          96,
          3,
          {430, 954, 954, 0, 528, 528},
          {0xd9dbcb3a, 0x5e4e25b9, 0x5de8bc73, 0x3d80f8c0, 0x68d62f7c,
           0x50ebf0b5},
          {0xba7efc92, 0xc44126c4, 0xa5824709, 0x4d7c01ed, 0x85bb0370,
           0x6911f513},
      },
      {
          "stereo_postfilter",
          2,
          64000,
          180,
          3,
          {0, 480, 480, 480, 480, 480},
          {0x5cdf3e0c, 0x6fc9dad3, 0x940608ff, 0x49c09e95, 0xe63d158e,
           0x418b6748},
          {0x042e972d, 0x547d2986, 0x8c2eee5d, 0x9d667d9a, 0xbb76e17f,
           0xdb241031},
      },
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    failures += run_case(mode, &cases[i]);
  }

  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr,
          "celt_decoder_postfilter_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
