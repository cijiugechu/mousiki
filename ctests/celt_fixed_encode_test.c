#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opus_custom.h"
#include "opus_defines.h"

#ifdef FIXED_POINT

#define SAMPLE_RATE 48000
#define FRAME_SIZE 960
#define MAX_PACKET_SIZE 1276
#define MAX_FRAMES 2

typedef struct {
  const char *name;
  int channels;
  int frames;
  int max_bytes;
  int bitrate;
  int complexity;
  int vbr;
  int lsb_depth;
  int expected_packet_len[MAX_FRAMES];
  uint32_t expected_packet_hash[MAX_FRAMES];
  uint32_t expected_pcm_hash[MAX_FRAMES];
} EncodeCase;

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

static void fill_case_pcm(int case_id, opus_int16 *pcm, int frames, int channels,
                          int frame_size) {
  for (int f = 0; f < frames; ++f) {
    for (int i = 0; i < frame_size; ++i) {
      int base = (f * frame_size + i) * channels;
      if (case_id == 0) {
        opus_int16 value = (opus_int16)(((i * 7 + f * 13) % 2000) - 1000);
        pcm[base] = value;
      } else if (case_id == 1) {
        opus_int16 left = (opus_int16)(((i * 3) % 4000) - 2000);
        opus_int16 right = (opus_int16)(((i * 5) % 4000) - 2000);
        pcm[base] = left;
        pcm[base + 1] = right;
      } else {
        opus_int16 left = (opus_int16)((i & 1) ? 30000 : -30000);
        opus_int16 right = (opus_int16)((i & 1) ? -28000 : 28000);
        pcm[base] = left;
        pcm[base + 1] = right;
      }
    }
  }
}

static int run_case(const OpusCustomMode *mode, const EncodeCase *test,
                    int case_id) {
  int failures = 0;
  int err = OPUS_OK;

  OpusCustomEncoder *enc =
      opus_custom_encoder_create(mode, test->channels, &err);
  if (err != OPUS_OK || enc == NULL) {
    fprintf(stderr, "%s: encoder create failed: %d\n", test->name, err);
    return 1;
  }
  OpusCustomDecoder *dec =
      opus_custom_decoder_create(mode, test->channels, &err);
  if (err != OPUS_OK || dec == NULL) {
    fprintf(stderr, "%s: decoder create failed: %d\n", test->name, err);
    opus_custom_encoder_destroy(enc);
    return 1;
  }

  if (opus_custom_encoder_ctl(enc, OPUS_SET_BITRATE(test->bitrate)) != OPUS_OK ||
      opus_custom_encoder_ctl(enc, OPUS_SET_VBR(test->vbr)) != OPUS_OK ||
      opus_custom_encoder_ctl(enc, OPUS_SET_COMPLEXITY(test->complexity)) !=
          OPUS_OK ||
      opus_custom_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(test->lsb_depth)) !=
          OPUS_OK) {
    fprintf(stderr, "%s: encoder ctl failed\n", test->name);
    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
    return 1;
  }

  const int pcm_len = test->frames * test->channels * FRAME_SIZE;
  opus_int16 *pcm =
      (opus_int16 *)calloc((size_t)pcm_len, sizeof(opus_int16));
  opus_int16 *decoded =
      (opus_int16 *)calloc((size_t)(FRAME_SIZE * test->channels),
                           sizeof(opus_int16));
  if (pcm == NULL || decoded == NULL) {
    fprintf(stderr, "%s: allocation failed\n", test->name);
    free(pcm);
    free(decoded);
    opus_custom_decoder_destroy(dec);
    opus_custom_encoder_destroy(enc);
    return 1;
  }

  fill_case_pcm(case_id, pcm, test->frames, test->channels, FRAME_SIZE);

  unsigned char packet[MAX_PACKET_SIZE];
  const int pcm_stride = test->channels * FRAME_SIZE;
  const int dump_enabled =
      getenv("CELT_FIXED_ENC_DUMP") != NULL &&
      getenv("CELT_FIXED_ENC_DUMP")[0] != '\0' &&
      getenv("CELT_FIXED_ENC_DUMP")[0] != '0';

  for (int f = 0; f < test->frames; ++f) {
    const opus_int16 *frame = pcm + f * pcm_stride;
    int packet_len =
        opus_custom_encode(enc, frame, FRAME_SIZE, packet, test->max_bytes);
    if (packet_len < 0) {
      fprintf(stderr, "%s: encode failed on frame %d: %d\n", test->name, f,
              packet_len);
      failures++;
      continue;
    }
    if (packet_len > test->max_bytes) {
      fprintf(stderr, "%s: packet too large on frame %d: %d\n", test->name, f,
              packet_len);
      failures++;
    }
    if (test->expected_packet_len[f] != 0 &&
        packet_len != test->expected_packet_len[f]) {
      fprintf(stderr,
              "%s: packet length mismatch on frame %d: got %d expected %d\n",
              test->name, f, packet_len, test->expected_packet_len[f]);
      failures++;
    }

    int decoded_len =
        opus_custom_decode(dec, packet, packet_len, decoded, FRAME_SIZE);
    if (decoded_len != FRAME_SIZE) {
      fprintf(stderr, "%s: decode failed on frame %d: %d\n", test->name, f,
              decoded_len);
      failures++;
      continue;
    }

    uint32_t packet_hash = fnv1a_bytes(packet, (size_t)packet_len);
    uint32_t pcm_hash = fnv1a_pcm_le(decoded, FRAME_SIZE * test->channels);
    if (dump_enabled) {
      printf("%s frame %d packet_len=%d packet_hash=0x%08x pcm_hash=0x%08x\n",
             test->name, f, packet_len, packet_hash, pcm_hash);
    }
    if (test->expected_packet_hash[f] != 0 &&
        packet_hash != test->expected_packet_hash[f]) {
      fprintf(stderr,
              "%s: packet hash mismatch on frame %d: got 0x%08x expected "
              "0x%08x\n",
              test->name, f, packet_hash, test->expected_packet_hash[f]);
      failures++;
    }
    if (test->expected_pcm_hash[f] != 0 &&
        pcm_hash != test->expected_pcm_hash[f]) {
      fprintf(stderr,
              "%s: pcm hash mismatch on frame %d: got 0x%08x expected 0x%08x\n",
              test->name, f, pcm_hash, test->expected_pcm_hash[f]);
      failures++;
    }
  }

  free(pcm);
  free(decoded);
  opus_custom_decoder_destroy(dec);
  opus_custom_encoder_destroy(enc);
  return failures;
}

int main(void) {
  int err = OPUS_OK;
  OpusCustomMode *mode =
      opus_custom_mode_create(SAMPLE_RATE, FRAME_SIZE, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "failed to create custom mode: %d\n", err);
    return 1;
  }

  EncodeCase cases[] = {
      {
          "mono_low_budget",
          1,
          2,
          15,
          12000,
          5,
          0,
          16,
          {15, 15},
          {0x95f3664b, 0x69d9242e},
          {0xb73ab8b3, 0x8bb8515b},
      },
      {
          "stereo_normal",
          2,
          1,
          120,
          64000,
          7,
          0,
          16,
          {120, 0},
          {0x01cc11e3, 0},
          {0x18c0a2b5, 0},
      },
      {
          "stereo_large_amp",
          2,
          1,
          120,
          64000,
          7,
          0,
          16,
          {120, 0},
          {0x76902e9b, 0},
          {0x78c40e00, 0},
      },
  };

  int failures = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    failures += run_case(mode, &cases[i], (int)i);
  }

  opus_custom_mode_destroy(mode);
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr, "celt_fixed_encode_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
