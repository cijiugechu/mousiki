#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opus.h"

#define FRAME_SIZE 960
#define SAMPLE_RATE 48000
#define CHANNELS 2
#define APPLICATION OPUS_APPLICATION_AUDIO
#define BITRATE 64000

#define MAX_PACKET_SIZE (3 * 1276)

static const unsigned char kMagic[8] = {'O', 'P', 'U', 'S',
                                        'P', 'K', 'T', '1'};

static int write_le16(FILE *file, uint16_t value) {
  unsigned char buf[2];
  buf[0] = (unsigned char)(value & 0xFF);
  buf[1] = (unsigned char)((value >> 8) & 0xFF);
  return fwrite(buf, sizeof(buf), 1, file) == 1;
}

static int write_le32(FILE *file, uint32_t value) {
  unsigned char buf[4];
  buf[0] = (unsigned char)(value & 0xFF);
  buf[1] = (unsigned char)((value >> 8) & 0xFF);
  buf[2] = (unsigned char)((value >> 16) & 0xFF);
  buf[3] = (unsigned char)((value >> 24) & 0xFF);
  return fwrite(buf, sizeof(buf), 1, file) == 1;
}

static int write_header(FILE *file) {
  if (fwrite(kMagic, sizeof(kMagic), 1, file) != 1) {
    return 0;
  }
  if (!write_le32(file, SAMPLE_RATE)) {
    return 0;
  }
  if (!write_le16(file, CHANNELS)) {
    return 0;
  }
  if (!write_le16(file, FRAME_SIZE)) {
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: opus_packet_encode <input.pcm> <output.opuspkt>\n");
    return EXIT_FAILURE;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[2];

  FILE *fin = fopen(input_path, "rb");
  if (fin == NULL) {
    fprintf(stderr, "failed to open input file: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  FILE *fout = fopen(output_path, "wb");
  if (fout == NULL) {
    fprintf(stderr, "failed to open output file: %s\n", strerror(errno));
    fclose(fin);
    return EXIT_FAILURE;
  }

  int err = 0;
  OpusEncoder *encoder =
      opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION, &err);
  if (err < 0 || encoder == NULL) {
    fprintf(stderr, "failed to create encoder: %s\n", opus_strerror(err));
    fclose(fin);
    fclose(fout);
    return EXIT_FAILURE;
  }

  err = opus_encoder_ctl(encoder, OPUS_SET_BITRATE(BITRATE));
  if (err < 0) {
    fprintf(stderr, "failed to set bitrate: %s\n", opus_strerror(err));
    opus_encoder_destroy(encoder);
    fclose(fin);
    fclose(fout);
    return EXIT_FAILURE;
  }

  if (!write_header(fout)) {
    fprintf(stderr, "failed to write packet header\n");
    opus_encoder_destroy(encoder);
    fclose(fin);
    fclose(fout);
    return EXIT_FAILURE;
  }

  unsigned char pcm_bytes[FRAME_SIZE * CHANNELS * 2];
  opus_int16 input[FRAME_SIZE * CHANNELS];
  unsigned char packet[MAX_PACKET_SIZE];

  while (1) {
    size_t samples = fread(pcm_bytes, sizeof(short) * CHANNELS, FRAME_SIZE, fin);
    if (samples != FRAME_SIZE) {
      break;
    }

    for (int i = 0; i < FRAME_SIZE * CHANNELS; ++i) {
      input[i] = (opus_int16)(pcm_bytes[2 * i + 1] << 8 | pcm_bytes[2 * i]);
    }

    int packet_len =
        opus_encode(encoder, input, FRAME_SIZE, packet, MAX_PACKET_SIZE);
    if (packet_len < 0) {
      fprintf(stderr, "encode failed: %s\n", opus_strerror(packet_len));
      opus_encoder_destroy(encoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    if (packet_len > UINT16_MAX) {
      fprintf(stderr, "packet length too large: %d\n", packet_len);
      opus_encoder_destroy(encoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    if (!write_le16(fout, (uint16_t)packet_len)) {
      fprintf(stderr, "failed to write packet length\n");
      opus_encoder_destroy(encoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    if (fwrite(packet, 1, (size_t)packet_len, fout) != (size_t)packet_len) {
      fprintf(stderr, "failed to write packet bytes\n");
      opus_encoder_destroy(encoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }
  }

  opus_encoder_destroy(encoder);
  fclose(fin);
  fclose(fout);
  return EXIT_SUCCESS;
}
