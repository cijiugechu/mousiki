#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opus.h"

#define FRAME_SIZE 960
#define SAMPLE_RATE 48000
#define CHANNELS 2

#define MAX_FRAME_SIZE (6 * FRAME_SIZE)
#define MAX_PACKET_SIZE (3 * 1276)

static const unsigned char kMagic[8] = {'O', 'P', 'U', 'S',
                                        'P', 'K', 'T', '1'};

static int read_le16(FILE *file, uint16_t *value, int *eof) {
  int b0 = fgetc(file);
  if (b0 == EOF) {
    *eof = 1;
    return 0;
  }
  int b1 = fgetc(file);
  if (b1 == EOF) {
    return -1;
  }
  *value = (uint16_t)((b1 << 8) | b0);
  *eof = 0;
  return 1;
}

static int read_le32(FILE *file, uint32_t *value) {
  int b0 = fgetc(file);
  int b1 = fgetc(file);
  int b2 = fgetc(file);
  int b3 = fgetc(file);
  if (b0 == EOF || b1 == EOF || b2 == EOF || b3 == EOF) {
    return 0;
  }
  *value =
      (uint32_t)(b0 | (b1 << 8) | (b2 << 16) | ((uint32_t)b3 << 24));
  return 1;
}

static int read_header(FILE *file) {
  unsigned char magic[8];
  if (fread(magic, sizeof(magic), 1, file) != 1) {
    return 0;
  }
  if (memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return 0;
  }

  uint32_t sample_rate = 0;
  uint16_t channels = 0;
  uint16_t frame_size = 0;
  if (!read_le32(file, &sample_rate)) {
    return 0;
  }
  int eof = 0;
  if (read_le16(file, &channels, &eof) <= 0 || eof) {
    return 0;
  }
  if (read_le16(file, &frame_size, &eof) <= 0 || eof) {
    return 0;
  }

  if (sample_rate != SAMPLE_RATE || channels != CHANNELS ||
      frame_size != FRAME_SIZE) {
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: opus_packet_decode <input.opuspkt> <output.pcm>\n");
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

  if (!read_header(fin)) {
    fprintf(stderr, "invalid packet header\n");
    fclose(fin);
    fclose(fout);
    return EXIT_FAILURE;
  }

  int err = 0;
  OpusDecoder *decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &err);
  if (err < 0 || decoder == NULL) {
    fprintf(stderr, "failed to create decoder: %s\n", opus_strerror(err));
    fclose(fin);
    fclose(fout);
    return EXIT_FAILURE;
  }

  unsigned char packet[MAX_PACKET_SIZE];
  opus_int16 output[MAX_FRAME_SIZE * CHANNELS];
  unsigned char pcm_bytes[MAX_FRAME_SIZE * CHANNELS * 2];

  while (1) {
    uint16_t packet_len = 0;
    int eof = 0;
    int status = read_le16(fin, &packet_len, &eof);
    if (status == 0 && eof) {
      break;
    }
    if (status < 0) {
      fprintf(stderr, "failed to read packet length\n");
      opus_decoder_destroy(decoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }
    if (packet_len == 0) {
      break;
    }
    if (packet_len > MAX_PACKET_SIZE) {
      fprintf(stderr, "packet length too large: %u\n", packet_len);
      opus_decoder_destroy(decoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    if (fread(packet, 1, packet_len, fin) != packet_len) {
      fprintf(stderr, "failed to read packet bytes\n");
      opus_decoder_destroy(decoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    int decoded = opus_decode(decoder, packet, packet_len, output, MAX_FRAME_SIZE,
                              0);
    if (decoded < 0) {
      fprintf(stderr, "decode failed: %s\n", opus_strerror(decoded));
      opus_decoder_destroy(decoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }

    for (int i = 0; i < decoded * CHANNELS; ++i) {
      pcm_bytes[2 * i] = (unsigned char)(output[i] & 0xFF);
      pcm_bytes[2 * i + 1] = (unsigned char)((output[i] >> 8) & 0xFF);
    }
    if (fwrite(pcm_bytes, sizeof(short), decoded * CHANNELS, fout) !=
        (size_t)(decoded * CHANNELS)) {
      fprintf(stderr, "failed to write pcm output\n");
      opus_decoder_destroy(decoder);
      fclose(fin);
      fclose(fout);
      return EXIT_FAILURE;
    }
  }

  opus_decoder_destroy(decoder);
  fclose(fin);
  fclose(fout);
  return EXIT_SUCCESS;
}
