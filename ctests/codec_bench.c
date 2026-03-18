#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "opus.h"

#define MAGIC_SIZE 8
#define MAX_PACKET_SIZE (3 * 1276)

static const unsigned char kMagic[MAGIC_SIZE] = {'O', 'P', 'U', 'S',
                                                 'B', 'E', 'N', '1'};

typedef enum {
  BITRATE_MODE_VBR = 0,
  BITRATE_MODE_CVBR = 1,
  BITRATE_MODE_CBR = 2,
} BitrateMode;

typedef enum {
  OUTPUT_TEXT,
  OUTPUT_CSV,
} OutputFormat;

typedef struct {
  int sample_rate;
  int channels;
  int frame_size;
  int application;
  const char *application_name;
  int bitrate;
  int complexity;
  BitrateMode bitrate_mode;
} EncodeConfig;

typedef struct {
  EncodeConfig encode;
  const char *input_path;
  const char *output_path;
  size_t max_frames;
  int has_max_frames;
} PacketArgs;

typedef struct {
  EncodeConfig encode;
  const char *input_path;
  size_t warmup;
  size_t measure;
  size_t max_frames;
  int has_max_frames;
  OutputFormat format;
  int header;
} EncodeBenchArgs;

typedef struct {
  const char *packet_path;
  size_t warmup;
  size_t measure;
  size_t max_frames;
  int has_max_frames;
  OutputFormat format;
  int header;
} DecodeBenchArgs;

typedef struct {
  uint32_t sample_rate;
  uint16_t channels;
  uint16_t frame_size;
  uint32_t application;
  int32_t bitrate;
  uint8_t complexity;
  uint8_t bitrate_mode;
  uint16_t reserved;
} PacketHeader;

typedef struct {
  size_t len;
  unsigned char *data;
} Packet;

typedef struct {
  PacketHeader header;
  Packet *packets;
  size_t count;
} PacketCorpus;

typedef struct {
  const char *implementation;
  const char *operation;
  EncodeConfig config;
  size_t frames;
  size_t warmup;
  size_t measure;
  double median_ns_per_frame;
  double p95_ns_per_frame;
  double median_packets_per_sec;
  double median_realtime_x;
} BenchResult;

static void usage(void) {
  fprintf(stderr,
          "usage:\n"
          "  codec_bench packets --input INPUT.pcm --output OUTPUT.opusbench "
          "[--sample-rate 48000] [--channels 2] [--frame-size 960]\n"
          "    [--application audio] [--bitrate 64000] [--complexity 10] "
          "[--bitrate-mode cvbr] [--max-frames N]\n"
          "  codec_bench encode --input INPUT.pcm [--sample-rate 48000] "
          "[--channels 2] [--frame-size 960]\n"
          "    [--application audio] [--bitrate 64000] [--complexity 10] "
          "[--bitrate-mode cvbr] [--warmup 3] [--measure 10] [--max-frames N]\n"
          "    [--format text|csv] [--no-header]\n"
          "  codec_bench decode --packets INPUT.opusbench [--warmup 3] "
          "[--measure 10] [--max-frames N] [--format text|csv] [--no-header]\n");
}

static int parse_i32(const char *value, int *out) {
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return 0;
  }
  *out = (int)parsed;
  return 1;
}

static int parse_usize(const char *value, size_t *out) {
  char *end = NULL;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    return 0;
  }
  *out = (size_t)parsed;
  return 1;
}

static int parse_application(const char *value, int *out, const char **name) {
  if (strcmp(value, "voip") == 0) {
    *out = OPUS_APPLICATION_VOIP;
    *name = "voip";
    return 1;
  }
  if (strcmp(value, "audio") == 0) {
    *out = OPUS_APPLICATION_AUDIO;
    *name = "audio";
    return 1;
  }
  if (strcmp(value, "restricted-lowdelay") == 0 ||
      strcmp(value, "restricted_lowdelay") == 0 ||
      strcmp(value, "lowdelay") == 0) {
    *out = OPUS_APPLICATION_RESTRICTED_LOWDELAY;
    *name = "restricted-lowdelay";
    return 1;
  }
  return 0;
}

static int parse_bitrate_mode(const char *value, BitrateMode *out) {
  if (strcmp(value, "vbr") == 0) {
    *out = BITRATE_MODE_VBR;
    return 1;
  }
  if (strcmp(value, "cvbr") == 0) {
    *out = BITRATE_MODE_CVBR;
    return 1;
  }
  if (strcmp(value, "cbr") == 0) {
    *out = BITRATE_MODE_CBR;
    return 1;
  }
  return 0;
}

static int parse_output_format(const char *value, OutputFormat *out) {
  if (strcmp(value, "text") == 0) {
    *out = OUTPUT_TEXT;
    return 1;
  }
  if (strcmp(value, "csv") == 0) {
    *out = OUTPUT_CSV;
    return 1;
  }
  return 0;
}

static EncodeConfig default_encode_config(void) {
  EncodeConfig config;
  config.sample_rate = 48000;
  config.channels = 2;
  config.frame_size = 960;
  config.application = OPUS_APPLICATION_AUDIO;
  config.application_name = "audio";
  config.bitrate = 64000;
  config.complexity = 10;
  config.bitrate_mode = BITRATE_MODE_CVBR;
  return config;
}

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

static int write_i32(FILE *file, int32_t value) {
  return write_le32(file, (uint32_t)value);
}

static int read_le16(FILE *file, uint16_t *value, int *eof) {
  int b0 = fgetc(file);
  if (b0 == EOF) {
    if (eof != NULL) {
      *eof = 1;
    }
    return 0;
  }
  int b1 = fgetc(file);
  if (b1 == EOF) {
    return 0;
  }
  *value = (uint16_t)((b1 << 8) | b0);
  if (eof != NULL) {
    *eof = 0;
  }
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
  *value = (uint32_t)(b0 | (b1 << 8) | (b2 << 16) | ((uint32_t)b3 << 24));
  return 1;
}

static int read_i32(FILE *file, int32_t *value) {
  uint32_t raw = 0;
  if (!read_le32(file, &raw)) {
    return 0;
  }
  *value = (int32_t)raw;
  return 1;
}

static int write_packet_header(FILE *file, const EncodeConfig *config) {
  if (fwrite(kMagic, sizeof(kMagic), 1, file) != 1) {
    return 0;
  }
  if (!write_le32(file, (uint32_t)config->sample_rate)) {
    return 0;
  }
  if (!write_le16(file, (uint16_t)config->channels)) {
    return 0;
  }
  if (!write_le16(file, (uint16_t)config->frame_size)) {
    return 0;
  }
  if (!write_le32(file, (uint32_t)config->application)) {
    return 0;
  }
  if (!write_i32(file, config->bitrate)) {
    return 0;
  }
  if (fputc(config->complexity, file) == EOF) {
    return 0;
  }
  if (fputc(config->bitrate_mode, file) == EOF) {
    return 0;
  }
  if (!write_le16(file, 0)) {
    return 0;
  }
  return 1;
}

static int read_packet_header(FILE *file, PacketHeader *header) {
  unsigned char magic[MAGIC_SIZE];
  if (fread(magic, sizeof(magic), 1, file) != 1) {
    return 0;
  }
  if (memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return 0;
  }
  if (!read_le32(file, &header->sample_rate)) {
    return 0;
  }
  int eof = 0;
  if (!read_le16(file, &header->channels, &eof) || eof) {
    return 0;
  }
  if (!read_le16(file, &header->frame_size, &eof) || eof) {
    return 0;
  }
  if (!read_le32(file, &header->application)) {
    return 0;
  }
  if (!read_i32(file, &header->bitrate)) {
    return 0;
  }
  int complexity = fgetc(file);
  int bitrate_mode = fgetc(file);
  if (complexity == EOF || bitrate_mode == EOF) {
    return 0;
  }
  header->complexity = (uint8_t)complexity;
  header->bitrate_mode = (uint8_t)bitrate_mode;
  if (!read_le16(file, &header->reserved, &eof) || eof) {
    return 0;
  }
  return 1;
}

static int create_encoder(const EncodeConfig *config, OpusEncoder **out) {
  int err = 0;
  OpusEncoder *encoder =
      opus_encoder_create(config->sample_rate, config->channels,
                          config->application, &err);
  if (err < 0 || encoder == NULL) {
    fprintf(stderr, "failed to create encoder: %s\n", opus_strerror(err));
    return 0;
  }
  err = opus_encoder_ctl(encoder, OPUS_SET_BITRATE(config->bitrate));
  if (err < 0) {
    fprintf(stderr, "failed to set bitrate: %s\n", opus_strerror(err));
    opus_encoder_destroy(encoder);
    return 0;
  }
  err = opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(config->complexity));
  if (err < 0) {
    fprintf(stderr, "failed to set complexity: %s\n", opus_strerror(err));
    opus_encoder_destroy(encoder);
    return 0;
  }
  if (config->bitrate_mode == BITRATE_MODE_CBR) {
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR(0));
    if (err >= 0) {
      err = opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(1));
    }
  } else if (config->bitrate_mode == BITRATE_MODE_CVBR) {
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR(1));
    if (err >= 0) {
      err = opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(1));
    }
  } else {
    err = opus_encoder_ctl(encoder, OPUS_SET_VBR(1));
    if (err >= 0) {
      err = opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(0));
    }
  }
  if (err < 0) {
    fprintf(stderr, "failed to set bitrate mode: %s\n", opus_strerror(err));
    opus_encoder_destroy(encoder);
    return 0;
  }
  *out = encoder;
  return 1;
}

static int load_pcm_samples(const char *path, opus_int16 **samples_out,
                            size_t *sample_count_out) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "failed to open input file: %s\n", strerror(errno));
    return 0;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fprintf(stderr, "failed to seek input file\n");
    fclose(file);
    return 0;
  }
  long size = ftell(file);
  if (size < 0) {
    fprintf(stderr, "failed to determine input size\n");
    fclose(file);
    return 0;
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fprintf(stderr, "failed to rewind input file\n");
    fclose(file);
    return 0;
  }
  if (size % 2 != 0) {
    fprintf(stderr, "pcm byte length must be even, got %ld\n", size);
    fclose(file);
    return 0;
  }
  size_t sample_count = (size_t)size / 2;
  opus_int16 *samples = (opus_int16 *)malloc(sample_count * sizeof(opus_int16));
  if (samples == NULL) {
    fprintf(stderr, "failed to allocate pcm buffer\n");
    fclose(file);
    return 0;
  }
  unsigned char *bytes = (unsigned char *)malloc((size_t)size);
  if (bytes == NULL) {
    fprintf(stderr, "failed to allocate pcm byte buffer\n");
    free(samples);
    fclose(file);
    return 0;
  }
  if (fread(bytes, 1, (size_t)size, file) != (size_t)size) {
    fprintf(stderr, "failed to read pcm file\n");
    free(bytes);
    free(samples);
    fclose(file);
    return 0;
  }
  for (size_t i = 0; i < sample_count; ++i) {
    samples[i] = (opus_int16)(bytes[2 * i] | (bytes[2 * i + 1] << 8));
  }
  free(bytes);
  fclose(file);
  *samples_out = samples;
  *sample_count_out = sample_count;
  return 1;
}

static size_t frame_count_for_pcm(size_t sample_count, const EncodeConfig *config) {
  size_t frame_samples = (size_t)config->frame_size * (size_t)config->channels;
  if (frame_samples == 0) {
    return 0;
  }
  return sample_count / frame_samples;
}

static int write_corpus_file(const char *path, const EncodeConfig *config,
                             const opus_int16 *samples, size_t frames) {
  FILE *file = fopen(path, "wb");
  if (file == NULL) {
    fprintf(stderr, "failed to open output file: %s\n", strerror(errno));
    return 0;
  }
  if (!write_packet_header(file, config)) {
    fprintf(stderr, "failed to write packet corpus header\n");
    fclose(file);
    return 0;
  }

  OpusEncoder *encoder = NULL;
  if (!create_encoder(config, &encoder)) {
    fclose(file);
    return 0;
  }

  size_t frame_samples = (size_t)config->frame_size * (size_t)config->channels;
  unsigned char packet[MAX_PACKET_SIZE];
  for (size_t frame = 0; frame < frames; ++frame) {
    size_t offset = frame * frame_samples;
    int packet_len = opus_encode(encoder, samples + offset, config->frame_size,
                                 packet, MAX_PACKET_SIZE);
    if (packet_len < 0) {
      fprintf(stderr, "encode failed: %s\n", opus_strerror(packet_len));
      opus_encoder_destroy(encoder);
      fclose(file);
      return 0;
    }
    if (!write_le16(file, (uint16_t)packet_len) ||
        fwrite(packet, 1, (size_t)packet_len, file) != (size_t)packet_len) {
      fprintf(stderr, "failed to write packet corpus payload\n");
      opus_encoder_destroy(encoder);
      fclose(file);
      return 0;
    }
  }

  opus_encoder_destroy(encoder);
  fclose(file);
  return 1;
}

static int load_packet_corpus(const char *path, PacketCorpus *corpus) {
  memset(corpus, 0, sizeof(*corpus));
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "failed to open packet corpus: %s\n", strerror(errno));
    return 0;
  }
  if (!read_packet_header(file, &corpus->header)) {
    fprintf(stderr, "invalid packet corpus header\n");
    fclose(file);
    return 0;
  }

  size_t capacity = 0;
  while (1) {
    uint16_t len = 0;
    int eof = 0;
    int status = read_le16(file, &len, &eof);
    if (!status && eof) {
      break;
    }
    if (!status) {
      fprintf(stderr, "failed to read packet length\n");
      fclose(file);
      return 0;
    }
    if (corpus->count == capacity) {
      size_t new_capacity = capacity == 0 ? 64 : capacity * 2;
      Packet *new_packets =
          (Packet *)realloc(corpus->packets, new_capacity * sizeof(Packet));
      if (new_packets == NULL) {
        fprintf(stderr, "failed to grow packet corpus\n");
        fclose(file);
        return 0;
      }
      corpus->packets = new_packets;
      capacity = new_capacity;
    }
    corpus->packets[corpus->count].len = len;
    corpus->packets[corpus->count].data = (unsigned char *)malloc(len);
    if (corpus->packets[corpus->count].data == NULL) {
      fprintf(stderr, "failed to allocate packet buffer\n");
      fclose(file);
      return 0;
    }
    if (fread(corpus->packets[corpus->count].data, 1, len, file) != len) {
      fprintf(stderr, "failed to read packet bytes\n");
      fclose(file);
      return 0;
    }
    corpus->count += 1;
  }

  fclose(file);
  return 1;
}

static void free_packet_corpus(PacketCorpus *corpus) {
  if (corpus->packets == NULL) {
    return;
  }
  for (size_t i = 0; i < corpus->count; ++i) {
    free(corpus->packets[i].data);
  }
  free(corpus->packets);
  corpus->packets = NULL;
  corpus->count = 0;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *lhs, const void *rhs) {
  uint64_t a = *(const uint64_t *)lhs;
  uint64_t b = *(const uint64_t *)rhs;
  return (a > b) - (a < b);
}

static uint64_t percentile_u64(const uint64_t *values, size_t count,
                               size_t percentile) {
  if (count == 0) {
    return 0;
  }
  if (count == 1) {
    return values[0];
  }
  size_t idx = (count - 1) * percentile / 100;
  return values[idx];
}

static void print_result(const BenchResult *result, OutputFormat format,
                         int header) {
  if (format == OUTPUT_TEXT) {
    printf("implementation=%s\n", result->implementation);
    printf("operation=%s\n", result->operation);
    printf("sample_rate=%d\n", result->config.sample_rate);
    printf("channels=%d\n", result->config.channels);
    printf("frame_size=%d\n", result->config.frame_size);
    printf("application=%s\n", result->config.application_name);
    printf("bitrate=%d\n", result->config.bitrate);
    printf("complexity=%d\n", result->config.complexity);
    printf("bitrate_mode=%s\n",
           result->config.bitrate_mode == BITRATE_MODE_VBR
               ? "vbr"
               : result->config.bitrate_mode == BITRATE_MODE_CVBR ? "cvbr"
                                                                  : "cbr");
    printf("frames=%zu\n", result->frames);
    printf("warmup_iters=%zu\n", result->warmup);
    printf("measure_iters=%zu\n", result->measure);
    printf("median_ns_per_frame=%.3f\n", result->median_ns_per_frame);
    printf("p95_ns_per_frame=%.3f\n", result->p95_ns_per_frame);
    printf("median_packets_per_sec=%.3f\n", result->median_packets_per_sec);
    printf("median_realtime_x=%.3f\n", result->median_realtime_x);
    return;
  }

  if (header) {
    printf("implementation,operation,sample_rate,channels,frame_size,application,"
           "bitrate,complexity,bitrate_mode,frames,warmup_iters,measure_iters,"
           "median_ns_per_frame,p95_ns_per_frame,median_packets_per_sec,"
           "median_realtime_x\n");
  }
  printf("%s,%s,%d,%d,%d,%s,%d,%d,%s,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f\n",
         result->implementation, result->operation, result->config.sample_rate,
         result->config.channels, result->config.frame_size,
         result->config.application_name, result->config.bitrate,
         result->config.complexity,
         result->config.bitrate_mode == BITRATE_MODE_VBR
             ? "vbr"
             : result->config.bitrate_mode == BITRATE_MODE_CVBR ? "cvbr" : "cbr",
         result->frames, result->warmup, result->measure,
         result->median_ns_per_frame, result->p95_ns_per_frame,
         result->median_packets_per_sec, result->median_realtime_x);
}

static int run_encode_iteration(const EncodeConfig *config,
                                const opus_int16 *samples, size_t frames,
                                size_t frame_samples, uint64_t *elapsed_ns) {
  OpusEncoder *encoder = NULL;
  if (!create_encoder(config, &encoder)) {
    return 0;
  }

  unsigned char packet[MAX_PACKET_SIZE];
  uint64_t start = now_ns();
  for (size_t frame = 0; frame < frames; ++frame) {
    size_t offset = frame * frame_samples;
    int packet_len = opus_encode(encoder, samples + offset, config->frame_size,
                                 packet, MAX_PACKET_SIZE);
    if (packet_len < 0) {
      fprintf(stderr, "encode failed: %s\n", opus_strerror(packet_len));
      opus_encoder_destroy(encoder);
      return 0;
    }
  }
  *elapsed_ns = now_ns() - start;
  opus_encoder_destroy(encoder);
  return 1;
}

static int benchmark_encode(const EncodeBenchArgs *args, BenchResult *result) {
  opus_int16 *samples = NULL;
  size_t sample_count = 0;
  if (!load_pcm_samples(args->input_path, &samples, &sample_count)) {
    return 0;
  }
  size_t total_frames = frame_count_for_pcm(sample_count, &args->encode);
  if (total_frames == 0) {
    fprintf(stderr, "pcm input does not contain a complete frame\n");
    free(samples);
    return 0;
  }
  size_t frames =
      args->has_max_frames && args->max_frames < total_frames ? args->max_frames
                                                              : total_frames;
  size_t frame_samples =
      (size_t)args->encode.frame_size * (size_t)args->encode.channels;
  uint64_t *measurements =
      (uint64_t *)malloc(args->measure * sizeof(uint64_t));
  if (measurements == NULL) {
    fprintf(stderr, "failed to allocate benchmark measurements\n");
    free(samples);
    return 0;
  }

  uint64_t ignored = 0;
  for (size_t i = 0; i < args->warmup; ++i) {
    if (!run_encode_iteration(&args->encode, samples, frames, frame_samples,
                              &ignored)) {
      free(measurements);
      free(samples);
      return 0;
    }
  }
  for (size_t i = 0; i < args->measure; ++i) {
    if (!run_encode_iteration(&args->encode, samples, frames, frame_samples,
                              &measurements[i])) {
      free(measurements);
      free(samples);
      return 0;
    }
  }

  qsort(measurements, args->measure, sizeof(uint64_t), cmp_u64);
  uint64_t median = percentile_u64(measurements, args->measure, 50);
  uint64_t p95 = percentile_u64(measurements, args->measure, 95);
  double median_ns_per_frame = (double)median / (double)frames;
  double frames_per_sec = 1000000000.0 / median_ns_per_frame;

  result->implementation = "c";
  result->operation = "encode";
  result->config = args->encode;
  result->frames = frames;
  result->warmup = args->warmup;
  result->measure = args->measure;
  result->median_ns_per_frame = median_ns_per_frame;
  result->p95_ns_per_frame = (double)p95 / (double)frames;
  result->median_packets_per_sec = frames_per_sec;
  result->median_realtime_x =
      frames_per_sec * (double)args->encode.frame_size /
      (double)args->encode.sample_rate;

  free(measurements);
  free(samples);
  return 1;
}

static int run_decode_iteration(const PacketCorpus *corpus,
                                const EncodeConfig *config, size_t frames,
                                uint64_t *elapsed_ns) {
  int err = 0;
  OpusDecoder *decoder =
      opus_decoder_create(config->sample_rate, config->channels, &err);
  if (err < 0 || decoder == NULL) {
    fprintf(stderr, "failed to create decoder: %s\n", opus_strerror(err));
    return 0;
  }

  size_t max_frame_size = (size_t)config->frame_size * 6U;
  opus_int16 *pcm =
      (opus_int16 *)malloc(max_frame_size * (size_t)config->channels *
                           sizeof(opus_int16));
  if (pcm == NULL) {
    fprintf(stderr, "failed to allocate decode output buffer\n");
    opus_decoder_destroy(decoder);
    return 0;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < frames; ++i) {
    int decoded = opus_decode(decoder, corpus->packets[i].data,
                              (opus_int32)corpus->packets[i].len, pcm,
                              (int)max_frame_size, 0);
    if (decoded < 0) {
      fprintf(stderr, "decode failed: %s\n", opus_strerror(decoded));
      free(pcm);
      opus_decoder_destroy(decoder);
      return 0;
    }
  }
  *elapsed_ns = now_ns() - start;

  free(pcm);
  opus_decoder_destroy(decoder);
  return 1;
}

static int benchmark_decode(const DecodeBenchArgs *args, BenchResult *result) {
  PacketCorpus corpus;
  if (!load_packet_corpus(args->packet_path, &corpus)) {
    return 0;
  }
  if (corpus.count == 0) {
    fprintf(stderr, "packet corpus is empty\n");
    free_packet_corpus(&corpus);
    return 0;
  }

  EncodeConfig config = default_encode_config();
  config.sample_rate = (int)corpus.header.sample_rate;
  config.channels = (int)corpus.header.channels;
  config.frame_size = (int)corpus.header.frame_size;
  config.application = (int)corpus.header.application;
  config.application_name =
      config.application == OPUS_APPLICATION_VOIP
          ? "voip"
          : config.application == OPUS_APPLICATION_AUDIO ? "audio"
                                                         : "restricted-lowdelay";
  config.bitrate = corpus.header.bitrate;
  config.complexity = (int)corpus.header.complexity;
  config.bitrate_mode = (BitrateMode)corpus.header.bitrate_mode;

  size_t frames =
      args->has_max_frames && args->max_frames < corpus.count ? args->max_frames
                                                              : corpus.count;
  uint64_t *measurements =
      (uint64_t *)malloc(args->measure * sizeof(uint64_t));
  if (measurements == NULL) {
    fprintf(stderr, "failed to allocate benchmark measurements\n");
    free_packet_corpus(&corpus);
    return 0;
  }

  uint64_t ignored = 0;
  for (size_t i = 0; i < args->warmup; ++i) {
    if (!run_decode_iteration(&corpus, &config, frames, &ignored)) {
      free(measurements);
      free_packet_corpus(&corpus);
      return 0;
    }
  }
  for (size_t i = 0; i < args->measure; ++i) {
    if (!run_decode_iteration(&corpus, &config, frames, &measurements[i])) {
      free(measurements);
      free_packet_corpus(&corpus);
      return 0;
    }
  }

  qsort(measurements, args->measure, sizeof(uint64_t), cmp_u64);
  uint64_t median = percentile_u64(measurements, args->measure, 50);
  uint64_t p95 = percentile_u64(measurements, args->measure, 95);
  double median_ns_per_frame = (double)median / (double)frames;
  double frames_per_sec = 1000000000.0 / median_ns_per_frame;

  result->implementation = "c";
  result->operation = "decode";
  result->config = config;
  result->frames = frames;
  result->warmup = args->warmup;
  result->measure = args->measure;
  result->median_ns_per_frame = median_ns_per_frame;
  result->p95_ns_per_frame = (double)p95 / (double)frames;
  result->median_packets_per_sec = frames_per_sec;
  result->median_realtime_x =
      frames_per_sec * (double)config.frame_size / (double)config.sample_rate;

  free(measurements);
  free_packet_corpus(&corpus);
  return 1;
}

static int parse_packets_args(int argc, char **argv, PacketArgs *args) {
  args->encode = default_encode_config();
  args->input_path = NULL;
  args->output_path = NULL;
  args->max_frames = 0;
  args->has_max_frames = 0;

  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      args->input_path = argv[++i];
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      args->output_path = argv[++i];
    } else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.sample_rate)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.channels)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--frame-size") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.frame_size)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--application") == 0 && i + 1 < argc) {
      if (!parse_application(argv[++i], &args->encode.application,
                             &args->encode.application_name)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--bitrate") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.bitrate)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--complexity") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.complexity)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--bitrate-mode") == 0 && i + 1 < argc) {
      if (!parse_bitrate_mode(argv[++i], &args->encode.bitrate_mode)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
      args->has_max_frames = parse_usize(argv[++i], &args->max_frames);
      if (!args->has_max_frames) {
        return 0;
      }
    } else {
      return 0;
    }
  }

  return args->input_path != NULL && args->output_path != NULL;
}

static int parse_encode_args(int argc, char **argv, EncodeBenchArgs *args) {
  args->encode = default_encode_config();
  args->input_path = NULL;
  args->warmup = 3;
  args->measure = 10;
  args->max_frames = 0;
  args->has_max_frames = 0;
  args->format = OUTPUT_TEXT;
  args->header = 1;

  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      args->input_path = argv[++i];
    } else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.sample_rate)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.channels)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--frame-size") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.frame_size)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--application") == 0 && i + 1 < argc) {
      if (!parse_application(argv[++i], &args->encode.application,
                             &args->encode.application_name)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--bitrate") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.bitrate)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--complexity") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &args->encode.complexity)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--bitrate-mode") == 0 && i + 1 < argc) {
      if (!parse_bitrate_mode(argv[++i], &args->encode.bitrate_mode)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      if (!parse_usize(argv[++i], &args->warmup)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
      if (!parse_usize(argv[++i], &args->measure)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
      args->has_max_frames = parse_usize(argv[++i], &args->max_frames);
      if (!args->has_max_frames) {
        return 0;
      }
    } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
      if (!parse_output_format(argv[++i], &args->format)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--no-header") == 0) {
      args->header = 0;
    } else {
      return 0;
    }
  }

  return args->input_path != NULL;
}

static int parse_decode_args(int argc, char **argv, DecodeBenchArgs *args) {
  args->packet_path = NULL;
  args->warmup = 3;
  args->measure = 10;
  args->max_frames = 0;
  args->has_max_frames = 0;
  args->format = OUTPUT_TEXT;
  args->header = 1;

  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--packets") == 0 && i + 1 < argc) {
      args->packet_path = argv[++i];
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      if (!parse_usize(argv[++i], &args->warmup)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
      if (!parse_usize(argv[++i], &args->measure)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
      args->has_max_frames = parse_usize(argv[++i], &args->max_frames);
      if (!args->has_max_frames) {
        return 0;
      }
    } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
      if (!parse_output_format(argv[++i], &args->format)) {
        return 0;
      }
    } else if (strcmp(argv[i], "--no-header") == 0) {
      args->header = 0;
    } else {
      return 0;
    }
  }

  return args->packet_path != NULL;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return EXIT_FAILURE;
  }

  if (strcmp(argv[1], "packets") == 0) {
    PacketArgs args;
    if (!parse_packets_args(argc, argv, &args)) {
      usage();
      return EXIT_FAILURE;
    }
    opus_int16 *samples = NULL;
    size_t sample_count = 0;
    if (!load_pcm_samples(args.input_path, &samples, &sample_count)) {
      return EXIT_FAILURE;
    }
    size_t total_frames = frame_count_for_pcm(sample_count, &args.encode);
    if (total_frames == 0) {
      fprintf(stderr, "pcm input does not contain a complete frame\n");
      free(samples);
      return EXIT_FAILURE;
    }
    size_t frames =
        args.has_max_frames && args.max_frames < total_frames ? args.max_frames
                                                              : total_frames;
    if (!write_corpus_file(args.output_path, &args.encode, samples, frames)) {
      free(samples);
      return EXIT_FAILURE;
    }
    printf("wrote %zu packets to %s\n", frames, args.output_path);
    free(samples);
    return EXIT_SUCCESS;
  }

  if (strcmp(argv[1], "encode") == 0) {
    EncodeBenchArgs args;
    BenchResult result;
    if (!parse_encode_args(argc, argv, &args)) {
      usage();
      return EXIT_FAILURE;
    }
    if (!benchmark_encode(&args, &result)) {
      return EXIT_FAILURE;
    }
    print_result(&result, args.format, args.header);
    return EXIT_SUCCESS;
  }

  if (strcmp(argv[1], "decode") == 0) {
    DecodeBenchArgs args;
    BenchResult result;
    if (!parse_decode_args(argc, argv, &args)) {
      usage();
      return EXIT_FAILURE;
    }
    if (!benchmark_decode(&args, &result)) {
      return EXIT_FAILURE;
    }
    print_result(&result, args.format, args.header);
    return EXIT_SUCCESS;
  }

  usage();
  return EXIT_FAILURE;
}
