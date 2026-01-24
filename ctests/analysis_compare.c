#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "analysis.h"

#define FRAME_SIZE 960
#define SAMPLE_RATE 48000
#define CHANNELS 2
#define DEFAULT_FRAMES 64

static int want_e_history(void) {
  const char *value = getenv("ANALYSIS_TRACE_E_HISTORY");
  if (!value || value[0] == '\0') {
    return 0;
  }
  return strcmp(value, "0") != 0;
}

static int want_angle_bits(void) {
  const char *value = getenv("ANALYSIS_TRACE_ANGLE_BITS");
  if (!value || value[0] == '\0') {
    return 0;
  }
  return strcmp(value, "0") != 0;
}

static int want_band_tonality_bits(void) {
  const char *value = getenv("ANALYSIS_TRACE_BAND_TONALITY_BITS");
  if (!value || value[0] == '\0') {
    return 0;
  }
  return strcmp(value, "0") != 0;
}

static void dump_analysis_info(int frame_idx, const AnalysisInfo *info) {
  printf("analysis_info[%d].valid=%d\n", frame_idx, info->valid);
  printf("analysis_info[%d].tonality=%.9e\n", frame_idx, (double)info->tonality);
  printf("analysis_info[%d].tonality_slope=%.9e\n", frame_idx,
         (double)info->tonality_slope);
  printf("analysis_info[%d].noisiness=%.9e\n", frame_idx,
         (double)info->noisiness);
  printf("analysis_info[%d].activity=%.9e\n", frame_idx,
         (double)info->activity);
  printf("analysis_info[%d].music_prob=%.9e\n", frame_idx,
         (double)info->music_prob);
  printf("analysis_info[%d].music_prob_min=%.9e\n", frame_idx,
         (double)info->music_prob_min);
  printf("analysis_info[%d].music_prob_max=%.9e\n", frame_idx,
         (double)info->music_prob_max);
  printf("analysis_info[%d].bandwidth=%d\n", frame_idx, info->bandwidth);
  printf("analysis_info[%d].activity_probability=%.9e\n", frame_idx,
         (double)info->activity_probability);
  printf("analysis_info[%d].max_pitch_ratio=%.9e\n", frame_idx,
         (double)info->max_pitch_ratio);
  for (int i = 0; i < LEAK_BANDS; ++i) {
    printf("analysis_info[%d].leak_boost[%d]=%u\n", frame_idx, i,
           (unsigned int)info->leak_boost[i]);
  }
}

static void dump_analysis_state(int frame_idx, const TonalityAnalysisState *analysis) {
  int last_e = analysis->E_count == 0 ? NB_FRAMES - 1 : analysis->E_count - 1;

  printf("analysis_state[%d].count=%d\n", frame_idx, analysis->count);
  printf("analysis_state[%d].e_count=%d\n", frame_idx, analysis->E_count);
  printf("analysis_state[%d].analysis_offset=%d\n", frame_idx,
         analysis->analysis_offset);
  printf("analysis_state[%d].mem_fill=%d\n", frame_idx, analysis->mem_fill);
  printf("analysis_state[%d].write_pos=%d\n", frame_idx, analysis->write_pos);
  printf("analysis_state[%d].read_pos=%d\n", frame_idx, analysis->read_pos);
  printf("analysis_state[%d].read_subframe=%d\n", frame_idx,
         analysis->read_subframe);
  printf("analysis_state[%d].hp_ener_accum=%.9e\n", frame_idx,
         (double)analysis->hp_ener_accum);
  printf("analysis_state[%d].e_tracker=%.9e\n", frame_idx,
         (double)analysis->Etracker);
  printf("analysis_state[%d].low_e_count=%.9e\n", frame_idx,
         (double)analysis->lowECount);
  printf("analysis_state[%d].prev_tonality=%.9e\n", frame_idx,
         (double)analysis->prev_tonality);
  printf("analysis_state[%d].prev_bandwidth=%d\n", frame_idx,
         analysis->prev_bandwidth);
  printf("analysis_state[%d].initialized=%d\n", frame_idx,
         analysis->initialized);

  for (int i = 0; i < 3; ++i) {
    printf("analysis_state[%d].downmix_state[%d]=%.9e\n", frame_idx, i,
           (double)analysis->downmix_state[i]);
  }
  for (int i = 0; i < 240; ++i) {
    printf("analysis_state[%d].angle[%d]=%.9e\n", frame_idx, i,
           (double)analysis->angle[i]);
  }
  for (int i = 0; i < 240; ++i) {
    printf("analysis_state[%d].d_angle[%d]=%.9e\n", frame_idx, i,
           (double)analysis->d_angle[i]);
  }
  for (int i = 0; i < 240; ++i) {
    printf("analysis_state[%d].d2_angle[%d]=%.9e\n", frame_idx, i,
           (double)analysis->d2_angle[i]);
  }
  if (want_angle_bits()) {
    for (int i = 0; i < 240; ++i) {
      union {
        float f;
        uint32_t u;
      } bits;
      bits.f = analysis->angle[i];
      printf("analysis_state[%d].angle_bits[%d]=0x%08x\n", frame_idx, i,
             (unsigned int)bits.u);
    }
    for (int i = 0; i < 240; ++i) {
      union {
        float f;
        uint32_t u;
      } bits;
      bits.f = analysis->d_angle[i];
      printf("analysis_state[%d].d_angle_bits[%d]=0x%08x\n", frame_idx, i,
             (unsigned int)bits.u);
    }
    for (int i = 0; i < 240; ++i) {
      union {
        float f;
        uint32_t u;
      } bits;
      bits.f = analysis->d2_angle[i];
      printf("analysis_state[%d].d2_angle_bits[%d]=0x%08x\n", frame_idx, i,
             (unsigned int)bits.u);
    }
  }
  for (int i = 0; i < NB_TBANDS; ++i) {
    printf("analysis_state[%d].E_last[%d]=%.9e\n", frame_idx, i,
           (double)analysis->E[last_e][i]);
  }
  if (want_e_history()) {
    for (int t = 0; t < NB_FRAMES; ++t) {
      for (int i = 0; i < NB_TBANDS; ++i) {
        union {
          float f;
          uint32_t u;
        } bits;
        bits.f = analysis->E[t][i];
        printf("analysis_state[%d].E_hist_bits[%d][%d]=0x%08x\n", frame_idx, t,
               i, (unsigned int)bits.u);
      }
    }
  }
  for (int i = 0; i < NB_TBANDS; ++i) {
    printf("analysis_state[%d].logE_last[%d]=%.9e\n", frame_idx, i,
           (double)analysis->logE[last_e][i]);
  }
  for (int i = 0; i < NB_TBANDS; ++i) {
    printf("analysis_state[%d].lowE[%d]=%.9e\n", frame_idx, i,
           (double)analysis->lowE[i]);
  }
  for (int i = 0; i < NB_TBANDS; ++i) {
    printf("analysis_state[%d].highE[%d]=%.9e\n", frame_idx, i,
           (double)analysis->highE[i]);
  }
  for (int i = 0; i < NB_TBANDS + 1; ++i) {
    printf("analysis_state[%d].meanE[%d]=%.9e\n", frame_idx, i,
           (double)analysis->meanE[i]);
  }
  for (int i = 0; i < NB_TBANDS; ++i) {
    printf("analysis_state[%d].prev_band_tonality[%d]=%.9e\n", frame_idx, i,
           (double)analysis->prev_band_tonality[i]);
  }
  if (want_band_tonality_bits()) {
    for (int i = 0; i < NB_TBANDS; ++i) {
      union {
        float f;
        uint32_t u;
      } bits;
      bits.f = analysis->prev_band_tonality[i];
      printf("analysis_state[%d].prev_band_tonality_bits[%d]=0x%08x\n",
             frame_idx, i, (unsigned int)bits.u);
    }
  }
  for (int i = 0; i < 32; ++i) {
    printf("analysis_state[%d].mem[%d]=%.9e\n", frame_idx, i,
           (double)analysis->mem[i]);
  }
  for (int i = 0; i < 8; ++i) {
    printf("analysis_state[%d].cmean[%d]=%.9e\n", frame_idx, i,
           (double)analysis->cmean[i]);
  }
  for (int i = 0; i < 9; ++i) {
    printf("analysis_state[%d].std[%d]=%.9e\n", frame_idx, i,
           (double)analysis->std[i]);
  }
}

static int parse_frame_limit(const char *arg, int *out_frames) {
  char *end = NULL;
  long value = strtol(arg, &end, 10);
  if (end == arg || *end != '\0' || value <= 0 || value > 1000000) {
    return 0;
  }
  *out_frames = (int)value;
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: analysis_compare <input.pcm> [frames]\n");
    return EXIT_FAILURE;
  }

  const char *input_path = argv[1];
  int max_frames = DEFAULT_FRAMES;
  if (argc == 3 && !parse_frame_limit(argv[2], &max_frames)) {
    fprintf(stderr, "invalid frame count: %s\n", argv[2]);
    return EXIT_FAILURE;
  }

  FILE *fin = fopen(input_path, "rb");
  if (fin == NULL) {
    fprintf(stderr, "failed to open input file: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  int err = 0;
  OpusCustomMode *mode =
      opus_custom_mode_create(SAMPLE_RATE, FRAME_SIZE, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "opus_custom_mode_create failed: %d\n", err);
    fclose(fin);
    return EXIT_FAILURE;
  }

  TonalityAnalysisState analysis;
  tonality_analysis_init(&analysis, SAMPLE_RATE);

  opus_int16 pcm[FRAME_SIZE * CHANNELS];
  int frame_idx = 0;
  while (frame_idx < max_frames) {
    size_t samples =
        fread(pcm, sizeof(opus_int16) * CHANNELS, FRAME_SIZE, fin);
    if (samples != FRAME_SIZE) {
      break;
    }

    AnalysisInfo info;
    memset(&info, 0, sizeof(info));
    info.valid = 0;
    run_analysis(&analysis, mode, pcm, FRAME_SIZE, FRAME_SIZE, 0, -2, CHANNELS,
                 SAMPLE_RATE, 16, downmix_int, &info);
    dump_analysis_info(frame_idx, &info);
    dump_analysis_state(frame_idx, &analysis);
    frame_idx++;
  }

#ifdef CUSTOM_MODES
  opus_custom_mode_destroy(mode);
#endif
  fclose(fin);
  return EXIT_SUCCESS;
}
