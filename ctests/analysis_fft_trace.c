#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "analysis.h"
#include "float_cast.h"
#include "kiss_fft.h"
#include "mathops.h"
#include "modes.h"

#define FRAME_SIZE 960
#define SAMPLE_RATE 48000
#define CHANNELS 2
#define DEFAULT_FRAMES 64
#define INITIAL_MEM_FILL 240
#define SCALE_ENER(e) ((1.f / 32768 / 32768) * (e))

static const float analysis_window[240] = {
    0.000043f, 0.000171f, 0.000385f, 0.000685f, 0.001071f, 0.001541f, 0.002098f, 0.002739f,
    0.003466f, 0.004278f, 0.005174f, 0.006156f, 0.007222f, 0.008373f, 0.009607f, 0.010926f,
    0.012329f, 0.013815f, 0.015385f, 0.017037f, 0.018772f, 0.020590f, 0.022490f, 0.024472f,
    0.026535f, 0.028679f, 0.030904f, 0.033210f, 0.035595f, 0.038060f, 0.040604f, 0.043227f,
    0.045928f, 0.048707f, 0.051564f, 0.054497f, 0.057506f, 0.060591f, 0.063752f, 0.066987f,
    0.070297f, 0.073680f, 0.077136f, 0.080665f, 0.084265f, 0.087937f, 0.091679f, 0.095492f,
    0.099373f, 0.103323f, 0.107342f, 0.111427f, 0.115579f, 0.119797f, 0.124080f, 0.128428f,
    0.132839f, 0.137313f, 0.141849f, 0.146447f, 0.151105f, 0.155823f, 0.160600f, 0.165435f,
    0.170327f, 0.175276f, 0.180280f, 0.185340f, 0.190453f, 0.195619f, 0.200838f, 0.206107f,
    0.211427f, 0.216797f, 0.222215f, 0.227680f, 0.233193f, 0.238751f, 0.244353f, 0.250000f,
    0.255689f, 0.261421f, 0.267193f, 0.273005f, 0.278856f, 0.284744f, 0.290670f, 0.296632f,
    0.302628f, 0.308658f, 0.314721f, 0.320816f, 0.326941f, 0.333097f, 0.339280f, 0.345492f,
    0.351729f, 0.357992f, 0.364280f, 0.370590f, 0.376923f, 0.383277f, 0.389651f, 0.396044f,
    0.402455f, 0.408882f, 0.415325f, 0.421783f, 0.428254f, 0.434737f, 0.441231f, 0.447736f,
    0.454249f, 0.460770f, 0.467298f, 0.473832f, 0.480370f, 0.486912f, 0.493455f, 0.500000f,
    0.506545f, 0.513088f, 0.519630f, 0.526168f, 0.532702f, 0.539230f, 0.545751f, 0.552264f,
    0.558769f, 0.565263f, 0.571746f, 0.578217f, 0.584675f, 0.591118f, 0.597545f, 0.603956f,
    0.610349f, 0.616723f, 0.623077f, 0.629410f, 0.635720f, 0.642008f, 0.648271f, 0.654508f,
    0.660720f, 0.666903f, 0.673059f, 0.679184f, 0.685279f, 0.691342f, 0.697372f, 0.703368f,
    0.709330f, 0.715256f, 0.721144f, 0.726995f, 0.732807f, 0.738579f, 0.744311f, 0.750000f,
    0.755647f, 0.761249f, 0.766807f, 0.772320f, 0.777785f, 0.783203f, 0.788573f, 0.793893f,
    0.799162f, 0.804381f, 0.809547f, 0.814660f, 0.819720f, 0.824724f, 0.829673f, 0.834565f,
    0.839400f, 0.844177f, 0.848895f, 0.853553f, 0.858151f, 0.862687f, 0.867161f, 0.871572f,
    0.875920f, 0.880203f, 0.884421f, 0.888573f, 0.892658f, 0.896677f, 0.900627f, 0.904508f,
    0.908321f, 0.912063f, 0.915735f, 0.919335f, 0.922864f, 0.926320f, 0.929703f, 0.933013f,
    0.936248f, 0.939409f, 0.942494f, 0.945503f, 0.948436f, 0.951293f, 0.954072f, 0.956773f,
    0.959396f, 0.961940f, 0.964405f, 0.966790f, 0.969096f, 0.971321f, 0.973465f, 0.975528f,
    0.977510f, 0.979410f, 0.981228f, 0.982963f, 0.984615f, 0.986185f, 0.987671f, 0.989074f,
    0.990393f, 0.991627f, 0.992778f, 0.993844f, 0.994826f, 0.995722f, 0.996534f, 0.997261f,
    0.997902f, 0.998459f, 0.998929f, 0.999315f, 0.999615f, 0.999829f, 0.999957f, 1.000000f,
};

static const int tbands[NB_TBANDS + 1] = {
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
};

static opus_val32 silk_resampler_down2_hp(opus_val32 *S, opus_val32 *out,
                                          const opus_val32 *in, int inLen) {
  int k;
  int len2 = inLen / 2;
  opus_val32 in32;
  opus_val32 out32;
  opus_val32 out32_hp;
  opus_val32 Y;
  opus_val32 X;
  opus_val64 hp_ener = 0;

  for (k = 0; k < len2; k++) {
    in32 = in[2 * k];

    Y = SUB32(in32, S[0]);
    X = MULT16_32_Q15(QCONST16(0.6074371f, 15), Y);
    out32 = ADD32(S[0], X);
    S[0] = ADD32(in32, X);
    out32_hp = out32;

    in32 = in[2 * k + 1];

    Y = SUB32(in32, S[1]);
    X = MULT16_32_Q15(QCONST16(0.15063f, 15), Y);
    out32 = ADD32(out32, S[1]);
    out32 = ADD32(out32, X);
    S[1] = ADD32(in32, X);

    Y = SUB32(-in32, S[2]);
    X = MULT16_32_Q15(QCONST16(0.15063f, 15), Y);
    out32_hp = ADD32(out32_hp, S[2]);
    out32_hp = ADD32(out32_hp, X);
    S[2] = ADD32(-in32, X);

    hp_ener += SHR64(out32_hp * (opus_val64)out32_hp, 8);
    out[k] = HALF32(out32);
  }
#ifdef FIXED_POINT
  hp_ener = hp_ener >> (2 * SIG_SHIFT);
  if (hp_ener > 2147483647) {
    hp_ener = 2147483647;
  }
#endif
  return (opus_val32)hp_ener;
}

static opus_val32 downmix_and_resample(downmix_func downmix, const void *_x,
                                       opus_val32 *y, opus_val32 S[3],
                                       int subframe, int offset, int c1,
                                       int c2, int C, int Fs) {
  int j;
  opus_val32 ret = 0;

  if (subframe == 0) {
    return 0;
  }
  if (Fs == 48000) {
    subframe *= 2;
    offset *= 2;
  } else if (Fs == 16000) {
    subframe = subframe * 2 / 3;
    offset = offset * 2 / 3;
  }

  opus_val32 *tmp = (opus_val32 *)malloc(sizeof(opus_val32) * subframe);
  if (tmp == NULL) {
    fprintf(stderr, "downmix_and_resample: out of memory\n");
    exit(EXIT_FAILURE);
  }

  downmix(_x, tmp, subframe, offset, c1, c2, C);
  if ((c2 == -2 && C == 2) || c2 > -1) {
    for (j = 0; j < subframe; j++) {
      tmp[j] = HALF32(tmp[j]);
    }
  }
  if (Fs == 48000) {
    ret = silk_resampler_down2_hp(S, y, tmp, subframe);
  } else if (Fs == 24000) {
    OPUS_COPY(y, tmp, subframe);
  } else if (Fs == 16000) {
    opus_val32 *tmp3x =
        (opus_val32 *)malloc(sizeof(opus_val32) * 3 * subframe);
    if (tmp3x == NULL) {
      fprintf(stderr, "downmix_and_resample: out of memory\n");
      free(tmp);
      exit(EXIT_FAILURE);
    }
    for (j = 0; j < subframe; j++) {
      tmp3x[3 * j] = tmp[j];
      tmp3x[3 * j + 1] = tmp[j];
      tmp3x[3 * j + 2] = tmp[j];
    }
    silk_resampler_down2_hp(S, y, tmp3x, 3 * subframe);
    free(tmp3x);
  }
  free(tmp);
#ifndef FIXED_POINT
  ret *= 1.f / 32768 / 32768;
#endif
  return ret;
}

typedef struct {
  int bins[480];
  int bin_count;
  int all_bins;
} TraceConfig;

typedef struct {
  int bands[NB_TBANDS];
  int band_count;
  int all_bands;
  int enabled;
  int has_frame;
  int frame;
  int want_bits;
} BandTraceConfig;

static int env_truthy(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return 0;
  }
  return strcmp(value, "0") != 0;
}

static int bin28_trace_enabled(void) {
  return env_truthy("ANALYSIS_TRACE_BIN28");
}

static int bin28_trace_frame_matches(int frame_idx) {
  const char *value = getenv("ANALYSIS_TRACE_BIN28_FRAME");
  if (value == NULL || value[0] == '\0') {
    return 1;
  }
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return 1;
  }
  return frame_idx == (int)parsed;
}

static int fft_trace_bits_enabled(void) {
  return env_truthy("ANALYSIS_TRACE_FFT_BITS");
}

static void init_trace_config(TraceConfig *cfg) {
  const char *env = getenv("ANALYSIS_TRACE_BINS");
  cfg->bin_count = 0;
  cfg->all_bins = 0;
  if (env == NULL || env[0] == '\0') {
    cfg->bins[cfg->bin_count++] = 1;
    cfg->bins[cfg->bin_count++] = 61;
    return;
  }
  if (strcmp(env, "all") == 0) {
    cfg->all_bins = 1;
    return;
  }
  const char *p = env;
  while (*p != '\0') {
    char *end = NULL;
    long value = strtol(p, &end, 10);
    if (end == p) {
      break;
    }
    if (value > 0 && value < 480 && cfg->bin_count < 480) {
      cfg->bins[cfg->bin_count++] = (int)value;
    }
    p = end;
    while (*p == ',' || *p == ' ' || *p == '\t') {
      p++;
    }
  }
  if (cfg->bin_count == 0 && !cfg->all_bins) {
    cfg->bins[cfg->bin_count++] = 1;
    cfg->bins[cfg->bin_count++] = 61;
  }
}

static int should_trace_bin(const TraceConfig *cfg, int bin) {
  if (cfg->all_bins) {
    return 1;
  }
  for (int i = 0; i < cfg->bin_count; ++i) {
    if (cfg->bins[i] == bin) {
      return 1;
    }
  }
  return 0;
}

static void init_band_trace_config(BandTraceConfig *cfg) {
  cfg->enabled = env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE")
      || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BANDS")
      || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_FRAME")
      || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BITS");
  cfg->band_count = 0;
  cfg->all_bands = 0;
  cfg->has_frame = 0;
  cfg->frame = 0;
  cfg->want_bits = env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BITS");
  if (!cfg->enabled) {
    return;
  }

  const char *frame_env = getenv("ANALYSIS_TRACE_TONALITY_SLOPE_FRAME");
  if (frame_env != NULL && frame_env[0] != '\0') {
    char *end = NULL;
    long value = strtol(frame_env, &end, 10);
    if (end != frame_env && *end == '\0' && value >= 0 && value < 1000000) {
      cfg->has_frame = 1;
      cfg->frame = (int)value;
    }
  }

  const char *bands_env = getenv("ANALYSIS_TRACE_TONALITY_SLOPE_BANDS");
  if (bands_env == NULL || bands_env[0] == '\0') {
    cfg->all_bands = 1;
    return;
  }
  if (strcmp(bands_env, "all") == 0) {
    cfg->all_bands = 1;
    return;
  }
  const char *p = bands_env;
  while (*p != '\0') {
    char *end = NULL;
    long value = strtol(p, &end, 10);
    if (end == p) {
      break;
    }
    if (value >= 0 && value < NB_TBANDS && cfg->band_count < NB_TBANDS) {
      cfg->bands[cfg->band_count++] = (int)value;
    }
    p = end;
    while (*p == ',' || *p == ' ' || *p == '\t') {
      p++;
    }
  }
  if (cfg->band_count == 0 && !cfg->all_bands) {
    cfg->all_bands = 1;
  }
}

static int should_trace_band(const BandTraceConfig *cfg, int band) {
  if (!cfg->enabled) {
    return 0;
  }
  if (cfg->all_bands) {
    return 1;
  }
  for (int i = 0; i < cfg->band_count; ++i) {
    if (cfg->bands[i] == band) {
      return 1;
    }
  }
  return 0;
}

static void dump_fft_trace(int frame_idx, int bin, const kiss_fft_cpx *in,
                           const kiss_fft_cpx *out, float x1r, float x1i,
                           float x2r, float x2i, float atan1, float atan2,
                           float angle, float angle2, float d_angle,
                           float d2_angle, float d_angle2, float d2_angle2,
                           int d2_angle_int, int d2_angle2_int, float bin_e,
                           float tonality, float tonality2) {
  int mirror = 480 - bin;
  union {
    float f;
    uint32_t u;
  } bits;
  printf("analysis_fft[%d].bin[%d].input_fft.r=%.9e\n", frame_idx, bin,
         (double)in[bin].r);
  printf("analysis_fft[%d].bin[%d].input_fft.i=%.9e\n", frame_idx, bin,
         (double)in[bin].i);
  printf("analysis_fft[%d].bin[%d].input_fft_mirror.r=%.9e\n", frame_idx, bin,
         (double)in[mirror].r);
  printf("analysis_fft[%d].bin[%d].input_fft_mirror.i=%.9e\n", frame_idx, bin,
         (double)in[mirror].i);
  printf("analysis_fft[%d].bin[%d].output_fft.r=%.9e\n", frame_idx, bin,
         (double)out[bin].r);
  printf("analysis_fft[%d].bin[%d].output_fft.i=%.9e\n", frame_idx, bin,
         (double)out[bin].i);
  printf("analysis_fft[%d].bin[%d].output_fft_mirror.r=%.9e\n", frame_idx, bin,
         (double)out[mirror].r);
  printf("analysis_fft[%d].bin[%d].output_fft_mirror.i=%.9e\n", frame_idx, bin,
         (double)out[mirror].i);
  if (fft_trace_bits_enabled()) {
    bits.f = in[bin].r;
    printf("analysis_fft[%d].bin[%d].input_fft_bits.r=0x%08x\n", frame_idx, bin,
           (unsigned int)bits.u);
    bits.f = in[bin].i;
    printf("analysis_fft[%d].bin[%d].input_fft_bits.i=0x%08x\n", frame_idx, bin,
           (unsigned int)bits.u);
    bits.f = in[mirror].r;
    printf("analysis_fft[%d].bin[%d].input_fft_mirror_bits.r=0x%08x\n",
           frame_idx, bin, (unsigned int)bits.u);
    bits.f = in[mirror].i;
    printf("analysis_fft[%d].bin[%d].input_fft_mirror_bits.i=0x%08x\n",
           frame_idx, bin, (unsigned int)bits.u);
    bits.f = out[bin].r;
    printf("analysis_fft[%d].bin[%d].output_fft_bits.r=0x%08x\n", frame_idx,
           bin, (unsigned int)bits.u);
    bits.f = out[bin].i;
    printf("analysis_fft[%d].bin[%d].output_fft_bits.i=0x%08x\n", frame_idx,
           bin, (unsigned int)bits.u);
    bits.f = out[mirror].r;
    printf("analysis_fft[%d].bin[%d].output_fft_mirror_bits.r=0x%08x\n",
           frame_idx, bin, (unsigned int)bits.u);
    bits.f = out[mirror].i;
    printf("analysis_fft[%d].bin[%d].output_fft_mirror_bits.i=0x%08x\n",
           frame_idx, bin, (unsigned int)bits.u);
    {
      float r2 = out[bin].r * out[bin].r;
      float mr2 = out[mirror].r * out[mirror].r;
      float i2 = out[bin].i * out[bin].i;
      float mi2 = out[mirror].i * out[mirror].i;
      float sum0 = r2 + mr2;
      float sum1 = i2 + mi2;
      float sum = sum0 + sum1;
      float scaled = SCALE_ENER(sum);
      printf("analysis_fft[%d].bin[%d].bin_e_r2=%.9e\n", frame_idx, bin,
             (double)r2);
      printf("analysis_fft[%d].bin[%d].bin_e_mr2=%.9e\n", frame_idx, bin,
             (double)mr2);
      printf("analysis_fft[%d].bin[%d].bin_e_i2=%.9e\n", frame_idx, bin,
             (double)i2);
      printf("analysis_fft[%d].bin[%d].bin_e_mi2=%.9e\n", frame_idx, bin,
             (double)mi2);
      printf("analysis_fft[%d].bin[%d].bin_e_sum0=%.9e\n", frame_idx, bin,
             (double)sum0);
      printf("analysis_fft[%d].bin[%d].bin_e_sum1=%.9e\n", frame_idx, bin,
             (double)sum1);
      printf("analysis_fft[%d].bin[%d].bin_e_sum=%.9e\n", frame_idx, bin,
             (double)sum);
      printf("analysis_fft[%d].bin[%d].bin_e_scaled=%.9e\n", frame_idx, bin,
             (double)scaled);
      bits.f = r2;
      printf("analysis_fft[%d].bin[%d].bin_e_r2_bits=0x%08x\n", frame_idx, bin,
             (unsigned int)bits.u);
      bits.f = mr2;
      printf("analysis_fft[%d].bin[%d].bin_e_mr2_bits=0x%08x\n", frame_idx, bin,
             (unsigned int)bits.u);
      bits.f = i2;
      printf("analysis_fft[%d].bin[%d].bin_e_i2_bits=0x%08x\n", frame_idx, bin,
             (unsigned int)bits.u);
      bits.f = mi2;
      printf("analysis_fft[%d].bin[%d].bin_e_mi2_bits=0x%08x\n", frame_idx, bin,
             (unsigned int)bits.u);
      bits.f = sum0;
      printf("analysis_fft[%d].bin[%d].bin_e_sum0_bits=0x%08x\n", frame_idx,
             bin, (unsigned int)bits.u);
      bits.f = sum1;
      printf("analysis_fft[%d].bin[%d].bin_e_sum1_bits=0x%08x\n", frame_idx,
             bin, (unsigned int)bits.u);
      bits.f = sum;
      printf("analysis_fft[%d].bin[%d].bin_e_sum_bits=0x%08x\n", frame_idx, bin,
             (unsigned int)bits.u);
      bits.f = scaled;
      printf("analysis_fft[%d].bin[%d].bin_e_scaled_bits=0x%08x\n", frame_idx,
             bin, (unsigned int)bits.u);
    }
  }
  printf("analysis_fft[%d].bin[%d].x1r=%.9e\n", frame_idx, bin, (double)x1r);
  printf("analysis_fft[%d].bin[%d].x1i=%.9e\n", frame_idx, bin, (double)x1i);
  printf("analysis_fft[%d].bin[%d].x2r=%.9e\n", frame_idx, bin, (double)x2r);
  printf("analysis_fft[%d].bin[%d].x2i=%.9e\n", frame_idx, bin, (double)x2i);
  printf("analysis_fft[%d].bin[%d].fast_atan2_x1=%.9e\n", frame_idx, bin,
         (double)atan1);
  printf("analysis_fft[%d].bin[%d].fast_atan2_x2=%.9e\n", frame_idx, bin,
         (double)atan2);
  printf("analysis_fft[%d].bin[%d].angle=%.9e\n", frame_idx, bin,
         (double)angle);
  printf("analysis_fft[%d].bin[%d].angle2=%.9e\n", frame_idx, bin,
         (double)angle2);
  printf("analysis_fft[%d].bin[%d].d_angle=%.9e\n", frame_idx, bin,
         (double)d_angle);
  printf("analysis_fft[%d].bin[%d].d2_angle=%.9e\n", frame_idx, bin,
         (double)d2_angle);
  printf("analysis_fft[%d].bin[%d].d_angle2=%.9e\n", frame_idx, bin,
         (double)d_angle2);
  printf("analysis_fft[%d].bin[%d].d2_angle2=%.9e\n", frame_idx, bin,
         (double)d2_angle2);
  printf("analysis_fft[%d].bin[%d].float2int_d2_angle=%d\n", frame_idx, bin,
         d2_angle_int);
  printf("analysis_fft[%d].bin[%d].float2int_d2_angle2=%d\n", frame_idx, bin,
         d2_angle2_int);
  printf("analysis_fft[%d].bin[%d].bin_e=%.9e\n", frame_idx, bin,
         (double)bin_e);
  bits.f = bin_e;
  printf("analysis_fft[%d].bin[%d].bin_e_bits=0x%08x\n", frame_idx, bin,
         (unsigned int)bits.u);
  printf("analysis_fft[%d].bin[%d].tonality=%.9e\n", frame_idx, bin,
         (double)tonality);
  bits.f = tonality;
  printf("analysis_fft[%d].bin[%d].tonality_bits=0x%08x\n", frame_idx, bin,
         (unsigned int)bits.u);
  printf("analysis_fft[%d].bin[%d].tonality2=%.9e\n", frame_idx, bin,
         (double)tonality2);
  bits.f = tonality2;
  printf("analysis_fft[%d].bin[%d].tonality2_bits=0x%08x\n", frame_idx, bin,
         (unsigned int)bits.u);
}

static void trace_fft_for_frame(const TonalityAnalysisState *analysis,
                                const OpusCustomMode *mode,
                                const opus_int16 *pcm, int frame_size,
                                int lsb_depth, const TraceConfig *cfg,
                                const BandTraceConfig *band_cfg,
                                int frame_idx) {
  TonalityAnalysisState tonal = *analysis;
  int len = frame_size;
  int offset = 0;
  int N = 480;
  int N2 = 240;

  if (!tonal.initialized) {
    tonal.mem_fill = INITIAL_MEM_FILL;
    tonal.initialized = 1;
  }

  if (tonal.Fs == 48000) {
    len /= 2;
    offset /= 2;
  } else if (tonal.Fs == 16000) {
    len = 3 * len / 2;
    offset = 3 * offset / 2;
  }

  tonal.hp_ener_accum +=
      (float)downmix_and_resample(downmix_int, pcm,
                                  &tonal.inmem[tonal.mem_fill],
                                  tonal.downmix_state,
                                  IMIN(len, ANALYSIS_BUF_SIZE - tonal.mem_fill),
                                  offset, 0, -2, CHANNELS, tonal.Fs);

  if (tonal.mem_fill + len < ANALYSIS_BUF_SIZE) {
    return;
  }

  int is_silence =
      is_digital_silence(tonal.inmem, ANALYSIS_BUF_SIZE, 1, lsb_depth);
  if (is_silence) {
    return;
  }

  kiss_fft_cpx input_fft[480];
  kiss_fft_cpx output_fft[480];
  float tonality_arr[240] = {0};
  float tonality2_arr[240] = {0};
  for (int i = 0; i < N2; ++i) {
    float w = analysis_window[i];
    input_fft[i].r = (kiss_fft_scalar)(w * tonal.inmem[i]);
    input_fft[i].i = (kiss_fft_scalar)(w * tonal.inmem[N2 + i]);
    input_fft[N - i - 1].r = (kiss_fft_scalar)(w * tonal.inmem[N - i - 1]);
    input_fft[N - i - 1].i = (kiss_fft_scalar)(w * tonal.inmem[N + N2 - i - 1]);
  }

  opus_fft(mode->mdct.kfft[0], input_fft, output_fft, tonal.arch);
  if (output_fft[0].r != output_fft[0].r) {
    return;
  }

  for (int i = 1; i < N2; ++i) {
    float x1r = (float)output_fft[i].r + output_fft[N - i].r;
    float x1i = (float)output_fft[i].i - output_fft[N - i].i;
    float x2r = (float)output_fft[i].i + output_fft[N - i].i;
    float x2i = (float)output_fft[N - i].r - output_fft[i].r;

    float atan1 = fast_atan2f(x1i, x1r);
    float angle = 0.5f / PI * atan1;
    float d_angle = angle - tonal.angle[i];
    float d2_angle = d_angle - tonal.d_angle[i];

    float atan2 = fast_atan2f(x2i, x2r);
    float angle2 = 0.5f / PI * atan2;
    float d_angle2 = angle2 - angle;
    float d2_angle2 = d_angle2 - d_angle;

    if ((i == 28 || i == 29) && bin28_trace_enabled()
        && bin28_trace_frame_matches(frame_idx)) {
      printf("analysis_bin%02d[%d].x2r=%.9e\n", i, frame_idx, (double)x2r);
      printf("analysis_bin%02d[%d].x2i=%.9e\n", i, frame_idx, (double)x2i);
      printf("analysis_bin%02d[%d].fast_atan2=%.9e\n", i, frame_idx, (double)atan2);
      printf("analysis_bin%02d[%d].angle2=%.9e\n", i, frame_idx, (double)angle2);
      printf("analysis_bin%02d[%d].d2_angle2=%.9e\n", i, frame_idx, (double)d2_angle2);
    }

    int d2_angle_int = float2int(d2_angle);
    int d2_angle2_int = float2int(d2_angle2);
    float mod1 = d2_angle - d2_angle_int;
    mod1 *= mod1;
    mod1 *= mod1;
    float mod2 = d2_angle2 - d2_angle2_int;
    mod2 *= mod2;
    mod2 *= mod2;
    float pi4 = PI * PI;
    pi4 *= pi4;
    float avg_mod = 0.25f * (tonal.d2_angle[i] + mod1 + 2.f * mod2);
    float tonality = 1.f / (1.f + 40.f * 16.f * pi4 * avg_mod) - .015f;
    float tonality2 = 1.f / (1.f + 40.f * 16.f * pi4 * mod2) - .015f;
    float bin_e = (float)output_fft[i].r * output_fft[i].r
        + output_fft[N - i].r * output_fft[N - i].r
        + output_fft[i].i * output_fft[i].i
        + output_fft[N - i].i * output_fft[N - i].i;
    bin_e = SCALE_ENER(bin_e);
    tonality_arr[i] = tonality;
    tonality2_arr[i] = tonality2;

    if (should_trace_bin(cfg, i)) {
      dump_fft_trace(frame_idx, i, input_fft, output_fft, x1r, x1i, x2r, x2i,
                     atan1, atan2, angle, angle2, d_angle, d2_angle, d_angle2,
                     d2_angle2, d2_angle_int, d2_angle2_int, bin_e, tonality,
                     tonality2);
    }
  }

  for (int i = 2; i < N2 - 1; ++i) {
    float tt = MIN32(tonality2_arr[i],
                     MAX32(tonality2_arr[i - 1], tonality2_arr[i + 1]));
    tonality_arr[i] = .9f * MAX32(tonality_arr[i], tt - .1f);
  }
  for (int i = 1; i < N2; ++i) {
    if (!should_trace_bin(cfg, i)) {
      continue;
    }
    union {
      float f;
      uint32_t u;
    } bits;
    printf("analysis_fft[%d].bin[%d].tonality_smoothed=%.9e\n", frame_idx, i,
           (double)tonality_arr[i]);
    bits.f = tonality_arr[i];
    printf("analysis_fft[%d].bin[%d].tonality_smoothed_bits=0x%08x\n",
           frame_idx, i, (unsigned int)bits.u);
  }

  if (band_cfg->enabled
      && (!band_cfg->has_frame || band_cfg->frame == frame_idx)) {
    float slope = 0.0f;
    for (int b = 0; b < NB_TBANDS; ++b) {
      float E = 0.0f;
      float tE = 0.0f;
      for (int i = tbands[b]; i < tbands[b + 1]; ++i) {
        float binE = (float)output_fft[i].r * output_fft[i].r
            + output_fft[N - i].r * output_fft[N - i].r
            + output_fft[i].i * output_fft[i].i
            + output_fft[N - i].i * output_fft[N - i].i;
        binE = SCALE_ENER(binE);
        E += binE;
        tE += binE * MAX32(0, tonality_arr[i]);
      }

      float L1 = 0.0f;
      float L2 = 0.0f;
      for (int i = 0; i < NB_FRAMES; ++i) {
        float bandE = tonal.E[i][b];
        if (i == tonal.E_count) {
          bandE = E;
        }
        L1 += (float)sqrt(bandE);
        L2 += bandE;
      }
      float stationarity =
          MIN16(0.99f, L1 / (float)sqrt(1e-15f + NB_FRAMES * L2));
      stationarity *= stationarity;
      stationarity *= stationarity;

      float energy_ratio = tE / (1e-15f + E);
      float stationarity_term = stationarity * tonal.prev_band_tonality[b];
      float band_tonality = MAX16(energy_ratio, stationarity_term);
      slope += band_tonality * (b - 8);

      if (should_trace_band(band_cfg, b)) {
        union {
          float f;
          uint32_t u;
        } bits;
        printf("analysis_tonality[%d].band[%d].E=%.9e\n", frame_idx, b,
               (double)E);
        printf("analysis_tonality[%d].band[%d].tE=%.9e\n", frame_idx, b,
               (double)tE);
        printf("analysis_tonality[%d].band[%d].energy_ratio=%.9e\n",
               frame_idx, b, (double)energy_ratio);
        printf("analysis_tonality[%d].band[%d].stationarity=%.9e\n",
               frame_idx, b, (double)stationarity);
        printf("analysis_tonality[%d].band[%d].stationarity_term=%.9e\n",
               frame_idx, b, (double)stationarity_term);
        printf("analysis_tonality[%d].band[%d].prev_band_tonality=%.9e\n",
               frame_idx, b, (double)tonal.prev_band_tonality[b]);
        printf("analysis_tonality[%d].band[%d].band_tonality=%.9e\n",
               frame_idx, b, (double)band_tonality);
        printf("analysis_tonality[%d].band[%d].slope_acc=%.9e\n", frame_idx,
               b, (double)slope);
        if (band_cfg->want_bits) {
          bits.f = E;
          printf("analysis_tonality[%d].band[%d].E_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = tE;
          printf("analysis_tonality[%d].band[%d].tE_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = energy_ratio;
          printf("analysis_tonality[%d].band[%d].energy_ratio_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = stationarity;
          printf("analysis_tonality[%d].band[%d].stationarity_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = stationarity_term;
          printf("analysis_tonality[%d].band[%d].stationarity_term_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = tonal.prev_band_tonality[b];
          printf("analysis_tonality[%d].band[%d].prev_band_tonality_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = band_tonality;
          printf("analysis_tonality[%d].band[%d].band_tonality_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
          bits.f = slope;
          printf("analysis_tonality[%d].band[%d].slope_acc_bits=0x%08x\n",
                 frame_idx, b, (unsigned int)bits.u);
        }
      }
    }
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
    fprintf(stderr, "usage: analysis_fft_trace <input.pcm> [frames]\n");
    return EXIT_FAILURE;
  }

  const char *input_path = argv[1];
  int max_frames = DEFAULT_FRAMES;
  if (argc == 3 && !parse_frame_limit(argv[2], &max_frames)) {
    fprintf(stderr, "invalid frame count: %s\n", argv[2]);
    return EXIT_FAILURE;
  }

  TraceConfig trace_cfg;
  init_trace_config(&trace_cfg);
  BandTraceConfig band_cfg;
  init_band_trace_config(&band_cfg);

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

    TonalityAnalysisState snapshot = analysis;
    trace_fft_for_frame(&snapshot, mode, pcm, FRAME_SIZE, 16, &trace_cfg,
                        &band_cfg, frame_idx);

    AnalysisInfo info;
    memset(&info, 0, sizeof(info));
    info.valid = 0;
    run_analysis(&analysis, mode, pcm, FRAME_SIZE, FRAME_SIZE, 0, -2, CHANNELS,
                 SAMPLE_RATE, 16, downmix_int, &info);
    frame_idx++;
  }

#ifdef CUSTOM_MODES
  opus_custom_mode_destroy(mode);
#endif
  fclose(fin);
  return EXIT_SUCCESS;
}
