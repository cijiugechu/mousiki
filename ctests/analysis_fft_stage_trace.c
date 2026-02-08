#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>

#include "analysis.h"
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "mathops.h"
#include "fft_twiddles_48000_960.h"

#define FRAME_SIZE 960
#define SAMPLE_RATE 48000
#define CHANNELS 2
#define DEFAULT_FRAMES 64
#define INITIAL_MEM_FILL 240

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

typedef struct {
  int bins[480];
  int bin_count;
  int all_bins;
  int frame_filter;
  int has_frame_filter;
  int bfly_stage;
  int has_bfly_stage;
  int bfly_index;
  int has_bfly_index;
  int bfly_hex;
  int twiddle_dump;
  int bitrev_src;
} TraceConfig;

static void init_trace_config(TraceConfig *cfg) {
  const char *env = getenv("ANALYSIS_TRACE_BINS");
  const char *frame_env = getenv("ANALYSIS_TRACE_FRAME");
  const char *bfly_env = getenv("ANALYSIS_TRACE_BFLY_STAGE");
  const char *bfly_index_env = getenv("ANALYSIS_TRACE_BFLY_INDEX");
  const char *hex_env = getenv("ANALYSIS_TRACE_BFLY_HEX");
  const char *twiddle_env = getenv("ANALYSIS_TRACE_TWIDDLES");
  const char *bitrev_env = getenv("ANALYSIS_TRACE_BITREV_SRC");
  cfg->bin_count = 0;
  cfg->all_bins = 0;
  cfg->has_frame_filter = 0;
  cfg->frame_filter = 0;
  cfg->has_bfly_stage = 0;
  cfg->bfly_stage = 0;
  cfg->has_bfly_index = 0;
  cfg->bfly_index = 0;
  cfg->bfly_hex = 0;
  cfg->twiddle_dump = 0;
  cfg->bitrev_src = 0;

  if (frame_env != NULL && frame_env[0] != '\0') {
    char *end = NULL;
    long value = strtol(frame_env, &end, 10);
    if (end != frame_env && *end == '\0' && value >= 0 && value <= 1000000) {
      cfg->has_frame_filter = 1;
      cfg->frame_filter = (int)value;
    }
  }

  if (bfly_env != NULL && bfly_env[0] != '\0') {
    char *end = NULL;
    long value = strtol(bfly_env, &end, 10);
    if (end != bfly_env && *end == '\0' && value >= 0 && value <= 1000) {
      cfg->has_bfly_stage = 1;
      cfg->bfly_stage = (int)value;
    }
  }
  if (bfly_index_env != NULL && bfly_index_env[0] != '\0') {
    char *end = NULL;
    long value = strtol(bfly_index_env, &end, 10);
    if (end != bfly_index_env && *end == '\0' && value >= 0 && value <= 1000000) {
      cfg->has_bfly_index = 1;
      cfg->bfly_index = (int)value;
    }
  }
  if (hex_env != NULL && hex_env[0] != '\0') {
    cfg->bfly_hex = 1;
  }
  if (twiddle_env != NULL && twiddle_env[0] != '\0') {
    cfg->twiddle_dump = 1;
  }
  if (bitrev_env != NULL && bitrev_env[0] != '\0') {
    cfg->bitrev_src = 1;
  }

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

static int should_trace_frame(const TraceConfig *cfg, int frame_idx) {
  if (!cfg->has_frame_filter) {
    return 1;
  }
  return frame_idx == cfg->frame_filter;
}

static void dump_bitrev(int frame_idx, const TraceConfig *cfg,
                        const kiss_fft_cpx *buf) {
  if (!should_trace_frame(cfg, frame_idx)) {
    return;
  }
  for (int bin = 0; bin < 480; ++bin) {
    if (!should_trace_bin(cfg, bin)) {
      continue;
    }
    printf("fft_stage[%d].bitrev.bin[%d].r=%.9e\n", frame_idx, bin,
           (double)buf[bin].r);
    printf("fft_stage[%d].bitrev.bin[%d].i=%.9e\n", frame_idx, bin,
           (double)buf[bin].i);
  }
}

static void dump_bitrev_src(int frame_idx, const TraceConfig *cfg,
                            const kiss_fft_cpx *input,
                            const kiss_fft_state *st) {
  if (!cfg->bitrev_src) {
    return;
  }
  if (!should_trace_frame(cfg, frame_idx)) {
    return;
  }
  for (int i = 0; i < st->nfft; ++i) {
    int dest = (int)st->bitrev[i];
    if (!should_trace_bin(cfg, dest)) {
      continue;
    }
    float scaled_r = S_MUL2(input[i].r, st->scale);
    float scaled_i = S_MUL2(input[i].i, st->scale);
    union {
      float f;
      uint32_t u;
    } bits;
    printf("fft_stage[%d].bitrev_src.bin[%d].src=%d\n", frame_idx, dest, i);
    printf("fft_stage[%d].bitrev_src.bin[%d].input.r=%.9e\n", frame_idx, dest,
           (double)input[i].r);
    printf("fft_stage[%d].bitrev_src.bin[%d].input.i=%.9e\n", frame_idx, dest,
           (double)input[i].i);
    bits.f = input[i].r;
    printf("fft_stage[%d].bitrev_src.bin[%d].input_bits.r=0x%08x\n", frame_idx,
           dest, bits.u);
    bits.f = input[i].i;
    printf("fft_stage[%d].bitrev_src.bin[%d].input_bits.i=0x%08x\n", frame_idx,
           dest, bits.u);
    printf("fft_stage[%d].bitrev_src.bin[%d].scaled.r=%.9e\n", frame_idx, dest,
           (double)scaled_r);
    printf("fft_stage[%d].bitrev_src.bin[%d].scaled.i=%.9e\n", frame_idx, dest,
           (double)scaled_i);
    bits.f = scaled_r;
    printf("fft_stage[%d].bitrev_src.bin[%d].scaled_bits.r=0x%08x\n", frame_idx,
           dest, bits.u);
    bits.f = scaled_i;
    printf("fft_stage[%d].bitrev_src.bin[%d].scaled_bits.i=0x%08x\n", frame_idx,
           dest, bits.u);
  }
}

static void dump_stage(int frame_idx, int stage_idx, const TraceConfig *cfg,
                       const kiss_fft_cpx *buf) {
  if (!should_trace_frame(cfg, frame_idx)) {
    return;
  }
  for (int bin = 0; bin < 480; ++bin) {
    if (!should_trace_bin(cfg, bin)) {
      continue;
    }
    printf("fft_stage[%d].stage[%d].bin[%d].r=%.9e\n", frame_idx, stage_idx,
           bin, (double)buf[bin].r);
    printf("fft_stage[%d].stage[%d].bin[%d].i=%.9e\n", frame_idx, stage_idx,
           bin, (double)buf[bin].i);
  }
}

static int g_bfly_enabled = 0;
static int g_bfly_stage = 0;
static int g_bfly_frame = 0;
static int g_bfly_index = 0;
static int g_bfly_hex = 0;
static int g_bfly_index_filter = -1;

static int bfly_current_index(void) { return g_bfly_index; }
static int bfly_detail_index(void) {
  return g_bfly_index_filter >= 0 ? g_bfly_index_filter : 4;
}

static void bfly_trace_begin(int frame_idx, int stage_idx,
                             const TraceConfig *cfg) {
  if (!cfg->has_bfly_stage || cfg->bfly_stage != stage_idx ||
      !should_trace_frame(cfg, frame_idx)) {
    g_bfly_enabled = 0;
    g_bfly_hex = 0;
    g_bfly_index_filter = -1;
    return;
  }
  g_bfly_enabled = 1;
  g_bfly_stage = stage_idx;
  g_bfly_frame = frame_idx;
  g_bfly_index = 0;
  g_bfly_hex = cfg->bfly_hex;
  g_bfly_index_filter = cfg->has_bfly_index ? cfg->bfly_index : -1;
}

static void bfly_trace_end(void) {
  g_bfly_enabled = 0;
  g_bfly_hex = 0;
}

static void dump_bfly(const kiss_fft_cpx *before, const kiss_fft_cpx *after,
                      int count) {
  if (!g_bfly_enabled) {
    return;
  }
  int idx = g_bfly_index++;
  for (int i = 0; i < count; ++i) {
    printf("fft_stage[%d].stage[%d].bfly[%d].in[%d].r=%.9e\n", g_bfly_frame,
           g_bfly_stage, idx, i, (double)before[i].r);
    printf("fft_stage[%d].stage[%d].bfly[%d].in[%d].i=%.9e\n", g_bfly_frame,
           g_bfly_stage, idx, i, (double)before[i].i);
    printf("fft_stage[%d].stage[%d].bfly[%d].out[%d].r=%.9e\n", g_bfly_frame,
           g_bfly_stage, idx, i, (double)after[i].r);
    printf("fft_stage[%d].stage[%d].bfly[%d].out[%d].i=%.9e\n", g_bfly_frame,
           g_bfly_stage, idx, i, (double)after[i].i);
  }
}

static void dump_bfly_value(const char *label, const kiss_fft_cpx *value,
                            int bfly_idx) {
  if (!g_bfly_enabled) {
    return;
  }
  printf("fft_stage[%d].stage[%d].bfly[%d].%s.r=%.9e\n", g_bfly_frame,
         g_bfly_stage, bfly_idx, label, (double)value->r);
  printf("fft_stage[%d].stage[%d].bfly[%d].%s.i=%.9e\n", g_bfly_frame,
         g_bfly_stage, bfly_idx, label, (double)value->i);
}

static void dump_bfly_bits(const char *label, const kiss_fft_cpx *value,
                           int bfly_idx) {
  if (!g_bfly_enabled || !g_bfly_hex) {
    return;
  }
  union {
    float f;
    uint32_t u;
  } r, i;
  r.f = (float)value->r;
  i.f = (float)value->i;
  printf("fft_stage[%d].stage[%d].bfly[%d].%s.bits.r=0x%08x\n", g_bfly_frame,
         g_bfly_stage, bfly_idx, label, r.u);
  printf("fft_stage[%d].stage[%d].bfly[%d].%s.bits.i=0x%08x\n", g_bfly_frame,
         g_bfly_stage, bfly_idx, label, i.u);
}

static void dump_twiddles(const TraceConfig *cfg, const kiss_fft_state *st) {
  if (!cfg->twiddle_dump) {
    return;
  }
  printf("fft_twiddles.nfft=%d\n", st->nfft);
  for (int i = 0; i < st->nfft; ++i) {
    union {
      float f;
      uint32_t u;
    } r, im;
    r.f = (float)st->twiddles[i].r;
    im.f = (float)st->twiddles[i].i;
    printf("fft_twiddles[%d].bits.r=0x%08x\n", i, r.u);
    printf("fft_twiddles[%d].bits.i=0x%08x\n", i, im.u);
  }
}

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

static void compute_bitrev_table(int fout, opus_int16 *f,
                                 const size_t fstride, int in_stride,
                                 opus_int16 *factors,
                                 const kiss_fft_state *st) {
  const int p = *factors++;
  const int m = *factors++;
  (void)st;

  if (m == 1) {
    for (int j = 0; j < p; j++) {
      *f = fout + j;
      f += fstride * in_stride;
    }
  } else {
    for (int j = 0; j < p; j++) {
      compute_bitrev_table(fout, f, fstride * p, in_stride, factors, st);
      f += fstride * in_stride;
      fout += m;
    }
  }
}

static int kf_factor(int n, opus_int16 *facbuf) {
  int p = 4;
  int stages = 0;
  int nbak = n;

  do {
    while (n % p) {
      switch (p) {
        case 4:
          p = 2;
          break;
        case 2:
          p = 3;
          break;
        default:
          p += 2;
          break;
      }
      if (p > 32000 || (opus_int32)p * (opus_int32)p > n) {
        p = n;
      }
    }
    n /= p;
#ifdef RADIX_TWO_ONLY
    if (p != 2 && p != 4)
#else
    if (p > 5)
#endif
    {
      return 0;
    }
    facbuf[2 * stages] = p;
    if (p == 2 && stages > 1) {
      facbuf[2 * stages] = 4;
      facbuf[2] = 2;
    }
    stages++;
  } while (n > 1);
  n = nbak;
  for (int i = 0; i < stages / 2; i++) {
    int tmp = facbuf[2 * i];
    facbuf[2 * i] = facbuf[2 * (stages - i - 1)];
    facbuf[2 * (stages - i - 1)] = tmp;
  }
  for (int i = 0; i < stages; i++) {
    n /= facbuf[2 * i];
    facbuf[2 * i + 1] = n;
  }
  return 1;
}

static void compute_twiddles(kiss_twiddle_cpx *twiddles, int nfft,
                             int trace_phase) {
  if (nfft == 480) {
    for (int i = 0; i < nfft; ++i) {
      twiddles[i] = fft_twiddles48000_960[i];
    }
    if (trace_phase) {
      const double pi = 3.14159265358979323846264338327;
      for (int i = 0; i < nfft; ++i) {
        if (i != 120 && i != 240) {
          continue;
        }
        double phase = (-2 * pi / nfft) * i;
        union {
          double f;
          uint64_t u;
        } bits;
        bits.f = phase;
        printf("fft_twiddle_phase[%d].bits=0x%016" PRIx64 "\n", i, bits.u);
      }
    }
    return;
  }
  for (int i = 0; i < nfft; ++i) {
    const double pi = 3.14159265358979323846264338327;
    double phase = (-2 * pi / nfft) * i;
    if (trace_phase && (i == 120 || i == 240)) {
      union {
        double f;
        uint64_t u;
      } bits;
      bits.f = phase;
      printf("fft_twiddle_phase[%d].bits=0x%016" PRIx64 "\n", i, bits.u);
    }
    kf_cexp(twiddles + i, phase);
  }
}

static void kf_bfly2(kiss_fft_cpx *Fout, int m, int N) {
  kiss_fft_cpx *Fout2;
#ifdef CUSTOM_MODES
  if (m == 1) {
    for (int i = 0; i < N; i++) {
      kiss_fft_cpx t;
      kiss_fft_cpx before[2];
      if (g_bfly_enabled) {
        before[0] = Fout[0];
        before[1] = Fout[1];
      }
      Fout2 = Fout + 1;
      t = *Fout2;
      C_SUB(*Fout2, *Fout, t);
      C_ADDTO(*Fout, t);
      if (g_bfly_enabled) {
        kiss_fft_cpx after[2];
        after[0] = Fout[0];
        after[1] = Fout[1];
        dump_bfly(before, after, 2);
      }
      Fout += 2;
    }
  } else
#endif
  {
    celt_coef tw;
    tw = QCONST32(0.7071067812f, COEF_SHIFT - 1);
    celt_assert(m == 4);
    for (int i = 0; i < N; i++) {
      kiss_fft_cpx t;
      kiss_fft_cpx before[8];
      if (g_bfly_enabled) {
        for (int k = 0; k < 8; ++k) {
          before[k] = Fout[k];
        }
      }
      Fout2 = Fout + 4;
      t = Fout2[0];
      C_SUB(Fout2[0], Fout[0], t);
      C_ADDTO(Fout[0], t);

      t.r = S_MUL(ADD32_ovflw(Fout2[1].r, Fout2[1].i), tw);
      t.i = S_MUL(SUB32_ovflw(Fout2[1].i, Fout2[1].r), tw);
      C_SUB(Fout2[1], Fout[1], t);
      C_ADDTO(Fout[1], t);

      t.r = Fout2[2].i;
      t.i = NEG32_ovflw(Fout2[2].r);
      C_SUB(Fout2[2], Fout[2], t);
      C_ADDTO(Fout[2], t);

      t.r = S_MUL(SUB32_ovflw(Fout2[3].i, Fout2[3].r), tw);
      t.i = S_MUL(NEG32_ovflw(ADD32_ovflw(Fout2[3].i, Fout2[3].r)), tw);
      C_SUB(Fout2[3], Fout[3], t);
      C_ADDTO(Fout[3], t);
      if (g_bfly_enabled) {
        kiss_fft_cpx after[8];
        for (int k = 0; k < 8; ++k) {
          after[k] = Fout[k];
        }
        dump_bfly(before, after, 8);
      }
      Fout += 8;
    }
  }
}

static void kf_bfly4(kiss_fft_cpx *Fout, const size_t fstride,
                     const kiss_fft_state *st, int m, int N, int mm) {
  if (m == 1) {
    for (int i = 0; i < N; i++) {
      kiss_fft_cpx scratch0, scratch1, scratch1b;
      kiss_fft_cpx before[4];
      int bfly_idx = 0;
      if (g_bfly_enabled) {
        for (int k = 0; k < 4; ++k) {
          before[k] = Fout[k];
        }
        bfly_idx = bfly_current_index();
      }
      C_SUB(scratch0, *Fout, Fout[2]);
      C_ADDTO(*Fout, Fout[2]);
      C_ADD(scratch1, Fout[1], Fout[3]);
      if (g_bfly_enabled) {
        dump_bfly_value("scratch0", &scratch0, bfly_idx);
        dump_bfly_value("scratch1", &scratch1, bfly_idx);
      }
      C_SUB(Fout[2], *Fout, scratch1);
      C_ADDTO(*Fout, scratch1);
      C_SUB(scratch1b, Fout[1], Fout[3]);

      Fout[1].r = ADD32_ovflw(scratch0.r, scratch1b.i);
      Fout[1].i = SUB32_ovflw(scratch0.i, scratch1b.r);
      Fout[3].r = SUB32_ovflw(scratch0.r, scratch1b.i);
      Fout[3].i = ADD32_ovflw(scratch0.i, scratch1b.r);
      if (g_bfly_enabled) {
        dump_bfly_value("scratch1b", &scratch1b, bfly_idx);
      }
      if (g_bfly_enabled) {
        kiss_fft_cpx after[4];
        for (int k = 0; k < 4; ++k) {
          after[k] = Fout[k];
        }
        dump_bfly(before, after, 4);
      }
      Fout += 4;
    }
  } else {
    const int m2 = 2 * m;
    const int m3 = 3 * m;
    kiss_fft_cpx *Fout_beg = Fout;
    for (int i = 0; i < N; i++) {
      const kiss_twiddle_cpx *tw1, *tw2, *tw3;
      kiss_fft_cpx scratch[6];
      Fout = Fout_beg + i * mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (int j = 0; j < m; j++) {
        kiss_fft_cpx before[4];
        int bfly_idx = 0;
        if (g_bfly_enabled) {
          before[0] = Fout[0];
          before[1] = Fout[m];
          before[2] = Fout[m2];
          before[3] = Fout[m3];
          bfly_idx = bfly_current_index();
          kiss_fft_cpx tw1v;
          kiss_fft_cpx tw2v;
          kiss_fft_cpx tw3v;
          tw1v.r = tw1->r;
          tw1v.i = tw1->i;
          tw2v.r = tw2->r;
          tw2v.i = tw2->i;
          tw3v.r = tw3->r;
          tw3v.i = tw3->i;
          dump_bfly_value("mul_in0", &before[1], bfly_idx);
          dump_bfly_value("mul_in1", &before[2], bfly_idx);
          dump_bfly_value("mul_in2", &before[3], bfly_idx);
          dump_bfly_bits("mul_in0", &before[1], bfly_idx);
          dump_bfly_bits("mul_in1", &before[2], bfly_idx);
          dump_bfly_bits("mul_in2", &before[3], bfly_idx);
          dump_bfly_value("tw1", &tw1v, bfly_idx);
          dump_bfly_value("tw2", &tw2v, bfly_idx);
          dump_bfly_value("tw3", &tw3v, bfly_idx);
          dump_bfly_bits("tw1", &tw1v, bfly_idx);
          dump_bfly_bits("tw2", &tw2v, bfly_idx);
          dump_bfly_bits("tw3", &tw3v, bfly_idx);
        }
        C_MUL(scratch[0], Fout[m], *tw1);
        C_MUL(scratch[1], Fout[m2], *tw2);
        C_MUL(scratch[2], Fout[m3], *tw3);
        if (g_bfly_enabled) {
          dump_bfly_value("mul0", &scratch[0], bfly_idx);
          dump_bfly_value("mul1", &scratch[1], bfly_idx);
          dump_bfly_value("mul2", &scratch[2], bfly_idx);
          dump_bfly_bits("mul0", &scratch[0], bfly_idx);
          dump_bfly_bits("mul1", &scratch[1], bfly_idx);
          dump_bfly_bits("mul2", &scratch[2], bfly_idx);
        }

        C_SUB(scratch[5], *Fout, scratch[1]);
        C_ADDTO(*Fout, scratch[1]);
        C_ADD(scratch[3], scratch[0], scratch[2]);
        C_SUB(scratch[4], scratch[0], scratch[2]);
        if (g_bfly_enabled) {
          dump_bfly_value("scratch5", &scratch[5], bfly_idx);
          dump_bfly_value("scratch3", &scratch[3], bfly_idx);
          dump_bfly_value("scratch4", &scratch[4], bfly_idx);
        }
        C_SUB(Fout[m2], *Fout, scratch[3]);
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        C_ADDTO(*Fout, scratch[3]);

        Fout[m].r = ADD32_ovflw(scratch[5].r, scratch[4].i);
        Fout[m].i = SUB32_ovflw(scratch[5].i, scratch[4].r);
        Fout[m3].r = SUB32_ovflw(scratch[5].r, scratch[4].i);
        Fout[m3].i = ADD32_ovflw(scratch[5].i, scratch[4].r);
        if (g_bfly_enabled) {
          kiss_fft_cpx after[4];
          after[0] = Fout[0];
          after[1] = Fout[m];
          after[2] = Fout[m2];
          after[3] = Fout[m3];
          dump_bfly(before, after, 4);
        }
        ++Fout;
      }
    }
  }
}

#ifndef RADIX_TWO_ONLY
static void kf_bfly3(kiss_fft_cpx *Fout, const size_t fstride,
                     const kiss_fft_state *st, int m, int N, int mm) {
  const size_t m2 = 2 * (size_t)m;
  kiss_fft_cpx *Fout_beg = Fout;
  kiss_twiddle_cpx epi3;
#ifdef FIXED_POINT
  epi3.i = -QCONST32(0.86602540f, COEF_SHIFT - 1);
#else
  epi3 = st->twiddles[fstride * m];
#endif
  for (int i = 0; i < N; i++) {
    const kiss_twiddle_cpx *tw1, *tw2;
    kiss_fft_cpx scratch[5];
    Fout = Fout_beg + i * mm;
    tw1 = tw2 = st->twiddles;
    int k = m;
    do {
      kiss_fft_cpx before[3];
      if (g_bfly_enabled) {
        before[0] = Fout[0];
        before[1] = Fout[m];
        before[2] = Fout[m2];
      }
      C_MUL(scratch[1], Fout[m], *tw1);
      C_MUL(scratch[2], Fout[m2], *tw2);

      C_ADD(scratch[3], scratch[1], scratch[2]);
      C_SUB(scratch[0], scratch[1], scratch[2]);
      tw1 += fstride;
      tw2 += fstride * 2;

      Fout[m].r = SUB32_ovflw(Fout->r, HALF_OF(scratch[3].r));
      Fout[m].i = SUB32_ovflw(Fout->i, HALF_OF(scratch[3].i));

      C_MULBYSCALAR(scratch[0], epi3.i);

      C_ADDTO(*Fout, scratch[3]);

      Fout[m2].r = ADD32_ovflw(Fout[m].r, scratch[0].i);
      Fout[m2].i = SUB32_ovflw(Fout[m].i, scratch[0].r);

      Fout[m].r = SUB32_ovflw(Fout[m].r, scratch[0].i);
      Fout[m].i = ADD32_ovflw(Fout[m].i, scratch[0].r);

      if (g_bfly_enabled) {
        kiss_fft_cpx after[3];
        after[0] = Fout[0];
        after[1] = Fout[m];
        after[2] = Fout[m2];
        dump_bfly(before, after, 3);
      }
      ++Fout;
    } while (--k);
  }
}

static void kf_bfly5(kiss_fft_cpx *Fout, const size_t fstride,
                     const kiss_fft_state *st, int m, int N, int mm) {
  kiss_fft_cpx *Fout0;
  kiss_fft_cpx *Fout1;
  kiss_fft_cpx *Fout2;
  kiss_fft_cpx *Fout3;
  kiss_fft_cpx *Fout4;
  kiss_fft_cpx scratch[13];
  const kiss_twiddle_cpx *tw;
  kiss_twiddle_cpx ya, yb;
  kiss_fft_cpx *Fout_beg = Fout;

#ifdef FIXED_POINT
  ya.r = QCONST32(0.30901699f, COEF_SHIFT - 1);
  ya.i = -QCONST32(0.95105652f, COEF_SHIFT - 1);
  yb.r = -QCONST32(0.80901699f, COEF_SHIFT - 1);
  yb.i = -QCONST32(0.58778525f, COEF_SHIFT - 1);
#else
  ya = st->twiddles[fstride * m];
  yb = st->twiddles[fstride * 2 * m];
#endif
  tw = st->twiddles;

  for (int i = 0; i < N; i++) {
    Fout = Fout_beg + i * mm;
    Fout0 = Fout;
    Fout1 = Fout0 + m;
    Fout2 = Fout0 + 2 * m;
    Fout3 = Fout0 + 3 * m;
    Fout4 = Fout0 + 4 * m;

    for (int u = 0; u < m; ++u) {
      kiss_fft_cpx before[5];
      int bfly_idx = 0;
      if (g_bfly_enabled) {
        before[0] = *Fout0;
        before[1] = *Fout1;
        before[2] = *Fout2;
        before[3] = *Fout3;
        before[4] = *Fout4;
        bfly_idx = bfly_current_index();
        int idx0 = (int)(Fout0 - Fout_beg);
        int idx1 = (int)(Fout1 - Fout_beg);
        int idx2 = (int)(Fout2 - Fout_beg);
        int idx3 = (int)(Fout3 - Fout_beg);
        int idx4 = (int)(Fout4 - Fout_beg);
        printf("fft_stage[%d].stage[%d].bfly[%d].idx[0]=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, idx0);
        printf("fft_stage[%d].stage[%d].bfly[%d].idx[1]=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, idx1);
        printf("fft_stage[%d].stage[%d].bfly[%d].idx[2]=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, idx2);
        printf("fft_stage[%d].stage[%d].bfly[%d].idx[3]=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, idx3);
        printf("fft_stage[%d].stage[%d].bfly[%d].idx[4]=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, idx4);
        if (bfly_idx == 0) {
          dump_bfly_bits("ya", (const kiss_fft_cpx *)&ya, bfly_idx);
          dump_bfly_bits("yb", (const kiss_fft_cpx *)&yb, bfly_idx);
        }
      }
      scratch[0] = *Fout0;
      if (g_bfly_enabled) {
        dump_bfly_value("scratch0", &scratch[0], bfly_idx);
      }

      C_MUL(scratch[1], *Fout1, tw[u * fstride]);
      C_MUL(scratch[2], *Fout2, tw[2 * u * fstride]);
      C_MUL(scratch[3], *Fout3, tw[3 * u * fstride]);
      C_MUL(scratch[4], *Fout4, tw[4 * u * fstride]);
      if (g_bfly_enabled) {
        dump_bfly_value("scratch1", &scratch[1], bfly_idx);
        dump_bfly_value("scratch2", &scratch[2], bfly_idx);
        dump_bfly_value("scratch3", &scratch[3], bfly_idx);
        dump_bfly_value("scratch4", &scratch[4], bfly_idx);
        dump_bfly_bits("mul_in4", Fout4, bfly_idx);
        kiss_fft_cpx tw4;
        int tw4_idx = 4 * u * fstride;
        tw4.r = tw[tw4_idx].r;
        tw4.i = tw[tw4_idx].i;
        printf("fft_stage[%d].stage[%d].bfly[%d].tw4.idx=%d\n", g_bfly_frame,
               g_bfly_stage, bfly_idx, tw4_idx);
        dump_bfly_bits("tw4", &tw4, bfly_idx);
        dump_bfly_bits("scratch4", &scratch[4], bfly_idx);
      }

      C_ADD(scratch[7], scratch[1], scratch[4]);
      C_SUB(scratch[10], scratch[1], scratch[4]);
      C_ADD(scratch[8], scratch[2], scratch[3]);
      C_SUB(scratch[9], scratch[2], scratch[3]);
      if (g_bfly_enabled) {
        dump_bfly_value("scratch7", &scratch[7], bfly_idx);
        dump_bfly_value("scratch10", &scratch[10], bfly_idx);
        dump_bfly_value("scratch8", &scratch[8], bfly_idx);
        dump_bfly_value("scratch9", &scratch[9], bfly_idx);
        if (bfly_idx == bfly_detail_index()) {
          float s6a = S_MUL(scratch[10].i, ya.i);
          float s6b = S_MUL(scratch[9].i, yb.i);
          float s6sum = s6a + s6b;
          float s11ar = S_MUL(scratch[7].r, yb.r);
          float s11br = S_MUL(scratch[8].r, ya.r);
          float s11sum_r = s11ar + s11br;
          float s11ai = S_MUL(scratch[7].i, yb.r);
          float s11bi = S_MUL(scratch[8].i, ya.r);
          float s11sum_i = s11ai + s11bi;
          union {
            float f;
            uint32_t u;
          } term_bits;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_term0=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s6a);
          term_bits.f = s6a;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_term0_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_term1=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s6b);
          term_bits.f = s6b;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_term1_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_sum=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s6sum);
          term_bits.f = s6sum;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch6_sum_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term0_r=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11ar);
          term_bits.f = s11ar;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term0_r_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term1_r=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11br);
          term_bits.f = s11br;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term1_r_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_sum_r=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11sum_r);
          term_bits.f = s11sum_r;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_sum_r_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term0_i=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11ai);
          term_bits.f = s11ai;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term0_i_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term1_i=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11bi);
          term_bits.f = s11bi;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_term1_i_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_sum_i=%.9e\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (double)s11sum_i);
          term_bits.f = s11sum_i;
          printf("fft_stage[%d].stage[%d].bfly[%d].scratch11_sum_i_bits=0x%08x\n",
                 g_bfly_frame, g_bfly_stage, bfly_idx, (unsigned int)term_bits.u);
        }
      }

      Fout0->r = ADD32_ovflw(Fout0->r,
                             ADD32_ovflw(scratch[7].r, scratch[8].r));
      Fout0->i = ADD32_ovflw(Fout0->i,
                             ADD32_ovflw(scratch[7].i, scratch[8].i));

      scratch[5].r =
          ADD32_ovflw(scratch[0].r,
                      ADD32_ovflw(S_MUL(scratch[7].r, ya.r),
                                  S_MUL(scratch[8].r, yb.r)));
      scratch[5].i =
          ADD32_ovflw(scratch[0].i,
                      ADD32_ovflw(S_MUL(scratch[7].i, ya.r),
                                  S_MUL(scratch[8].i, yb.r)));

      scratch[6].r =
          ADD32_ovflw(S_MUL(scratch[10].i, ya.i), S_MUL(scratch[9].i, yb.i));
      scratch[6].i =
          NEG32_ovflw(ADD32_ovflw(S_MUL(scratch[10].r, ya.i),
                                  S_MUL(scratch[9].r, yb.i)));
      if (g_bfly_enabled) {
        dump_bfly_value("scratch5", &scratch[5], bfly_idx);
        dump_bfly_value("scratch6", &scratch[6], bfly_idx);
        dump_bfly_bits("scratch6", &scratch[6], bfly_idx);
      }

      C_SUB(*Fout1, scratch[5], scratch[6]);
      C_ADD(*Fout4, scratch[5], scratch[6]);

      scratch[11].r =
          ADD32_ovflw(scratch[0].r,
                      ADD32_ovflw(S_MUL(scratch[7].r, yb.r),
                                  S_MUL(scratch[8].r, ya.r)));
      scratch[11].i =
          ADD32_ovflw(scratch[0].i,
                      ADD32_ovflw(S_MUL(scratch[7].i, yb.r),
                                  S_MUL(scratch[8].i, ya.r)));
      scratch[12].r =
          SUB32_ovflw(S_MUL(scratch[9].i, ya.i), S_MUL(scratch[10].i, yb.i));
      scratch[12].i =
          SUB32_ovflw(S_MUL(scratch[10].r, yb.i), S_MUL(scratch[9].r, ya.i));
      if (g_bfly_enabled) {
        dump_bfly_value("scratch11", &scratch[11], bfly_idx);
        dump_bfly_value("scratch12", &scratch[12], bfly_idx);
        dump_bfly_bits("scratch11", &scratch[11], bfly_idx);
        dump_bfly_bits("scratch12", &scratch[12], bfly_idx);
      }

      C_ADD(*Fout2, scratch[11], scratch[12]);
      C_SUB(*Fout3, scratch[11], scratch[12]);

      if (g_bfly_enabled) {
        kiss_fft_cpx after[5];
        after[0] = *Fout0;
        after[1] = *Fout1;
        after[2] = *Fout2;
        after[3] = *Fout3;
        after[4] = *Fout4;
        dump_bfly(before, after, 5);
        if (g_bfly_hex && bfly_idx == bfly_detail_index()) {
          dump_bfly_bits("out0", &after[0], bfly_idx);
          dump_bfly_bits("out1", &after[1], bfly_idx);
          dump_bfly_bits("out2", &after[2], bfly_idx);
          dump_bfly_bits("out3", &after[3], bfly_idx);
          dump_bfly_bits("out4", &after[4], bfly_idx);
        }
      }
      ++Fout0;
      ++Fout1;
      ++Fout2;
      ++Fout3;
      ++Fout4;
    }
  }
}
#endif

static void trace_fft_impl(const kiss_fft_state *st, kiss_fft_cpx *fout,
                           const TraceConfig *cfg, int frame_idx) {
  int m2;
  int m;
  int p;
  int L = 0;
  int fstride[MAXFACTORS];
  int shift = st->shift > 0 ? st->shift : 0;

  fstride[0] = 1;
  do {
    p = st->factors[2 * L];
    m = st->factors[2 * L + 1];
    fstride[L + 1] = fstride[L] * p;
    L++;
  } while (m != 1);

  int stages = L;
  m = st->factors[2 * L - 1];
  for (int i = L - 1; i >= 0; i--) {
    if (i != 0) {
      m2 = st->factors[2 * i - 1];
    } else {
      m2 = 1;
    }
    int applied_stage = stages - 1 - i;
    bfly_trace_begin(frame_idx, applied_stage, cfg);
    switch (st->factors[2 * i]) {
      case 2:
        kf_bfly2(fout, m, fstride[i]);
        break;
      case 4:
        kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2);
        break;
#ifndef RADIX_TWO_ONLY
      case 3:
        kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2);
        break;
      case 5:
        kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2);
        break;
#endif
      default:
        break;
    }
    bfly_trace_end();
    dump_stage(frame_idx, applied_stage, cfg, fout);
    m = m2;
  }
}

static float fft_scale(int nfft) {
  /* Match the static-mode scale literals to keep FFT outputs bit-identical. */
  switch (nfft) {
    case 60:
      return 0.016666667f;
    case 120:
      return 0.008333333f;
    case 240:
      return 0.004166667f;
    case 480:
      return 0.002083333f;
    default:
      return 1.f / nfft;
  }
}

static int trace_fft_state_init(kiss_fft_state *st, int nfft,
                                const TraceConfig *cfg) {
  st->nfft = nfft;
  st->scale = fft_scale(nfft);
  st->shift = -1;
  st->arch_fft = NULL;
  if (!kf_factor(nfft, st->factors)) {
    return 0;
  }

  opus_int16 *bitrev = (opus_int16 *)malloc(sizeof(opus_int16) * nfft);
  kiss_twiddle_cpx *twiddles =
      (kiss_twiddle_cpx *)malloc(sizeof(kiss_twiddle_cpx) * nfft);
  if (bitrev == NULL || twiddles == NULL) {
    free(bitrev);
    free(twiddles);
    return 0;
  }
  compute_twiddles(twiddles, nfft, cfg->twiddle_dump);
  compute_bitrev_table(0, bitrev, 1, 1, st->factors, st);
  st->bitrev = bitrev;
  st->twiddles = twiddles;
  return 1;
}

static void trace_fft_state_destroy(kiss_fft_state *st) {
  free((void *)st->bitrev);
  free((void *)st->twiddles);
}

static void trace_fft_for_frame(const kiss_fft_state *st,
                                const opus_int16 *pcm, int frame_size,
                                int lsb_depth, const TraceConfig *cfg,
                                int frame_idx, TonalityAnalysisState *analysis) {
  int len = frame_size;
  int offset = 0;
  int N = 480;
  int N2 = 240;

  if (!analysis->initialized) {
    analysis->mem_fill = INITIAL_MEM_FILL;
    analysis->initialized = 1;
  }

  if (analysis->Fs == 48000) {
    len /= 2;
    offset /= 2;
  } else if (analysis->Fs == 16000) {
    len = 3 * len / 2;
    offset = 3 * offset / 2;
  }

  analysis->hp_ener_accum +=
      (float)downmix_and_resample(downmix_int, pcm,
                                  &analysis->inmem[analysis->mem_fill],
                                  analysis->downmix_state,
                                  IMIN(len, ANALYSIS_BUF_SIZE - analysis->mem_fill),
                                  offset, 0, -2, CHANNELS, analysis->Fs);

  if (analysis->mem_fill + len < ANALYSIS_BUF_SIZE) {
    return;
  }

  int is_silence =
      is_digital_silence(analysis->inmem, ANALYSIS_BUF_SIZE, 1, lsb_depth);
  if (is_silence) {
    return;
  }

  kiss_fft_cpx input_fft[480];
  kiss_fft_cpx fout[480];
  for (int i = 0; i < N2; ++i) {
    float w = analysis_window[i];
    input_fft[i].r = (kiss_fft_scalar)(w * analysis->inmem[i]);
    input_fft[i].i = (kiss_fft_scalar)(w * analysis->inmem[N2 + i]);
    input_fft[N - i - 1].r = (kiss_fft_scalar)(w * analysis->inmem[N - i - 1]);
    input_fft[N - i - 1].i =
        (kiss_fft_scalar)(w * analysis->inmem[N + N2 - i - 1]);
  }

  for (int i = 0; i < st->nfft; ++i) {
    kiss_fft_cpx x = input_fft[i];
    fout[st->bitrev[i]].r = S_MUL2(x.r, st->scale);
    fout[st->bitrev[i]].i = S_MUL2(x.i, st->scale);
  }
  dump_bitrev_src(frame_idx, cfg, input_fft, st);
  dump_bitrev(frame_idx, cfg, fout);
  trace_fft_impl(st, fout, cfg, frame_idx);
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
    fprintf(stderr, "usage: analysis_fft_stage_trace <input.pcm> [frames]\n");
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

  FILE *fin = fopen(input_path, "rb");
  if (fin == NULL) {
    fprintf(stderr, "failed to open input file: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  kiss_fft_state st;
  if (!trace_fft_state_init(&st, 480, &trace_cfg)) {
    fprintf(stderr, "failed to initialize FFT state\n");
    fclose(fin);
    return EXIT_FAILURE;
  }
  dump_twiddles(&trace_cfg, &st);

  int err = 0;
  OpusCustomMode *mode =
      opus_custom_mode_create(SAMPLE_RATE, FRAME_SIZE, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "opus_custom_mode_create failed: %d\n", err);
    trace_fft_state_destroy(&st);
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
    trace_fft_for_frame(&st, pcm, FRAME_SIZE, 16, &trace_cfg, frame_idx,
                        &snapshot);

    AnalysisInfo info;
    memset(&info, 0, sizeof(info));
    info.valid = 0;
    run_analysis(&analysis, mode, pcm, FRAME_SIZE, FRAME_SIZE, 0, -2, CHANNELS,
                 SAMPLE_RATE, 16, downmix_int, &info);
    frame_idx++;
  }

  trace_fft_state_destroy(&st);
#ifdef CUSTOM_MODES
  opus_custom_mode_destroy(mode);
#endif
  fclose(fin);
  return EXIT_SUCCESS;
}
