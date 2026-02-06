#include <stdio.h>
#include <string.h>

#include "mathops.h"

#ifdef FIXED_POINT

enum {
  kDecayLen = 320,
  kRatioLen = 144
};

typedef struct {
  const char *name;
  int period;
  int amp;
  int phase;
  int invert;
} WaveSpec;

static void fill_periodic(opus_val16 *x, int len, WaveSpec spec) {
  int half = spec.period / 2;
  for (int i = 0; i < len; ++i) {
    int pos = (i + spec.phase) % spec.period;
    int tri = pos < half ? pos : spec.period - pos;
    int centered = 2 * tri - half;
    int v = centered * spec.amp / half;
    if (spec.invert) {
      v = -v;
    }
    x[i] = (opus_val16)v;
  }
}

static int check_i32(const char *label, opus_val32 got, opus_val32 expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, (int)got,
            (int)expected);
    return 1;
  }
  return 0;
}

static int check_i16(const char *label, opus_val16 got, opus_val16 expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, (int)got,
            (int)expected);
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

/* Mirrors the fixed-point decoder PLC decay expression:
   decay = celt_sqrt(frac_div32(SHR32(E1, 1), E2)); */
static void compute_decay_terms(const opus_val16 *exc_tail, int exc_length,
                                opus_val32 *e1, opus_val32 *e2,
                                opus_val16 *decay) {
  int shift =
      IMAX(0, 2 * celt_zlog2(celt_maxabs16(exc_tail, exc_length)) - 20);
  int decay_length = exc_length >> 1;
  opus_val32 E1 = 1;
  opus_val32 E2 = 1;
  for (int i = 0; i < decay_length; ++i) {
    opus_val16 a = exc_tail[exc_length - decay_length + i];
    opus_val16 b = exc_tail[exc_length - 2 * decay_length + i];
    E1 += SHR32(MULT16_16(a, a), shift);
    E2 += SHR32(MULT16_16(b, b), shift);
  }
  E1 = MIN32(E1, E2);
  *e1 = E1;
  *e2 = E2;
  *decay = (opus_val16)celt_sqrt(frac_div32(SHR32(E1, 1), E2));
}

/* Mirrors the fixed-point decoder attenuation ratio expression:
   ratio = celt_sqrt(frac_div32(SHR32(S1,1)+1, S2+1)); */
static void compute_ratio_terms(const opus_val16 *old_sig,
                                const opus_val16 *new_sig, int len,
                                opus_val32 *s1, opus_val32 *s2,
                                opus_val16 *ratio) {
  opus_val32 S1 = 0;
  opus_val32 S2 = 0;
  for (int i = 0; i < len; ++i) {
    S1 += SHR32(MULT16_16(old_sig[i], old_sig[i]), 10);
    S2 += SHR32(MULT16_16(new_sig[i], new_sig[i]), 10);
  }
  *s1 = S1;
  *s2 = S2;
  *ratio = (opus_val16)celt_sqrt(frac_div32(SHR32(S1, 1) + 1, S2 + 1));
}

int main(void) {
  int failures = 0;

  /* Decoder-like excitation windows used to derive PLC decay terms. */
  {
    opus_val16 exc[kDecayLen];
    opus_val32 e1 = 0;
    opus_val32 e2 = 0;
    opus_val16 decay = 0;

    fill_periodic(exc, kDecayLen,
                  (WaveSpec){"steady_tri", 96, 14000, 0, 0});
    compute_decay_terms(exc, kDecayLen, &e1, &e2, &decay);
    failures += check_i32("decay_case_a_e1", e1, 148945491);
    failures += check_i32("decay_case_a_e2", e2, 172083276);
    failures += check_i16("decay_case_a", decay, 30486);

    fill_periodic(exc, kDecayLen,
                  (WaveSpec){"tail_drop", 96, 14000, 0, 0});
    for (int i = kDecayLen / 2; i < kDecayLen; ++i) {
      exc[i] = (opus_val16)(exc[i] / 4);
    }
    compute_decay_terms(exc, kDecayLen, &e1, &e2, &decay);
    failures += check_i32("decay_case_b_e1", e1, 9306492);
    failures += check_i32("decay_case_b_e2", e2, 172083276);
    failures += check_i16("decay_case_b", decay, 7620);
    failures += check_true("decay_case_b_lt_case_a", decay < 30486);

    fill_periodic(exc, kDecayLen,
                  (WaveSpec){"flat_energy_halves", 64, 3000, 7, 1});
    for (int i = 0; i < kDecayLen / 3; ++i) {
      exc[i] = 0;
    }
    compute_decay_terms(exc, kDecayLen, &e1, &e2, &decay);
    failures += check_i32("decay_case_c_e1", e1, 45578035);
    failures += check_i32("decay_case_c_e2", e2, 45578035);
    failures += check_i16("decay_case_c", decay, 32767);
  }

  /* Decoder-like synthesis/reference windows used for attenuation ratio. */
  {
    opus_val16 old_sig[kRatioLen];
    opus_val16 new_sig[kRatioLen];
    opus_val32 s1 = 0;
    opus_val32 s2 = 0;
    opus_val16 ratio = 0;

    fill_periodic(old_sig, kRatioLen,
                  (WaveSpec){"old_low_energy", 72, 7000, 0, 0});
    fill_periodic(new_sig, kRatioLen,
                  (WaveSpec){"new_high_energy", 72, 12000, 11, 1});
    compute_ratio_terms(old_sig, new_sig, kRatioLen, &s1, &s2, &ratio);
    failures += check_i32("ratio_case_a_s1", s1, 2299964);
    failures += check_i32("ratio_case_a_s2", s2, 6759820);
    failures += check_i16("ratio_case_a", ratio, 19113);

    fill_periodic(old_sig, kRatioLen,
                  (WaveSpec){"old_high_energy", 72, 12000, 11, 1});
    fill_periodic(new_sig, kRatioLen,
                  (WaveSpec){"new_low_energy", 72, 7000, 0, 0});
    compute_ratio_terms(old_sig, new_sig, kRatioLen, &s1, &s2, &ratio);
    failures += check_i32("ratio_case_b_s1", s1, 6759820);
    failures += check_i32("ratio_case_b_s2", s2, 2299964);
    failures += check_i16("ratio_case_b", ratio, 32767);

    memset(old_sig, 0, sizeof(old_sig));
    memset(new_sig, 0, sizeof(new_sig));
    compute_ratio_terms(old_sig, new_sig, kRatioLen, &s1, &s2, &ratio);
    failures += check_i32("ratio_case_c_s1", s1, 0);
    failures += check_i32("ratio_case_c_s2", s2, 0);
    failures += check_i16("ratio_case_c", ratio, 32767);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "celt_decoder_math_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
