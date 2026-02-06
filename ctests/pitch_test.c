#include <stdio.h>
#include <string.h>

#include "arch.h"
#include "pitch.h"

#ifdef FIXED_POINT

enum {
  kLen = 64,
  kHalfLen = kLen / 2,
  kMaxPitch = 128,
  kSearchOffset = 48,
  kMaxPeriod = 120,
  kMinPeriodEarly = 100,
  kMinPeriodUpdate = 20,
  kN = 160
};

static const opus_val16 kExpectedPdZero[kHalfLen] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const opus_val16 kExpectedPdLarge[kHalfLen] = {
  -704, -1025, -407, 405, 147, -318, -127, 177,
  -159, -398, -127, 177, -159, -398, -127, 177,
  -159, -398, -127, 177, -159, -398, -127, 177,
  -159, -398, -127, 177, -159, -398, -127, 177
};

static void fill_triangle_i16(opus_val16 *buf, int len, int period, int amp) {
  for (int i = 0; i < len; ++i) {
    int pos = i % period;
    int half = period / 2;
    int v = pos < half ? pos : (period - pos);
    int sample = v * amp / half;
    buf[i] = (opus_val16)sample;
  }
}

static void fill_pattern_sig(celt_sig *buf, int len, int period, int amp) {
  for (int i = 0; i < len; ++i) {
    int v = (i % period) - (period / 2);
    buf[i] = (celt_sig)(v * (amp / (period / 2)));
  }
}

static int check_i16_array(const char *label, const opus_val16 *got,
                           const opus_val16 *expected, int len) {
  int failures = 0;
  for (int i = 0; i < len; ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "%s[%d] mismatch: got %d expected %d\n",
              label, i, (int)got[i], (int)expected[i]);
      failures++;
    }
  }
  return failures;
}

static int check_int(const char *label, int got, int expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, got, expected);
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

static int check_int_in_range(const char *label, int value, int lo, int hi) {
  if (value < lo || value > hi) {
    fprintf(stderr, "%s out of range: got %d expected [%d, %d]\n",
            label, value, lo, hi);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;

  /* pitch_downsample: silence in stereo exercises maxabs<1 and C==2 branch. */
  celt_sig x0[kLen];
  celt_sig x1[kLen];
  celt_sig *x[2] = { x0, x1 };
  opus_val16 x_lp[kHalfLen];
  memset(x0, 0, sizeof(x0));
  memset(x1, 0, sizeof(x1));
  memset(x_lp, 0, sizeof(x_lp));
  pitch_downsample(x, x_lp, kLen, 2, 0);
  failures += check_i16_array("pitch_downsample_zero", x_lp, kExpectedPdZero,
                              kHalfLen);

  /* pitch_downsample: large single-channel input exercises shift>0 path. */
  fill_pattern_sig(x0, kLen, 8, 1 << 22);
  pitch_downsample(x, x_lp, kLen, 1, 0);
  failures += check_i16_array("pitch_downsample_large", x_lp,
                              kExpectedPdLarge, kHalfLen);

  /* pitch_downsample: opposite stereo channels should cancel to zero. */
  fill_pattern_sig(x0, kLen, 8, 1 << 18);
  for (int i = 0; i < kLen; ++i) {
    x1[i] = -x0[i];
  }
  pitch_downsample(x, x_lp, kLen, 2, 0);
  failures += check_i16_array("pitch_downsample_cancel", x_lp, kExpectedPdZero,
                              kHalfLen);

  /* pitch_search: low amplitude keeps shift at 0. */
  opus_val16 y[kLen + kMaxPitch];
  opus_val16 x_lp_search[kLen];
  int pitch = -1;
  memset(y, 0, sizeof(y));
  fill_triangle_i16(x_lp_search, kLen, 16, 100);
  memcpy(&y[kSearchOffset], x_lp_search, sizeof(x_lp_search));
  pitch_search(x_lp_search, y, kLen, kMaxPitch, &pitch, 0);
  failures += check_int("pitch_search_low", pitch, 96);

  /* pitch_search: large amplitude triggers downshift path. */
  memset(y, 0, sizeof(y));
  fill_triangle_i16(x_lp_search, kLen, 16, 20000);
  memcpy(&y[kSearchOffset], x_lp_search, sizeof(x_lp_search));
  pitch_search(x_lp_search, y, kLen, kMaxPitch, &pitch, 0);
  failures += check_int("pitch_search_high", pitch, 96);

  /* pitch_search: unmatched reference still returns a valid pitch index. */
  memset(y, 0, sizeof(y));
  fill_triangle_i16(x_lp_search, kLen, 16, 1200);
  pitch_search(x_lp_search, y, kLen, kMaxPitch, &pitch, 0);
  failures += check_int_in_range("pitch_search_unmatched_range", pitch, 0,
                                 kMaxPitch - 1);

  /* remove_doubling: early-break when minperiod is large. */
  opus_val16 xbuf[kMaxPeriod + kN];
  opus_val16 pg = 0;
  int t0 = 80;
  fill_triangle_i16(xbuf, kMaxPeriod + kN, 40, 2000);
  pg = remove_doubling(xbuf, kMaxPeriod, kMinPeriodEarly, kN, &t0, 80, 16384, 0);
  failures += check_int("remove_doubling_early_t0", t0, 100);
  failures += check_int("remove_doubling_early_pg", pg, 32767);

  /* remove_doubling: update path with smaller minperiod. */
  t0 = 80;
  fill_triangle_i16(xbuf, kMaxPeriod + kN, 40, 2000);
  pg = remove_doubling(xbuf, kMaxPeriod, kMinPeriodUpdate, kN, &t0, 40, 8192, 0);
  failures += check_int("remove_doubling_update_t0", t0, 80);
  failures += check_int("remove_doubling_update_pg", pg, 32767);

  /* remove_doubling: large initial lag should clamp back to legal bounds. */
  t0 = 999;
  fill_triangle_i16(xbuf, kMaxPeriod + kN, 48, 3000);
  pg = remove_doubling(xbuf, kMaxPeriod, 30, kN, &t0, 32, 0, 0);
  failures += check_int_in_range("remove_doubling_clamp_t0", t0, 30,
                                 kMaxPeriod - 1);
  failures += check_int_in_range("remove_doubling_clamp_pg", pg, 0, 32767);
  failures += check_true("remove_doubling_clamp_nonzero_period", t0 > 0);

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "pitch_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
