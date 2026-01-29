#include <stdio.h>
#include <string.h>

#include "celt.h"
#include "pitch.h"

#ifdef FIXED_POINT

enum {
  kHistory = 40,
  kN = 12,
  kOverlap = 5
};

static const celt_coef kWindow[kOverlap] = {
    QCONST16(0.05f, 15), QCONST16(0.25f, 15), QCONST16(0.5f, 15),
    QCONST16(0.75f, 15), QCONST16(0.9f, 15)};

static const opus_val16 kGains[3][3] = {
    {QCONST16(0.3066406250f, 15), QCONST16(0.2170410156f, 15),
     QCONST16(0.1296386719f, 15)},
    {QCONST16(0.4638671875f, 15), QCONST16(0.2680664062f, 15),
     QCONST16(0.f, 15)},
    {QCONST16(0.7998046875f, 15), QCONST16(0.1000976562f, 15),
     QCONST16(0.f, 15)}};

static int check_sig_array(const char *label, const opus_val32 *got,
                           const opus_val32 *expected, int len) {
  int failures = 0;
  for (int i = 0; i < len; ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "%s[%d] mismatch: got %d expected %d\n", label, i,
              (int)got[i], (int)expected[i]);
      failures++;
    }
  }
  return failures;
}

static void comb_filter_const_reference(opus_val32 *y, const opus_val32 *x,
                                        int T, int N, opus_val16 g10,
                                        opus_val16 g11, opus_val16 g12) {
  opus_val32 x0, x1, x2, x3, x4;
  int i;
  x4 = x[-T - 2];
  x3 = x[-T - 1];
  x2 = x[-T];
  x1 = x[-T + 1];
  for (i = 0; i < N; ++i) {
    x0 = x[i - T + 2];
    y[i] = x[i] + MULT_COEF_32(g10, x2) + MULT_COEF_32(g11, ADD32(x1, x3)) +
           MULT_COEF_32(g12, ADD32(x0, x4));
    y[i] = SUB32(y[i], 1);
    y[i] = SATURATE(y[i], SIG_SAT);
    x4 = x3;
    x3 = x2;
    x2 = x1;
    x1 = x0;
  }
}

static void comb_filter_reference(opus_val32 *y, const opus_val32 *x, int T0,
                                  int T1, int N, opus_val16 g0,
                                  opus_val16 g1, int tapset0,
                                  int tapset1, const celt_coef *window,
                                  int overlap) {
  int i;
  opus_val16 g00, g01, g02, g10, g11, g12;
  opus_val32 x0, x1, x2, x3, x4;

  if (g0 == 0 && g1 == 0) {
    memcpy(y, x, N * sizeof(*y));
    return;
  }

  T0 = IMAX(T0, COMBFILTER_MINPERIOD);
  T1 = IMAX(T1, COMBFILTER_MINPERIOD);

  g00 = MULT_COEF_TAPS(g0, kGains[tapset0][0]);
  g01 = MULT_COEF_TAPS(g0, kGains[tapset0][1]);
  g02 = MULT_COEF_TAPS(g0, kGains[tapset0][2]);
  g10 = MULT_COEF_TAPS(g1, kGains[tapset1][0]);
  g11 = MULT_COEF_TAPS(g1, kGains[tapset1][1]);
  g12 = MULT_COEF_TAPS(g1, kGains[tapset1][2]);

  x1 = x[-T1 + 1];
  x2 = x[-T1];
  x3 = x[-T1 - 1];
  x4 = x[-T1 - 2];

  if (g0 == g1 && T0 == T1 && tapset0 == tapset1) {
    overlap = 0;
  }

  for (i = 0; i < overlap; ++i) {
    celt_coef f;
    x0 = x[i - T1 + 2];
    f = MULT_COEF(window[i], window[i]);
    y[i] = x[i] +
           MULT_COEF_32(MULT_COEF((COEF_ONE - f), g00), x[i - T0]) +
           MULT_COEF_32(MULT_COEF((COEF_ONE - f), g01),
                        ADD32(x[i - T0 + 1], x[i - T0 - 1])) +
           MULT_COEF_32(MULT_COEF((COEF_ONE - f), g02),
                        ADD32(x[i - T0 + 2], x[i - T0 - 2])) +
           MULT_COEF_32(MULT_COEF(f, g10), x2) +
           MULT_COEF_32(MULT_COEF(f, g11), ADD32(x1, x3)) +
           MULT_COEF_32(MULT_COEF(f, g12), ADD32(x0, x4));
    y[i] = SUB32(y[i], 3);
    y[i] = SATURATE(y[i], SIG_SAT);
    x4 = x3;
    x3 = x2;
    x2 = x1;
    x1 = x0;
  }

  if (g1 == 0) {
    if (overlap < N) {
      memcpy(y + overlap, x + overlap, (N - overlap) * sizeof(*y));
    }
    return;
  }

  if (overlap < N) {
    comb_filter_const_reference(y + overlap, x + overlap, T1, N - overlap,
                                g10, g11, g12);
  }
}

static void fill_pattern(opus_val32 *buf, int len) {
  for (int i = 0; i < len; ++i) {
    int base = (i % 11) - 5;
    int bump = (i & 1) ? 200 : -200;
    buf[i] = base * 900 + bump;
  }
}

int main(void) {
  int failures = 0;
  opus_val32 xbuf[kHistory + kN + 8];
  opus_val32 *x = xbuf + kHistory;

  fill_pattern(xbuf, (int)(sizeof(xbuf) / sizeof(xbuf[0])));

  /* comb_filter_const: fixed bias + saturation path. */
  {
    opus_val32 y[kN];
    opus_val32 expected[kN];
    opus_val16 g10 = QCONST16(0.6f, 15);
    opus_val16 g11 = QCONST16(-0.2f, 15);
    opus_val16 g12 = QCONST16(0.1f, 15);

    comb_filter_const(y, x, 20, kN, g10, g11, g12, 0);
    comb_filter_const_reference(expected, x, 20, kN, g10, g11, g12);
    failures += check_sig_array("comb_filter_const", y, expected, kN);
  }

  /* comb_filter: overlap + constant tail with negative gain. */
  {
    opus_val32 y[kN];
    opus_val32 expected[kN];
    opus_val16 g0 = QCONST16(0.65f, 15);
    opus_val16 g1 = QCONST16(-0.35f, 15);

    comb_filter(y, x, 18, 26, kN, g0, g1, 0, 2, kWindow, kOverlap, 0);
    comb_filter_reference(expected, x, 18, 26, kN, g0, g1, 0, 2, kWindow,
                          kOverlap);
    failures += check_sig_array("comb_filter_overlap", y, expected, kN);
  }

  /* comb_filter: g1 == 0 copies tail after overlap. */
  {
    opus_val32 y[kN];
    opus_val32 expected[kN];
    opus_val16 g0 = QCONST16(0.45f, 15);
    opus_val16 g1 = 0;

    comb_filter(y, x, 21, 24, kN, g0, g1, 1, 1, kWindow, kOverlap, 0);
    comb_filter_reference(expected, x, 21, 24, kN, g0, g1, 1, 1, kWindow,
                          kOverlap);
    failures += check_sig_array("comb_filter_g1_zero", y, expected, kN);
  }

  /* comb_filter: zero gains copy input. */
  {
    opus_val32 y[kN];
    opus_val32 expected[kN];
    memcpy(expected, x, sizeof(expected));
    comb_filter(y, x, 15, 15, kN, 0, 0, 0, 0, kWindow, kOverlap, 0);
    failures += check_sig_array("comb_filter_zero", y, expected, kN);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "comb_filter_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
