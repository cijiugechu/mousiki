#include <stdio.h>
#include <string.h>

#include "celt.h"

#ifdef FIXED_POINT

enum { kNFast = 8, kNSlow = 10 };

static int check_sig_array(const char *label, const celt_sig *got,
                           const celt_sig *expected, int len) {
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

static int check_sig(const char *label, celt_sig got, celt_sig expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, (int)got,
            (int)expected);
    return 1;
  }
  return 0;
}

static void preemphasis_reference(const opus_res *pcmp, celt_sig *inp, int N,
                                  int CC, int upsample,
                                  const opus_val16 *coef, celt_sig *mem,
                                  int clip) {
  int i;
  opus_val16 coef0 = coef[0];
  celt_sig m = *mem;

  if (coef[1] == 0 && upsample == 1 && !clip) {
    for (i = 0; i < N; ++i) {
      celt_sig x = RES2SIG(pcmp[CC * i]);
      inp[i] = x - m;
      m = MULT16_32_Q15(coef0, x);
    }
    *mem = m;
    return;
  }

  {
    int Nu = N / upsample;
    if (upsample != 1) {
      memset(inp, 0, N * sizeof(*inp));
    }
    for (i = 0; i < Nu; ++i) {
      inp[i * upsample] = RES2SIG(pcmp[CC * i]);
    }
  }

  for (i = 0; i < N; ++i) {
    celt_sig x = inp[i];
    inp[i] = x - m;
    m = MULT16_32_Q15(coef0, x);
  }

  *mem = m;
}

int main(void) {
  int failures = 0;

  /* Fast path: coef1 == 0, upsample == 1, clip == 0. */
  {
    opus_res pcm[kNFast] = {1000, -2000, 1500, -500, 700, -900, 300, -100};
    celt_sig out[kNFast];
    celt_sig expected[kNFast];
    celt_sig mem = 1234;
    celt_sig mem_expected = 1234;
    opus_val16 coef[4] = {QCONST16(0.85f, 15), 0, 0, 0};

    celt_preemphasis(pcm, out, kNFast, 1, 1, coef, &mem, 0);
    preemphasis_reference(pcm, expected, kNFast, 1, 1, coef, &mem_expected, 0);

    failures += check_sig_array("preemphasis_fast", out, expected, kNFast);
    failures += check_sig("preemphasis_fast_mem", mem, mem_expected);
  }

  /* Slow path: upsample != 1 (clip ignored in fixed build). */
  {
    opus_res pcm[kNSlow / 2] = {1200, -800, 600, -400, 200};
    celt_sig out[kNSlow];
    celt_sig expected[kNSlow];
    celt_sig mem = -2222;
    celt_sig mem_expected = -2222;
    opus_val16 coef[4] = {QCONST16(0.6f, 15), 0, 0, 0};

    celt_preemphasis(pcm, out, kNSlow, 1, 2, coef, &mem, 1);
    preemphasis_reference(pcm, expected, kNSlow, 1, 2, coef, &mem_expected, 1);

    failures += check_sig_array("preemphasis_upsample", out, expected, kNSlow);
    failures += check_sig("preemphasis_upsample_mem", mem, mem_expected);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "prefilter_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
