#include <stdio.h>
#include <string.h>

#include "arch.h"
#include "bands.h"
#include "vq.h"

#ifdef FIXED_POINT

static int check_i16_array(const char *label, const celt_norm *got,
                           const celt_norm *expected, int len) {
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

static int check_u32(const char *label, unsigned got, unsigned expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %u expected %u\n", label, got, expected);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;

  /* renormalise_vector: zero-energy path keeps zeros. */
  {
    celt_norm x[8] = {0};
    const celt_norm expected[8] = {0};
    renormalise_vector(x, 8, Q31ONE, 0);
    failures += check_i16_array("renorm_zero", x, expected, 8);
  }

  /* renormalise_vector: mixed signs and moderate energy. */
  {
    celt_norm x[8] = {1000, -2000, 3000, -4000, 500, -600, 700, -800};
    const celt_norm expected[8] = {2908, -5816, 8724, -11632, 1454, -1745,
                                   2036, -2326};
    renormalise_vector(x, 8, Q31ONE, 0);
    failures += check_i16_array("renorm_mixed", x, expected, 8);
  }

  /* renormalise_vector: larger magnitudes to exercise scaling. */
  {
    celt_norm x[4] = {30000, -30000, 20000, -10000};
    const celt_norm expected[4] = {10248, -10248, 6832, -3416};
    renormalise_vector(x, 4, Q31ONE, 0);
    failures += check_i16_array("renorm_large", x, expected, 4);
  }

  /* alg_quant/alg_unquant: rotation path with partial collapse mask. */
  {
    enum { N = 8, K = 3, B = 2 };
    unsigned char buffer[256];
    celt_norm x[N] = {12000, -8000, 6000, -4000, 2000, 0, -1000, 500};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {10117, -9412, 8794, 318, 0, 0, 0, 0};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_NORMAL, B, &enc, Q31ONE, 1, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_NORMAL, B, &dec, Q31ONE);

    failures += check_u32("alg_case1_mask", mask, 1);
    failures += check_u32("alg_case1_umask", umask, 1);
    failures += check_i16_array("alg_case1_q", xq, expected, N);
    failures += check_i16_array("alg_case1_u", xu, expected, N);
  }

  /* alg_quant/alg_unquant: spread none with sparse pulses across blocks. */
  {
    enum { N = 10, K = 3, B = 5 };
    unsigned char buffer[256];
    celt_norm x[N] = {0, 16000, -16000, 8000, -8000, 4000, 0, 2000, -2000, 0};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {0, 9459, -9458, 9459, 0, 0, 0, 0, 0, 0};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_NONE, B, &enc, Q31ONE, 1, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_NONE, B, &dec, Q31ONE);

    failures += check_u32("alg_case2_mask", mask, 3);
    failures += check_u32("alg_case2_umask", umask, 3);
    failures += check_i16_array("alg_case2_q", xq, expected, N);
    failures += check_i16_array("alg_case2_u", xu, expected, N);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "vq_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
