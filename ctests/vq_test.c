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

  /* renormalise_vector: gain scaling. */
  {
    celt_norm x[4] = {30000, -30000, 20000, -10000};
    const celt_norm expected[4] = {5124, -5124, 3416, -1708};
    renormalise_vector(x, 4, 1073741824, 0);
    failures += check_i16_array("renorm_gain_half", x, expected, 4);
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

  /* alg_quant: resynth disabled leaves rotated residual, unquant stays stable. */
  {
    enum { N = 8, K = 3, B = 2 };
    unsigned char buffer[256];
    celt_norm x[N] = {12000, -8000, 6000, -4000, 2000, 0, -1000, 500};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected_q[N] = {11429, 8383, 6429, 4217,
                                     1993, 209, 975, 534};
    const celt_norm expected_u[N] = {10117, -9412, 8794, 318,
                                     0, 0, 0, 0};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_NORMAL, B, &enc, Q31ONE, 0, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_NORMAL, B, &dec, Q31ONE);

    failures += check_u32("alg_case1_resynth0_mask", mask, 1);
    failures += check_u32("alg_case1_resynth0_umask", umask, 1);
    failures += check_i16_array("alg_case1_resynth0_q", xq, expected_q, N);
    failures += check_i16_array("alg_case1_resynth0_u", xu, expected_u, N);
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

  /* alg_quant/alg_unquant: early rotation return via 2*K >= N and B=1. */
  {
    enum { N = 4, K = 2, B = 1 };
    unsigned char buffer[256];
    celt_norm x[N] = {8000, -4000, 2000, -1000};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {11585, -11584, 0, 0};
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

    failures += check_u32("alg_case3_mask", mask, 1);
    failures += check_u32("alg_case3_umask", umask, 1);
    failures += check_i16_array("alg_case3_q", xq, expected, N);
    failures += check_i16_array("alg_case3_u", xu, expected, N);
  }

  /* alg_quant/alg_unquant: minimal N/K boundary. */
  {
    enum { N = 2, K = 1, B = 1 };
    unsigned char buffer[256];
    celt_norm x[N] = {12000, -8000};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {16383, 0};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_AGGRESSIVE, B, &enc, Q31ONE, 1, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_AGGRESSIVE, B, &dec, Q31ONE);

    failures += check_u32("alg_case4_mask", mask, 1);
    failures += check_u32("alg_case4_umask", umask, 1);
    failures += check_i16_array("alg_case4_q", xq, expected, N);
    failures += check_i16_array("alg_case4_u", xu, expected, N);
  }

  /* alg_quant/alg_unquant: spread light with stride2 rotation path. */
  {
    enum { N = 16, K = 3, B = 2 };
    unsigned char buffer[256];
    celt_norm x[N] = {5000, -4000, 3000, -2000, 1000, -500, 250, -125,
                      6000, -5000, 4000, -3000, 2000, -1000, 500, -250};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {9376, 1011, 108, -501, -51, -57, 507, 28,
                                   10390, -8309, -901, -611, 449, -5, 538,
                                   -483};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_LIGHT, B, &enc, Q31ONE, 1, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_LIGHT, B, &dec, Q31ONE);

    failures += check_u32("alg_case5_mask", mask, 3);
    failures += check_u32("alg_case5_umask", umask, 3);
    failures += check_i16_array("alg_case5_q", xq, expected, N);
    failures += check_i16_array("alg_case5_u", xu, expected, N);
  }

  /* alg_quant/alg_unquant: aggressive spread with non-unity gain. */
  {
    enum { N = 16, K = 3, B = 2 };
    unsigned char buffer[256];
    celt_norm x[N] = {0, 10000, -9000, 8000, -7000, 6000, -5000, 4000,
                      -3000, 2000, -1000, 500, -250, 125, -60, 30};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {-1153, 6368, -3849, 2819, 157, 1165,
                                   -687, 805, 0, 0, 0, 0, 0, 0, 0, 0};
    ec_enc enc;
    ec_dec dec;
    unsigned mask;
    unsigned umask;

    memcpy(xq, x, sizeof(x));
    ec_enc_init(&enc, buffer, sizeof(buffer));
    mask = alg_quant(xq, N, K, SPREAD_AGGRESSIVE, B, &enc, 1073741824, 1, 0);
    ec_enc_done(&enc);

    ec_dec_init(&dec, buffer, sizeof(buffer));
    umask = alg_unquant(xu, N, K, SPREAD_AGGRESSIVE, B, &dec, 1073741824);

    failures += check_u32("alg_case6_mask", mask, 1);
    failures += check_u32("alg_case6_umask", umask, 1);
    failures += check_i16_array("alg_case6_q", xq, expected, N);
    failures += check_i16_array("alg_case6_u", xu, expected, N);
  }

  /* alg_quant/alg_unquant: pre-search path with K > N/2. */
  {
    enum { N = 8, K = 6, B = 2 };
    unsigned char buffer[256];
    celt_norm x[N] = {16000, 8000, 4000, 2000, 1000, 500, 250, 125};
    celt_norm xq[N];
    celt_norm xu[N];
    const celt_norm expected[N] = {13135, 8757, 4378, 0, 0, 0, 0, 0};
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

    failures += check_u32("alg_case7_mask", mask, 1);
    failures += check_u32("alg_case7_umask", umask, 1);
    failures += check_i16_array("alg_case7_q", xq, expected, N);
    failures += check_i16_array("alg_case7_u", xu, expected, N);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "vq_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
