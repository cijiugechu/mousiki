#include <stdio.h>

#include "mathops.h"

#ifdef FIXED_POINT

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

int main(void) {
  int failures = 0;

  /* celt_sqrt: zero and saturation cases plus typical inputs. */
  failures += check_i32("celt_sqrt_zero", celt_sqrt(0), 0);
  failures += check_i32("celt_sqrt_1m", celt_sqrt(1 << 20), 1024);
  failures += check_i32("celt_sqrt_16m", celt_sqrt(1 << 24), 4096);
  failures += check_i32("celt_sqrt_sat", celt_sqrt(1 << 30), 32767);
  failures += check_i32("celt_sqrt_sat_plus", celt_sqrt((1 << 30) + 12345),
                        32767);

  /* frac_div32: zero input, rounding, sign handling, and saturation. */
  failures += check_i32("frac_div32_zero", frac_div32(0, 32768), 0);
  failures += check_i32("frac_div32_half", frac_div32(32768, 65536),
                        1073741808);
  failures += check_i32("frac_div32_neg_half", frac_div32(-32768, 65536),
                        -1073741824);
  failures += check_i32("frac_div32_sat_hi", frac_div32(65536, 32768),
                        2147483647);
  failures += check_i32("frac_div32_near_one",
                        frac_div32(1073741824, 1073741824), 2147483632);
  failures += check_i32("frac_div32_q29_half",
                        frac_div32_q29(32768, 65536), 268435452);
  failures += check_i32("frac_div32_q29_sat_input",
                        frac_div32_q29(65536, 32768), 1073741820);

  /* celt_rcp: very small/medium/large inputs exercise ilog2 branches. */
  failures += check_i32("celt_rcp_min", celt_rcp(1), 2147418112);
  failures += check_i32("celt_rcp_1000", celt_rcp(1000), 2147456);
  failures += check_i32("celt_rcp_half", celt_rcp(16384), 131068);
  failures += check_i32("celt_rcp_one", celt_rcp(32768), 65534);
  failures += check_i32("celt_rcp_high", celt_rcp(40000), 53688);

  /* celt_cos_norm: cover wrap, boundary, and sign branches. */
  failures += check_i16("celt_cos_norm_0", celt_cos_norm(0), 32767);
  failures += check_i16("celt_cos_norm_q", celt_cos_norm(0x4000), 23171);
  failures += check_i16("celt_cos_norm_pi2", celt_cos_norm(0x8000), 0);
  failures += check_i16("celt_cos_norm_3q", celt_cos_norm(0xC000), -23171);
  failures += check_i16("celt_cos_norm_pi", celt_cos_norm(0x10000), -32767);
  failures += check_i16("celt_cos_norm_fold", celt_cos_norm(0x18000), 0);
  failures += check_i16("celt_cos_norm_wrap", celt_cos_norm(0x20000), 32767);
  failures += check_i16("celt_cos_norm_mask", celt_cos_norm(0x1FFFF), 32767);

  /* celt_ilog2: boundary and high-range values. */
  {
    struct {
      opus_int32 input;
      opus_int16 expected;
    } ilog2_cases[] = {
        {1, 0},         {2, 1},         {3, 1},         {4, 2},
        {5, 2},         {7, 2},         {8, 3},         {9, 3},
        {15, 3},        {16, 4},        {17, 4},        {31, 4},
        {32, 5},        {33, 5},        {255, 7},       {256, 8},
        {257, 8},       {2147483647, 30},
    };
    for (size_t i = 0; i < sizeof(ilog2_cases) / sizeof(ilog2_cases[0]); ++i) {
      char label[64];
      snprintf(label, sizeof(label), "celt_ilog2_%d",
               (int)ilog2_cases[i].input);
      failures += check_i16(label, celt_ilog2(ilog2_cases[i].input),
                            ilog2_cases[i].expected);
    }
  }

  /* celt_log2: zero path, sub-1 inputs, and larger magnitudes. */
  {
    struct {
      opus_val32 input;
      opus_val16 expected;
    } log2_cases[] = {
        {0, -32767},     {1, -14336},     {2, -13312},     {100, -7533},
        {1000, -4131},   {8192, -1024},   {16384, 0},      {20000, 295},
        {24576, 599},    {32768, 1024},   {40000, 1319},   {49152, 1623},
        {65536, 2048},   {131072, 3072},  {1048576, 6144},
    };
    for (size_t i = 0; i < sizeof(log2_cases) / sizeof(log2_cases[0]); ++i) {
      char label[64];
      snprintf(label, sizeof(label), "celt_log2_%d",
               (int)log2_cases[i].input);
      failures += check_i16(label, celt_log2(log2_cases[i].input),
                            log2_cases[i].expected);
    }
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "mathops_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
