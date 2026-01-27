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

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "mathops_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
