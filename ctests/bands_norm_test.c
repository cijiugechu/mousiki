#include <stdio.h>
#include <string.h>

#include "bands.h"
#include "modes.h"

#ifdef FIXED_POINT

static void init_dummy_mode(OpusCustomMode *mode, const opus_int16 *e_bands,
                            int nb_ebands, int short_mdct_size) {
  memset(mode, 0, sizeof(*mode));
  mode->Fs = 48000;
  mode->nbEBands = nb_ebands;
  mode->effEBands = nb_ebands;
  mode->eBands = e_bands;
  mode->shortMdctSize = short_mdct_size;
}

static int check_i32_array(const char *label, const celt_sig *got,
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

int main(void) {
  int failures = 0;
  const opus_int16 e_bands[] = {0, 1, 2, 4};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, 3, 4);

  /* normalise_bands: cover shift>0 and shift<=0 paths via bandE scale. */
  celt_sig freq[4] = {1000, -2000, 3000, -4000};
  celt_norm x_norm[4] = {0};
  celt_ener bandE[3] = {1 << 10, 1 << 20, 1 << 15};
  const celt_norm expected_norm[4] = {15984, -31, 1500, -2000};

  normalise_bands(&mode, freq, x_norm, bandE, 3, 1, 1);
  failures += check_i16_array("normalise_bands", x_norm, expected_norm, 4);

  /* denormalise_bands: exercise shift>31, shift<0, and shift>=0 branches. */
  celt_norm x_in[4] = {1234, -2345, 3456, -4567};
  celt_sig freq_out[4] = {0};
  celt_glog bandLogE[3] = {-(18 << DB_SHIFT), (17 << DB_SHIFT), (10 << DB_SHIFT)};
  const celt_sig expected_denorm[4] = {0, -153681920, 884682, -1169081};

  denormalise_bands(&mode, x_in, freq_out, bandLogE, 0, 3, 1, 1, 0);
  failures += check_i32_array("denormalise_bands", freq_out, expected_denorm, 4);

  /* denormalise_bands: downsample path zeroes tail. */
  memset(freq_out, 0, sizeof(freq_out));
  denormalise_bands(&mode, x_in, freq_out, bandLogE, 0, 3, 1, 2, 0);
  {
    const celt_sig expected_downsample[4] = {0, -153681920, 0, 0};
    failures += check_i32_array("denormalise_downsample", freq_out,
                                expected_downsample, 4);
  }

  /* denormalise_bands: silence path zeros everything. */
  freq_out[0] = 1;
  freq_out[1] = -1;
  freq_out[2] = 2;
  freq_out[3] = -2;
  denormalise_bands(&mode, x_in, freq_out, bandLogE, 0, 3, 1, 1, 1);
  {
    const celt_sig expected_silence[4] = {0, 0, 0, 0};
    failures += check_i32_array("denormalise_silence", freq_out,
                                expected_silence, 4);
  }

  return failures ? 1 : 0;
}
#else
int main(void) {
  fprintf(stderr, "bands_norm_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
