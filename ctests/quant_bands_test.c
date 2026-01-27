#include <stdio.h>
#include <string.h>

#include "arch.h"
#include "entdec.h"
#include "entenc.h"
#include "modes.h"
#include "quant_bands.h"
#include "rate.h"

#ifdef FIXED_POINT

static void init_dummy_mode(OpusCustomMode *mode, const opus_int16 *e_bands,
                            const opus_int16 *logN, int nb_ebands,
                            int short_mdct_size) {
  memset(mode, 0, sizeof(*mode));
  mode->Fs = 48000;
  mode->nbEBands = nb_ebands;
  mode->effEBands = nb_ebands;
  mode->eBands = e_bands;
  mode->logN = logN;
  mode->shortMdctSize = short_mdct_size;
}

static int check_i32(const char *label, opus_int32 got, opus_int32 expected) {
  if (got != expected) {
    fprintf(stderr, "%s mismatch: got %d expected %d\n", label, (int)got,
            (int)expected);
    return 1;
  }
  return 0;
}

static int check_i32_array(const char *label, const celt_glog *got,
                           const celt_glog *expected, int len) {
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

static int test_amp2log2_fixed(void) {
  enum { kChannels = 2, kNbEBands = 3, kShortMdct = 8 };
  const opus_int16 e_bands[kNbEBands + 1] = {0, 1, 2, 4};
  const opus_int16 logN[kNbEBands] = {6, 6, 6};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, logN, kNbEBands, kShortMdct);

  celt_ener bandE[kNbEBands * kChannels];
  celt_glog bandLogE[kNbEBands * kChannels];
  memset(bandE, 0, sizeof(bandE));
  memset(bandLogE, 0, sizeof(bandLogE));

  bandE[0] = 1 << 12;
  bandE[1] = 0;
  bandE[2] = 1 << 8;
  bandE[3] = 1 << 10;
  bandE[4] = 1 << 9;
  bandE[5] = 1;

  amp2Log2(&mode, 2, kNbEBands, bandE, bandLogE, kChannels);

  int failures = 0;
  for (int c = 0; c < kChannels; ++c) {
    for (int b = 0; b < 2; ++b) {
      int idx = b + c * kNbEBands;
      celt_glog expected =
          celt_log2_db(bandE[idx]) -
          SHL32((celt_glog)eMeans[b], DB_SHIFT - 4) + GCONST(2.f);
      char label[64];
      snprintf(label, sizeof(label), "amp2log2_c%d_b%d", c, b);
      failures += check_i32(label, bandLogE[idx], expected);
    }
    failures +=
        check_i32("amp2log2_tail", bandLogE[2 + c * kNbEBands], -GCONST(14.f));
  }

  return failures;
}

static int run_coarse_roundtrip(const CELTMode *mode, const celt_glog *eBands,
                                const celt_glog *old_init, int C, int LM,
                                opus_uint32 budget, int force_intra,
                                int nbAvailableBytes, int two_pass,
                                int loss_rate, int lfe,
                                celt_glog *out_old) {
  enum { kBufferSize = 256 };
  const int len = C * mode->nbEBands;
  unsigned char buffer[kBufferSize];
  ec_enc enc;
  celt_glog old_enc[len];
  celt_glog error[len];

  opus_uint32 storage_bytes = budget / 8;
  if (storage_bytes == 0) {
    storage_bytes = 1;
  }
  ec_enc_init(&enc, buffer, storage_bytes);
  memcpy(old_enc, old_init, len * sizeof(old_enc[0]));
  memset(error, 0, len * sizeof(error[0]));

  opus_val32 delayedIntra = 0;
  quant_coarse_energy(mode, 0, mode->nbEBands, mode->nbEBands, eBands, old_enc,
                      budget, error, &enc, C, LM, nbAvailableBytes, force_intra,
                      &delayedIntra, two_pass, loss_rate, lfe);
  ec_enc_done(&enc);

  ec_dec dec;
  ec_dec_init(&dec, buffer, storage_bytes);

  celt_glog old_dec[len];
  memcpy(old_dec, old_init, len * sizeof(old_dec[0]));
  int intra = 0;
  int tell = ec_tell(&dec);
  if (tell + 3 <= (int)budget) {
    intra = ec_dec_bit_logp(&dec, 3);
  }
  unquant_coarse_energy(mode, 0, mode->nbEBands, old_dec, intra, &dec, C, LM);

  int failures = check_i32_array("coarse_roundtrip", old_dec, old_enc, len);
  if (out_old) {
    memcpy(out_old, old_enc, len * sizeof(old_enc[0]));
  }
  return failures;
}

static int test_coarse_roundtrip_cases(void) {
  enum { kChannels = 2, kNbEBands = 4, kShortMdct = 8 };
  const opus_int16 e_bands[kNbEBands + 1] = {0, 1, 2, 3, 4};
  const opus_int16 logN[kNbEBands] = {6, 6, 6, 6};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, logN, kNbEBands, kShortMdct);

  celt_glog eBands[kChannels * kNbEBands] = {
      GCONST(6.0f), GCONST(1.0f), GCONST(4.0f), GCONST(0.5f),
      GCONST(5.0f), GCONST(2.5f), GCONST(3.0f), GCONST(-1.0f),
  };
  celt_glog old_init[kChannels * kNbEBands] = {
      GCONST(0.0f), GCONST(-3.0f), GCONST(1.5f), GCONST(-2.0f),
      GCONST(1.0f), GCONST(-1.0f), GCONST(0.25f), GCONST(-4.0f),
  };

  int failures = 0;
  failures += run_coarse_roundtrip(&mode, eBands, old_init, kChannels, 1, 192,
                                   0, 24, 0, 0, 0, NULL);

  celt_glog old_lfe[kChannels * kNbEBands];
  celt_glog old_non_lfe[kChannels * kNbEBands];
  failures += run_coarse_roundtrip(&mode, eBands, old_init, kChannels, 1, 16, 1,
                                   2, 0, 0, 1, old_lfe);
  failures += run_coarse_roundtrip(&mode, eBands, old_init, kChannels, 1, 16, 1,
                                   2, 0, 0, 0, old_non_lfe);

  {
    enum { kLfeBands = 3 };
    const opus_int16 e_bands_lfe[kLfeBands + 1] = {0, 1, 2, 3};
    const opus_int16 logN_lfe[kLfeBands] = {6, 6, 6};
    OpusCustomMode lfe_mode;
    init_dummy_mode(&lfe_mode, e_bands_lfe, logN_lfe, kLfeBands, kShortMdct);

    celt_glog eBands_lfe[kLfeBands] = {
        GCONST(1.5f),
        GCONST(2.5f),
        GCONST(4.0f),
    };
    celt_glog old_init_lfe[kLfeBands] = {0, 0, 0};
    celt_glog old_lfe_simple[kLfeBands];
    celt_glog old_non_lfe_simple[kLfeBands];

    failures += run_coarse_roundtrip(&lfe_mode, eBands_lfe, old_init_lfe, 1, 0,
                                     32, 1, 4, 0, 0, 1,
                                     old_lfe_simple);
    failures += run_coarse_roundtrip(&lfe_mode, eBands_lfe, old_init_lfe, 1, 0,
                                     32, 1, 4, 0, 0, 0,
                                     old_non_lfe_simple);

    if (old_lfe_simple[2] > old_non_lfe_simple[2]) {
      fprintf(stderr, "lfe clamp mismatch: band 2 got %d non-lfe %d\n",
              (int)old_lfe_simple[2], (int)old_non_lfe_simple[2]);
      failures++;
    }
  }

  return failures;
}

static int test_fine_roundtrip_cases(void) {
  enum { kChannels = 1, kNbEBands = 4, kShortMdct = 8 };
  const opus_int16 e_bands[kNbEBands + 1] = {0, 1, 2, 3, 4};
  const opus_int16 logN[kNbEBands] = {6, 6, 6, 6};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, logN, kNbEBands, kShortMdct);

  celt_glog old_init[kNbEBands] = {
      GCONST(-2.0f), GCONST(1.5f), GCONST(-0.5f), GCONST(3.0f),
  };
  celt_glog error[kNbEBands] = {
      GCONST(-0.75f), GCONST(0.25f), GCONST(0.9f), GCONST(-0.1f),
  };
  int fine_quant[kNbEBands] = {0, 1, MAX_FINE_BITS, 2};
  int fine_priority[kNbEBands] = {0, 1, 0, 1};

  enum { kBufferSize = 128, kStorageBytes = 16 };
  unsigned char buffer[kBufferSize];
  ec_enc enc;
  ec_enc_init(&enc, buffer, kStorageBytes);

  celt_glog old_enc[kNbEBands];
  celt_glog error_enc[kNbEBands];
  memcpy(old_enc, old_init, sizeof(old_enc));
  memcpy(error_enc, error, sizeof(error_enc));
  quant_fine_energy(&mode, 0, kNbEBands, old_enc, error_enc, fine_quant, &enc,
                    kChannels);
  quant_energy_finalise(&mode, 0, kNbEBands, old_enc, error_enc, fine_quant,
                        fine_priority, 2, &enc, kChannels);
  ec_enc_done(&enc);

  ec_dec dec;
  ec_dec_init(&dec, buffer, kStorageBytes);

  celt_glog old_dec[kNbEBands];
  memcpy(old_dec, old_init, sizeof(old_dec));
  unquant_fine_energy(&mode, 0, kNbEBands, old_dec, fine_quant, &dec,
                      kChannels);
  unquant_energy_finalise(&mode, 0, kNbEBands, old_dec, fine_quant,
                          fine_priority, 2, &dec, kChannels);

  return check_i32_array("fine_roundtrip", old_dec, old_enc, kNbEBands);
}

int main(void) {
  int failures = 0;
  failures += test_amp2log2_fixed();
  failures += test_coarse_roundtrip_cases();
  failures += test_fine_roundtrip_cases();
  return failures ? 1 : 0;
}

#else
int main(void) {
  fprintf(stderr, "quant_bands_test requires FIXED_POINT build; skipping.\n");
  return 0;
}
#endif
