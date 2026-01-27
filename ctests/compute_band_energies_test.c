#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bands.h"
#include "mathops.h"
#include "modes.h"

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

#ifdef FIXED_POINT
static celt_ener compute_expected_band_energy_fixed(const CELTMode *mode,
                                                    const celt_sig *spectrum,
                                                    int band, int channel,
                                                    int lm, int *out_shift) {
  const opus_int16 *e_bands = mode->eBands;
  const int n = mode->shortMdctSize << lm;
  const int band_start = e_bands[band] << lm;
  const int band_end = e_bands[band + 1] << lm;
  const int band_len = band_end - band_start;
  const celt_sig *channel_base = &spectrum[channel * n];
  opus_val32 maxval = celt_maxabs32(&channel_base[band_start], band_len);
  if (maxval <= 0) {
    if (out_shift) {
      *out_shift = 0;
    }
    return EPSILON;
  }

  int shift = celt_ilog2(maxval) - 14;
  int shift2 = (((mode->logN[band] >> BITRES) + lm + 1) >> 1);
  int branch_shift = shift;
  int j = band_start;
  opus_val32 sum = 0;

  if (shift > 0) {
    do {
      sum = ADD32(sum, SHR32(MULT16_16(EXTRACT16(SHR32(channel_base[j], shift)),
                                       EXTRACT16(SHR32(channel_base[j], shift))),
                             2 * shift2));
    } while (++j < band_end);
  } else {
    do {
      sum = ADD32(sum, SHR32(MULT16_16(EXTRACT16(SHL32(channel_base[j], -shift)),
                                       EXTRACT16(SHL32(channel_base[j], -shift))),
                             2 * shift2));
    } while (++j < band_end);
  }

  shift = branch_shift + shift2;
  while (sum < (1 << 28)) {
    sum <<= 2;
    shift -= 1;
  }

  if (out_shift) {
    *out_shift = branch_shift;
  }

  return EPSILON + VSHR32(celt_sqrt(sum), -shift);
}
#endif

int main(void) {
#ifdef FIXED_POINT
  enum {
    kChannels = 2,
    kLm = 1,
    kShortMdct = 4,
    kNbEBands = 2,
    kN = kShortMdct << kLm,
    kSpectrumLen = kChannels * kN
  };

  const opus_int16 e_bands[kNbEBands + 1] = {0, 2, 4};
  const opus_int16 logN[kNbEBands] = {8, 8};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, logN, kNbEBands, kShortMdct);

  celt_sig spectrum[kSpectrumLen];
  memset(spectrum, 0, sizeof(spectrum));

  /* Channel 0: band 0 -> large values (shift > 0), band 1 -> small values (shift <= 0). */
  spectrum[0] = 1 << 20;
  spectrum[1] = -(1 << 20);
  spectrum[2] = 1 << 19;
  spectrum[3] = -(1 << 19);
  spectrum[4] = 1 << 8;
  spectrum[5] = -(1 << 8);
  spectrum[6] = 1 << 7;
  spectrum[7] = -(1 << 7);

  celt_ener band_e[kNbEBands * kChannels];
  compute_band_energies(&mode, spectrum, band_e, mode.nbEBands, kChannels,
                        kLm, 0);

  int failures = 0;
  int shift_band0 = 0;
  int shift_band1 = 0;
  celt_ener expected_band0 =
      compute_expected_band_energy_fixed(&mode, spectrum, 0, 0, kLm,
                                         &shift_band0);
  celt_ener expected_band1 =
      compute_expected_band_energy_fixed(&mode, spectrum, 1, 0, kLm,
                                         &shift_band1);

  if (shift_band0 <= 0) {
    fprintf(stderr, "expected shift > 0 for band 0, got %d\n", shift_band0);
    failures++;
  }
  if (shift_band1 > 0) {
    fprintf(stderr, "expected shift <= 0 for band 1, got %d\n", shift_band1);
    failures++;
  }

  if (band_e[0] != expected_band0) {
    fprintf(stderr, "band 0 mismatch: got %d expected %d\n",
            (int)band_e[0], (int)expected_band0);
    failures++;
  }
  if (band_e[1] != expected_band1) {
    fprintf(stderr, "band 1 mismatch: got %d expected %d\n",
            (int)band_e[1], (int)expected_band1);
    failures++;
  }

  for (int b = 0; b < kNbEBands; ++b) {
    int idx = b + 1 * mode.nbEBands;
    if (band_e[idx] != EPSILON) {
      fprintf(stderr, "channel 1 band %d expected EPSILON got %d\n", b,
              (int)band_e[idx]);
      failures++;
    }
  }

  /* Extra-large amplitude case to stress scaling/shift logic. */
  celt_sig spectrum_big[kSpectrumLen];
  memset(spectrum_big, 0, sizeof(spectrum_big));
  spectrum_big[0] = 1 << 25;
  spectrum_big[1] = -(1 << 25);
  spectrum_big[2] = 1 << 24;
  spectrum_big[3] = -(1 << 24);
  spectrum_big[4] = 1 << 23;
  spectrum_big[5] = -(1 << 23);
  spectrum_big[6] = 1 << 22;
  spectrum_big[7] = -(1 << 22);

  compute_band_energies(&mode, spectrum_big, band_e, mode.nbEBands, kChannels,
                        kLm, 0);

  int shift_big_band0 = 0;
  int shift_big_band1 = 0;
  celt_ener expected_big_band0 =
      compute_expected_band_energy_fixed(&mode, spectrum_big, 0, 0, kLm,
                                         &shift_big_band0);
  celt_ener expected_big_band1 =
      compute_expected_band_energy_fixed(&mode, spectrum_big, 1, 0, kLm,
                                         &shift_big_band1);

  if (shift_big_band0 <= 0 || shift_big_band1 <= 0) {
    fprintf(stderr, "expected shift > 0 for big bands, got %d/%d\n",
            shift_big_band0, shift_big_band1);
    failures++;
  }
  if (band_e[0] != expected_big_band0) {
    fprintf(stderr, "big band 0 mismatch: got %d expected %d\n",
            (int)band_e[0], (int)expected_big_band0);
    failures++;
  }
  if (band_e[1] != expected_big_band1) {
    fprintf(stderr, "big band 1 mismatch: got %d expected %d\n",
            (int)band_e[1], (int)expected_big_band1);
    failures++;
  }

  return failures ? 1 : 0;
#else
  static int nearly_equal(opus_val32 a, opus_val32 b, opus_val32 tol) {
    return fabsf(a - b) <= tol;
  }
  enum {
    kChannels = 2,
    kLm = 1,
    kShortMdct = 4,
    kNbEBands = 2,
    kN = kShortMdct << kLm,
    kSpectrumLen = kChannels * kN
  };

  const opus_int16 e_bands[kNbEBands + 1] = {0, 2, 4};
  const opus_int16 logN[kNbEBands] = {8, 8};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, logN, kNbEBands, kShortMdct);

  celt_sig spectrum[kSpectrumLen];
  for (int idx = 0; idx < kSpectrumLen; ++idx) {
    spectrum[idx] = (celt_sig)sinf((float)idx * 0.13f - 0.5f);
  }

  celt_ener band_e[kNbEBands * kChannels];
  compute_band_energies(&mode, spectrum, band_e, mode.nbEBands, kChannels, kLm,
                        0);

  int failures = 0;
  for (int c = 0; c < kChannels; ++c) {
    for (int b = 0; b < mode.nbEBands; ++b) {
      int start = (mode.eBands[b] << kLm) + c * kN;
      int stop = (mode.eBands[b + 1] << kLm) + c * kN;
      opus_val32 sum = 0.0f;
      for (int i = start; i < stop; ++i) {
        sum += spectrum[i] * spectrum[i];
      }
      opus_val32 expected = sqrtf(1e-27f + sum);
      int idx = b + c * mode.nbEBands;
      if (!nearly_equal(band_e[idx], expected, 1e-6f)) {
        fprintf(stderr,
                "mismatch channel %d band %d: got %g expected %g\n",
                c, b, (double)band_e[idx], (double)expected);
        failures++;
      }
    }
  }

  return failures ? 1 : 0;
#endif
}
