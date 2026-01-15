#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bands.h"
#include "modes.h"

static void init_dummy_mode(OpusCustomMode *mode, const opus_int16 *e_bands,
                            int nb_ebands, int short_mdct_size) {
  memset(mode, 0, sizeof(*mode));
  mode->Fs = 48000;
  mode->nbEBands = nb_ebands;
  mode->effEBands = nb_ebands;
  mode->eBands = e_bands;
  mode->shortMdctSize = short_mdct_size;
}

static int nearly_equal(opus_val32 a, opus_val32 b, opus_val32 tol) {
  return fabsf(a - b) <= tol;
}

int main(void) {
  enum {
    kChannels = 2,
    kLm = 1,
    kShortMdct = 4,
    kNbEBands = 2,
    kN = kShortMdct << kLm,
    kSpectrumLen = kChannels * kN
  };

  const opus_int16 e_bands[kNbEBands + 1] = {0, 2, 4};
  OpusCustomMode mode;
  init_dummy_mode(&mode, e_bands, kNbEBands, kShortMdct);

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
}
