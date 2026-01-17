#include <stdio.h>

#include "celt.h"
#include "modes.h"
#include "opus_custom.h"

int main(void) {
  enum { kSampleRate = 48000, kFrameSize = 120 };

  int err = 0;
  OpusCustomMode *mode = opus_custom_mode_create(kSampleRate, kFrameSize, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "opus_custom_mode_create failed: %d\n", err);
    return 1;
  }

  const int lm = 0;
  const int is_transient = 0;
  const int coded_channels = 1;
  const int output_channels = 1;
  const int start = 0;
  const int eff_end = mode->effEBands;
  const int downsample = 1;
  const int silence = 0;
  const int n = mode->shortMdctSize << lm;
  const int nb_ebands = mode->nbEBands;
  const int shift = is_transient ? mode->maxLM : mode->maxLM - lm;
  const int mdct_len = mode->mdct.n >> shift;
  const int n2 = mdct_len >> 1;
  const int output_len = (mode->overlap >> 1) + n2;

  celt_norm x[n];
  for (int i = 0; i < n; ++i) {
    x[i] = (celt_norm)(i * 0.01f - 0.5f);
  }

  celt_glog old_band_e[nb_ebands];
  for (int i = 0; i < nb_ebands; ++i) {
    old_band_e[i] = (celt_glog)(0.5f + 0.01f * i);
  }

  celt_sig output[output_len];
  for (int i = 0; i < output_len; ++i) {
    output[i] = 0.0f;
  }

  celt_sig *out_syn[1] = {output};
  celt_synthesis(
      mode,
      x,
      out_syn,
      old_band_e,
      start,
      eff_end,
      coded_channels,
      output_channels,
      is_transient,
      lm,
      downsample,
      silence,
      0);

  for (int i = 0; i < output_len; ++i) {
    printf("celt_synthesis_out[%d]=%.9e\n", i, (double)output[i]);
  }

#ifdef CUSTOM_MODES
  opus_custom_mode_destroy(mode);
#endif
  return 0;
}
