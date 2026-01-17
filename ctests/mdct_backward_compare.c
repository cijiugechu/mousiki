#include <math.h>
#include <stdio.h>

#include "mdct.h"
#include "modes.h"
#include "opus_custom.h"

int main(void) {
  enum { kStride = 1 };

  int err = 0;
  OpusCustomMode *mode = opus_custom_mode_create(48000, 120, &err);
  if (mode == NULL || err != OPUS_OK) {
    fprintf(stderr, "opus_custom_mode_create failed: %d\n", err);
    return 1;
  }

  const int overlap = mode->overlap;
  const int n = mode->mdct.n;
  const int n2 = n / 2;
  const int output_len = (overlap / 2 + n2) > overlap ? (overlap / 2 + n2)
                                                      : overlap;

  const celt_coef *window = mode->window;
  if (window == NULL) {
    fprintf(stderr, "mode window is null\n");
    return 1;
  }

  kiss_fft_scalar input[n2];
  for (int i = 0; i < n2; ++i) {
    input[i] = cosf((float)i * 0.19f);
  }

  kiss_fft_scalar output[output_len];
  for (int i = 0; i < output_len; ++i) {
    output[i] = 0.0f;
  }

  clt_mdct_backward(&mode->mdct, input, output, window, overlap, 0, kStride, 0);

  for (int i = 0; i < output_len; ++i) {
    printf("mdct_backward_out[%d]=%.9e\n", i, (double)output[i]);
  }

#ifdef CUSTOM_MODES
  opus_custom_mode_destroy(mode);
#endif
  return 0;
}
