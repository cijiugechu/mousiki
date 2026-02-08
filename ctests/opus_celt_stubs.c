#include "arch.h"
#include "entcode.h"
#include "entdec.h"
#include "entenc.h"
#include "quant_bands.h"
#include "vq.h"

#ifdef FIXED_POINT
const signed char eMeans[25] = {
    103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74,
    76, 71, 60, 60, 60, 60, 60,
};
#else
const opus_val16 eMeans[25] = {0};
#endif

unsigned alg_quant(celt_norm *X, int N, int K, int spread, int B, ec_enc *enc,
                   opus_val32 gain, int resynth, int arch) {
  (void)X;
  (void)N;
  (void)K;
  (void)spread;
  (void)B;
  (void)enc;
  (void)gain;
  (void)resynth;
  (void)arch;
  return 0;
}

unsigned alg_unquant(celt_norm *X, int N, int K, int spread, int B,
                     ec_dec *dec, opus_val32 gain) {
  (void)X;
  (void)N;
  (void)K;
  (void)spread;
  (void)B;
  (void)dec;
  (void)gain;
  return 0;
}

void renormalise_vector(celt_norm *X, int N, opus_val32 gain, int arch) {
  (void)X;
  (void)N;
  (void)gain;
  (void)arch;
}

int stereo_itheta(const celt_norm *X, const celt_norm *Y, int stereo, int N,
                  int arch) {
  (void)X;
  (void)Y;
  (void)stereo;
  (void)N;
  (void)arch;
  return 0;
}

unsigned ec_decode(ec_dec *_this, unsigned _ft) {
  (void)_this;
  (void)_ft;
  return 0;
}

void ec_dec_update(ec_dec *_this, unsigned _fl, unsigned _fh, unsigned _ft) {
  (void)_this;
  (void)_fl;
  (void)_fh;
  (void)_ft;
}

int ec_dec_bit_logp(ec_dec *_this, unsigned _logp) {
  (void)_this;
  (void)_logp;
  return 0;
}

opus_uint32 ec_dec_uint(ec_dec *_this, opus_uint32 _ft) {
  (void)_this;
  (void)_ft;
  return 0;
}

opus_uint32 ec_dec_bits(ec_dec *_this, unsigned _ftb) {
  (void)_this;
  (void)_ftb;
  return 0;
}

void ec_encode(ec_enc *_this, unsigned _fl, unsigned _fh, unsigned _ft) {
  (void)_this;
  (void)_fl;
  (void)_fh;
  (void)_ft;
}

void ec_enc_bit_logp(ec_enc *_this, int _val, unsigned _logp) {
  (void)_this;
  (void)_val;
  (void)_logp;
}

void ec_enc_uint(ec_enc *_this, opus_uint32 _fl, opus_uint32 _ft) {
  (void)_this;
  (void)_fl;
  (void)_ft;
}

void ec_enc_bits(ec_enc *_this, opus_uint32 _fl, unsigned _ftb) {
  (void)_this;
  (void)_fl;
  (void)_ftb;
}

opus_uint32 ec_tell_frac(ec_ctx *_this) {
  (void)_this;
  return 0;
}
