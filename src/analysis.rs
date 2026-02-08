#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

#[cfg(test)]
extern crate std;

#[cfg(test)]
use alloc::vec::Vec;

use core::cmp::{max, min};
use core::f32::consts::{LOG2_E, PI};

use libm::{fmaf, floorf, log, log10f, logf, sqrt, sqrtf};

use crate::celt::{
    AnalysisInfo, CELT_SIG_SCALE, KissFftCpx, KissFftState, OpusCustomMode, celt_maxabs32,
    fast_atan2f, float2int, opus_fft, opus_select_arch,
};
use crate::mlp::{
    LAYER0, LAYER1, LAYER2, MAX_NEURONS, analysis_compute_dense, analysis_compute_gru,
};

const NB_FRAMES: usize = 8;
const NB_TBANDS: usize = 18;
const ANALYSIS_BUF_SIZE: usize = 720;
const ANALYSIS_COUNT_MAX: i32 = 10_000;
const DETECT_SIZE: usize = 100;
const NB_TONAL_SKIP_BANDS: usize = 9;
const TRANSITION_PENALTY: f32 = 10.0;
const LEAK_BANDS: usize = 19;
const SCALE_ENER: f32 = 1.0 / (32_768.0 * 32_768.0);
const INITIAL_MEM_FILL: usize = 240;

#[inline]
fn mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
    fmaf(a, b, c)
}

#[inline]
fn bin_energy_sum(r: f32, i: f32, mirror_r: f32, mirror_i: f32) -> f32 {
    let mr2 = mirror_r * mirror_r;
    let sum_r = mul_add_f32(r, r, mr2);
    let sum_i = mul_add_f32(i, i, sum_r);
    mul_add_f32(mirror_i, mirror_i, sum_i)
}

const DCT_TABLE: [f32; 128] = [
    0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000,
    0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000, 0.250_000,
    0.351_851, 0.338_330, 0.311_806, 0.273_300, 0.224_292, 0.166_664, 0.102_631, 0.034_654,
    -0.034_654, -0.102_631, -0.166_664, -0.224_292, -0.273_300, -0.311_806, -0.338_330, -0.351_851,
    0.346_760, 0.293_969, 0.196_424, 0.068_975, -0.068_975, -0.196_424, -0.293_969, -0.346_760,
    -0.346_760, -0.293_969, -0.196_424, -0.068_975, 0.068_975, 0.196_424, 0.293_969, 0.346_760,
    0.338_330, 0.224_292, 0.034_654, -0.166_664, -0.311_806, -0.351_851, -0.273_300, -0.102_631,
    0.102_631, 0.273_300, 0.351_851, 0.311_806, 0.166_664, -0.034_654, -0.224_292, -0.338_330,
    0.326_641, 0.135_299, -0.135_299, -0.326_641, -0.326_641, -0.135_299, 0.135_299, 0.326_641,
    0.326_641, 0.135_299, -0.135_299, -0.326_641, -0.326_641, -0.135_299, 0.135_299, 0.326_641,
    0.311_806, 0.034_654, -0.273_300, -0.338_330, -0.102_631, 0.224_292, 0.351_851, 0.166_664,
    -0.166_664, -0.351_851, -0.224_292, 0.102_631, 0.338_330, 0.273_300, -0.034_654, -0.311_806,
    0.293_969, -0.068_975, -0.346_760, -0.196_424, 0.196_424, 0.346_760, 0.068_975, -0.293_969,
    -0.293_969, 0.068_975, 0.346_760, 0.196_424, -0.196_424, -0.346_760, -0.068_975, 0.293_969,
    0.273_300, -0.166_664, -0.338_330, 0.034_654, 0.351_851, 0.102_631, -0.311_806, -0.224_292,
    0.224_292, 0.311_806, -0.102_631, -0.351_851, -0.034_654, 0.338_330, 0.166_664, -0.273_300,
];

const ANALYSIS_WINDOW: [f32; 240] = [
    0.000_043, 0.000_171, 0.000_385, 0.000_685, 0.001_071, 0.001_541, 0.002_098, 0.002_739,
    0.003_466, 0.004_278, 0.005_174, 0.006_156, 0.007_222, 0.008_373, 0.009_607, 0.010_926,
    0.012_329, 0.013_815, 0.015_385, 0.017_037, 0.018_772, 0.020_590, 0.022_490, 0.024_472,
    0.026_535, 0.028_679, 0.030_904, 0.033_210, 0.035_595, 0.038_060, 0.040_604, 0.043_227,
    0.045_928, 0.048_707, 0.051_564, 0.054_497, 0.057_506, 0.060_591, 0.063_752, 0.066_987,
    0.070_297, 0.073_680, 0.077_136, 0.080_665, 0.084_265, 0.087_937, 0.091_679, 0.095_492,
    0.099_373, 0.103_323, 0.107_342, 0.111_427, 0.115_579, 0.119_797, 0.124_080, 0.128_428,
    0.132_839, 0.137_313, 0.141_849, 0.146_447, 0.151_105, 0.155_823, 0.160_600, 0.165_435,
    0.170_327, 0.175_276, 0.180_280, 0.185_340, 0.190_453, 0.195_619, 0.200_838, 0.206_107,
    0.211_427, 0.216_797, 0.222_215, 0.227_680, 0.233_193, 0.238_751, 0.244_353, 0.250_000,
    0.255_689, 0.261_421, 0.267_193, 0.273_005, 0.278_856, 0.284_744, 0.290_670, 0.296_632,
    0.302_628, 0.308_658, 0.314_721, 0.320_816, 0.326_941, 0.333_097, 0.339_280, 0.345_492,
    0.351_729, 0.357_992, 0.364_280, 0.370_590, 0.376_923, 0.383_277, 0.389_651, 0.396_044,
    0.402_455, 0.408_882, 0.415_325, 0.421_783, 0.428_254, 0.434_737, 0.441_231, 0.447_736,
    0.454_249, 0.460_770, 0.467_298, 0.473_832, 0.480_370, 0.486_912, 0.493_455, 0.500_000,
    0.506_545, 0.513_088, 0.519_630, 0.526_168, 0.532_702, 0.539_230, 0.545_751, 0.552_264,
    0.558_769, 0.565_263, 0.571_746, 0.578_217, 0.584_675, 0.591_118, 0.597_545, 0.603_956,
    0.610_349, 0.616_723, 0.623_077, 0.629_410, 0.635_720, 0.642_008, 0.648_271, 0.654_508,
    0.660_720, 0.666_903, 0.673_059, 0.679_184, 0.685_279, 0.691_342, 0.697_372, 0.703_368,
    0.709_330, 0.715_256, 0.721_144, 0.726_995, 0.732_807, 0.738_579, 0.744_311, 0.750_000,
    0.755_647, 0.761_249, 0.766_807, 0.772_320, 0.777_785, 0.783_203, 0.788_573, 0.793_893,
    0.799_162, 0.804_381, 0.809_547, 0.814_660, 0.819_720, 0.824_724, 0.829_673, 0.834_565,
    0.839_400, 0.844_177, 0.848_895, 0.853_553, 0.858_151, 0.862_687, 0.867_161, 0.871_572,
    0.875_920, 0.880_203, 0.884_421, 0.888_573, 0.892_658, 0.896_677, 0.900_627, 0.904_508,
    0.908_321, 0.912_063, 0.915_735, 0.919_335, 0.922_864, 0.926_320, 0.929_703, 0.933_013,
    0.936_248, 0.939_409, 0.942_494, 0.945_503, 0.948_436, 0.951_293, 0.954_072, 0.956_773,
    0.959_396, 0.961_940, 0.964_405, 0.966_790, 0.969_096, 0.971_321, 0.973_465, 0.975_528,
    0.977_510, 0.979_410, 0.981_228, 0.982_963, 0.984_615, 0.986_185, 0.987_671, 0.989_074,
    0.990_393, 0.991_627, 0.992_778, 0.993_844, 0.994_826, 0.995_722, 0.996_534, 0.997_261,
    0.997_902, 0.998_459, 0.998_929, 0.999_315, 0.999_615, 0.999_829, 0.999_957, 1.000_000,
];

#[cfg(test)]
mod fft_trace {
    extern crate std;

    use std::env;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::vec::Vec;

    use super::{KissFftCpx, SCALE_ENER};

    pub(crate) struct TraceConfig {
        bins: Vec<usize>,
        all_bins: bool,
        want_bits: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static TRACE_FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn set_frame_index(idx: usize) {
        TRACE_FRAME_INDEX.store(idx, Ordering::Relaxed);
    }

    pub(crate) fn frame_index() -> usize {
        TRACE_FRAME_INDEX.load(Ordering::Relaxed)
    }

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env::var_os("ANALYSIS_TRACE").is_some()
                    || env::var_os("ANALYSIS_TRACE_BINS").is_some()
                    || env_truthy("ANALYSIS_TRACE_FFT_BITS");
                if !enabled {
                    return None;
                }
                match env::var("ANALYSIS_TRACE_BINS") {
                    Ok(value) => Some(parse_bins(&value)),
                    Err(_) => Some(default_bins()),
                }
            })
            .as_ref()
    }

    pub(crate) fn should_trace_bin(cfg: &TraceConfig, bin: usize) -> bool {
        if cfg.all_bins {
            return true;
        }
        cfg.bins.iter().any(|&value| value == bin)
    }

    fn default_bins() -> TraceConfig {
        let mut bins = Vec::new();
        bins.push(1);
        bins.push(61);
        TraceConfig {
            bins,
            all_bins: false,
            want_bits: env_truthy("ANALYSIS_TRACE_FFT_BITS"),
        }
    }

    fn parse_bins(value: &str) -> TraceConfig {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return default_bins();
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return TraceConfig {
                bins: Vec::new(),
                all_bins: true,
                want_bits: env_truthy("ANALYSIS_TRACE_FFT_BITS"),
            };
        }
        let mut bins = Vec::new();
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == '\t') {
            if token.is_empty() {
                continue;
            }
            if let Ok(bin) = token.parse::<usize>() {
                if (1..240).contains(&bin) {
                    bins.push(bin);
                }
            }
        }
        if bins.is_empty() {
            default_bins()
        } else {
            TraceConfig {
                bins,
                all_bins: false,
                want_bits: env_truthy("ANALYSIS_TRACE_FFT_BITS"),
            }
        }
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dump_bin(
        frame_idx: usize,
        bin: usize,
        input_fft: &[KissFftCpx; 480],
        output_fft: &[KissFftCpx; 480],
        x1r: f32,
        x1i: f32,
        x2r: f32,
        x2i: f32,
        atan1: f32,
        atan2: f32,
        angle: f32,
        angle2: f32,
        d_angle: f32,
        d2_angle: f32,
        d_angle2: f32,
        d2_angle2: f32,
        d2_angle_int: i32,
        d2_angle2_int: i32,
        bin_e: f32,
        tonality: f32,
        tonality2: f32,
    ) {
        let mirror = 480 - bin;
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].input_fft.r={:.9e}",
            input_fft[bin].r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].input_fft.i={:.9e}",
            input_fft[bin].i as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].input_fft_mirror.r={:.9e}",
            input_fft[mirror].r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].input_fft_mirror.i={:.9e}",
            input_fft[mirror].i as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].output_fft.r={:.9e}",
            output_fft[bin].r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].output_fft.i={:.9e}",
            output_fft[bin].i as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].output_fft_mirror.r={:.9e}",
            output_fft[mirror].r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].output_fft_mirror.i={:.9e}",
            output_fft[mirror].i as f64
        );
        if let Some(cfg) = config() {
            if cfg.want_bits {
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].input_fft_bits.r=0x{:08x}",
                    input_fft[bin].r.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].input_fft_bits.i=0x{:08x}",
                    input_fft[bin].i.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].input_fft_mirror_bits.r=0x{:08x}",
                    input_fft[mirror].r.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].input_fft_mirror_bits.i=0x{:08x}",
                    input_fft[mirror].i.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].output_fft_bits.r=0x{:08x}",
                    output_fft[bin].r.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].output_fft_bits.i=0x{:08x}",
                    output_fft[bin].i.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].output_fft_mirror_bits.r=0x{:08x}",
                    output_fft[mirror].r.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].output_fft_mirror_bits.i=0x{:08x}",
                    output_fft[mirror].i.to_bits()
                );
                let r2 = output_fft[bin].r * output_fft[bin].r;
                let mr2 = output_fft[mirror].r * output_fft[mirror].r;
                let i2 = output_fft[bin].i * output_fft[bin].i;
                let mi2 = output_fft[mirror].i * output_fft[mirror].i;
                let sum0 = r2 + mr2;
                let sum1 = i2 + mi2;
                let sum = sum0 + sum1;
                let scaled = sum * SCALE_ENER;
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_r2={:.9e}",
                    r2 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_mr2={:.9e}",
                    mr2 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_i2={:.9e}",
                    i2 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_mi2={:.9e}",
                    mi2 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum0={:.9e}",
                    sum0 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum1={:.9e}",
                    sum1 as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum={:.9e}",
                    sum as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_scaled={:.9e}",
                    scaled as f64
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_r2_bits=0x{:08x}",
                    r2.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_mr2_bits=0x{:08x}",
                    mr2.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_i2_bits=0x{:08x}",
                    i2.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_mi2_bits=0x{:08x}",
                    mi2.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum0_bits=0x{:08x}",
                    sum0.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum1_bits=0x{:08x}",
                    sum1.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_sum_bits=0x{:08x}",
                    sum.to_bits()
                );
                std::println!(
                    "analysis_fft[{frame_idx}].bin[{bin}].bin_e_scaled_bits=0x{:08x}",
                    scaled.to_bits()
                );
            }
        }
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].x1r={:.9e}",
            x1r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].x1i={:.9e}",
            x1i as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].x2r={:.9e}",
            x2r as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].x2i={:.9e}",
            x2i as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].fast_atan2_x1={:.9e}",
            atan1 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].fast_atan2_x2={:.9e}",
            atan2 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].angle={:.9e}",
            angle as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].angle2={:.9e}",
            angle2 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].d_angle={:.9e}",
            d_angle as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].d2_angle={:.9e}",
            d2_angle as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].d_angle2={:.9e}",
            d_angle2 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].d2_angle2={:.9e}",
            d2_angle2 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].float2int_d2_angle={d2_angle_int}"
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].float2int_d2_angle2={d2_angle2_int}"
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].bin_e={:.9e}",
            bin_e as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].bin_e_bits=0x{:08x}",
            bin_e.to_bits()
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality={:.9e}",
            tonality as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality_bits=0x{:08x}",
            tonality.to_bits()
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality2={:.9e}",
            tonality2 as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality2_bits=0x{:08x}",
            tonality2.to_bits()
        );
    }

    pub(crate) fn dump_tonality_smoothed(frame_idx: usize, bin: usize, tonality: f32) {
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality_smoothed={:.9e}",
            tonality as f64
        );
        std::println!(
            "analysis_fft[{frame_idx}].bin[{bin}].tonality_smoothed_bits=0x{:08x}",
            tonality.to_bits()
        );
    }
}

#[cfg(test)]
mod tonality_trace {
    extern crate std;

    use std::env;
    use std::sync::OnceLock;
    use std::vec::Vec;

    use super::{NB_TBANDS, TBANDS};

    pub(crate) struct TraceConfig {
        bands: Vec<usize>,
        all_bands: bool,
        frame: Option<usize>,
        want_bits: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE")
                    || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BANDS")
                    || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_FRAME")
                    || env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BITS");
                if !enabled {
                    return None;
                }
                let frame = env::var("ANALYSIS_TRACE_TONALITY_SLOPE_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let bands = match env::var("ANALYSIS_TRACE_TONALITY_SLOPE_BANDS") {
                    Ok(value) => parse_bands(&value),
                    Err(_) => default_bands(),
                };
                Some(TraceConfig {
                    bands: bands.bands,
                    all_bands: bands.all_bands,
                    frame,
                    want_bits: env_truthy("ANALYSIS_TRACE_TONALITY_SLOPE_BITS"),
                })
            })
            .as_ref()
    }

    pub(crate) fn should_trace_band(cfg: &TraceConfig, band: usize) -> bool {
        if cfg.all_bands {
            return true;
        }
        cfg.bands.iter().any(|&value| value == band)
    }

    pub(crate) fn should_trace_bin(cfg: &TraceConfig, bin: usize) -> bool {
        if cfg.all_bands {
            return true;
        }
        cfg.bands.iter().any(|&band| bin >= TBANDS[band] && bin < TBANDS[band + 1])
    }

    pub(crate) fn frame_matches(cfg: &TraceConfig, frame: usize) -> bool {
        cfg.frame.map_or(true, |value| value == frame)
    }

    pub(crate) fn want_bits(cfg: &TraceConfig) -> bool {
        cfg.want_bits
    }

    pub(crate) fn dump_band(
        cfg: &TraceConfig,
        frame_idx: usize,
        band: usize,
        band_e: f32,
        t_e: f32,
        energy_ratio: f32,
        energy_ratio_num: f32,
        energy_ratio_denom: f32,
        stationarity: f32,
        stationarity_term: f32,
        prev_band_tonality: f32,
        band_tonality: f32,
        slope_pre: f32,
        slope_term: f32,
        slope_acc: f32,
    ) {
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].E={:.9e}",
            band_e as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].tE={:.9e}",
            t_e as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].energy_ratio_num={:.9e}",
            energy_ratio_num as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].energy_ratio_denom={:.9e}",
            energy_ratio_denom as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].energy_ratio={:.9e}",
            energy_ratio as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].stationarity={:.9e}",
            stationarity as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].stationarity_term={:.9e}",
            stationarity_term as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].prev_band_tonality={:.9e}",
            prev_band_tonality as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].band_tonality={:.9e}",
            band_tonality as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].slope_pre={:.9e}",
            slope_pre as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].slope_term={:.9e}",
            slope_term as f64
        );
        std::println!(
            "analysis_tonality[{frame_idx}].band[{band}].slope_acc={:.9e}",
            slope_acc as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].E_bits=0x{:08x}",
                band_e.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].tE_bits=0x{:08x}",
                t_e.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].energy_ratio_num_bits=0x{:08x}",
                energy_ratio_num.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].energy_ratio_denom_bits=0x{:08x}",
                energy_ratio_denom.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].energy_ratio_bits=0x{:08x}",
                energy_ratio.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].stationarity_bits=0x{:08x}",
                stationarity.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].stationarity_term_bits=0x{:08x}",
                stationarity_term.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].prev_band_tonality_bits=0x{:08x}",
                prev_band_tonality.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].band_tonality_bits=0x{:08x}",
                band_tonality.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].slope_pre_bits=0x{:08x}",
                slope_pre.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].slope_term_bits=0x{:08x}",
                slope_term.to_bits()
            );
            std::println!(
                "analysis_tonality[{frame_idx}].band[{band}].slope_acc_bits=0x{:08x}",
                slope_acc.to_bits()
            );
        }
    }

    pub(crate) fn dump_bin(
        cfg: &TraceConfig,
        frame_idx: usize,
        band: usize,
        bin: usize,
        bin_e: f32,
        tonality: f32,
        tonality_clamped: f32,
        t_e_term: f32,
        t_e_pre: f32,
        t_e_acc: f32,
    ) {
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].binE={:.9e}",
            bin_e as f64
        );
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tonality={:.9e}",
            tonality as f64
        );
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tonality_clamped={:.9e}",
            tonality_clamped as f64
        );
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_term={:.9e}",
            t_e_term as f64
        );
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_pre={:.9e}",
            t_e_pre as f64
        );
        std::println!(
            "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_acc={:.9e}",
            t_e_acc as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].binE_bits=0x{:08x}",
                bin_e.to_bits()
            );
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tonality_bits=0x{:08x}",
                tonality.to_bits()
            );
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tonality_clamped_bits=0x{:08x}",
                tonality_clamped.to_bits()
            );
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_term_bits=0x{:08x}",
                t_e_term.to_bits()
            );
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_pre_bits=0x{:08x}",
                t_e_pre.to_bits()
            );
            std::println!(
                "analysis_tonality_bin[{frame_idx}].band[{band}].bin[{bin}].tE_acc_bits=0x{:08x}",
                t_e_acc.to_bits()
            );
        }
    }

    pub(crate) fn dump_bin_raw(
        cfg: &TraceConfig,
        frame_idx: usize,
        bin: usize,
        x1r: f32,
        x1i: f32,
        x2r: f32,
        x2i: f32,
        atan1: f32,
        angle: f32,
        d_angle: f32,
        atan2: f32,
        angle2: f32,
        d_angle2: f32,
        d2_angle: f32,
        d2_angle2: f32,
        d2_angle_int: i32,
        d2_angle2_int: i32,
        mod1_pre: f32,
        mod2_pre: f32,
        mod1: f32,
        mod2: f32,
        avg_mod: f32,
        tonality_raw: f32,
        tonality2: f32,
    ) {
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].x1r={:.9e}",
            x1r as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].x1i={:.9e}",
            x1i as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].x2r={:.9e}",
            x2r as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].x2i={:.9e}",
            x2i as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].atan1={:.9e}",
            atan1 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].angle={:.9e}",
            angle as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d_angle={:.9e}",
            d_angle as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].atan2={:.9e}",
            atan2 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].angle2={:.9e}",
            angle2 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d_angle2={:.9e}",
            d_angle2 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle={:.9e}",
            d2_angle as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle2={:.9e}",
            d2_angle2 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle_int={d2_angle_int}"
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle2_int={d2_angle2_int}"
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod1_pre={:.9e}",
            mod1_pre as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod2_pre={:.9e}",
            mod2_pre as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod1={:.9e}",
            mod1 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod2={:.9e}",
            mod2 as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].avg_mod={:.9e}",
            avg_mod as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].tonality={:.9e}",
            tonality_raw as f64
        );
        std::println!(
            "analysis_tonality_raw[{frame_idx}].bin[{bin}].tonality2={:.9e}",
            tonality2 as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].x1r_bits=0x{:08x}",
                x1r.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].x1i_bits=0x{:08x}",
                x1i.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].x2r_bits=0x{:08x}",
                x2r.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].x2i_bits=0x{:08x}",
                x2i.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].atan1_bits=0x{:08x}",
                atan1.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].angle_bits=0x{:08x}",
                angle.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].d_angle_bits=0x{:08x}",
                d_angle.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].atan2_bits=0x{:08x}",
                atan2.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].angle2_bits=0x{:08x}",
                angle2.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].d_angle2_bits=0x{:08x}",
                d_angle2.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle_bits=0x{:08x}",
                d2_angle.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].d2_angle2_bits=0x{:08x}",
                d2_angle2.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod1_pre_bits=0x{:08x}",
                mod1_pre.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod2_pre_bits=0x{:08x}",
                mod2_pre.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod1_bits=0x{:08x}",
                mod1.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].mod2_bits=0x{:08x}",
                mod2.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].avg_mod_bits=0x{:08x}",
                avg_mod.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].tonality_bits=0x{:08x}",
                tonality_raw.to_bits()
            );
            std::println!(
                "analysis_tonality_raw[{frame_idx}].bin[{bin}].tonality2_bits=0x{:08x}",
                tonality2.to_bits()
            );
        }
    }

    pub(crate) fn dump_bin_smooth(
        cfg: &TraceConfig,
        frame_idx: usize,
        bin: usize,
        tonality_pre: f32,
        tonality2_prev: f32,
        tonality2: f32,
        tonality2_next: f32,
        tt: f32,
        tonality_smoothed: f32,
    ) {
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality_pre={:.9e}",
            tonality_pre as f64
        );
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2_prev={:.9e}",
            tonality2_prev as f64
        );
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2={:.9e}",
            tonality2 as f64
        );
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2_next={:.9e}",
            tonality2_next as f64
        );
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tt={:.9e}",
            tt as f64
        );
        std::println!(
            "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality_smoothed={:.9e}",
            tonality_smoothed as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality_pre_bits=0x{:08x}",
                tonality_pre.to_bits()
            );
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2_prev_bits=0x{:08x}",
                tonality2_prev.to_bits()
            );
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2_bits=0x{:08x}",
                tonality2.to_bits()
            );
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality2_next_bits=0x{:08x}",
                tonality2_next.to_bits()
            );
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tt_bits=0x{:08x}",
                tt.to_bits()
            );
            std::println!(
                "analysis_tonality_smooth[{frame_idx}].bin[{bin}].tonality_smoothed_bits=0x{:08x}",
                tonality_smoothed.to_bits()
            );
        }
    }

    pub(crate) fn dump_stationarity_sample(
        cfg: &TraceConfig,
        frame_idx: usize,
        band: usize,
        idx: usize,
        band_e: f32,
        sqrt_e: f32,
    ) {
        std::println!(
            "analysis_stationarity[{frame_idx}].band[{band}].E[{idx}]={:.9e}",
            band_e as f64
        );
        std::println!(
            "analysis_stationarity[{frame_idx}].band[{band}].sqrtE[{idx}]={:.9e}",
            sqrt_e as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_stationarity[{frame_idx}].band[{band}].E_bits[{idx}]=0x{:08x}",
                band_e.to_bits()
            );
            std::println!(
                "analysis_stationarity[{frame_idx}].band[{band}].sqrtE_bits[{idx}]=0x{:08x}",
                sqrt_e.to_bits()
            );
        }
    }

    pub(crate) fn dump_stationarity_totals(
        cfg: &TraceConfig,
        frame_idx: usize,
        band: usize,
        l1: f32,
        l2: f32,
        denom: f32,
    ) {
        std::println!(
            "analysis_stationarity[{frame_idx}].band[{band}].L1={:.9e}",
            l1 as f64
        );
        std::println!(
            "analysis_stationarity[{frame_idx}].band[{band}].L2={:.9e}",
            l2 as f64
        );
        std::println!(
            "analysis_stationarity[{frame_idx}].band[{band}].denom={:.9e}",
            denom as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_stationarity[{frame_idx}].band[{band}].L1_bits=0x{:08x}",
                l1.to_bits()
            );
            std::println!(
                "analysis_stationarity[{frame_idx}].band[{band}].L2_bits=0x{:08x}",
                l2.to_bits()
            );
            std::println!(
                "analysis_stationarity[{frame_idx}].band[{band}].denom_bits=0x{:08x}",
                denom.to_bits()
            );
        }
    }

    struct BandSelection {
        bands: Vec<usize>,
        all_bands: bool,
    }

    fn default_bands() -> BandSelection {
        BandSelection {
            bands: Vec::new(),
            all_bands: true,
        }
    }

    fn parse_bands(value: &str) -> BandSelection {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return default_bands();
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return default_bands();
        }
        let mut bands = Vec::new();
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == '\t') {
            if token.is_empty() {
                continue;
            }
            if let Ok(band) = token.parse::<usize>() {
                if band < NB_TBANDS {
                    bands.push(band);
                }
            }
        }
        if bands.is_empty() {
            default_bands()
        } else {
            BandSelection {
                bands,
                all_bands: false,
            }
        }
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod bin28_trace {
    extern crate std;

    use std::env;

    pub(crate) fn enabled(frame_idx: usize) -> bool {
        let enabled = env::var_os("ANALYSIS_TRACE_BIN28").is_some();
        if !enabled {
            return false;
        }
        let frame = env::var("ANALYSIS_TRACE_BIN28_FRAME")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        frame.map_or(true, |value| value == frame_idx)
    }
}

#[cfg(test)]
mod pitch_ratio_trace {
    extern crate std;

    use std::env;
    use std::sync::OnceLock;
    use std::vec::Vec;

    use super::NB_TBANDS;

    pub(crate) struct TraceConfig {
        bands: Vec<usize>,
        all_bands: bool,
        frame: Option<usize>,
        want_bits: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env_truthy("ANALYSIS_TRACE_PITCH_RATIO")
                    || env_truthy("ANALYSIS_TRACE_PITCH_RATIO_BANDS")
                    || env_truthy("ANALYSIS_TRACE_PITCH_RATIO_FRAME")
                    || env_truthy("ANALYSIS_TRACE_PITCH_RATIO_BITS");
                if !enabled {
                    return None;
                }
                let frame = env::var("ANALYSIS_TRACE_PITCH_RATIO_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let bands = match env::var("ANALYSIS_TRACE_PITCH_RATIO_BANDS") {
                    Ok(value) => parse_bands(&value),
                    Err(_) => default_bands(),
                };
                Some(TraceConfig {
                    bands: bands.bands,
                    all_bands: bands.all_bands,
                    frame,
                    want_bits: env_truthy("ANALYSIS_TRACE_PITCH_RATIO_BITS"),
                })
            })
            .as_ref()
    }

    pub(crate) fn should_trace_band(cfg: &TraceConfig, band: usize) -> bool {
        if cfg.all_bands {
            return true;
        }
        cfg.bands.iter().any(|&value| value == band)
    }

    pub(crate) fn frame_matches(cfg: &TraceConfig, frame: usize) -> bool {
        cfg.frame.map_or(true, |value| value == frame)
    }

    pub(crate) fn dump_band(
        cfg: &TraceConfig,
        frame_idx: usize,
        band: usize,
        band_start: usize,
        band_end: usize,
        band_e: f32,
        max_e: f32,
        below_total: f32,
        above_total: f32,
    ) {
        let below_bucket = i32::from(band_start < 64);
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].band_start={band_start}"
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].band_end={band_end}"
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].below_bucket={below_bucket}"
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].bandE={:.9e}",
            band_e as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].maxE={:.9e}",
            max_e as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].below_total={:.9e}",
            below_total as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].band[{band}].above_total={:.9e}",
            above_total as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].band[{band}].bandE_bits=0x{:08x}",
                band_e.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].band[{band}].maxE_bits=0x{:08x}",
                max_e.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].band[{band}].below_bits=0x{:08x}",
                below_total.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].band[{band}].above_bits=0x{:08x}",
                above_total.to_bits()
            );
        }
    }

    pub(crate) fn dump_high_band(
        cfg: &TraceConfig,
        frame_idx: usize,
        hp_ener: f32,
        e_high: f32,
        above_total: f32,
    ) {
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].high.hp_ener={:.9e}",
            hp_ener as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].high.e_high={:.9e}",
            e_high as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].high.above_total={:.9e}",
            above_total as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].high.hp_ener_bits=0x{:08x}",
                hp_ener.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].high.e_high_bits=0x{:08x}",
                e_high.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].high.above_bits=0x{:08x}",
                above_total.to_bits()
            );
        }
    }

    pub(crate) fn dump_total(
        cfg: &TraceConfig,
        frame_idx: usize,
        below_total: f32,
        above_total: f32,
        ratio: f32,
    ) {
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].below_total={:.9e}",
            below_total as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].above_total={:.9e}",
            above_total as f64
        );
        std::println!(
            "analysis_pitch_ratio[{frame_idx}].ratio={:.9e}",
            ratio as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].below_bits=0x{:08x}",
                below_total.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].above_bits=0x{:08x}",
                above_total.to_bits()
            );
            std::println!(
                "analysis_pitch_ratio[{frame_idx}].ratio_bits=0x{:08x}",
                ratio.to_bits()
            );
        }
    }

    struct BandSelection {
        bands: Vec<usize>,
        all_bands: bool,
    }

    fn default_bands() -> BandSelection {
        BandSelection {
            bands: Vec::new(),
            all_bands: true,
        }
    }

    fn parse_bands(value: &str) -> BandSelection {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return default_bands();
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return default_bands();
        }
        let mut bands = Vec::new();
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == '\t') {
            if token.is_empty() {
                continue;
            }
            if let Ok(band) = token.parse::<usize>() {
                if band < NB_TBANDS {
                    bands.push(band);
                }
            }
        }
        if bands.is_empty() {
            default_bands()
        } else {
            BandSelection {
                bands,
                all_bands: false,
            }
        }
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod activity_trace {
    extern crate std;

    use std::env;
    use std::sync::OnceLock;

    use super::{NB_FRAMES, NB_TBANDS};

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
        want_bits: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env_truthy("ANALYSIS_TRACE_ACTIVITY")
                    || env_truthy("ANALYSIS_TRACE_ACTIVITY_FRAME")
                    || env_truthy("ANALYSIS_TRACE_ACTIVITY_BITS");
                if !enabled {
                    return None;
                }
                let frame = env::var("ANALYSIS_TRACE_ACTIVITY_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                Some(TraceConfig {
                    frame,
                    want_bits: env_truthy("ANALYSIS_TRACE_ACTIVITY_BITS"),
                })
            })
            .as_ref()
    }

    pub(crate) fn frame_matches(cfg: &TraceConfig, frame_idx: usize) -> bool {
        cfg.frame.map_or(true, |value| value == frame_idx)
    }

    pub(crate) fn dump_features(cfg: &TraceConfig, frame_idx: usize, features: &[f32]) {
        std::println!(
            "analysis_activity[{frame_idx}].features.len={}",
            features.len()
        );
        for (idx, &value) in features.iter().enumerate() {
            std::println!(
                "analysis_activity[{frame_idx}].features[{idx}]={:.9e}",
                value as f64
            );
        if cfg.want_bits {
                std::println!(
                    "analysis_activity[{frame_idx}].features_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_layer(cfg: &TraceConfig, frame_idx: usize, label: &str, values: &[f32]) {
        std::println!(
            "analysis_activity[{frame_idx}].{label}.len={}",
            values.len()
        );
        for (idx, &value) in values.iter().enumerate() {
            std::println!(
                "analysis_activity[{frame_idx}].{label}[{idx}]={:.9e}",
                value as f64
            );
            if cfg.want_bits {
                std::println!(
                    "analysis_activity[{frame_idx}].{label}_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_probs(cfg: &TraceConfig, frame_idx: usize, probs: &[f32]) {
        std::println!(
            "analysis_activity[{frame_idx}].frame_probs.len={}",
            probs.len()
        );
        for (idx, &value) in probs.iter().enumerate() {
            std::println!(
                "analysis_activity[{frame_idx}].frame_probs[{idx}]={:.9e}",
                value as f64
            );
            if cfg.want_bits {
                std::println!(
                    "analysis_activity[{frame_idx}].frame_probs_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_scalar(cfg: &TraceConfig, frame_idx: usize, label: &str, value: f32) {
        std::println!(
            "analysis_activity[{frame_idx}].{label}={:.9e}",
            value as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_activity[{frame_idx}].{label}_bits=0x{:08x}",
                value.to_bits()
            );
        }
    }

    pub(crate) fn dump_band_scalar(
        cfg: &TraceConfig,
        frame_idx: usize,
        label: &str,
        band: usize,
        value: f32,
    ) {
        std::println!(
            "analysis_activity[{frame_idx}].{label}[{band}]={:.9e}",
            value as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_activity[{frame_idx}].{label}_bits[{band}]=0x{:08x}",
                value.to_bits()
            );
        }
    }

    pub(crate) fn dump_log_hist(cfg: &TraceConfig, frame_idx: usize, log_hist: &[[f32; NB_TBANDS]]) {
        std::println!(
            "analysis_activity[{frame_idx}].log_e_hist.len={NB_FRAMES}"
        );
        for (t, bands) in log_hist.iter().enumerate() {
            for (idx, &value) in bands.iter().enumerate() {
                std::println!(
                    "analysis_activity[{frame_idx}].log_e_hist[{t}][{idx}]={:.9e}",
                    value as f64
                );
                if cfg.want_bits {
                    std::println!(
                        "analysis_activity[{frame_idx}].log_e_hist_bits[{t}][{idx}]=0x{:08x}",
                        value.to_bits()
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod hp_resampler_trace {
    extern crate std;

    use std::env;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::vec::Vec;

    const INACTIVE_FRAME: usize = usize::MAX;

    pub(crate) struct TraceConfig {
        samples: Vec<usize>,
        all_samples: bool,
        frame: Option<usize>,
        call: Option<usize>,
        want_bits: bool,
    }

    pub(crate) struct TraceCall {
        pub(crate) cfg: &'static TraceConfig,
        pub(crate) frame_idx: usize,
        pub(crate) call_idx: usize,
    }

    struct SampleSelection {
        samples: Vec<usize>,
        all_samples: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static TRACE_FRAME_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static CURRENT_FRAME: AtomicUsize = AtomicUsize::new(INACTIVE_FRAME);
    static CALL_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env_truthy("ANALYSIS_TRACE_HP_RESAMPLER")
                    || env_truthy("ANALYSIS_TRACE_HP_RESAMPLER_FRAME")
                    || env_truthy("ANALYSIS_TRACE_HP_RESAMPLER_CALL")
                    || env_truthy("ANALYSIS_TRACE_HP_RESAMPLER_SAMPLES")
                    || env_truthy("ANALYSIS_TRACE_HP_RESAMPLER_BITS");
                if !enabled {
                    return None;
                }
                let frame = env::var("ANALYSIS_TRACE_HP_RESAMPLER_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let call = env::var("ANALYSIS_TRACE_HP_RESAMPLER_CALL")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let samples = match env::var("ANALYSIS_TRACE_HP_RESAMPLER_SAMPLES") {
                    Ok(value) => parse_samples(&value),
                    Err(_) => SampleSelection {
                        samples: Vec::new(),
                        all_samples: true,
                    },
                };
                Some(TraceConfig {
                    samples: samples.samples,
                    all_samples: samples.all_samples,
                    frame,
                    call,
                    want_bits: env_truthy("ANALYSIS_TRACE_HP_RESAMPLER_BITS"),
                })
            })
            .as_ref()
    }

    pub(crate) fn begin_frame() {
        if config().is_none() {
            return;
        }
        let frame_idx = TRACE_FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        CURRENT_FRAME.store(frame_idx, Ordering::Relaxed);
        CALL_INDEX.store(0, Ordering::Relaxed);
    }

    pub(crate) fn reset_frame() {
        if config().is_none() {
            return;
        }
        CURRENT_FRAME.store(INACTIVE_FRAME, Ordering::Relaxed);
        CALL_INDEX.store(0, Ordering::Relaxed);
    }

    pub(crate) fn begin_call() -> Option<TraceCall> {
        let cfg = config()?;
        let frame_idx = CURRENT_FRAME.load(Ordering::Relaxed);
        if frame_idx == INACTIVE_FRAME {
            return None;
        }
        if cfg.frame.map_or(false, |value| value != frame_idx) {
            return None;
        }
        let call_idx = CALL_INDEX.fetch_add(1, Ordering::Relaxed);
        if cfg.call.map_or(false, |value| value != call_idx) {
            return None;
        }
        Some(TraceCall {
            cfg,
            frame_idx,
            call_idx,
        })
    }

    pub(crate) fn should_trace_sample(cfg: &TraceConfig, sample: usize) -> bool {
        if cfg.all_samples {
            return true;
        }
        cfg.samples.iter().any(|&value| value == sample)
    }

    pub(crate) fn dump_sample(
        cfg: &TraceConfig,
        frame_idx: usize,
        call_idx: usize,
        k: usize,
        out32_hp: f32,
        s2: f32,
    ) {
        std::println!(
            "analysis_hp_resampler[{frame_idx}].call[{call_idx}].k[{k}].out32_hp={:.9e}",
            out32_hp as f64
        );
        std::println!(
            "analysis_hp_resampler[{frame_idx}].call[{call_idx}].k[{k}].s2={:.9e}",
            s2 as f64
        );
        if cfg.want_bits {
            std::println!(
                "analysis_hp_resampler[{frame_idx}].call[{call_idx}].k[{k}].out32_hp_bits=0x{:08x}",
                out32_hp.to_bits()
            );
            std::println!(
                "analysis_hp_resampler[{frame_idx}].call[{call_idx}].k[{k}].s2_bits=0x{:08x}",
                s2.to_bits()
            );
        }
    }

    fn parse_samples(value: &str) -> SampleSelection {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return SampleSelection {
                samples: Vec::new(),
                all_samples: true,
            };
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return SampleSelection {
                samples: Vec::new(),
                all_samples: true,
            };
        }
        let mut samples = Vec::new();
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == '\t') {
            if token.is_empty() {
                continue;
            }
            if let Ok(sample) = token.parse::<usize>() {
                if sample < 480 {
                    samples.push(sample);
                }
            }
        }
        if samples.is_empty() {
            SampleSelection {
                samples,
                all_samples: true,
            }
        } else {
            SampleSelection {
                samples,
                all_samples: false,
            }
        }
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }
}

const TBANDS: [usize; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

// Keep the literal to match the C M_PI value used in tonality analysis (do not replace).
#[allow(clippy::approx_constant)]
const M_PI_F64: f64 = 3.141592653;
const LEAKAGE_OFFSET: f32 = 2.5;
const LEAKAGE_SLOPE: f32 = 2.0;
const STD_FEATURE_BIAS: [f32; 9] = [
    5.684_947, 3.475_288, 1.770_634, 1.599_784, 3.773_215, 2.163_313, 1.260_756, 1.116_868,
    1.918_795,
];

// Use f64 intermediates to mirror the C sqrt(double) path; keep the cast/lint for parity.
#[allow(clippy::cast_lossless)]
fn stationarity_denom(l2: f32) -> f32 {
    sqrt(1e-15f64 + (NB_FRAMES as f64) * (l2 as f64)) as f32
}

// Compute pi^4 with the C literal and f64 order, then downcast to match reference bits.
// Keep this helper to preserve C bit-exactness (do not rewrite with core::f32::consts::PI).
fn pi4_f32() -> f32 {
    (M_PI_F64 * M_PI_F64 * M_PI_F64 * M_PI_F64) as f32
}

#[inline]
fn accumulate_t_e(acc: f32, bin_e: f32, tonality_clamped: f32) -> f32 {
    // Use FMA to mirror the C compiler's contraction for tE accumulation.
    fmaf(bin_e, tonality_clamped, acc)
}

pub(crate) trait DownmixInput {
    fn downmix(
        &self,
        output: &mut [f32],
        subframe: usize,
        offset: usize,
        c1: i32,
        c2: i32,
        channels: i32,
    );
}

impl DownmixInput for [f32] {
    fn downmix(
        &self,
        output: &mut [f32],
        subframe: usize,
        offset: usize,
        c1: i32,
        c2: i32,
        channels: i32,
    ) {
        let channels = channels as usize;
        debug_assert!(channels > 0);
        debug_assert!(c1 >= 0 && (c1 as usize) < channels);
        debug_assert!(output.len() >= subframe);
        let base = offset * channels;
        debug_assert!(base + subframe * channels <= self.len());

        for j in 0..subframe {
            let idx = base + j * channels + c1 as usize;
            let mut sample = self[idx] * CELT_SIG_SCALE;
            if c2 > -1 {
                let idx2 = base + j * channels + c2 as usize;
                sample += self[idx2] * CELT_SIG_SCALE;
            } else if c2 == -2 {
                for c in 1..channels {
                    sample += self[base + j * channels + c] * CELT_SIG_SCALE;
                }
            }
            output[j] = sample;
        }
    }
}

impl DownmixInput for [i16] {
    fn downmix(
        &self,
        output: &mut [f32],
        subframe: usize,
        offset: usize,
        c1: i32,
        c2: i32,
        channels: i32,
    ) {
        let channels = channels as usize;
        debug_assert!(channels > 0);
        debug_assert!(c1 >= 0 && (c1 as usize) < channels);
        debug_assert!(output.len() >= subframe);
        let base = offset * channels;
        debug_assert!(base + subframe * channels <= self.len());

        for j in 0..subframe {
            let idx = base + j * channels + c1 as usize;
            let mut sample = f32::from(self[idx]);
            if c2 > -1 {
                let idx2 = base + j * channels + c2 as usize;
                sample += f32::from(self[idx2]);
            } else if c2 == -2 {
                for c in 1..channels {
                    sample += f32::from(self[base + j * channels + c]);
                }
            }
            output[j] = sample;
        }
    }
}

impl DownmixInput for [i32] {
    fn downmix(
        &self,
        output: &mut [f32],
        subframe: usize,
        offset: usize,
        c1: i32,
        c2: i32,
        channels: i32,
    ) {
        let channels = channels as usize;
        debug_assert!(channels > 0);
        debug_assert!(c1 >= 0 && (c1 as usize) < channels);
        debug_assert!(output.len() >= subframe);
        let base = offset * channels;
        debug_assert!(base + subframe * channels <= self.len());

        for j in 0..subframe {
            let idx = base + j * channels + c1 as usize;
            let mut sample = self[idx] as f32 * (1.0 / 256.0);
            if c2 > -1 {
                let idx2 = base + j * channels + c2 as usize;
                sample += self[idx2] as f32 * (1.0 / 256.0);
            } else if c2 == -2 {
                for c in 1..channels {
                    sample += self[base + j * channels + c] as f32 * (1.0 / 256.0);
                }
            }
            output[j] = sample;
        }
    }
}

#[derive(Debug)]
pub(crate) struct TonalityAnalysisState {
    pub arch: i32,
    pub application: i32,
    pub fs: i32,
    angle: [f32; 240],
    d_angle: [f32; 240],
    d2_angle: [f32; 240],
    inmem: [f32; ANALYSIS_BUF_SIZE],
    mem_fill: usize,
    prev_band_tonality: [f32; NB_TBANDS],
    prev_tonality: f32,
    prev_bandwidth: i32,
    e: [[f32; NB_TBANDS]; NB_FRAMES],
    log_e: [[f32; NB_TBANDS]; NB_FRAMES],
    low_e: [f32; NB_TBANDS],
    high_e: [f32; NB_TBANDS],
    mean_e: [f32; NB_TBANDS + 1],
    mem: [f32; 32],
    cmean: [f32; 8],
    std: [f32; 9],
    e_tracker: f32,
    low_e_count: f32,
    e_count: usize,
    count: i32,
    analysis_offset: i32,
    write_pos: usize,
    read_pos: usize,
    read_subframe: i32,
    hp_ener_accum: f32,
    initialized: bool,
    rnn_state: [f32; MAX_NEURONS],
    downmix_state: [f32; 3],
    info: [AnalysisInfo; DETECT_SIZE],
    kfft: KissFftState,
}

impl TonalityAnalysisState {
    #[must_use]
    pub(crate) fn new(fs: i32) -> Self {
        let mut state = Self {
            arch: opus_select_arch(),
            application: 0,
            fs,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [0.0; ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            e: [[0.0; NB_TBANDS]; NB_FRAMES],
            log_e: [[0.0; NB_TBANDS]; NB_FRAMES],
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            e_tracker: 0.0,
            low_e_count: 0.0,
            e_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: false,
            rnn_state: [0.0; MAX_NEURONS],
            downmix_state: [0.0; 3],
            info: core::array::from_fn(|_| AnalysisInfo::default()),
            kfft: KissFftState::new(480),
        };
        state.reset();
        state
    }

    pub(crate) fn reset(&mut self) {
        self.angle.fill(0.0);
        self.d_angle.fill(0.0);
        self.d2_angle.fill(0.0);
        self.inmem.fill(0.0);
        self.mem_fill = 0;
        self.prev_band_tonality.fill(0.0);
        self.prev_tonality = 0.0;
        self.prev_bandwidth = 0;
        self.e = [[0.0; NB_TBANDS]; NB_FRAMES];
        self.log_e = [[0.0; NB_TBANDS]; NB_FRAMES];
        self.low_e.fill(0.0);
        self.high_e.fill(0.0);
        self.mean_e.fill(0.0);
        self.mem.fill(0.0);
        self.cmean.fill(0.0);
        self.std.fill(0.0);
        self.e_tracker = 0.0;
        self.low_e_count = 0.0;
        self.e_count = 0;
        self.count = 0;
        self.analysis_offset = 0;
        self.write_pos = 0;
        self.read_pos = 0;
        self.read_subframe = 0;
        self.hp_ener_accum = 0.0;
        self.initialized = false;
        self.rnn_state.fill(0.0);
        self.downmix_state.fill(0.0);
        self.info = core::array::from_fn(|_| AnalysisInfo::default());
    }

    #[inline]
    #[must_use]
    pub(crate) fn snapshot_read_state(&self) -> (usize, i32) {
        (self.read_pos, self.read_subframe)
    }

    #[inline]
    pub(crate) fn restore_read_state(&mut self, state: (usize, i32)) {
        self.read_pos = state.0;
        self.read_subframe = state.1;
    }
}

pub(crate) fn tonality_analysis_init(state: &mut TonalityAnalysisState, fs: i32) {
    *state = TonalityAnalysisState::new(fs);
}

pub(crate) fn tonality_analysis_reset(state: &mut TonalityAnalysisState) {
    state.reset();
}

fn is_digital_silence32(pcm: &[f32], frame_size: usize, channels: usize, lsb_depth: i32) -> bool {
    let total = frame_size * channels;
    if pcm.len() < total {
        return false;
    }
    let sample_max = celt_maxabs32(&pcm[..total]);
    sample_max <= 1.0 / (1_i32 << lsb_depth) as f32
}

fn silk_resampler_down2_hp(s: &mut [f32; 3], out: &mut [f32], input: &[f32]) -> f32 {
    let len2 = input.len() / 2;
    debug_assert!(out.len() >= len2);
    let mut hp_ener = 0.0f32;
    #[cfg(test)]
    let trace_call = hp_resampler_trace::begin_call();

    for k in 0..len2 {
        let in_even = input[2 * k];
        let y = in_even - s[0];
        let x = 0.607_437_1 * y;
        let mut out32 = s[0] + x;
        s[0] = in_even + x;
        let mut out32_hp = out32;

        let in_odd = input[2 * k + 1];
        let y = in_odd - s[1];
        let x = 0.150_63 * y;
        out32 += s[1];
        out32 += x;
        s[1] = in_odd + x;

        let y = -in_odd - s[2];
        let x = 0.150_63 * y;
        out32_hp += s[2];
        out32_hp += x;
        s[2] = -in_odd + x;

        #[cfg(test)]
        if let Some(trace) = trace_call.as_ref() {
            if hp_resampler_trace::should_trace_sample(trace.cfg, k) {
                hp_resampler_trace::dump_sample(
                    trace.cfg,
                    trace.frame_idx,
                    trace.call_idx,
                    k,
                    out32_hp,
                    s[2],
                );
            }
        }

        hp_ener = mul_add_f32(out32_hp, out32_hp, hp_ener);
        out[k] = 0.5 * out32;
    }

    hp_ener * SCALE_ENER
}

fn downmix_and_resample<PCM: DownmixInput + ?Sized>(
    pcm: &PCM,
    y: &mut [f32],
    s: &mut [f32; 3],
    mut subframe: usize,
    mut offset: i32,
    c1: i32,
    c2: i32,
    channels: i32,
    fs: i32,
) -> f32 {
    if subframe == 0 {
        return 0.0;
    }

    debug_assert!(offset >= 0, "downmix offset must be non-negative");
    if fs == 48_000 {
        subframe *= 2;
        offset *= 2;
    } else if fs == 16_000 {
        subframe = subframe * 2 / 3;
        offset = offset * 2 / 3;
    }

    let mut tmp = [0.0f32; 960];
    debug_assert!(subframe <= tmp.len());
    pcm.downmix(
        &mut tmp[..subframe],
        subframe,
        offset as usize,
        c1,
        c2,
        channels,
    );

    if (c2 == -2 && channels == 2) || c2 > -1 {
        for value in tmp.iter_mut().take(subframe) {
            *value *= 0.5;
        }
    }

    if fs == 48_000 {
        silk_resampler_down2_hp(s, y, &tmp[..subframe])
    } else if fs == 24_000 {
        y[..subframe].copy_from_slice(&tmp[..subframe]);
        0.0
    } else {
        let mut tmp3x = [0.0f32; 1440];
        debug_assert!(3 * subframe <= tmp3x.len());
        for j in 0..subframe {
            let value = tmp[j];
            tmp3x[3 * j] = value;
            tmp3x[3 * j + 1] = value;
            tmp3x[3 * j + 2] = value;
        }
        silk_resampler_down2_hp(s, y, &tmp3x[..3 * subframe])
    }
}

pub(crate) fn tonality_get_info(
    tonal: &mut TonalityAnalysisState,
    info_out: &mut AnalysisInfo,
    len: usize,
) {
    let mut pos = tonal.read_pos as i32;
    let mut curr_lookahead = tonal.write_pos as i32 - tonal.read_pos as i32;
    if curr_lookahead < 0 {
        curr_lookahead += DETECT_SIZE as i32;
    }

    tonal.read_subframe += (len as i32) / (tonal.fs / 400);
    while tonal.read_subframe >= 8 {
        tonal.read_subframe -= 8;
        tonal.read_pos = (tonal.read_pos + 1) % DETECT_SIZE;
    }

    if len as i32 > tonal.fs / 50 && pos as usize != tonal.write_pos {
        pos += 1;
        if pos == DETECT_SIZE as i32 {
            pos = 0;
        }
    }
    if pos as usize == tonal.write_pos {
        pos -= 1;
    }
    if pos < 0 {
        pos = DETECT_SIZE as i32 - 1;
    }
    let pos0 = pos as usize;
    *info_out = tonal.info[pos0].clone();
    if !info_out.valid {
        return;
    }

    let mut tonality_max = info_out.tonality;
    let mut tonality_avg = info_out.tonality;
    let mut tonality_count = 1;
    let mut bandwidth_span = 6;

    let mut cursor = pos0;
    for _ in 0..3 {
        cursor = (cursor + 1) % DETECT_SIZE;
        if cursor == tonal.write_pos {
            break;
        }
        tonality_max = tonality_max.max(tonal.info[cursor].tonality);
        tonality_avg += tonal.info[cursor].tonality;
        tonality_count += 1;
        info_out.bandwidth = max(info_out.bandwidth, tonal.info[cursor].bandwidth);
        bandwidth_span -= 1;
    }

    cursor = pos0;
    for _ in 0..bandwidth_span {
        if cursor == 0 {
            cursor = DETECT_SIZE - 1;
        } else {
            cursor -= 1;
        }
        if cursor == tonal.write_pos {
            break;
        }
        info_out.bandwidth = max(info_out.bandwidth, tonal.info[cursor].bandwidth);
    }
    info_out.tonality = (tonality_avg / tonality_count as f32).max(tonality_max - 0.2);

    let mut mpos = pos0;
    let mut vpos = pos0;
    if curr_lookahead > 15 {
        mpos = (mpos + 5) % DETECT_SIZE;
        vpos = (vpos + 1) % DETECT_SIZE;
    }

    let mut prob_min = 1.0f32;
    let mut prob_max = 0.0f32;
    let vad_prob = tonal.info[vpos].activity_probability;
    let mut prob_count = vad_prob.max(0.1);
    let mut prob_avg = prob_count * tonal.info[mpos].music_prob;

    loop {
        mpos = (mpos + 1) % DETECT_SIZE;
        if mpos == tonal.write_pos {
            break;
        }
        vpos = (vpos + 1) % DETECT_SIZE;
        if vpos == tonal.write_pos {
            break;
        }
        let pos_vad = tonal.info[vpos].activity_probability;
        let delta = vad_prob - pos_vad;
        let min_term = mul_add_f32(-TRANSITION_PENALTY, delta, prob_avg) / prob_count;
        let max_term = mul_add_f32(TRANSITION_PENALTY, delta, prob_avg) / prob_count;
        prob_min = prob_min.min(min_term);
        prob_max = prob_max.max(max_term);
        prob_count += pos_vad.max(0.1);
        prob_avg = mul_add_f32(pos_vad.max(0.1), tonal.info[mpos].music_prob, prob_avg);
    }

    info_out.music_prob = prob_avg / prob_count;
    prob_min = prob_min.min(prob_avg / prob_count);
    prob_max = prob_max.max(prob_avg / prob_count);
    prob_min = prob_min.max(0.0);
    prob_max = prob_max.min(1.0);

    if curr_lookahead < 10 {
        let mut pmin = prob_min;
        let mut pmax = prob_max;
        cursor = pos0;
        for _ in 0..min(tonal.count.saturating_sub(1), 15) {
            if cursor == 0 {
                cursor = DETECT_SIZE - 1;
            } else {
                cursor -= 1;
            }
            if cursor == tonal.write_pos {
                break;
            }
            pmin = pmin.min(tonal.info[cursor].music_prob);
            pmax = pmax.max(tonal.info[cursor].music_prob);
        }
        pmin = (pmin - 0.1 * vad_prob).max(0.0);
        pmax = (pmax + 0.1 * vad_prob).min(1.0);
        let weight = 1.0 - 0.1 * curr_lookahead as f32;
        prob_min = mul_add_f32(weight, pmin - prob_min, prob_min);
        prob_max = mul_add_f32(weight, pmax - prob_max, prob_max);
    }

    info_out.music_prob_min = prob_min;
    info_out.music_prob_max = prob_max;
}

fn tonality_analysis<PCM: DownmixInput + ?Sized>(
    tonal: &mut TonalityAnalysisState,
    _celt_mode: &OpusCustomMode<'_>,
    x: &PCM,
    len: usize,
    mut offset: i32,
    c1: i32,
    c2: i32,
    channels: i32,
    lsb_depth: i32,
) {
    let mut len = len as i32;
    if !tonal.initialized {
        tonal.mem_fill = INITIAL_MEM_FILL;
        tonal.initialized = true;
    }
    let alpha = 1.0 / min(10, 1 + tonal.count) as f32;
    let alpha_e = 1.0 / min(25, 1 + tonal.count) as f32;
    let mut alpha_e2 = 1.0 / min(100, 1 + tonal.count) as f32;
    if tonal.count <= 1 {
        alpha_e2 = 1.0;
    }

    if tonal.fs == 48_000 {
        len /= 2;
        offset /= 2;
    } else if tonal.fs == 16_000 {
        len = 3 * len / 2;
        offset = 3 * offset / 2;
    }

    #[cfg(test)]
    {
        if hp_resampler_trace::config().is_some() {
            let will_process = tonal.mem_fill + (len as usize) >= ANALYSIS_BUF_SIZE;
            if will_process {
                hp_resampler_trace::begin_frame();
            } else {
                hp_resampler_trace::reset_frame();
            }
        }
    }

    #[cfg(test)]
    let activity_cfg = activity_trace::config();
    #[cfg(test)]
    let activity_frame_idx = fft_trace::frame_index();
    #[cfg(test)]
    let activity_trace_enabled = activity_cfg
        .map(|cfg| activity_trace::frame_matches(cfg, activity_frame_idx))
        .unwrap_or(false);

    // Capture the accumulated high-pass energy for this analysis window (matches C).
    let hp_ener = {
        let avail = min(len as usize, ANALYSIS_BUF_SIZE - tonal.mem_fill);
        let ret = downmix_and_resample(
            x,
            &mut tonal.inmem[tonal.mem_fill..],
            &mut tonal.downmix_state,
            avail,
            offset,
            c1,
            c2,
            channels,
            tonal.fs,
        );
        tonal.hp_ener_accum += ret;
        tonal.hp_ener_accum
    };

    if tonal.mem_fill + (len as usize) < ANALYSIS_BUF_SIZE {
        tonal.mem_fill += len as usize;
        return;
    }

    let info_slot = tonal.write_pos;
    tonal.write_pos = (tonal.write_pos + 1) % DETECT_SIZE;

    let is_silence = is_digital_silence32(&tonal.inmem, ANALYSIS_BUF_SIZE, 1, lsb_depth);

    let mut input_fft = [KissFftCpx::default(); 480];
    let mut output_fft = [KissFftCpx::default(); 480];
    let mut tonality = [0.0f32; 240];
    let mut noisiness = [0.0f32; 240];

    for i in 0..240 {
        let w = ANALYSIS_WINDOW[i];
        input_fft[i].r = w * tonal.inmem[i];
        input_fft[i].i = w * tonal.inmem[240 + i];
        input_fft[479 - i].r = w * tonal.inmem[479 - i];
        input_fft[479 - i].i = w * tonal.inmem[720 - i - 1];
    }

    tonal
        .inmem
        .copy_within(ANALYSIS_BUF_SIZE - 240..ANALYSIS_BUF_SIZE, 0);
    let remaining = len as usize - (ANALYSIS_BUF_SIZE - tonal.mem_fill);
    tonal.hp_ener_accum = downmix_and_resample(
        x,
        &mut tonal.inmem[240..],
        &mut tonal.downmix_state,
        remaining,
        offset + (ANALYSIS_BUF_SIZE - tonal.mem_fill) as i32,
        c1,
        c2,
        channels,
        tonal.fs,
    );
    tonal.mem_fill = 240 + remaining;

    if is_silence {
        let prev_pos = (tonal.write_pos + DETECT_SIZE - 2) % DETECT_SIZE;
        let prev_info = tonal.info[prev_pos].clone();
        tonal.info[info_slot] = prev_info;
        return;
    }

    let info = &mut tonal.info[info_slot];
    #[cfg(test)]
    {
        if std::env::var_os("ANALYSIS_TRACE_STAGE").is_some()
            || std::env::var_os("KISS_FFT_TRACE_STAGE").is_some()
        {
            crate::celt::set_fft_trace_frame(fft_trace::frame_index());
        }
    }
    opus_fft(&tonal.kfft, &input_fft, &mut output_fft);
    if output_fft[0].r.is_nan() {
        info.valid = false;
        return;
    }

    let mut tonality2 = [0.0f32; 240];
    let mut band_tonality = [0.0f32; NB_TBANDS];
    let mut log_e = [0.0f32; NB_TBANDS];
    let mut bfcc = [0.0f32; 8];
    let mut features = [0.0f32; 25];
    let mut frame_noisiness = 0.0f32;
    let mut frame_stationarity = 0.0f32;
    let mut frame_tonality = 0.0f32;
    let mut max_frame_tonality = 0.0f32;
    let mut slope = 0.0f32;
    let mut relative_e = 0.0f32;
    let mut frame_loudness = 0.0f32;
    let mut noise_floor;
    let mut mid_e = [0.0f32; 8];
    let mut spec_variability = 0.0f32;
    let mut band_log2 = [0.0f32; NB_TBANDS + 1];
    let mut leakage_from = [0.0f32; NB_TBANDS + 1];
    let mut leakage_to = [0.0f32; NB_TBANDS + 1];
    let mut layer_out = [0.0f32; MAX_NEURONS];
    let mut is_masked = [false; NB_TBANDS + 1];
    let mut below_max_pitch;
    let mut above_max_pitch;

    let pi4 = pi4_f32();
    for i in 1..240 {
        let x1r = output_fft[i].r + output_fft[480 - i].r;
        let x1i = output_fft[i].i - output_fft[480 - i].i;
        let x2r = output_fft[i].i + output_fft[480 - i].i;
        let x2i = output_fft[480 - i].r - output_fft[i].r;

        let atan1 = fast_atan2f(x1i, x1r);
        let angle = 0.5 / PI * atan1;
        let d_angle = angle - tonal.angle[i];
        let d2_angle = d_angle - tonal.d_angle[i];

        let atan2 = fast_atan2f(x2i, x2r);
        let angle2 = 0.5 / PI * atan2;
        let d_angle2 = angle2 - angle;
        let d2_angle2 = d_angle2 - d_angle;

        #[cfg(test)]
        if (i == 28 || i == 29) && bin28_trace::enabled(fft_trace::frame_index()) {
            let frame_idx = fft_trace::frame_index();
            std::println!("analysis_bin{i:02}[{frame_idx}].x2r={:.9e}", x2r as f64);
            std::println!("analysis_bin{i:02}[{frame_idx}].x2i={:.9e}", x2i as f64);
            std::println!(
                "analysis_bin{i:02}[{frame_idx}].fast_atan2={:.9e}",
                atan2 as f64
            );
            std::println!(
                "analysis_bin{i:02}[{frame_idx}].angle2={:.9e}",
                angle2 as f64
            );
            std::println!(
                "analysis_bin{i:02}[{frame_idx}].d2_angle2={:.9e}",
                d2_angle2 as f64
            );
        }

        let d2_angle_int = float2int(d2_angle);
        let mod1_pre = d2_angle - d2_angle_int as f32;
        noisiness[i] = mod1_pre.abs();
        let mut mod1 = mod1_pre;
        mod1 *= mod1;
        mod1 *= mod1;

        let d2_angle2_int = float2int(d2_angle2);
        let mod2_pre = d2_angle2 - d2_angle2_int as f32;
        noisiness[i] += mod2_pre.abs();
        let mut mod2 = mod2_pre;
        mod2 *= mod2;
        mod2 *= mod2;

        let avg_mod = 0.25 * (tonal.d2_angle[i] + mod1 + 2.0 * mod2);
        let scale = (40.0_f32 * 16.0_f32) * pi4;
        let denom = mul_add_f32(scale, avg_mod, 1.0);
        let denom2 = mul_add_f32(scale, mod2, 1.0);
        tonality[i] = 1.0 / denom - 0.015;
        tonality2[i] = 1.0 / denom2 - 0.015;

        #[cfg(test)]
        if let Some(cfg) = tonality_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if tonality_trace::frame_matches(cfg, frame_idx)
                && tonality_trace::should_trace_bin(cfg, i)
            {
                tonality_trace::dump_bin_raw(
                    cfg,
                    frame_idx,
                    i,
                    x1r,
                    x1i,
                    x2r,
                    x2i,
                    atan1,
                    angle,
                    d_angle,
                    atan2,
                    angle2,
                    d_angle2,
                    d2_angle,
                    d2_angle2,
                    d2_angle_int,
                    d2_angle2_int,
                    mod1_pre,
                    mod2_pre,
                    mod1,
                    mod2,
                    avg_mod,
                    tonality[i],
                    tonality2[i],
                );
            }
        }

        #[cfg(test)]
        if let Some(cfg) = fft_trace::config() {
            if fft_trace::should_trace_bin(cfg, i) {
                let frame_idx = fft_trace::frame_index();
                let mirror = 480 - i;
                let sum = bin_energy_sum(
                    output_fft[i].r,
                    output_fft[i].i,
                    output_fft[mirror].r,
                    output_fft[mirror].i,
                );
                let bin_e = sum * SCALE_ENER;
                fft_trace::dump_bin(
                    frame_idx,
                    i,
                    &input_fft,
                    &output_fft,
                    x1r,
                    x1i,
                    x2r,
                    x2i,
                    atan1,
                    atan2,
                    angle,
                    angle2,
                    d_angle,
                    d2_angle,
                    d_angle2,
                    d2_angle2,
                    d2_angle_int,
                    d2_angle2_int,
                    bin_e,
                    tonality[i],
                    tonality2[i],
                );
            }
        }

        tonal.angle[i] = angle2;
        tonal.d_angle[i] = d_angle2;
        tonal.d2_angle[i] = mod2;
    }

    for i in 2..239 {
        let tonality_pre = tonality[i];
        let tonality2_prev = tonality2[i - 1];
        let tonality2_cur = tonality2[i];
        let tonality2_next = tonality2[i + 1];
        let tt = tonality2[i].min(tonality2[i - 1].max(tonality2[i + 1]));
        tonality[i] = 0.9 * tonality[i].max(tt - 0.1);
        #[cfg(test)]
        if let Some(cfg) = tonality_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if tonality_trace::frame_matches(cfg, frame_idx)
                && tonality_trace::should_trace_bin(cfg, i)
            {
                tonality_trace::dump_bin_smooth(
                    cfg,
                    frame_idx,
                    i,
                    tonality_pre,
                    tonality2_prev,
                    tonality2_cur,
                    tonality2_next,
                    tt,
                    tonality[i],
                );
            }
        }
    }

    #[cfg(test)]
    if let Some(cfg) = fft_trace::config() {
        let frame_idx = fft_trace::frame_index();
        for i in 1..240 {
            if fft_trace::should_trace_bin(cfg, i) {
                fft_trace::dump_tonality_smoothed(frame_idx, i, tonality[i]);
            }
        }
    }

    info.activity = 0.0;
    if tonal.count == 0 {
        tonal.low_e.fill(1e10);
        tonal.high_e.fill(-1e10);
    }

    // First band (DC).
    let e = {
        let x1r = 2.0 * output_fft[0].r;
        let x2r = 2.0 * output_fft[0].i;
        let mut sum = mul_add_f32(x1r, x1r, x2r * x2r);
        for i in 1..4 {
            let bin_e = bin_energy_sum(
                output_fft[i].r,
                output_fft[i].i,
                output_fft[480 - i].r,
                output_fft[480 - i].i,
            );
            sum += bin_e;
        }
        sum * SCALE_ENER
    };
    band_log2[0] = 0.5 * LOG2_E * logf(e + 1e-10);

    for b in 0..NB_TBANDS {
        let mut band_e = 0.0f32;
        let mut t_e = 0.0f32;
        let mut n_e = 0.0f32;
        let band_start = TBANDS[b];
        let band_end = TBANDS[b + 1];
        for i in band_start..band_end {
            let bin_e = bin_energy_sum(
                output_fft[i].r,
                output_fft[i].i,
                output_fft[480 - i].r,
                output_fft[480 - i].i,
            ) * SCALE_ENER;
            let tonality_val = tonality[i];
            let tonality_clamped = tonality_val.max(0.0);
            let t_e_term = bin_e * tonality_clamped;
            #[cfg(test)]
            if let Some(cfg) = tonality_trace::config() {
                let frame_idx = fft_trace::frame_index();
                if tonality_trace::frame_matches(cfg, frame_idx)
                    && tonality_trace::should_trace_band(cfg, b)
                {
                    let t_e_pre = t_e;
                    let t_e_acc_next = t_e_pre + t_e_term;
                    tonality_trace::dump_bin(
                        cfg,
                        frame_idx,
                        b,
                        i,
                        bin_e,
                        tonality_val,
                        tonality_clamped,
                        t_e_term,
                        t_e_pre,
                        t_e_acc_next,
                    );
                }
            }
            band_e += bin_e;
            t_e = accumulate_t_e(t_e, bin_e, tonality_clamped);
            let noisiness_term = 0.5 - noisiness[i];
            n_e = mul_add_f32(bin_e * 2.0, noisiness_term, n_e);
        }

        tonal.e[tonal.e_count][b] = band_e;
        let band_noisiness = n_e / (1e-15 + band_e);
        frame_noisiness += band_noisiness;
        #[cfg(test)]
        if activity_trace_enabled {
            let cfg = activity_cfg.expect("activity cfg");
            activity_trace::dump_band_scalar(cfg, activity_frame_idx, "band_e", b, band_e);
            activity_trace::dump_band_scalar(cfg, activity_frame_idx, "band_n_e", b, n_e);
            activity_trace::dump_band_scalar(
                cfg,
                activity_frame_idx,
                "band_noisiness",
                b,
                band_noisiness,
            );
        }

        frame_loudness += sqrtf(band_e + 1e-10);
        let band_e_eps = band_e + 1e-10;
        let log_e_val = log(band_e_eps as f64) as f32;
        log_e[b] = log_e_val;
        band_log2[b + 1] = 0.5 * LOG2_E * log_e_val;
        tonal.log_e[tonal.e_count][b] = log_e[b];

        if tonal.count == 0 {
            tonal.high_e[b] = log_e[b];
            tonal.low_e[b] = log_e[b];
        }
        if tonal.high_e[b] > tonal.low_e[b] + 7.5 {
            if tonal.high_e[b] - log_e[b] > log_e[b] - tonal.low_e[b] {
                tonal.high_e[b] -= 0.01;
            } else {
                tonal.low_e[b] += 0.01;
            }
        }
        if log_e[b] > tonal.high_e[b] {
            tonal.high_e[b] = log_e[b];
            tonal.low_e[b] = (tonal.high_e[b] - 15.0).max(tonal.low_e[b]);
        } else if log_e[b] < tonal.low_e[b] {
            tonal.low_e[b] = log_e[b];
            tonal.high_e[b] = (tonal.low_e[b] + 15.0).min(tonal.high_e[b]);
        }
        let denom = 1e-5 + (tonal.high_e[b] - tonal.low_e[b]);
        relative_e += (log_e[b] - tonal.low_e[b]) / denom;

        let mut l1 = 0.0f32;
        let mut l2 = 0.0f32;
        for i in 0..NB_FRAMES {
            let band_e = tonal.e[i][b];
            // Use f64 sqrt to mirror the C double path; keep the cast for parity.
            #[allow(clippy::cast_lossless)]
            let sqrt_e = sqrt(band_e as f64) as f32;
            #[cfg(test)]
            if let Some(cfg) = tonality_trace::config() {
                let frame_idx = fft_trace::frame_index();
                if tonality_trace::frame_matches(cfg, frame_idx)
                    && tonality_trace::should_trace_band(cfg, b)
                {
                    tonality_trace::dump_stationarity_sample(
                        cfg,
                        frame_idx,
                        b,
                        i,
                        band_e,
                        sqrt_e,
                    );
                }
            }
            l1 += sqrt_e;
            l2 += band_e;
        }
        let denom = stationarity_denom(l2);
        #[cfg(test)]
        if let Some(cfg) = tonality_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if tonality_trace::frame_matches(cfg, frame_idx)
                && tonality_trace::should_trace_band(cfg, b)
            {
                tonality_trace::dump_stationarity_totals(
                    cfg,
                    frame_idx,
                    b,
                    l1,
                    l2,
                    denom,
                );
            }
        }
        let mut stationarity = (l1 / denom).min(0.99);
        stationarity = stationarity * stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;

        let energy_ratio_num = t_e.max(0.0);
        let energy_ratio_denom = 1e-15 + band_e;
        let energy_ratio = energy_ratio_num / energy_ratio_denom;
        let stationarity_term = stationarity * tonal.prev_band_tonality[b];
        band_tonality[b] = energy_ratio.max(stationarity_term);
        frame_tonality += band_tonality[b];
        if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
            let idx = b + NB_TONAL_SKIP_BANDS - NB_TBANDS;
            frame_tonality -= band_tonality[idx];
        }
        let weight = (1.0_f64 + 0.03_f64 * (b as f64 - NB_TBANDS as f64)) as f32;
        let weighted = (weight as f64 * frame_tonality as f64) as f32;
        max_frame_tonality = max_frame_tonality.max(weighted);
        let slope_delta = (b as i32 - 8) as f32;
        let slope_pre = slope;
        let slope_term = band_tonality[b] * slope_delta;
        slope = slope + slope_term;
        #[cfg(test)]
        if let Some(cfg) = tonality_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if tonality_trace::frame_matches(cfg, frame_idx)
                && tonality_trace::should_trace_band(cfg, b)
            {
                tonality_trace::dump_band(
                    cfg,
                    frame_idx,
                    b,
                    band_e,
                    t_e,
                    energy_ratio,
                    energy_ratio_num,
                    energy_ratio_denom,
                    stationarity,
                    stationarity_term,
                    tonal.prev_band_tonality[b],
                    band_tonality[b],
                    slope_pre,
                    slope_term,
                    slope,
                );
                std::println!(
                    "analysis_tonality_accum[{frame_idx}].band[{b}].frame_tonality={:.9e}",
                    frame_tonality as f64
                );
                std::println!(
                    "analysis_tonality_accum[{frame_idx}].band[{b}].weight={:.9e}",
                    weight as f64
                );
                std::println!(
                    "analysis_tonality_accum[{frame_idx}].band[{b}].weighted={:.9e}",
                    weighted as f64
                );
                std::println!(
                    "analysis_tonality_accum[{frame_idx}].band[{b}].max_frame_tonality={:.9e}",
                    max_frame_tonality as f64
                );
                if tonality_trace::want_bits(cfg) {
                    std::println!(
                        "analysis_tonality_accum[{frame_idx}].band[{b}].frame_tonality_bits=0x{:08x}",
                        frame_tonality.to_bits()
                    );
                    std::println!(
                        "analysis_tonality_accum[{frame_idx}].band[{b}].weight_bits=0x{:08x}",
                        weight.to_bits()
                    );
                    std::println!(
                        "analysis_tonality_accum[{frame_idx}].band[{b}].weighted_bits=0x{:08x}",
                        weighted.to_bits()
                    );
                    std::println!(
                        "analysis_tonality_accum[{frame_idx}].band[{b}].max_frame_tonality_bits=0x{:08x}",
                        max_frame_tonality.to_bits()
                    );
                }
            }
        }
        tonal.prev_band_tonality[b] = band_tonality[b];
    }

    leakage_from[0] = band_log2[0];
    leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
    for b in 1..=NB_TBANDS {
        let leak_slope = LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0;
        leakage_from[b] = (leakage_from[b - 1] + leak_slope).min(band_log2[b]);
        leakage_to[b] = (leakage_to[b - 1] - leak_slope).max(band_log2[b] - LEAKAGE_OFFSET);
    }
    for b in (0..NB_TBANDS).rev() {
        let leak_slope = LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0;
        leakage_from[b] = (leakage_from[b + 1] + leak_slope).min(leakage_from[b]);
        leakage_to[b] = (leakage_to[b + 1] - leak_slope).max(leakage_to[b]);
    }
    #[allow(clippy::assertions_on_constants, clippy::int_plus_one)]
    {
        debug_assert!(NB_TBANDS + 1 <= LEAK_BANDS);
    }
    for b in 0..=NB_TBANDS {
        let boost = (leakage_to[b] - band_log2[b]).max(0.0)
            + (band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET)).max(0.0);
        info.leak_boost[b] = floorf(boost * 64.0 + 0.5).min(255.0) as u8;
    }
    info.leak_boost
        .iter_mut()
        .skip(NB_TBANDS + 1)
        .for_each(|value| *value = 0);

    for i in 0..NB_FRAMES {
        let mut mindist = f32::MAX;
        for j in 0..NB_FRAMES {
            if i == j {
                continue;
            }
            let mut dist = 0.0f32;
            for k in 0..NB_TBANDS {
                let tmp = tonal.log_e[i][k] - tonal.log_e[j][k];
                dist = fmaf(tmp, tmp, dist);
            }
            mindist = mindist.min(dist);
        }
        spec_variability += mindist;
    }
    spec_variability = sqrtf(spec_variability / (NB_FRAMES * NB_TBANDS) as f32);
    let mut bandwidth_mask = 0.0f32;
    let mut bandwidth = 0usize;
    let mut max_e = 0.0f32;
    let depth = (lsb_depth - 8).max(0);
    noise_floor = 5.7e-4 / (1 << depth) as f32;
    noise_floor *= noise_floor;
    below_max_pitch = 0.0;
    above_max_pitch = 0.0;
    for b in 0..NB_TBANDS {
        let mut band_e = 0.0f32;
        let band_start = TBANDS[b];
        let band_end = TBANDS[b + 1];
        for i in band_start..band_end {
            let bin_e = bin_energy_sum(
                output_fft[i].r,
                output_fft[i].i,
                output_fft[480 - i].r,
                output_fft[480 - i].i,
            );
            band_e += bin_e;
        }
        band_e *= SCALE_ENER;
        max_e = max_e.max(band_e);
        if band_start < 64 {
            below_max_pitch += band_e;
        } else {
            above_max_pitch += band_e;
        }
        #[cfg(test)]
        if let Some(cfg) = pitch_ratio_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if pitch_ratio_trace::frame_matches(cfg, frame_idx)
                && pitch_ratio_trace::should_trace_band(cfg, b)
            {
                pitch_ratio_trace::dump_band(
                    cfg,
                    frame_idx,
                    b,
                    band_start,
                    band_end,
                    band_e,
                    max_e,
                    below_max_pitch,
                    above_max_pitch,
                );
            }
        }
        tonal.mean_e[b] = ((1.0 - alpha_e2) * tonal.mean_e[b]).max(band_e);
        let em = tonal.mean_e[b].max(band_e);
        if band_e * 1e9 > max_e
            && (em > 3.0 * noise_floor * (band_end - band_start) as f32
                || band_e > noise_floor * (band_end - band_start) as f32)
        {
            bandwidth = b + 1;
        }
        let threshold = if tonal.prev_bandwidth >= (b + 1) as i32 {
            0.01
        } else {
            0.05
        } * bandwidth_mask;
        is_masked[b] = band_e < threshold;
        bandwidth_mask = (0.05 * bandwidth_mask).max(band_e);
    }

    if tonal.fs == 48_000 {
        let mut e_high = hp_ener * (1.0 / (60.0 * 60.0));
        if e_high < 0.0 {
            e_high = 0.0;
        }
        let noise_ratio = if tonal.prev_bandwidth == 20 {
            10.0
        } else {
            30.0
        };
        above_max_pitch += e_high;
        #[cfg(test)]
        if let Some(cfg) = pitch_ratio_trace::config() {
            let frame_idx = fft_trace::frame_index();
            if pitch_ratio_trace::frame_matches(cfg, frame_idx) {
                pitch_ratio_trace::dump_high_band(cfg, frame_idx, hp_ener, e_high, above_max_pitch);
            }
        }
        tonal.mean_e[NB_TBANDS] = ((1.0 - alpha_e2) * tonal.mean_e[NB_TBANDS]).max(e_high);
        let em = tonal.mean_e[NB_TBANDS].max(e_high);
        if em > 3.0 * noise_ratio * noise_floor * 160.0
            || e_high > noise_ratio * noise_floor * 160.0
        {
            bandwidth = 20;
        }
        let threshold = if tonal.prev_bandwidth == 20 {
            0.01
        } else {
            0.05
        } * bandwidth_mask;
        is_masked[NB_TBANDS] = e_high < threshold;
    }

    info.max_pitch_ratio = if above_max_pitch > below_max_pitch {
        below_max_pitch / above_max_pitch
    } else {
        1.0
    };
    #[cfg(test)]
    if let Some(cfg) = pitch_ratio_trace::config() {
        let frame_idx = fft_trace::frame_index();
        if pitch_ratio_trace::frame_matches(cfg, frame_idx) {
            pitch_ratio_trace::dump_total(
                cfg,
                frame_idx,
                below_max_pitch,
                above_max_pitch,
                info.max_pitch_ratio,
            );
        }
    }

    if bandwidth == 20 && is_masked[NB_TBANDS] {
        bandwidth -= 2;
    } else if bandwidth > 0 && bandwidth <= NB_TBANDS && is_masked[bandwidth - 1] {
        bandwidth -= 1;
    }
    if tonal.count <= 2 {
        bandwidth = 20;
    }
    frame_loudness = 20.0 * log10f(frame_loudness);
    tonal.e_tracker = (tonal.e_tracker - 0.003).max(frame_loudness);
    tonal.low_e_count *= 1.0 - alpha_e;
    if frame_loudness < tonal.e_tracker - 30.0 {
        tonal.low_e_count += alpha_e;
    }

    for i in 0..8 {
        let mut sum = 0.0f32;
        for b in 0..16 {
            sum = fmaf(DCT_TABLE[i * 16 + b], log_e[b], sum);
        }
        bfcc[i] = sum;
    }
    for i in 0..8 {
        let mut sum = 0.0f32;
        for b in 0..16 {
            let mid = 0.5 * (tonal.high_e[b] + tonal.low_e[b]);
            sum = fmaf(DCT_TABLE[i * 16 + b], mid, sum);
        }
        mid_e[i] = sum;
    }

    frame_stationarity /= NB_TBANDS as f32;
    relative_e /= NB_TBANDS as f32;
    if tonal.count < 10 {
        relative_e = 0.5;
    }
    frame_noisiness /= NB_TBANDS as f32;
    info.activity = mul_add_f32(1.0 - frame_noisiness, relative_e, frame_noisiness);
    frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
    let prev_term = tonal.prev_tonality * 0.8;
    frame_tonality = frame_tonality.max(prev_term);
    #[cfg(test)]
    if let Some(cfg) = tonality_trace::config() {
        let frame_idx = fft_trace::frame_index();
        if tonality_trace::frame_matches(cfg, frame_idx) {
            std::println!(
                "analysis_tonality_frame[{frame_idx}].max_frame_tonality={:.9e}",
                max_frame_tonality as f64
            );
            std::println!(
                "analysis_tonality_frame[{frame_idx}].frame_tonality_pre={:.9e}",
                (max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32) as f64
            );
            std::println!(
                "analysis_tonality_frame[{frame_idx}].prev_tonality={:.9e}",
                tonal.prev_tonality as f64
            );
            std::println!(
                "analysis_tonality_frame[{frame_idx}].prev_term={:.9e}",
                prev_term as f64
            );
            std::println!(
                "analysis_tonality_frame[{frame_idx}].frame_tonality={:.9e}",
                frame_tonality as f64
            );
            if tonality_trace::want_bits(cfg) {
                std::println!(
                    "analysis_tonality_frame[{frame_idx}].max_frame_tonality_bits=0x{:08x}",
                    max_frame_tonality.to_bits()
                );
                std::println!(
                    "analysis_tonality_frame[{frame_idx}].frame_tonality_pre_bits=0x{:08x}",
                    (max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32).to_bits()
                );
                std::println!(
                    "analysis_tonality_frame[{frame_idx}].prev_tonality_bits=0x{:08x}",
                    tonal.prev_tonality.to_bits()
                );
                std::println!(
                    "analysis_tonality_frame[{frame_idx}].prev_term_bits=0x{:08x}",
                    prev_term.to_bits()
                );
                std::println!(
                    "analysis_tonality_frame[{frame_idx}].frame_tonality_bits=0x{:08x}",
                    frame_tonality.to_bits()
                );
            }
        }
    }
    tonal.prev_tonality = frame_tonality;

    slope /= 64.0;
    info.tonality_slope = slope;

    tonal.e_count = (tonal.e_count + 1) % NB_FRAMES;
    tonal.count = min(tonal.count + 1, ANALYSIS_COUNT_MAX);
    info.tonality = frame_tonality;

    for i in 0..4 {
        let bfcc_mem = bfcc[i] + tonal.mem[i + 24];
        let mem_sum = tonal.mem[i] + tonal.mem[i + 16];
        let mut sum = mul_add_f32(-0.122_99, bfcc_mem, 0.491_95 * mem_sum);
        sum = mul_add_f32(0.696_93, tonal.mem[i + 8], sum);
        sum = mul_add_f32(-1.4349, tonal.cmean[i], sum);
        features[i] = sum;
    }

    for i in 0..4 {
        let update = alpha * bfcc[i];
        tonal.cmean[i] = mul_add_f32(1.0 - alpha, tonal.cmean[i], update);
    }

    for i in 0..4 {
        let bfcc_delta = bfcc[i] - tonal.mem[i + 24];
        let mem_delta = tonal.mem[i] - tonal.mem[i + 16];
        let tail = 0.316_23 * mem_delta;
        features[4 + i] = mul_add_f32(0.632_46, bfcc_delta, tail);
    }
    for i in 0..3 {
        let bfcc_sum = bfcc[i] + tonal.mem[i + 24];
        let mem_sum = tonal.mem[i] + tonal.mem[i + 16];
        let part = mul_add_f32(0.534_52, bfcc_sum, -0.267_26 * mem_sum);
        features[8 + i] = mul_add_f32(-0.534_52, tonal.mem[i + 8], part);
    }

    if tonal.count > 5 {
        for i in 0..9 {
            let update = alpha * features[i] * features[i];
            tonal.std[i] = mul_add_f32(1.0 - alpha, tonal.std[i], update);
        }
    }
    for i in 0..4 {
        features[i] = bfcc[i] - mid_e[i];
    }

    for i in 0..8 {
        tonal.mem[i + 24] = tonal.mem[i + 16];
        tonal.mem[i + 16] = tonal.mem[i + 8];
        tonal.mem[i + 8] = tonal.mem[i];
        tonal.mem[i] = bfcc[i];
    }

    for i in 0..9 {
        let std_sqrt = sqrt(tonal.std[i] as f64) as f32;
        features[11 + i] = std_sqrt - STD_FEATURE_BIAS[i];
    }
    features[18] = spec_variability - 0.78;
    features[20] = info.tonality - 0.154_723;
    features[21] = info.activity - 0.724_643;
    features[22] = frame_stationarity - 0.743_717;
    features[23] = info.tonality_slope + 0.069_216;
    features[24] = tonal.low_e_count - 0.067_930;

    #[cfg(test)]
    let mut rnn_pre: Vec<f32> = Vec::new();
    #[cfg(test)]
    if activity_trace_enabled {
        let cfg = activity_cfg.expect("activity cfg");
        rnn_pre.extend_from_slice(&tonal.rnn_state[..LAYER1.nb_neurons]);
        activity_trace::dump_layer(cfg, activity_frame_idx, "mem_pre", &tonal.mem);
        activity_trace::dump_layer(cfg, activity_frame_idx, "log_e", &log_e);
        activity_trace::dump_log_hist(cfg, activity_frame_idx, &tonal.log_e);
        activity_trace::dump_layer(cfg, activity_frame_idx, "bfcc", &bfcc);
        activity_trace::dump_layer(cfg, activity_frame_idx, "mid_e", &mid_e);
        activity_trace::dump_features(cfg, activity_frame_idx, &features);
        activity_trace::dump_scalar(cfg, activity_frame_idx, "frame_noisiness", frame_noisiness);
        activity_trace::dump_scalar(cfg, activity_frame_idx, "relative_e", relative_e);
        activity_trace::dump_scalar(cfg, activity_frame_idx, "activity", info.activity);
        activity_trace::dump_layer(cfg, activity_frame_idx, "rnn_pre", &rnn_pre);
    }

    analysis_compute_dense(&LAYER0, &mut layer_out, &features);
    #[cfg(test)]
    if activity_trace_enabled {
        let cfg = activity_cfg.expect("activity cfg");
        activity_trace::dump_layer(
            cfg,
            activity_frame_idx,
            "layer0",
            &layer_out[..LAYER0.nb_neurons],
        );
    }
    #[cfg(test)]
    crate::mlp::set_gru_trace_frame(fft_trace::frame_index());
    analysis_compute_gru(&LAYER1, &mut tonal.rnn_state, &layer_out);
    #[cfg(test)]
    if activity_trace_enabled {
        let cfg = activity_cfg.expect("activity cfg");
        activity_trace::dump_layer(
            cfg,
            activity_frame_idx,
            "rnn_post",
            &tonal.rnn_state[..LAYER1.nb_neurons],
        );
    }
    let mut frame_probs = [0.0f32; 2];
    analysis_compute_dense(&LAYER2, &mut frame_probs, &tonal.rnn_state);
    #[cfg(test)]
    if activity_trace_enabled {
        let cfg = activity_cfg.expect("activity cfg");
        activity_trace::dump_probs(cfg, activity_frame_idx, &frame_probs);
    }

    info.activity_probability = frame_probs[1];
    info.music_prob = frame_probs[0];
    info.bandwidth = bandwidth as i32;
    tonal.prev_bandwidth = bandwidth as i32;
    info.noisiness = frame_noisiness;
    info.valid = true;
}

pub(crate) fn run_analysis<PCM: DownmixInput + ?Sized>(
    analysis: &mut TonalityAnalysisState,
    celt_mode: &OpusCustomMode<'_>,
    analysis_pcm: Option<&PCM>,
    mut analysis_frame_size: usize,
    frame_size: usize,
    c1: i32,
    c2: i32,
    channels: i32,
    fs: i32,
    lsb_depth: i32,
    analysis_info: &mut AnalysisInfo,
) {
    analysis_frame_size &= !1;
    if let Some(pcm) = analysis_pcm {
        let max_analysis = min(((DETECT_SIZE - 5) * fs as usize) / 50, analysis_frame_size);
        let mut pcm_len = max_analysis as i32 - analysis.analysis_offset;
        let mut offset = analysis.analysis_offset;
        while pcm_len > 0 {
            let chunk = min(fs / 50, pcm_len);
            tonality_analysis(
                analysis,
                celt_mode,
                pcm,
                chunk as usize,
                offset,
                c1,
                c2,
                channels,
                lsb_depth,
            );
            offset += fs / 50;
            pcm_len -= fs / 50;
        }
        analysis.analysis_offset = max_analysis as i32;
        analysis.analysis_offset -= frame_size as i32;
    }

    tonality_get_info(analysis, analysis_info, frame_size);
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;
    use std::env;
    use std::fs::File;
    use std::io::{self, Read};
    use std::path::PathBuf;

    use libm::fmaf;

    use crate::celt::{AnalysisInfo, opus_custom_mode_find_static, set_fft_trace_frame};
    use crate::analysis::NB_FRAMES;

    use super::{TonalityAnalysisState, accumulate_t_e, run_analysis, tonality_analysis_reset};
    use super::fft_trace;

    #[test]
    fn run_analysis_without_pcm_keeps_info_invalid() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut state = TonalityAnalysisState::new(48_000);
        let mut info = AnalysisInfo::default();

        run_analysis(
            &mut state,
            &mode,
            None::<&[i16]>,
            960,
            960,
            0,
            -1,
            1,
            48_000,
            16,
            &mut info,
        );

        assert!(!info.valid);
    }

    #[test]
    fn non_silent_input_produces_analysis() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut state = TonalityAnalysisState::new(48_000);
        let mut info = AnalysisInfo::default();
        let pcm: Vec<i16> = (0..960).map(|i| (i as i16).wrapping_mul(13)).collect();

        run_analysis(
            &mut state,
            &mode,
            Some(pcm.as_slice()),
            960,
            960,
            0,
            -1,
            1,
            48_000,
            16,
            &mut info,
        );

        assert!(info.valid);
    }

    #[test]
    fn t_e_accumulation_uses_fma() {
        let mut seed: u32 = 0x1234_5678;
        let mut found = false;
        for _ in 0..200_000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let acc_bits = (seed & 0x007f_ffff) | 0x3f00_0000;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let bin_bits = (seed & 0x007f_ffff) | 0x3f80_0000;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let ton_bits = (seed & 0x007f_ffff) | 0x3f00_0000;

            let acc = f32::from_bits(acc_bits);
            let bin = f32::from_bits(bin_bits);
            let ton = f32::from_bits(ton_bits);

            let fused = fmaf(bin, ton, acc);
            let unfused = acc + bin * ton;
            if fused.to_bits() != unfused.to_bits() {
                let got = accumulate_t_e(acc, bin, ton);
                assert_eq!(got.to_bits(), fused.to_bits());
                found = true;
                break;
            }
        }
        assert!(found, "expected to find FMA-sensitive inputs");
    }

    #[test]
    fn reset_clears_runtime_state() {
        let mut state = TonalityAnalysisState::new(48_000);
        state.mem_fill = 123;
        state.count = 42;
        tonality_analysis_reset(&mut state);
        assert_eq!(state.mem_fill, 0);
        assert_eq!(state.count, 0);
    }

    #[test]
    fn stationarity_denom_matches_reference_bits() {
        let l2 = 6.127_892_860e-9_f32;
        let denom = super::stationarity_denom(l2);
        assert_eq!(denom.to_bits(), 0x3968_2ac1);
    }

    #[test]
    fn pi4_matches_reference_bits() {
        let pi4 = super::pi4_f32();
        assert_eq!(pi4.to_bits(), 0x42c2_d174);
    }

    #[test]
    fn bin_energy_sum_fma_bits_match_reference() {
        // Frame 12 bin 33 FFT output bits from analysis trace; this ensures
        // the fused multiply-add accumulation matches opus-c behavior.
        let r = f32::from_bits(0xbb22_d304);
        let i = f32::from_bits(0x38b4_0d90);
        let mr = f32::from_bits(0x3b22_d305);
        let mi = f32::from_bits(0x38b4_0d80);
        let sum = super::bin_energy_sum(r, i, mr, mi);
        assert_eq!(sum.to_bits(), 0x374f_5ed1);
        let scaled = sum * super::SCALE_ENER;
        assert_eq!(scaled.to_bits(), 0x284f_5ed1);
    }

    fn dump_analysis_info(frame_idx: usize, info: &AnalysisInfo) {
        std::println!(
            "analysis_info[{frame_idx}].valid={}",
            if info.valid { 1 } else { 0 }
        );
        std::println!(
            "analysis_info[{frame_idx}].tonality={:.9e}",
            info.tonality as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].tonality_slope={:.9e}",
            info.tonality_slope as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].noisiness={:.9e}",
            info.noisiness as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].activity={:.9e}",
            info.activity as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].music_prob={:.9e}",
            info.music_prob as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].music_prob_min={:.9e}",
            info.music_prob_min as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].music_prob_max={:.9e}",
            info.music_prob_max as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].bandwidth={}",
            info.bandwidth
        );
        std::println!(
            "analysis_info[{frame_idx}].activity_probability={:.9e}",
            info.activity_probability as f64
        );
        std::println!(
            "analysis_info[{frame_idx}].max_pitch_ratio={:.9e}",
            info.max_pitch_ratio as f64
        );
        for (idx, &value) in info.leak_boost.iter().enumerate() {
            std::println!("analysis_info[{frame_idx}].leak_boost[{idx}]={value}");
        }
    }

    fn want_e_history() -> bool {
        match env::var("ANALYSIS_TRACE_E_HISTORY") {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    fn want_angle_bits() -> bool {
        match env::var("ANALYSIS_TRACE_ANGLE_BITS") {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    fn want_band_tonality_bits() -> bool {
        match env::var("ANALYSIS_TRACE_BAND_TONALITY_BITS") {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    fn dump_analysis_state(frame_idx: usize, state: &TonalityAnalysisState) {
        let last_idx = if state.e_count == 0 {
            NB_FRAMES - 1
        } else {
            state.e_count - 1
        };

        std::println!("analysis_state[{frame_idx}].count={}", state.count);
        std::println!("analysis_state[{frame_idx}].e_count={}", state.e_count);
        std::println!(
            "analysis_state[{frame_idx}].analysis_offset={}",
            state.analysis_offset
        );
        std::println!(
            "analysis_state[{frame_idx}].mem_fill={}",
            state.mem_fill
        );
        std::println!(
            "analysis_state[{frame_idx}].write_pos={}",
            state.write_pos
        );
        std::println!(
            "analysis_state[{frame_idx}].read_pos={}",
            state.read_pos
        );
        std::println!(
            "analysis_state[{frame_idx}].read_subframe={}",
            state.read_subframe
        );
        std::println!(
            "analysis_state[{frame_idx}].hp_ener_accum={:.9e}",
            state.hp_ener_accum as f64
        );
        std::println!(
            "analysis_state[{frame_idx}].e_tracker={:.9e}",
            state.e_tracker as f64
        );
        std::println!(
            "analysis_state[{frame_idx}].low_e_count={:.9e}",
            state.low_e_count as f64
        );
        std::println!(
            "analysis_state[{frame_idx}].prev_tonality={:.9e}",
            state.prev_tonality as f64
        );
        std::println!(
            "analysis_state[{frame_idx}].prev_bandwidth={}",
            state.prev_bandwidth
        );
        std::println!(
            "analysis_state[{frame_idx}].initialized={}",
            if state.initialized { 1 } else { 0 }
        );

        for (idx, value) in state.downmix_state.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].downmix_state[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.angle.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].angle[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.d_angle.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].d_angle[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.d2_angle.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].d2_angle[{idx}]={:.9e}",
                *value as f64
            );
        }
        if want_angle_bits() {
            for (idx, value) in state.angle.iter().enumerate() {
                std::println!(
                    "analysis_state[{frame_idx}].angle_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
            for (idx, value) in state.d_angle.iter().enumerate() {
                std::println!(
                    "analysis_state[{frame_idx}].d_angle_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
            for (idx, value) in state.d2_angle.iter().enumerate() {
                std::println!(
                    "analysis_state[{frame_idx}].d2_angle_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
        }
        for (idx, value) in state.e[last_idx].iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].E_last[{idx}]={:.9e}",
                *value as f64
            );
        }
        if want_e_history() {
            for (t, band_values) in state.e.iter().enumerate() {
                for (idx, value) in band_values.iter().enumerate() {
                    std::println!(
                        "analysis_state[{frame_idx}].E_hist_bits[{t}][{idx}]=0x{:08x}",
                        value.to_bits()
                    );
                }
            }
        }
        for (idx, value) in state.log_e[last_idx].iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].logE_last[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.low_e.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].lowE[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.high_e.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].highE[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.mean_e.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].meanE[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.prev_band_tonality.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].prev_band_tonality[{idx}]={:.9e}",
                *value as f64
            );
        }
        if want_band_tonality_bits() {
            for (idx, value) in state.prev_band_tonality.iter().enumerate() {
                std::println!(
                    "analysis_state[{frame_idx}].prev_band_tonality_bits[{idx}]=0x{:08x}",
                    value.to_bits()
                );
            }
        }
        for (idx, value) in state.mem.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].mem[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.cmean.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].cmean[{idx}]={:.9e}",
                *value as f64
            );
        }
        for (idx, value) in state.std.iter().enumerate() {
            std::println!(
                "analysis_state[{frame_idx}].std[{idx}]={:.9e}",
                *value as f64
            );
        }
    }

    #[test]
    fn analysis_compare_output() {
        let Some(path) = env::var_os("ANALYSIS_PCM").map(PathBuf::from) else {
            return;
        };
        let max_frames = env::var("ANALYSIS_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);

        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut state = TonalityAnalysisState::new(48_000);

        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(err) => {
                std::eprintln!(
                    "analysis_compare_output: failed to open {path:?}: {err}"
                );
                return;
            }
        };

        const FRAME_SIZE: usize = 960;
        const CHANNELS: usize = 2;
        let mut input_bytes = vec![0u8; FRAME_SIZE * CHANNELS * 2];
        let mut pcm = vec![0i16; FRAME_SIZE * CHANNELS];

        for frame_idx in 0..max_frames {
            match file.read_exact(&mut input_bytes) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("analysis_compare_output: read failed: {err}"),
            }

            for (sample, chunk) in pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
                *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            }

            let mut info = AnalysisInfo::default();
            fft_trace::set_frame_index(frame_idx);
            if env::var_os("ANALYSIS_TRACE_STAGE").is_some() {
                set_fft_trace_frame(frame_idx);
            }
            run_analysis(
                &mut state,
                &mode,
                Some(pcm.as_slice()),
                FRAME_SIZE,
                FRAME_SIZE,
                0,
                -2,
                CHANNELS as i32,
                48_000,
                16,
                &mut info,
            );
            dump_analysis_info(frame_idx, &info);
            dump_analysis_state(frame_idx, &state);
        }
    }

    #[test]
    fn analysis_fft_trace_output() {
        let Some(path) = env::var_os("ANALYSIS_PCM").map(PathBuf::from) else {
            return;
        };
        let trace_enabled = env::var_os("ANALYSIS_TRACE").is_some()
            || env::var_os("ANALYSIS_TRACE_BINS").is_some()
            || env::var_os("ANALYSIS_TRACE_TWIDDLES").is_some()
            || env::var_os("KISS_FFT_TRACE_TWIDDLES").is_some();
        if !trace_enabled {
            return;
        }
        let max_frames = env::var("ANALYSIS_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);

        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut state = TonalityAnalysisState::new(48_000);

        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(err) => {
                std::eprintln!(
                    "analysis_fft_trace_output: failed to open {path:?}: {err}"
                );
                return;
            }
        };

        const FRAME_SIZE: usize = 960;
        const CHANNELS: usize = 2;
        let mut input_bytes = vec![0u8; FRAME_SIZE * CHANNELS * 2];
        let mut pcm = vec![0i16; FRAME_SIZE * CHANNELS];

        for frame_idx in 0..max_frames {
            match file.read_exact(&mut input_bytes) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("analysis_fft_trace_output: read failed: {err}"),
            }

            for (sample, chunk) in pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
                *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            }

            fft_trace::set_frame_index(frame_idx);

            let mut info = AnalysisInfo::default();
            run_analysis(
                &mut state,
                &mode,
                Some(pcm.as_slice()),
                FRAME_SIZE,
                FRAME_SIZE,
                0,
                -2,
                CHANNELS as i32,
                48_000,
                16,
                &mut info,
            );
        }
    }
}
