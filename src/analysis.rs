#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use core::cmp::{max, min};
use core::f32::consts::{LOG2_E, PI};

use libm::{floorf, log10f, logf, sqrtf};

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

const TBANDS: [usize; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

const LEAKAGE_OFFSET: f32 = 2.5;
const LEAKAGE_SLOPE: f32 = 2.0;
const STD_FEATURE_BIAS: [f32; 9] = [
    5.684_947, 3.475_288, 1.770_634, 1.599_784, 3.773_215, 2.163_313, 1.260_756, 1.116_868,
    1.918_795,
];

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

        hp_ener += out32_hp * out32_hp;
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
        let pos_vad = tonal.info[vpos].activity_probability.max(0.0);
        prob_min =
            prob_min.min((prob_avg - TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count);
        prob_max =
            prob_max.max((prob_avg + TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count);
        prob_count += pos_vad.max(0.1);
        prob_avg += pos_vad.max(0.1) * tonal.info[mpos].music_prob;
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
        prob_min += (1.0 - 0.1 * curr_lookahead as f32) * (pmin - prob_min);
        prob_max += (1.0 - 0.1 * curr_lookahead as f32) * (pmax - prob_max);
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
        ret
    };

    if tonal.mem_fill + (len as usize) < ANALYSIS_BUF_SIZE {
        tonal.mem_fill += len as usize;
        tonal.hp_ener_accum = hp_ener;
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

    for i in 1..240 {
        let x1r = output_fft[i].r + output_fft[480 - i].r;
        let x1i = output_fft[i].i - output_fft[480 - i].i;
        let x2r = output_fft[i].i + output_fft[480 - i].i;
        let x2i = output_fft[480 - i].r - output_fft[i].r;

        let angle = 0.5 / PI * fast_atan2f(x1i, x1r);
        let d_angle = angle - tonal.angle[i];
        let d2_angle = d_angle - tonal.d_angle[i];

        let angle2 = 0.5 / PI * fast_atan2f(x2i, x2r);
        let d_angle2 = angle2 - angle;
        let d2_angle2 = d_angle2 - d_angle;

        let mut mod1 = d2_angle - float2int(d2_angle) as f32;
        noisiness[i] = mod1.abs();
        mod1 *= mod1;
        mod1 *= mod1;

        let mut mod2 = d2_angle2 - float2int(d2_angle2) as f32;
        noisiness[i] += mod2.abs();
        mod2 *= mod2;
        mod2 *= mod2;

        let avg_mod = 0.25 * (tonal.d2_angle[i] + mod1 + 2.0 * mod2);
        tonality[i] = 1.0 / (1.0 + 40.0 * 16.0 * PI * PI * PI * PI * avg_mod) - 0.015;
        tonality2[i] = 1.0 / (1.0 + 40.0 * 16.0 * PI * PI * PI * PI * mod2) - 0.015;

        tonal.angle[i] = angle2;
        tonal.d_angle[i] = d_angle2;
        tonal.d2_angle[i] = mod2;
    }

    for i in 2..239 {
        let tt = tonality2[i].min(tonality2[i - 1].max(tonality2[i + 1]));
        tonality[i] = 0.9 * tonality[i].max(tt - 0.1);
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
        let mut sum = x1r * x1r + x2r * x2r;
        for i in 1..4 {
            let bin_e = output_fft[i].r * output_fft[i].r
                + output_fft[480 - i].r * output_fft[480 - i].r
                + output_fft[i].i * output_fft[i].i
                + output_fft[480 - i].i * output_fft[480 - i].i;
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
            let bin_e = output_fft[i].r * output_fft[i].r
                + output_fft[480 - i].r * output_fft[480 - i].r
                + output_fft[i].i * output_fft[i].i
                + output_fft[480 - i].i * output_fft[480 - i].i;
            let bin_e = bin_e * SCALE_ENER;
            band_e += bin_e;
            t_e += bin_e * tonality[i].max(0.0);
            n_e += bin_e * 2.0 * (0.5 - noisiness[i]);
        }

        tonal.e[tonal.e_count][b] = band_e;
        frame_noisiness += n_e / (1e-15 + band_e);

        frame_loudness += sqrtf(band_e + 1e-10);
        log_e[b] = logf(band_e + 1e-10);
        band_log2[b + 1] = 0.5 * LOG2_E * logf(band_e + 1e-10);
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
        relative_e += (log_e[b] - tonal.low_e[b]) / (1e-5 + tonal.high_e[b] - tonal.low_e[b]);

        let mut l1 = 0.0f32;
        let mut l2 = 0.0f32;
        for i in 0..NB_FRAMES {
            l1 += sqrtf(tonal.e[i][b]);
            l2 += tonal.e[i][b];
        }
        let mut stationarity = (l1 / sqrtf(1e-15 + NB_FRAMES as f32 * l2)).min(0.99);
        stationarity = stationarity * stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;

        band_tonality[b] = t_e.max(0.0) / (1e-15 + band_e);
        band_tonality[b] = band_tonality[b].max(stationarity * tonal.prev_band_tonality[b]);
        frame_tonality += band_tonality[b];
        if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
            let idx = b + NB_TONAL_SKIP_BANDS - NB_TBANDS;
            frame_tonality -= band_tonality[idx];
        }
        max_frame_tonality =
            max_frame_tonality.max((1.0 + 0.03 * (b as f32 - NB_TBANDS as f32)) * frame_tonality);
        slope += band_tonality[b] * (b as f32 - 8.0);
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
                dist += tmp * tmp;
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
            let bin_e = output_fft[i].r * output_fft[i].r
                + output_fft[480 - i].r * output_fft[480 - i].r
                + output_fft[i].i * output_fft[i].i
                + output_fft[480 - i].i * output_fft[480 - i].i;
            band_e += bin_e;
        }
        band_e *= SCALE_ENER;
        max_e = max_e.max(band_e);
        if band_start < 64 {
            below_max_pitch += band_e;
        } else {
            above_max_pitch += band_e;
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
        let mut e_high = tonal.hp_ener_accum * (1.0 / (60.0 * 60.0));
        if e_high < 0.0 {
            e_high = 0.0;
        }
        let noise_ratio = if tonal.prev_bandwidth == 20 {
            10.0
        } else {
            30.0
        };
        above_max_pitch += e_high;
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
            sum += DCT_TABLE[i * 16 + b] * log_e[b];
        }
        bfcc[i] = sum;
    }
    for i in 0..8 {
        let mut sum = 0.0f32;
        for b in 0..16 {
            sum += DCT_TABLE[i * 16 + b] * 0.5 * (tonal.high_e[b] + tonal.low_e[b]);
        }
        mid_e[i] = sum;
    }

    frame_stationarity /= NB_TBANDS as f32;
    relative_e /= NB_TBANDS as f32;
    if tonal.count < 10 {
        relative_e = 0.5;
    }
    frame_noisiness /= NB_TBANDS as f32;
    info.activity = frame_noisiness + (1.0 - frame_noisiness) * relative_e;
    frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
    frame_tonality = frame_tonality.max(tonal.prev_tonality * 0.8);
    tonal.prev_tonality = frame_tonality;

    slope /= 64.0;
    info.tonality_slope = slope;

    tonal.e_count = (tonal.e_count + 1) % NB_FRAMES;
    tonal.count = min(tonal.count + 1, ANALYSIS_COUNT_MAX);
    info.tonality = frame_tonality;

    for i in 0..4 {
        features[i] = -0.122_99 * (bfcc[i] + tonal.mem[i + 24])
            + 0.491_95 * (tonal.mem[i] + tonal.mem[i + 16])
            + 0.696_93 * tonal.mem[i + 8]
            - 1.4349 * tonal.cmean[i];
    }

    for i in 0..4 {
        tonal.cmean[i] = (1.0 - alpha) * tonal.cmean[i] + alpha * bfcc[i];
    }

    for i in 0..4 {
        features[4 + i] = 0.632_46 * (bfcc[i] - tonal.mem[i + 24])
            + 0.316_23 * (tonal.mem[i] - tonal.mem[i + 16]);
    }
    for i in 0..3 {
        features[8 + i] = 0.534_52 * (bfcc[i] + tonal.mem[i + 24])
            - 0.267_26 * (tonal.mem[i] + tonal.mem[i + 16])
            - 0.534_52 * tonal.mem[i + 8];
    }

    if tonal.count > 5 {
        for i in 0..9 {
            tonal.std[i] = (1.0 - alpha) * tonal.std[i] + alpha * features[i] * features[i];
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
        features[11 + i] = sqrtf(tonal.std[i]) - STD_FEATURE_BIAS[i];
    }
    features[18] = spec_variability - 0.78;
    features[20] = info.tonality - 0.154_723;
    features[21] = info.activity - 0.724_643;
    features[22] = frame_stationarity - 0.743_717;
    features[23] = info.tonality_slope + 0.069_216;
    features[24] = tonal.low_e_count - 0.067_930;

    analysis_compute_dense(&LAYER0, &mut layer_out, &features);
    analysis_compute_gru(&LAYER1, &mut tonal.rnn_state, &layer_out);
    let mut frame_probs = [0.0f32; 2];
    analysis_compute_dense(&LAYER2, &mut frame_probs, &tonal.rnn_state);

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
    use alloc::vec::Vec;

    use crate::celt::{AnalysisInfo, opus_custom_mode_find_static};

    use super::{TonalityAnalysisState, run_analysis, tonality_analysis_reset};

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
    fn reset_clears_runtime_state() {
        let mut state = TonalityAnalysisState::new(48_000);
        state.mem_fill = 123;
        state.count = 42;
        tonality_analysis_reset(&mut state);
        assert_eq!(state.mem_fill, 0);
        assert_eq!(state.count, 0);
    }
}
