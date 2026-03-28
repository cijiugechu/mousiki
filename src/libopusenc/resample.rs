extern crate std;

use alloc::vec;
use alloc::vec::Vec;

const QUALITY_MAP: [QualityMapping; 11] = [
    QualityMapping::new(8, 4, 0.830, 0.860, WindowFunction::Kaiser6),
    QualityMapping::new(16, 4, 0.850, 0.880, WindowFunction::Kaiser6),
    QualityMapping::new(32, 4, 0.882, 0.910, WindowFunction::Kaiser6),
    QualityMapping::new(48, 8, 0.895, 0.917, WindowFunction::Kaiser8),
    QualityMapping::new(64, 8, 0.921, 0.940, WindowFunction::Kaiser8),
    QualityMapping::new(80, 16, 0.922, 0.940, WindowFunction::Kaiser10),
    QualityMapping::new(96, 16, 0.940, 0.945, WindowFunction::Kaiser10),
    QualityMapping::new(128, 16, 0.950, 0.950, WindowFunction::Kaiser10),
    QualityMapping::new(160, 16, 0.960, 0.960, WindowFunction::Kaiser10),
    QualityMapping::new(192, 32, 0.968, 0.968, WindowFunction::Kaiser12),
    QualityMapping::new(256, 32, 0.975, 0.975, WindowFunction::Kaiser12),
];

// Upstream libopusenc enables RESAMPLE_FULL_SINC_TABLE in configure.ac/config.h.
// Match that build configuration so the port selects the same direct sinc kernels.
const RESAMPLE_FULL_SINC_TABLE: bool = true;

const KAISER12_TABLE: [f64; 68] = [
    0.99859849,
    1.00000000,
    0.99859849,
    0.99440475,
    0.98745105,
    0.97779076,
    0.96549770,
    0.95066529,
    0.93340547,
    0.91384741,
    0.89213598,
    0.86843014,
    0.84290116,
    0.81573067,
    0.78710866,
    0.75723148,
    0.72629970,
    0.69451601,
    0.66208321,
    0.62920216,
    0.59606986,
    0.56287762,
    0.52980938,
    0.49704014,
    0.46473455,
    0.43304576,
    0.40211431,
    0.37206735,
    0.34301800,
    0.31506490,
    0.28829195,
    0.26276832,
    0.23854851,
    0.21567274,
    0.19416736,
    0.17404546,
    0.15530766,
    0.13794294,
    0.12192957,
    0.10723616,
    0.09382272,
    0.08164178,
    0.07063950,
    0.06075685,
    0.05193064,
    0.04409466,
    0.03718069,
    0.03111947,
    0.02584161,
    0.02127838,
    0.01736250,
    0.01402878,
    0.01121463,
    0.00886058,
    0.00691064,
    0.00531256,
    0.00401805,
    0.00298291,
    0.00216702,
    0.00153438,
    0.00105297,
    0.00069463,
    0.00043489,
    0.00025272,
    0.00013031,
    0.0000527734,
    0.00001000,
    0.00000000,
];

const KAISER10_TABLE: [f64; 36] = [
    0.99537781, 1.00000000, 0.99537781, 0.98162644, 0.95908712, 0.92831446, 0.89005583, 0.84522401,
    0.79486424, 0.74011713, 0.68217934, 0.62226347, 0.56155915, 0.50119680, 0.44221549, 0.38553619,
    0.33194107, 0.28205962, 0.23636152, 0.19515633, 0.15859932, 0.12670280, 0.09935205, 0.07632451,
    0.05731132, 0.04193980, 0.02979584, 0.02044510, 0.01345224, 0.00839739, 0.00488951, 0.00257636,
    0.00115101, 0.00035515, 0.00000000, 0.00000000,
];

const KAISER8_TABLE: [f64; 36] = [
    0.99635258, 1.00000000, 0.99635258, 0.98548012, 0.96759014, 0.94302200, 0.91223751, 0.87580811,
    0.83439927, 0.78875245, 0.73966538, 0.68797126, 0.63451750, 0.58014482, 0.52566725, 0.47185369,
    0.41941150, 0.36897272, 0.32108304, 0.27619388, 0.23465776, 0.19672670, 0.16255380, 0.13219758,
    0.10562887, 0.08273982, 0.06335451, 0.04724088, 0.03412321, 0.02369490, 0.01563093, 0.00959968,
    0.00527363, 0.00233883, 0.00050000, 0.00000000,
];

const KAISER6_TABLE: [f64; 36] = [
    0.99733006, 1.00000000, 0.99733006, 0.98935595, 0.97618418, 0.95799003, 0.93501423, 0.90755855,
    0.87598009, 0.84068475, 0.80211977, 0.76076565, 0.71712752, 0.67172623, 0.62508937, 0.57774224,
    0.53019925, 0.48295561, 0.43647969, 0.39120616, 0.34752997, 0.30580127, 0.26632152, 0.22934058,
    0.19505503, 0.16360756, 0.13508755, 0.10953262, 0.08693120, 0.06722600, 0.05031820, 0.03607231,
    0.02432151, 0.01487334, 0.00752000, 0.00000000,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResamplerError {
    Success = 0,
    AllocFailed = 1,
    BadState = 2,
    InvalidArg = 3,
    PtrOverlap = 4,
    Overflow = 5,
}

impl ResamplerError {
    #[must_use]
    pub(crate) const fn strerror(self) -> &'static str {
        match self {
            Self::Success => "Success.",
            Self::AllocFailed => "Memory allocation failed.",
            Self::BadState => "Bad resampler state.",
            Self::InvalidArg => "Invalid argument.",
            Self::PtrOverlap => "Input and output buffers overlap.",
            Self::Overflow => "Unknown error. Bad error code or strange version mismatch.",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct QualityMapping {
    base_length: u32,
    oversample: u32,
    downsample_bandwidth: f32,
    upsample_bandwidth: f32,
    window_func: WindowFunction,
}

impl QualityMapping {
    const fn new(
        base_length: u32,
        oversample: u32,
        downsample_bandwidth: f32,
        upsample_bandwidth: f32,
        window_func: WindowFunction,
    ) -> Self {
        Self {
            base_length,
            oversample,
            downsample_bandwidth,
            upsample_bandwidth,
            window_func,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WindowFunction {
    Kaiser12,
    Kaiser10,
    Kaiser8,
    Kaiser6,
}

impl WindowFunction {
    const fn table(self) -> &'static [f64] {
        match self {
            Self::Kaiser12 => &KAISER12_TABLE,
            Self::Kaiser10 => &KAISER10_TABLE,
            Self::Kaiser8 => &KAISER8_TABLE,
            Self::Kaiser6 => &KAISER6_TABLE,
        }
    }

    const fn oversample(self) -> usize {
        match self {
            Self::Kaiser12 => 64,
            Self::Kaiser10 | Self::Kaiser8 | Self::Kaiser6 => 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResamplerImpl {
    DirectSingle,
    DirectDouble,
    InterpolateSingle,
    InterpolateDouble,
    Zero,
}

#[derive(Debug, Clone)]
pub(crate) struct SpeexResampler {
    in_rate: u32,
    out_rate: u32,
    num_rate: u32,
    den_rate: u32,
    quality: i32,
    nb_channels: u32,
    filt_len: u32,
    mem_alloc_size: u32,
    buffer_size: u32,
    int_advance: u32,
    frac_advance: u32,
    cutoff: f32,
    oversample: u32,
    initialised: bool,
    started: bool,
    last_sample: Vec<i32>,
    samp_frac_num: Vec<u32>,
    magic_samples: Vec<u32>,
    mem: Vec<f32>,
    sinc_table: Vec<f32>,
    mode: ResamplerImpl,
    in_stride: u32,
    out_stride: u32,
}

impl SpeexResampler {
    pub(crate) fn new(
        nb_channels: u32,
        in_rate: u32,
        out_rate: u32,
        quality: i32,
    ) -> Result<Self, ResamplerError> {
        Self::new_frac(nb_channels, in_rate, out_rate, in_rate, out_rate, quality)
    }

    pub(crate) fn new_frac(
        nb_channels: u32,
        ratio_num: u32,
        ratio_den: u32,
        in_rate: u32,
        out_rate: u32,
        quality: i32,
    ) -> Result<Self, ResamplerError> {
        if nb_channels == 0 || ratio_num == 0 || ratio_den == 0 || !(0..=10).contains(&quality) {
            return Err(ResamplerError::InvalidArg);
        }
        let mut st = Self {
            in_rate: 0,
            out_rate: 0,
            num_rate: 0,
            den_rate: 0,
            quality: -1,
            nb_channels,
            filt_len: 0,
            mem_alloc_size: 0,
            buffer_size: 160,
            int_advance: 0,
            frac_advance: 0,
            cutoff: 1.0,
            oversample: 0,
            initialised: false,
            started: false,
            last_sample: vec![0; nb_channels as usize],
            samp_frac_num: vec![0; nb_channels as usize],
            magic_samples: vec![0; nb_channels as usize],
            mem: Vec::new(),
            sinc_table: Vec::new(),
            mode: ResamplerImpl::Zero,
            in_stride: 1,
            out_stride: 1,
        };
        st.set_quality(quality)?;
        st.set_rate_frac(ratio_num, ratio_den, in_rate, out_rate)?;
        st.update_filter()?;
        st.initialised = true;
        Ok(st)
    }

    #[must_use]
    pub(crate) const fn quality(&self) -> i32 {
        self.quality
    }

    pub(crate) fn set_quality(&mut self, quality: i32) -> Result<(), ResamplerError> {
        if !(0..=10).contains(&quality) {
            return Err(ResamplerError::InvalidArg);
        }
        if self.quality == quality {
            return Ok(());
        }
        self.quality = quality;
        if self.initialised {
            self.update_filter()?;
        }
        Ok(())
    }

    #[must_use]
    pub(crate) const fn rates(&self) -> (u32, u32) {
        (self.in_rate, self.out_rate)
    }

    pub(crate) fn set_rate(&mut self, in_rate: u32, out_rate: u32) -> Result<(), ResamplerError> {
        self.set_rate_frac(in_rate, out_rate, in_rate, out_rate)
    }

    pub(crate) fn set_rate_frac(
        &mut self,
        ratio_num: u32,
        ratio_den: u32,
        in_rate: u32,
        out_rate: u32,
    ) -> Result<(), ResamplerError> {
        if ratio_num == 0 || ratio_den == 0 {
            return Err(ResamplerError::InvalidArg);
        }
        if self.in_rate == in_rate
            && self.out_rate == out_rate
            && self.num_rate == ratio_num
            && self.den_rate == ratio_den
        {
            return Ok(());
        }
        let old_den = self.den_rate;
        self.in_rate = in_rate;
        self.out_rate = out_rate;
        self.num_rate = ratio_num;
        self.den_rate = ratio_den;

        let fact = compute_gcd(self.num_rate, self.den_rate);
        self.num_rate /= fact;
        self.den_rate /= fact;

        if old_den > 0 {
            for samp_frac_num in &mut self.samp_frac_num {
                *samp_frac_num = multiply_frac(*samp_frac_num, self.den_rate, old_den)?;
                if *samp_frac_num >= self.den_rate {
                    *samp_frac_num = self.den_rate - 1;
                }
            }
        }

        if self.initialised {
            self.update_filter()?;
        }
        Ok(())
    }

    #[must_use]
    pub(crate) const fn ratio(&self) -> (u32, u32) {
        (self.num_rate, self.den_rate)
    }

    pub(crate) fn set_input_stride(&mut self, stride: u32) {
        self.in_stride = stride;
    }

    #[must_use]
    pub(crate) const fn input_stride(&self) -> u32 {
        self.in_stride
    }

    pub(crate) fn set_output_stride(&mut self, stride: u32) {
        self.out_stride = stride;
    }

    #[must_use]
    pub(crate) const fn output_stride(&self) -> u32 {
        self.out_stride
    }

    #[must_use]
    pub(crate) const fn input_latency(&self) -> u32 {
        self.filt_len / 2
    }

    #[must_use]
    pub(crate) const fn output_latency(&self) -> u32 {
        ((self.filt_len / 2) * self.den_rate + (self.num_rate >> 1)) / self.num_rate
    }

    pub(crate) fn skip_zeros(&mut self) -> Result<(), ResamplerError> {
        let last = (self.filt_len / 2) as i32;
        for item in &mut self.last_sample {
            *item = last;
        }
        Ok(())
    }

    pub(crate) fn reset_mem(&mut self) -> Result<(), ResamplerError> {
        for last in &mut self.last_sample {
            *last = 0;
        }
        for magic in &mut self.magic_samples {
            *magic = 0;
        }
        for frac in &mut self.samp_frac_num {
            *frac = 0;
        }
        let used = self.nb_channels as usize * self.filt_len.saturating_sub(1) as usize;
        self.mem[..used].fill(0.0);
        Ok(())
    }

    pub(crate) fn process_float(
        &mut self,
        channel_index: u32,
        input: Option<&[f32]>,
        in_len: &mut u32,
        output: &mut [f32],
        out_len: &mut u32,
    ) -> Result<(), ResamplerError> {
        self.process(channel_index, input, in_len, output, out_len)
    }

    pub(crate) fn process_int(
        &mut self,
        channel_index: u32,
        input: Option<&[i16]>,
        in_len: &mut u32,
        output: &mut [i16],
        out_len: &mut u32,
    ) -> Result<(), ResamplerError> {
        self.process(channel_index, input, in_len, output, out_len)
    }

    pub(crate) fn process_interleaved_float(
        &mut self,
        input: Option<&[f32]>,
        in_len: &mut u32,
        output: &mut [f32],
        out_len: &mut u32,
    ) -> Result<(), ResamplerError> {
        self.process_interleaved(input, in_len, output, out_len, Self::process_float)
    }

    pub(crate) fn process_interleaved_int(
        &mut self,
        input: Option<&[i16]>,
        in_len: &mut u32,
        output: &mut [i16],
        out_len: &mut u32,
    ) -> Result<(), ResamplerError> {
        self.process_interleaved(input, in_len, output, out_len, Self::process_int)
    }

    fn process<TIn, TOut>(
        &mut self,
        channel_index: u32,
        input: Option<&[TIn]>,
        in_len: &mut u32,
        output: &mut [TOut],
        out_len: &mut u32,
    ) -> Result<(), ResamplerError>
    where
        TIn: ResamplerSampleIn,
        TOut: ResamplerSampleOut,
    {
        let channel = channel_index as usize;
        if channel >= self.nb_channels as usize {
            return Err(ResamplerError::InvalidArg);
        }
        let mut ilen = *in_len as usize;
        let mut olen = *out_len as usize;
        let filt_offs = self.filt_len.saturating_sub(1) as usize;
        let xlen = self.mem_alloc_size as usize - filt_offs;
        let istride = self.in_stride as usize;

        let mut out_cursor = 0usize;
        let mut in_cursor = 0usize;

        if self.magic_samples[channel] > 0 {
            let produced = self.process_magic::<TOut>(channel, output, out_cursor, olen)?;
            out_cursor += produced * self.out_stride as usize;
            olen -= produced;
        }

        if self.magic_samples[channel] == 0 {
            while ilen > 0 && olen > 0 {
                let mut ichunk = ilen.min(xlen);
                let mut ochunk = olen;
                let mem_offset = channel * self.mem_alloc_size as usize;
                for j in 0..ichunk {
                    self.mem[mem_offset + filt_offs + j] = input
                        .and_then(|src| src.get(in_cursor + j * istride))
                        .map_or(0.0, |sample| sample.to_resampler_f32());
                }
                self.process_native(channel, &mut ichunk, output, out_cursor, &mut ochunk)?;
                ilen -= ichunk;
                olen -= ochunk;
                out_cursor += ochunk * self.out_stride as usize;
                in_cursor += ichunk * istride;
            }
        }

        *in_len -= ilen as u32;
        *out_len -= olen as u32;
        if self.mode == ResamplerImpl::Zero {
            return Err(ResamplerError::AllocFailed);
        }
        Ok(())
    }

    fn process_interleaved<TIn, TOut>(
        &mut self,
        input: Option<&[TIn]>,
        in_len: &mut u32,
        output: &mut [TOut],
        out_len: &mut u32,
        process_fn: fn(
            &mut Self,
            u32,
            Option<&[TIn]>,
            &mut u32,
            &mut [TOut],
            &mut u32,
        ) -> Result<(), ResamplerError>,
    ) -> Result<(), ResamplerError>
    where
        TIn: ResamplerSampleIn,
        TOut: ResamplerSampleOut,
    {
        let bak_out_len = *out_len;
        let bak_in_len = *in_len;
        let istride = self.in_stride;
        let ostride = self.out_stride;
        self.in_stride = self.nb_channels;
        self.out_stride = self.nb_channels;

        let mut result = Ok(());
        for i in 0..self.nb_channels as usize {
            *out_len = bak_out_len;
            *in_len = bak_in_len;
            let out_slice = if i < output.len() {
                &mut output[i..]
            } else {
                result = Err(ResamplerError::InvalidArg);
                break;
            };
            let in_slice = input.map(|src| if i < src.len() { &src[i..] } else { &src[0..0] });
            if let Err(err) = process_fn(self, i as u32, in_slice, in_len, out_slice, out_len) {
                result = Err(err);
            }
        }

        self.in_stride = istride;
        self.out_stride = ostride;
        result
    }

    fn process_magic<TOut>(
        &mut self,
        channel: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> Result<usize, ResamplerError>
    where
        TOut: ResamplerSampleOut,
    {
        let mut tmp_in_len = self.magic_samples[channel] as usize;
        let mut produced = out_len;
        self.process_native(channel, &mut tmp_in_len, output, out_offset, &mut produced)?;
        self.magic_samples[channel] -= tmp_in_len as u32;
        if self.magic_samples[channel] > 0 {
            let mem_offset = channel * self.mem_alloc_size as usize;
            let n = self.filt_len as usize;
            for i in 0..self.magic_samples[channel] as usize {
                self.mem[mem_offset + n - 1 + i] = self.mem[mem_offset + n - 1 + i + tmp_in_len];
            }
        }
        Ok(produced)
    }

    fn process_native<TOut>(
        &mut self,
        channel: usize,
        in_len: &mut usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: &mut usize,
    ) -> Result<(), ResamplerError>
    where
        TOut: ResamplerSampleOut,
    {
        self.started = true;
        let produced = match self.mode {
            ResamplerImpl::DirectSingle => {
                self.resampler_basic_direct_single(channel, *in_len, output, out_offset, *out_len)
            }
            ResamplerImpl::DirectDouble => {
                self.resampler_basic_direct_double(channel, *in_len, output, out_offset, *out_len)
            }
            ResamplerImpl::InterpolateSingle => self
                .resampler_basic_interpolate_single(channel, *in_len, output, out_offset, *out_len),
            ResamplerImpl::InterpolateDouble => self
                .resampler_basic_interpolate_double(channel, *in_len, output, out_offset, *out_len),
            ResamplerImpl::Zero => {
                self.resampler_basic_zero(channel, *in_len, output, out_offset, *out_len)
            }
        };

        if self.last_sample[channel] < *in_len as i32 {
            *in_len = self.last_sample[channel] as usize;
        }
        *out_len = produced;
        self.last_sample[channel] -= *in_len as i32;

        let mem_offset = channel * self.mem_alloc_size as usize;
        let n = self.filt_len as usize;
        if n > 1 {
            let src_start = mem_offset + *in_len;
            let src_end = src_start + (n - 1);
            self.mem.copy_within(src_start..src_end, mem_offset);
        }
        Ok(())
    }

    fn resampler_basic_direct_single<TOut>(
        &mut self,
        channel: usize,
        in_len: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> usize
    where
        TOut: ResamplerSampleOut,
    {
        let n = self.filt_len as usize;
        let out_stride = self.out_stride as usize;
        let int_advance = self.int_advance;
        let frac_advance = self.frac_advance;
        let den_rate = self.den_rate;
        let mem_offset = channel * self.mem_alloc_size as usize;

        let mut out_sample = 0usize;
        let mut last_sample = self.last_sample[channel];
        let mut samp_frac_num = self.samp_frac_num[channel];
        while last_sample < in_len as i32 && out_sample < out_len {
            let sinc_offset = samp_frac_num as usize * n;
            let input_offset = mem_offset + last_sample as usize;
            let mut sum = 0.0f32;
            unsafe {
                // The resampler keeps `filt_len - 1 + in_len` samples in `mem`,
                // so this contiguous FIR window is guaranteed in-bounds here.
                let sinc_ptr = self.sinc_table.as_ptr().add(sinc_offset);
                let mem_ptr = self.mem.as_ptr().add(input_offset);
                for j in 0..n {
                    sum += *sinc_ptr.add(j) * *mem_ptr.add(j);
                }
            }
            output[out_offset + out_sample * out_stride].write_resampler_f32(sum);
            out_sample += 1;
            last_sample += int_advance as i32;
            samp_frac_num += frac_advance;
            if samp_frac_num >= den_rate {
                samp_frac_num -= den_rate;
                last_sample += 1;
            }
        }
        self.last_sample[channel] = last_sample;
        self.samp_frac_num[channel] = samp_frac_num;
        out_sample
    }

    fn resampler_basic_direct_double<TOut>(
        &mut self,
        channel: usize,
        in_len: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> usize
    where
        TOut: ResamplerSampleOut,
    {
        let n = self.filt_len as usize;
        let out_stride = self.out_stride as usize;
        let int_advance = self.int_advance;
        let frac_advance = self.frac_advance;
        let den_rate = self.den_rate;
        let mem_offset = channel * self.mem_alloc_size as usize;

        let mut out_sample = 0usize;
        let mut last_sample = self.last_sample[channel];
        let mut samp_frac_num = self.samp_frac_num[channel];
        while last_sample < in_len as i32 && out_sample < out_len {
            let sinc_offset = samp_frac_num as usize * n;
            let input_offset = mem_offset + last_sample as usize;
            let mut accum = [0.0f64; 4];
            let mut j = 0usize;
            unsafe {
                let sinc_ptr = self.sinc_table.as_ptr().add(sinc_offset);
                let mem_ptr = self.mem.as_ptr().add(input_offset);
                while j < n {
                    accum[0] += *sinc_ptr.add(j) as f64 * *mem_ptr.add(j) as f64;
                    accum[1] += *sinc_ptr.add(j + 1) as f64 * *mem_ptr.add(j + 1) as f64;
                    accum[2] += *sinc_ptr.add(j + 2) as f64 * *mem_ptr.add(j + 2) as f64;
                    accum[3] += *sinc_ptr.add(j + 3) as f64 * *mem_ptr.add(j + 3) as f64;
                    j += 4;
                }
            }
            output[out_offset + out_sample * out_stride]
                .write_resampler_f32((accum[0] + accum[1] + accum[2] + accum[3]) as f32);
            out_sample += 1;
            last_sample += int_advance as i32;
            samp_frac_num += frac_advance;
            if samp_frac_num >= den_rate {
                samp_frac_num -= den_rate;
                last_sample += 1;
            }
        }
        self.last_sample[channel] = last_sample;
        self.samp_frac_num[channel] = samp_frac_num;
        out_sample
    }

    fn resampler_basic_interpolate_single<TOut>(
        &mut self,
        channel: usize,
        in_len: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> usize
    where
        TOut: ResamplerSampleOut,
    {
        let n = self.filt_len as usize;
        let out_stride = self.out_stride as usize;
        let int_advance = self.int_advance;
        let frac_advance = self.frac_advance;
        let den_rate = self.den_rate;
        let oversample = self.oversample as usize;
        let mem_offset = channel * self.mem_alloc_size as usize;

        let mut out_sample = 0usize;
        let mut last_sample = self.last_sample[channel];
        let mut samp_frac_num = self.samp_frac_num[channel];
        while last_sample < in_len as i32 && out_sample < out_len {
            let input_offset = mem_offset + last_sample as usize;
            let offset = samp_frac_num as usize * oversample / den_rate as usize;
            let frac = ((samp_frac_num as usize * oversample) % den_rate as usize) as f32
                / den_rate as f32;
            let interp = cubic_coef(frac);
            let mut accum = [0.0f32; 4];
            let sinc_base = 4usize + oversample - offset;
            unsafe {
                let sinc_ptr = self.sinc_table.as_ptr();
                let mem_ptr = self.mem.as_ptr().add(input_offset);
                for j in 0..n {
                    let curr_in = *mem_ptr.add(j);
                    let sinc_index = sinc_base + j * oversample;
                    accum[0] += curr_in * *sinc_ptr.add(sinc_index - 2);
                    accum[1] += curr_in * *sinc_ptr.add(sinc_index - 1);
                    accum[2] += curr_in * *sinc_ptr.add(sinc_index);
                    accum[3] += curr_in * *sinc_ptr.add(sinc_index + 1);
                }
            }
            let sum = interp[0] * accum[0]
                + interp[1] * accum[1]
                + interp[2] * accum[2]
                + interp[3] * accum[3];
            output[out_offset + out_sample * out_stride].write_resampler_f32(sum);
            out_sample += 1;
            last_sample += int_advance as i32;
            samp_frac_num += frac_advance;
            if samp_frac_num >= den_rate {
                samp_frac_num -= den_rate;
                last_sample += 1;
            }
        }
        self.last_sample[channel] = last_sample;
        self.samp_frac_num[channel] = samp_frac_num;
        out_sample
    }

    fn resampler_basic_interpolate_double<TOut>(
        &mut self,
        channel: usize,
        in_len: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> usize
    where
        TOut: ResamplerSampleOut,
    {
        let n = self.filt_len as usize;
        let out_stride = self.out_stride as usize;
        let int_advance = self.int_advance;
        let frac_advance = self.frac_advance;
        let den_rate = self.den_rate;
        let oversample = self.oversample as usize;
        let mem_offset = channel * self.mem_alloc_size as usize;

        let mut out_sample = 0usize;
        let mut last_sample = self.last_sample[channel];
        let mut samp_frac_num = self.samp_frac_num[channel];
        while last_sample < in_len as i32 && out_sample < out_len {
            let input_offset = mem_offset + last_sample as usize;
            let offset = samp_frac_num as usize * oversample / den_rate as usize;
            let frac = ((samp_frac_num as usize * oversample) % den_rate as usize) as f32
                / den_rate as f32;
            let interp = cubic_coef(frac);
            let mut accum = [0.0f64; 4];
            let sinc_base = 4usize + oversample - offset;
            unsafe {
                let sinc_ptr = self.sinc_table.as_ptr();
                let mem_ptr = self.mem.as_ptr().add(input_offset);
                for j in 0..n {
                    let curr_in = *mem_ptr.add(j) as f64;
                    let sinc_index = sinc_base + j * oversample;
                    accum[0] += curr_in * *sinc_ptr.add(sinc_index - 2) as f64;
                    accum[1] += curr_in * *sinc_ptr.add(sinc_index - 1) as f64;
                    accum[2] += curr_in * *sinc_ptr.add(sinc_index) as f64;
                    accum[3] += curr_in * *sinc_ptr.add(sinc_index + 1) as f64;
                }
            }
            let sum = interp[0] as f64 * accum[0]
                + interp[1] as f64 * accum[1]
                + interp[2] as f64 * accum[2]
                + interp[3] as f64 * accum[3];
            output[out_offset + out_sample * out_stride].write_resampler_f32(sum as f32);
            out_sample += 1;
            last_sample += int_advance as i32;
            samp_frac_num += frac_advance;
            if samp_frac_num >= den_rate {
                samp_frac_num -= den_rate;
                last_sample += 1;
            }
        }
        self.last_sample[channel] = last_sample;
        self.samp_frac_num[channel] = samp_frac_num;
        out_sample
    }

    fn resampler_basic_zero<TOut>(
        &mut self,
        channel: usize,
        in_len: usize,
        output: &mut [TOut],
        out_offset: usize,
        out_len: usize,
    ) -> usize
    where
        TOut: ResamplerSampleOut,
    {
        let out_stride = self.out_stride as usize;
        let int_advance = self.int_advance;
        let frac_advance = self.frac_advance;
        let den_rate = self.den_rate;

        let mut out_sample = 0usize;
        let mut last_sample = self.last_sample[channel];
        let mut samp_frac_num = self.samp_frac_num[channel];
        while last_sample < in_len as i32 && out_sample < out_len {
            output[out_offset + out_sample * out_stride].write_resampler_f32(0.0);
            out_sample += 1;
            last_sample += int_advance as i32;
            samp_frac_num += frac_advance;
            if samp_frac_num >= den_rate {
                samp_frac_num -= den_rate;
                last_sample += 1;
            }
        }
        self.last_sample[channel] = last_sample;
        self.samp_frac_num[channel] = samp_frac_num;
        out_sample
    }

    fn update_filter(&mut self) -> Result<(), ResamplerError> {
        let old_length = self.filt_len;
        let old_alloc_size = self.mem_alloc_size;

        self.int_advance = self.num_rate / self.den_rate;
        self.frac_advance = self.num_rate % self.den_rate;
        self.oversample = QUALITY_MAP[self.quality as usize].oversample;
        self.filt_len = QUALITY_MAP[self.quality as usize].base_length;

        if self.num_rate > self.den_rate {
            self.cutoff = QUALITY_MAP[self.quality as usize].downsample_bandwidth
                * self.den_rate as f32
                / self.num_rate as f32;
            self.filt_len = multiply_frac(self.filt_len, self.num_rate, self.den_rate)?;
            self.filt_len = ((self.filt_len - 1) & !0x7) + 8;
            if 2 * self.den_rate < self.num_rate {
                self.oversample >>= 1;
            }
            if 4 * self.den_rate < self.num_rate {
                self.oversample >>= 1;
            }
            if 8 * self.den_rate < self.num_rate {
                self.oversample >>= 1;
            }
            if 16 * self.den_rate < self.num_rate {
                self.oversample >>= 1;
            }
            self.oversample = self.oversample.max(1);
        } else {
            self.cutoff = QUALITY_MAP[self.quality as usize].upsample_bandwidth;
        }

        let use_direct = if RESAMPLE_FULL_SINC_TABLE {
            true
        } else {
            self.filt_len.saturating_mul(self.den_rate)
                <= self
                    .filt_len
                    .saturating_mul(self.oversample)
                    .saturating_add(8)
        };
        let min_sinc_table_length = if use_direct {
            self.filt_len.saturating_mul(self.den_rate) as usize
        } else {
            self.filt_len
                .saturating_mul(self.oversample)
                .saturating_add(8) as usize
        };
        if self.sinc_table.len() < min_sinc_table_length {
            self.sinc_table.resize(min_sinc_table_length, 0.0);
        }

        if use_direct {
            for i in 0..self.den_rate as usize {
                for j in 0..self.filt_len as usize {
                    self.sinc_table[i * self.filt_len as usize + j] = sinc(
                        self.cutoff,
                        (j as i32 - self.filt_len as i32 / 2 + 1) as f32
                            - i as f32 / self.den_rate as f32,
                        self.filt_len as usize,
                        QUALITY_MAP[self.quality as usize].window_func,
                    );
                }
            }
            self.mode = if self.quality > 8 {
                ResamplerImpl::DirectDouble
            } else {
                ResamplerImpl::DirectSingle
            };
        } else {
            for i in -4i32..(self.oversample * self.filt_len + 4) as i32 {
                self.sinc_table[(i + 4) as usize] = sinc(
                    self.cutoff,
                    i as f32 / self.oversample as f32 - self.filt_len as f32 / 2.0,
                    self.filt_len as usize,
                    QUALITY_MAP[self.quality as usize].window_func,
                );
            }
            self.mode = if self.quality > 8 {
                ResamplerImpl::InterpolateDouble
            } else {
                ResamplerImpl::InterpolateSingle
            };
        }

        let min_alloc_size = self
            .filt_len
            .saturating_sub(1)
            .saturating_add(self.buffer_size);
        if min_alloc_size > self.mem_alloc_size {
            self.mem_alloc_size = min_alloc_size;
            self.mem.resize(
                self.nb_channels as usize * self.mem_alloc_size as usize,
                0.0,
            );
        }

        if !self.started {
            self.mem.fill(0.0);
        } else if self.filt_len > old_length {
            for i in (0..self.nb_channels as usize).rev() {
                let mut olen = old_length;
                if self.magic_samples[i] > 0 {
                    olen = old_length + 2 * self.magic_samples[i];
                    for j in (0..old_length as usize - 1 + self.magic_samples[i] as usize).rev() {
                        self.mem[i * self.mem_alloc_size as usize
                            + j
                            + self.magic_samples[i] as usize] =
                            self.mem[i * old_alloc_size as usize + j];
                    }
                    for j in 0..self.magic_samples[i] as usize {
                        self.mem[i * self.mem_alloc_size as usize + j] = 0.0;
                    }
                    self.magic_samples[i] = 0;
                }
                if self.filt_len > olen {
                    let mut j = 0usize;
                    while j < olen as usize - 1 {
                        self.mem
                            [i * self.mem_alloc_size as usize + (self.filt_len as usize - 2 - j)] =
                            self.mem[i * self.mem_alloc_size as usize + (olen as usize - 2 - j)];
                        j += 1;
                    }
                    while j < self.filt_len as usize - 1 {
                        self.mem
                            [i * self.mem_alloc_size as usize + (self.filt_len as usize - 2 - j)] =
                            0.0;
                        j += 1;
                    }
                    self.last_sample[i] += ((self.filt_len - olen) / 2) as i32;
                } else {
                    self.magic_samples[i] = (olen - self.filt_len) / 2;
                    for j in 0..self.filt_len as usize - 1 + self.magic_samples[i] as usize {
                        self.mem[i * self.mem_alloc_size as usize + j] = self.mem
                            [i * self.mem_alloc_size as usize + j + self.magic_samples[i] as usize];
                    }
                }
            }
        } else if self.filt_len < old_length {
            for i in 0..self.nb_channels as usize {
                let old_magic = self.magic_samples[i];
                self.magic_samples[i] = (old_length - self.filt_len) / 2;
                for j in 0..(self.filt_len as usize - 1
                    + self.magic_samples[i] as usize
                    + old_magic as usize)
                {
                    self.mem[i * self.mem_alloc_size as usize + j] = self.mem
                        [i * self.mem_alloc_size as usize + j + self.magic_samples[i] as usize];
                }
                self.magic_samples[i] += old_magic;
            }
        }

        Ok(())
    }
}

trait ResamplerSampleIn {
    fn to_resampler_f32(&self) -> f32;
}

impl ResamplerSampleIn for f32 {
    fn to_resampler_f32(&self) -> f32 {
        *self
    }
}

impl ResamplerSampleIn for i16 {
    fn to_resampler_f32(&self) -> f32 {
        *self as f32
    }
}

trait ResamplerSampleOut {
    fn write_resampler_f32(&mut self, value: f32);
}

impl ResamplerSampleOut for f32 {
    fn write_resampler_f32(&mut self, value: f32) {
        *self = value;
    }
}

impl ResamplerSampleOut for i16 {
    fn write_resampler_f32(&mut self, value: f32) {
        let rounded = value.round().clamp(i16::MIN as f32, i16::MAX as f32);
        *self = rounded as i16;
    }
}

fn multiply_frac(value: u32, num: u32, den: u32) -> Result<u32, ResamplerError> {
    let major = value / den;
    let remain = value % den;
    if remain > u32::MAX / num
        || major > u32::MAX / num
        || major.saturating_mul(num) > u32::MAX - remain.saturating_mul(num) / den
    {
        return Err(ResamplerError::Overflow);
    }
    Ok(remain * num / den + major * num)
}

const fn compute_gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = a;
        a = b;
        b = temp % b;
    }
    a
}

fn sinc(cutoff: f32, x: f32, n: usize, window_func: WindowFunction) -> f32 {
    let xx = x * cutoff;
    if x.abs() < 1e-6 {
        return cutoff;
    }
    if x.abs() > 0.5 * n as f32 {
        return 0.0;
    }
    cutoff * libm::sinf(core::f32::consts::PI * xx) / (core::f32::consts::PI * xx)
        * compute_func((2.0 * x / n as f32).abs(), window_func) as f32
}

fn compute_func(x: f32, func: WindowFunction) -> f64 {
    let y = x * func.oversample() as f32;
    let ind = libm::floorf(y) as usize;
    let frac = y - ind as f32;
    let interp3 = -0.1666666667 * frac + 0.1666666667 * frac * frac * frac;
    let interp2 = frac + 0.5 * frac * frac - 0.5 * frac * frac * frac;
    let interp0 = -0.3333333333 * frac + 0.5 * frac * frac - 0.1666666667 * frac * frac * frac;
    let interp1 = 1.0 - interp3 - interp2 - interp0;
    let table = func.table();
    interp0 as f64 * table[ind]
        + interp1 as f64 * table[ind + 1]
        + interp2 as f64 * table[ind + 2]
        + interp3 as f64 * table[ind + 3]
}

fn cubic_coef(frac: f32) -> [f32; 4] {
    let interp0 = -0.16667 * frac + 0.16667 * frac * frac * frac;
    let interp1 = frac + 0.5 * frac * frac - 0.5 * frac * frac * frac;
    let interp3 = -0.33333 * frac + 0.5 * frac * frac - 0.16667 * frac * frac * frac;
    let interp2 = 1.0 - interp0 - interp1 - interp3;
    [interp0, interp1, interp2, interp3]
}

#[cfg(test)]
mod tests {
    use super::{ResamplerError, SpeexResampler};

    const FLOAT_TOL: f32 = 1e-4;
    const COMPARE_TOL: f32 = 1e-5;
    const SENTINEL_FLOAT: f32 = -1234.5;
    const SENTINEL_INT: i16 = 0x5a5a;

    fn assert_close(expected: f32, actual: f32, tol: f32) {
        let diff = (expected - actual).abs();
        assert!(
            diff <= tol,
            "expected {expected:.9} got {actual:.9} (tol {tol:.9})"
        );
    }

    fn assert_all_close(expected: &[f32], actual: &[f32], tol: f32) {
        assert_eq!(expected.len(), actual.len());
        for (index, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "index {index}: expected {lhs:.9} got {rhs:.9} (tol {tol:.9})"
            );
        }
    }

    fn fill_impulse(buffer: &mut [f32], at: usize, amplitude: f32) {
        for (i, sample) in buffer.iter_mut().enumerate() {
            *sample = if i == at { amplitude } else { 0.0 };
        }
    }

    fn fill_sine(buffer: &mut [f32], scale: f32) {
        for (i, sample) in buffer.iter_mut().enumerate() {
            *sample = scale * libm::sinf(i as f32 * 0.17);
        }
    }

    fn fill_stereo_asymmetric(buffer: &mut [f32], frames: usize) {
        for i in 0..frames {
            buffer[2 * i] = if i == 0 { 1.0 } else { 0.0 };
            buffer[2 * i + 1] = if i < 24 { 0.5 } else { -0.5 };
        }
    }

    fn first_nonzero_index(buffer: &[f32], threshold: f32) -> Option<usize> {
        buffer.iter().position(|sample| sample.abs() > threshold)
    }

    fn peak_abs(buffer: &[f32]) -> f32 {
        buffer.iter().map(|sample| sample.abs()).fold(0.0, f32::max)
    }

    #[test]
    fn init_and_configuration_contracts_match_ctest() {
        assert_eq!(
            ResamplerError::InvalidArg,
            SpeexResampler::new(0, 44_100, 48_000, 5).unwrap_err()
        );
        assert_eq!(
            ResamplerError::InvalidArg,
            SpeexResampler::new_frac(1, 0, 1, 0, 1, 5).unwrap_err()
        );
        assert_eq!(
            ResamplerError::InvalidArg,
            SpeexResampler::new_frac(1, 1, 0, 1, 0, 5).unwrap_err()
        );
        assert_eq!(
            ResamplerError::InvalidArg,
            SpeexResampler::new(1, 44_100, 48_000, -1).unwrap_err()
        );
        assert_eq!(
            ResamplerError::InvalidArg,
            SpeexResampler::new(1, 44_100, 48_000, 11).unwrap_err()
        );

        let mut st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
        assert_eq!((44_100, 48_000), st.rates());
        assert_eq!((147, 160), st.ratio());

        st.set_quality(10).expect("set quality");
        assert_eq!(10, st.quality());

        assert_eq!(Err(ResamplerError::InvalidArg), st.set_quality(-1));
        assert_eq!(Err(ResamplerError::InvalidArg), st.set_quality(11));
        assert_eq!(10, st.quality());

        assert_eq!(
            Err(ResamplerError::InvalidArg),
            st.set_rate_frac(0, 1, 0, 1)
        );
        assert_eq!(
            Err(ResamplerError::InvalidArg),
            st.set_rate_frac(1, 0, 1, 0)
        );

        st.set_rate(48_000, 32_000).expect("set rate");
        assert_eq!((48_000, 32_000), st.rates());
        assert_eq!((3, 2), st.ratio());
        st.set_rate_frac(48_000, 32_000, 48_000, 32_000)
            .expect("set rate frac");

        st.set_input_stride(7);
        assert_eq!(7, st.input_stride());
        st.set_output_stride(9);
        assert_eq!(9, st.output_stride());
    }

    #[test]
    fn strerror_and_latency_match_ctest() {
        assert_eq!("Success.", ResamplerError::Success.strerror());
        assert_eq!("Invalid argument.", ResamplerError::InvalidArg.strerror());
        assert_eq!(
            "Input and output buffers overlap.",
            ResamplerError::PtrOverlap.strerror()
        );
        assert_eq!(
            "Unknown error. Bad error code or strange version mismatch.",
            ResamplerError::Overflow.strerror()
        );

        let st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
        let input_latency = st.input_latency();
        let output_latency = st.output_latency();
        assert!(input_latency > 0);
        assert!(output_latency > 0);

        let (ratio_num, ratio_den) = st.ratio();
        let expected_output_latency = (input_latency * ratio_den + (ratio_num >> 1)) / ratio_num;
        assert_eq!(expected_output_latency, output_latency);
    }

    #[test]
    fn mono_baseline_numeric_matches_ctest() {
        let indices = [0usize, 1, 2, 17, 63, 127, 235];
        let expected = [
            0.002053946,
            0.120288044,
            0.249981761,
            0.373679519,
            -0.322577417,
            0.667169452,
            -0.671027243,
        ];

        let mut st = SpeexResampler::new(1, 44_100, 48_000, 5).expect("resampler");
        let mut input = [0.0f32; 256];
        let mut output = [0.0f32; 512];
        fill_sine(&mut input, 0.8);
        st.skip_zeros().expect("skip zeros");

        let mut in_len = 256;
        let mut out_len = 512;
        st.process_float(0, Some(&input), &mut in_len, &mut output, &mut out_len)
            .expect("process");
        assert_eq!(256, in_len);
        assert_eq!(236, out_len);

        for (index, expected_value) in indices.into_iter().zip(expected) {
            assert_close(expected_value, output[index], FLOAT_TOL);
        }
    }

    #[test]
    fn partial_output_and_resume_match_ctest() {
        let mut full_state = SpeexResampler::new(1, 44_100, 48_000, 5).expect("full");
        let mut split_state = SpeexResampler::new(1, 44_100, 48_000, 5).expect("split");
        let mut input = [0.0f32; 256];
        let mut full_out = [0.0f32; 512];
        let mut part1_out = [SENTINEL_FLOAT; 16];
        let mut part2_out = [SENTINEL_FLOAT; 512];
        let mut combined = [0.0f32; 236];
        fill_sine(&mut input, 0.8);
        full_state.skip_zeros().expect("skip zeros");
        split_state.skip_zeros().expect("skip zeros");

        let mut full_in_len = 256;
        let mut full_out_len = 512;
        full_state
            .process_float(
                0,
                Some(&input),
                &mut full_in_len,
                &mut full_out,
                &mut full_out_len,
            )
            .expect("full process");
        assert_eq!(256, full_in_len);
        assert_eq!(236, full_out_len);

        let mut first_in_len = 256;
        let mut first_out_len = 10;
        split_state
            .process_float(
                0,
                Some(&input),
                &mut first_in_len,
                &mut part1_out,
                &mut first_out_len,
            )
            .expect("part1");
        assert_eq!(49, first_in_len);
        assert_eq!(10, first_out_len);
        for sample in &part1_out[10..] {
            assert_close(SENTINEL_FLOAT, *sample, 0.0);
        }

        let consumed_first = first_in_len as usize;
        let mut second_in_len = 256 - consumed_first as u32;
        let mut second_out_len = 512;
        split_state
            .process_float(
                0,
                Some(&input[consumed_first..]),
                &mut second_in_len,
                &mut part2_out,
                &mut second_out_len,
            )
            .expect("part2");
        assert_eq!(207, second_in_len);
        assert_eq!(226, second_out_len);

        combined[..10].copy_from_slice(&part1_out[..10]);
        combined[10..].copy_from_slice(&part2_out[..226]);
        assert_all_close(&full_out[..236], &combined, COMPARE_TOL);
    }

    #[test]
    fn skip_zeros_and_reset_mem_match_ctest() {
        let mut no_skip = SpeexResampler::new(1, 44_100, 48_000, 5).expect("no_skip");
        let mut skip = SpeexResampler::new(1, 44_100, 48_000, 5).expect("skip");
        let mut reset = SpeexResampler::new(1, 44_100, 48_000, 5).expect("reset");
        let mut fresh = SpeexResampler::new(1, 44_100, 48_000, 5).expect("fresh");
        let mut input = [0.0f32; 256];
        fill_sine(&mut input, 0.8);

        let mut out_no_skip = [0.0f32; 512];
        let mut in_len = 256;
        let mut out_len_no_skip = 512;
        no_skip
            .process_float(
                0,
                Some(&input),
                &mut in_len,
                &mut out_no_skip,
                &mut out_len_no_skip,
            )
            .expect("no skip process");
        assert_eq!(256, in_len);
        assert_eq!(279, out_len_no_skip);
        assert!(first_nonzero_index(&out_no_skip[..out_len_no_skip as usize], 1e-5).unwrap() > 0);
        assert!(peak_abs(&out_no_skip[..out_len_no_skip as usize]) > 0.1);

        skip.skip_zeros().expect("skip zeros");
        let mut out_skip = [0.0f32; 512];
        let mut in_len = 256;
        let mut out_len_skip = 512;
        skip.process_float(
            0,
            Some(&input),
            &mut in_len,
            &mut out_skip,
            &mut out_len_skip,
        )
        .expect("skip process");
        assert_eq!(256, in_len);
        assert_eq!(236, out_len_skip);
        assert_eq!(
            Some(0),
            first_nonzero_index(&out_skip[..out_len_skip as usize], 1e-5)
        );

        let mut out_after_reset = [0.0f32; 512];
        let mut in_len = 256;
        let mut out_len_reset = 512;
        reset
            .process_float(
                0,
                Some(&input),
                &mut in_len,
                &mut out_after_reset,
                &mut out_len_reset,
            )
            .expect("reset warmup");
        reset.reset_mem().expect("reset mem");

        out_after_reset.fill(0.0);
        in_len = 256;
        out_len_reset = 512;
        reset
            .process_float(
                0,
                Some(&input),
                &mut in_len,
                &mut out_after_reset,
                &mut out_len_reset,
            )
            .expect("after reset");

        let mut out_fresh = [0.0f32; 512];
        let mut in_len_fresh = 256;
        let mut out_len_fresh = 512;
        fresh
            .process_float(
                0,
                Some(&input),
                &mut in_len_fresh,
                &mut out_fresh,
                &mut out_len_fresh,
            )
            .expect("fresh process");

        assert_eq!(out_len_fresh, out_len_reset);
        assert_all_close(
            &out_fresh[..out_len_fresh as usize],
            &out_after_reset[..out_len_reset as usize],
            COMPARE_TOL,
        );
    }

    #[test]
    fn stateful_rate_change_matches_ctest() {
        let mut st = SpeexResampler::new(1, 48_000, 48_000, 5).expect("resampler");
        st.skip_zeros().expect("skip zeros");

        let mut impulse = [0.0f32; 128];
        let mut out = [0.0f32; 512];
        fill_impulse(&mut impulse, 0, 1.0);
        let mut in_len = 128;
        let mut out_len = 512;
        st.process_float(0, Some(&impulse), &mut in_len, &mut out, &mut out_len)
            .expect("48k process");
        assert!(out_len > 0);

        st.set_rate_frac(48_000, 32_000, 48_000, 32_000)
            .expect("set rate frac");
        assert_eq!((3, 2), st.ratio());
        assert_eq!(60, st.input_latency());
        assert_eq!(40, st.output_latency());

        fill_impulse(&mut impulse, 0, 1.0);
        out.fill(0.0);
        in_len = 128;
        out_len = 512;
        st.process_float(0, Some(&impulse), &mut in_len, &mut out, &mut out_len)
            .expect("32k process");
        assert_eq!(128, in_len);
        assert_eq!(72, out_len);
        assert_close(0.000699097, out[0], FLOAT_TOL);
        assert_close(-0.000801697, out[1], FLOAT_TOL);
        assert_close(0.000838438, out[2], FLOAT_TOL);
    }

    #[test]
    fn stride_processing_matches_ctest() {
        let mut reference = SpeexResampler::new(1, 16_000, 48_000, 5).expect("reference");
        let mut stride = SpeexResampler::new(1, 16_000, 48_000, 5).expect("stride");

        let mut compact_in = [0.0f32; 96];
        let mut sparse_in = [SENTINEL_FLOAT; 192];
        let mut reference_out = [0.0f32; 512];
        let mut sparse_out = [SENTINEL_FLOAT; 1536];
        fill_sine(&mut compact_in, 0.6);
        for (i, sample) in compact_in.iter().enumerate() {
            sparse_in[2 * i] = *sample;
        }

        reference.skip_zeros().expect("reference skip");
        stride.skip_zeros().expect("stride skip");
        stride.set_input_stride(2);
        stride.set_output_stride(3);
        assert_eq!(2, stride.input_stride());
        assert_eq!(3, stride.output_stride());

        let mut in_len_ref = 96;
        let mut out_len_ref = 512;
        reference
            .process_float(
                0,
                Some(&compact_in),
                &mut in_len_ref,
                &mut reference_out,
                &mut out_len_ref,
            )
            .expect("reference process");

        let mut in_len_stride = 96;
        let mut out_len_stride = 512;
        stride
            .process_float(
                0,
                Some(&sparse_in),
                &mut in_len_stride,
                &mut sparse_out,
                &mut out_len_stride,
            )
            .expect("stride process");
        assert_eq!(in_len_ref, in_len_stride);
        assert_eq!(out_len_ref, out_len_stride);

        for i in 0..out_len_stride as usize {
            assert_close(reference_out[i], sparse_out[3 * i], COMPARE_TOL);
            assert_close(SENTINEL_FLOAT, sparse_out[3 * i + 1], 0.0);
            assert_close(SENTINEL_FLOAT, sparse_out[3 * i + 2], 0.0);
        }
        for sample in &sparse_out[out_len_stride as usize * 3..] {
            assert_close(SENTINEL_FLOAT, *sample, 0.0);
        }
    }

    #[test]
    fn interleaved_float_channel_isolation_matches_ctest() {
        let mut stereo = SpeexResampler::new(2, 44_100, 48_000, 5).expect("stereo");
        let mut left = SpeexResampler::new(1, 44_100, 48_000, 5).expect("left");
        let mut right = SpeexResampler::new(1, 44_100, 48_000, 5).expect("right");
        let mut interleaved_in = [0.0f32; 96];
        let mut left_in = [0.0f32; 48];
        let mut right_in = [0.0f32; 48];
        let mut stereo_out = [0.0f32; 512];
        let mut left_out = [0.0f32; 256];
        let mut right_out = [0.0f32; 256];
        fill_stereo_asymmetric(&mut interleaved_in, 48);
        for i in 0..48 {
            left_in[i] = interleaved_in[2 * i];
            right_in[i] = interleaved_in[2 * i + 1];
        }

        stereo.skip_zeros().expect("stereo skip");
        left.skip_zeros().expect("left skip");
        right.skip_zeros().expect("right skip");

        let mut stereo_in_len = 48;
        let mut stereo_out_len = 256;
        stereo
            .process_interleaved_float(
                Some(&interleaved_in),
                &mut stereo_in_len,
                &mut stereo_out,
                &mut stereo_out_len,
            )
            .expect("stereo process");

        let mut mono_in_len = 48;
        let mut left_out_len = 256;
        left.process_float(
            0,
            Some(&left_in),
            &mut mono_in_len,
            &mut left_out,
            &mut left_out_len,
        )
        .expect("left process");
        assert_eq!(48, mono_in_len);

        mono_in_len = 48;
        let mut right_out_len = 256;
        right
            .process_float(
                0,
                Some(&right_in),
                &mut mono_in_len,
                &mut right_out,
                &mut right_out_len,
            )
            .expect("right process");
        assert_eq!(48, mono_in_len);

        assert_eq!(left_out_len, stereo_out_len);
        assert_eq!(right_out_len, stereo_out_len);
        for i in 0..stereo_out_len as usize {
            assert_close(left_out[i], stereo_out[2 * i], 1e-6);
            assert_close(right_out[i], stereo_out[2 * i + 1], 1e-6);
        }
    }

    #[test]
    fn interleaved_int_smoke_matches_ctest() {
        let mut st = SpeexResampler::new(2, 16_000, 48_000, 5).expect("resampler");
        let mut input = [0i16; 256];
        let mut output = [SENTINEL_INT; 1032];
        for i in 0..128 {
            input[2 * i] = if i < 64 { 12_000 } else { -12_000 };
            input[2 * i + 1] = 0;
        }
        st.skip_zeros().expect("skip zeros");

        let mut in_len = 128;
        let mut out_len = 512;
        st.process_interleaved_int(Some(&input), &mut in_len, &mut output, &mut out_len)
            .expect("process");
        assert_eq!(128, in_len);
        assert!(out_len > 0);
        for i in 0..out_len as usize {
            assert_eq!(0, output[2 * i + 1]);
        }
        assert!(output[0] != 0 || output[2] != 0);
        for sample in &output[2 * out_len as usize..] {
            assert_eq!(SENTINEL_INT, *sample);
        }
    }
}
