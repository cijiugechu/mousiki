#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use super::mini_kfft::KissFftCpx;
use super::types::{CeltCoef, MdctLookup};
use libm::fmaf;

fn fold_input(input: &[f32], window: &[CeltCoef], overlap: usize, n2: usize) -> Vec<f32> {
    let n4 = n2 >> 1;
    let quarter_overlap = (overlap + 3) >> 2;
    let half_overlap = overlap >> 1;

    let mut folded = vec![0.0f32; n2];
    let mut yp = 0usize;

    let mut xp1 = half_overlap as isize;
    let mut xp2 = (half_overlap + n2 - 1) as isize;
    let mut wp1 = half_overlap as isize;
    let mut wp2 = half_overlap as isize - 1;

    let n2_isize = n2 as isize;

    for _ in 0..quarter_overlap {
        let a = input[(xp1 + n2_isize) as usize];
        let b = input[xp2 as usize];
        let c = input[xp1 as usize];
        let d = input[(xp2 - n2_isize) as usize];
        let w1 = window[wp1 as usize];
        let w2 = window[wp2 as usize];
        let mul_bw1 = b * w1;
        let mul_dw2 = d * w2;
        let re = fmaf(a, w2, mul_bw1);
        let im = fmaf(c, w1, -mul_dw2);
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
    }

    for _ in quarter_overlap..(n4 - quarter_overlap) {
        let re = input[xp2 as usize];
        let im = input[xp1 as usize];
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
    }

    wp1 = 0;
    wp2 = overlap as isize - 1;

    for _ in (n4 - quarter_overlap)..n4 {
        let a = input[(xp1 - n2_isize) as usize];
        let b = input[xp2 as usize];
        let c = input[xp1 as usize];
        let d = input[(xp2 + n2_isize) as usize];
        let w1 = window[wp1 as usize];
        let w2 = window[wp2 as usize];
        // Match C tail loop: re uses explicit mul/add, im may fuse on some targets.
        let mul_bw2 = b * w2;
        let re = fmaf(-a, w1, mul_bw2);
        let im = fmaf(c, w2, d * w1);
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
    }

    folded
}

fn mdct_fft_scale(n4: usize) -> f32 {
    // Match the exact float value produced by the C build for nfft=480.
    // This is derived from tracing `st->scale` in the reference implementation.
    if n4 == 480 {
        f32::from_bits(0x3b088887)
    } else {
        1.0 / n4 as f32
    }
}

#[cfg(test)]
type MdctTraceCtx<'a> = &'a mdct_trace::TraceContext;
#[cfg(not(test))]
type MdctTraceCtx<'a> = ();

fn pre_rotate_forward(
    folded: &[f32],
    twiddles: &[f32],
    n4: usize,
    #[cfg(test)] ctx: Option<MdctTraceCtx<'_>>,
    #[cfg(not(test))] _ctx: Option<MdctTraceCtx<'_>>,
) -> Vec<KissFftCpx> {
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![KissFftCpx::default(); n4];
    for i in 0..n4 {
        let re = folded[2 * i];
        let im = folded[2 * i + 1];
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let mul_re_t0 = re * t0;
        let mul_im_t1 = im * t1;
        let mul_im_t0 = im * t0;
        let mul_re_t1 = re * t1;
        let yr = fmaf(re, t0, -im * t1);
        let yi = fmaf(im, t0, re * t1);
        let yr_nf = mul_re_t0 - mul_im_t1;
        let yi_nf = mul_im_t0 + mul_re_t1;
        out[i] = KissFftCpx::new(yr, yi);
        #[cfg(test)]
        if let Some(ctx) = ctx {
            mdct_trace::dump_stage_pre_rotate_terms(
                ctx,
                i,
                t0,
                t1,
                re,
                im,
                mul_re_t0,
                mul_im_t1,
                mul_im_t0,
                mul_re_t1,
                yr,
                yi,
                yr_nf,
                yi_nf,
            );
        }
    }
    out
}

#[cfg(not(test))]
fn post_rotate_forward(freq: &[KissFftCpx], twiddles: &[f32], out: &mut [f32], stride: usize) {
    let n4 = freq.len();
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let n2 = n4 * 2;
    let mut left = 0usize;
    let mut right = (n2 - 1) * stride;
    for i in 0..n4 {
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        // Match C float path: allow contraction in the (a*b + c) sums.
        let yr = fmaf(freq[i].i, t1, -(freq[i].r * t0));
        let yi = fmaf(freq[i].r, t1, freq[i].i * t0);
        out[left] = yr;
        out[right] = yi;
        left += 2 * stride;
        if right >= 2 * stride {
            right -= 2 * stride;
        } else {
            right = 0;
        }
    }
}

#[cfg(test)]
fn post_rotate_forward(
    freq: &[KissFftCpx],
    twiddles: &[f32],
    out: &mut [f32],
    stride: usize,
    ctx: Option<&mdct_trace::TraceContext>,
) {
    let n4 = freq.len();
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let n2 = n4 * 2;
    let mut left = 0usize;
    let mut right = (n2 - 1) * stride;
    for i in 0..n4 {
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        // Match C float path: allow contraction in the (a*b + c) sums.
        let yr = fmaf(freq[i].i, t1, -(freq[i].r * t0));
        let yi = fmaf(freq[i].r, t1, freq[i].i * t0);
        #[cfg(test)]
        if let Some(ctx) = ctx {
            mdct_trace::dump_stage_post_rotate(ctx, i, t0, t1, freq[i].r, freq[i].i, yr, yi);
        }
        out[left] = yr;
        out[right] = yi;
        left += 2 * stride;
        if right >= 2 * stride {
            right -= 2 * stride;
        } else {
            right = 0;
        }
    }
}

fn pre_rotate_backward(input: &[f32], twiddles: &[f32], stride: usize) -> Vec<KissFftCpx> {
    let n2 = input.len() / stride;
    let n4 = n2 / 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![KissFftCpx::default(); n4];
    let stride_isize = stride as isize;
    let mut xp1 = 0isize;
    let mut xp2 = (n2 as isize - 1) * stride_isize;
    for i in 0..n4 {
        let x1 = input[xp1 as usize];
        let x2 = input[xp2 as usize];
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let re = x2 * t0 + x1 * t1;
        let im = x1 * t0 - x2 * t1;
        out[i] = KissFftCpx::new(re, im);
        xp1 += 2 * stride_isize;
        xp2 -= 2 * stride_isize;
    }
    out
}

fn post_rotate_backward(
    freq: &[KissFftCpx],
    twiddles: &[f32],
    out: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
) {
    let n4 = freq.len();
    let n2 = n4 * 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let half_overlap = overlap >> 1;
    let mut temp = vec![0.0f32; n2];

    let pairs = (n4 + 1) >> 1;
    for i in 0..pairs {
        let f_front = freq[i];
        let t0_front = cos_part[i];
        let t1_front = sin_part[i];
        let yr_front = f_front.r * t0_front + f_front.i * t1_front;
        let yi_front = f_front.r * t1_front - f_front.i * t0_front;

        let back_index = n4 - i - 1;
        let (yr_back, yi_back) = if back_index == i {
            (yr_front, yi_front)
        } else {
            let f_back = freq[back_index];
            let t0_back = cos_part[back_index];
            let t1_back = sin_part[back_index];
            (
                f_back.r * t0_back + f_back.i * t1_back,
                f_back.r * t1_back - f_back.i * t0_back,
            )
        };

        let front_even = 2 * i;
        let front_odd = front_even + 1;
        let back_even = n2.saturating_sub(2 * (i + 1));
        let back_odd = back_even + 1;

        temp[front_even] = yr_front;
        temp[front_odd] = yi_back;
        temp[back_even] = yr_back;
        temp[back_odd] = yi_front;
    }

    for (dst, src) in out[half_overlap..half_overlap + n2]
        .iter_mut()
        .zip(temp.iter())
    {
        *dst = *src;
    }

    if overlap == 0 {
        return;
    }

    for (offset, (&w1, &w2)) in window
        .iter()
        .zip(window.iter().rev())
        .take(overlap >> 1)
        .enumerate()
    {
        let yp1 = offset;
        let xp1 = overlap - 1 - offset;
        let x1 = out[xp1];
        let x2 = out[yp1];
        out[yp1] = x2 * w2 - x1 * w1;
        out[xp1] = x2 * w1 + x1 * w2;
    }
}

pub fn clt_mdct_forward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= overlap + n2);
    assert!(window.len() >= overlap);
    let required = (n2 - 1) * stride + 1;
    assert!(output.len() >= required);
    assert!(stride > 0);

    #[cfg(test)]
    let trace_ctx = mdct_trace::begin_call();

    let twiddles = lookup.twiddles(shift);
    #[cfg(test)]
    if let Some(ctx) = trace_ctx.as_ref() {
        if ctx.trace_in {
            mdct_trace::dump_input(ctx, input);
        }
        if let Some(detail_idx) = mdct_trace::window_detail_index() {
            mdct_trace::dump_window_detail(ctx, input, window, overlap, n2, detail_idx);
        } else if let Some(detail_idx) = mdct_trace::window_detail_tail_index() {
            mdct_trace::dump_window_detail_tail(ctx, input, window, overlap, n2, n4, detail_idx);
        }
    }
    let folded = fold_input(input, window, overlap, n2);
    #[cfg(test)]
    if let Some(ctx) = trace_ctx.as_ref() {
        if ctx.trace_window {
            mdct_trace::dump_window(ctx, &folded);
        }
    }
    #[cfg(test)]
    let mut spectrum = pre_rotate_forward(&folded, twiddles, n4, trace_ctx.as_ref());
    #[cfg(not(test))]
    let mut spectrum = pre_rotate_forward(&folded, twiddles, n4, None);
    // Match the C path: kiss_fft applies the 1/nfft scale before the butterfly stages.
    let scale = mdct_fft_scale(n4);
    #[cfg(test)]
    if let Some(ctx) = trace_ctx.as_ref() {
        mdct_trace::dump_stage_scale(ctx, scale);
        let mut scaled = spectrum.clone();
        for entry in scaled.iter_mut() {
            entry.r *= scale;
            entry.i *= scale;
        }
        mdct_trace::dump_stage_pre_rotate(ctx, &scaled);
    }
    let mut fft_out = vec![KissFftCpx::default(); n4];
    lookup.forward_plan(shift).fft(&spectrum, &mut fft_out);
    #[cfg(test)]
    if let Some(ctx) = trace_ctx.as_ref() {
        mdct_trace::dump_stage_fft(ctx, &fft_out);
    }
    #[cfg(not(test))]
    post_rotate_forward(&fft_out, twiddles, output, stride);
    #[cfg(test)]
    post_rotate_forward(&fft_out, twiddles, output, stride, trace_ctx.as_ref());
}

pub fn clt_mdct_backward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= stride * n2);
    assert!(window.len() >= overlap);
    let half_overlap = overlap >> 1;
    assert!(output.len() >= overlap);
    assert!(output.len() >= half_overlap + n2);
    assert!(stride > 0);

    let twiddles = lookup.twiddles(shift);
    output.fill(0.0);
    let pre = pre_rotate_backward(input, twiddles, stride);
    let mut fft_out = vec![KissFftCpx::default(); n4];
    lookup.inverse_plan(shift).ifft(&pre, &mut fft_out);
    post_rotate_backward(&fft_out, twiddles, output, window, overlap);
}

#[cfg(test)]
pub(crate) mod mdct_trace {
    extern crate std;

    use core::sync::atomic::{AtomicUsize, Ordering};
    use libm::fmaf;
    use std::env;
    use std::sync::OnceLock;

    use super::{CeltCoef, KissFftCpx};

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
        want_bits: bool,
        start: usize,
        count: usize,
        trace_in: bool,
        trace_window: bool,
        trace_stage: bool,
    }

    pub(crate) struct TraceContext {
        frame: usize,
        call: usize,
        channel: usize,
        block: usize,
        tag: usize,
        pub trace_in: bool,
        pub trace_window: bool,
        pub trace_stage: bool,
        want_bits: bool,
        start: usize,
        count: usize,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static CALL_INDEX: AtomicUsize = AtomicUsize::new(0);
    static CHANNEL_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static BLOCK_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static TAG_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn set_frame(frame_idx: usize) {
        FRAME_INDEX.store(frame_idx, Ordering::Relaxed);
        CALL_INDEX.store(0, Ordering::Relaxed);
    }

    pub(crate) fn set_call(channel: usize, block: usize) {
        CHANNEL_INDEX.store(channel, Ordering::Relaxed);
        BLOCK_INDEX.store(block, Ordering::Relaxed);
    }

    pub(crate) fn set_tag(tag: usize) {
        TAG_INDEX.store(tag, Ordering::Relaxed);
    }

    pub(crate) fn begin_call() -> Option<TraceContext> {
        let cfg = config()?;
        let frame = FRAME_INDEX.load(Ordering::Relaxed);
        if frame == usize::MAX {
            return None;
        }
        if let Some(target) = cfg.frame {
            if target != frame {
                return None;
            }
        }
        let call = CALL_INDEX.fetch_add(1, Ordering::Relaxed);
        let channel = CHANNEL_INDEX.load(Ordering::Relaxed);
        let block = BLOCK_INDEX.load(Ordering::Relaxed);
        let tag = TAG_INDEX.load(Ordering::Relaxed);
        Some(TraceContext {
            frame,
            call,
            channel,
            block,
            tag,
            trace_in: cfg.trace_in,
            trace_window: cfg.trace_window,
            trace_stage: cfg.trace_stage,
            want_bits: cfg.want_bits,
            start: cfg.start,
            count: cfg.count,
        })
    }

    pub(crate) fn context_indices(ctx: &TraceContext) -> (usize, usize, usize, usize, usize) {
        (ctx.frame, ctx.call, ctx.channel, ctx.block, ctx.tag)
    }

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let trace_in = env_truthy("CELT_TRACE_MDCT_IN");
                let trace_window = env_truthy("CELT_TRACE_MDCT_WINDOW");
                let trace_all = env_truthy("CELT_TRACE_MDCT");
                let mut want_in = trace_in;
                let mut want_window = trace_window;
                if trace_all && !trace_in && !trace_window {
                    want_in = true;
                    want_window = true;
                }
                if !want_in && !want_window {
                    let trace_stage = env_truthy("CELT_TRACE_MDCT_STAGE");
                    if !trace_stage {
                        return None;
                    }
                }
                let trace_stage = env_truthy("CELT_TRACE_MDCT_STAGE");
                let frame = env::var("CELT_TRACE_MDCT_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let want_bits = env_truthy("CELT_TRACE_MDCT_BITS");
                let start = env::var("CELT_TRACE_MDCT_START")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                let count = env::var("CELT_TRACE_MDCT_COUNT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(64);
                Some(TraceConfig {
                    frame,
                    want_bits,
                    start,
                    count,
                    trace_in: want_in,
                    trace_window: want_window,
                    trace_stage,
                })
            })
            .as_ref()
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    fn tag_name(tag: usize) -> &'static str {
        if tag == 1 { "mdct2" } else { "main" }
    }

    pub(crate) fn window_detail_index() -> Option<usize> {
        let enabled = env_truthy("CELT_TRACE_MDCT_WINDOW_DETAIL");
        if !enabled {
            return None;
        }
        let idx = env::var("CELT_TRACE_MDCT_WINDOW_DETAIL_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        Some(idx)
    }

    pub(crate) fn window_detail_tail_index() -> Option<usize> {
        let enabled = env_truthy("CELT_TRACE_MDCT_WINDOW_DETAIL_TAIL");
        if !enabled {
            return None;
        }
        let idx = env::var("CELT_TRACE_MDCT_WINDOW_DETAIL_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        Some(idx)
    }

    pub(crate) fn dump_input(ctx: &TraceContext, input: &[f32]) {
        let len = input.len();
        let start = ctx.start.min(len);
        let end = start.saturating_add(ctx.count).min(len);
        std::println!(
            "celt_mdct_in[{}].{}.call[{}].ch[{}].block[{}].len={}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            len
        );
        for i in start..end {
            let value = input[i];
            std::println!(
                "celt_mdct_in[{}].{}.call[{}].ch[{}].block[{}].idx[{}]={:.9}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value
            );
            if ctx.want_bits {
                std::println!(
                    "celt_mdct_in[{}].{}.call[{}].ch[{}].block[{}].idx_bits[{}]=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_window(ctx: &TraceContext, folded: &[f32]) {
        let len = folded.len();
        let start = ctx.start.min(len);
        let end = start.saturating_add(ctx.count).min(len);
        std::println!(
            "celt_mdct_win[{}].{}.call[{}].ch[{}].block[{}].len={}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            len
        );
        for i in start..end {
            let value = folded[i];
            std::println!(
                "celt_mdct_win[{}].{}.call[{}].ch[{}].block[{}].idx[{}]={:.9}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value
            );
            if ctx.want_bits {
                std::println!(
                    "celt_mdct_win[{}].{}.call[{}].ch[{}].block[{}].idx_bits[{}]=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_stage_pre_rotate(ctx: &TraceContext, spectrum: &[KissFftCpx]) {
        if !ctx.trace_stage {
            return;
        }
        let len = spectrum.len();
        let start = ctx.start.min(len);
        let end = start.saturating_add(ctx.count).min(len);
        for i in start..end {
            let value = spectrum[i];
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].r={:.9e}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value.r
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].i={:.9e}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value.i
            );
            if ctx.want_bits {
                std::println!(
                    "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].r=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.r.to_bits()
                );
                std::println!(
                    "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].i=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.i.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_stage_pre_rotate_terms(
        ctx: &TraceContext,
        idx: usize,
        t0: f32,
        t1: f32,
        re: f32,
        im: f32,
        mul_re_t0: f32,
        mul_im_t1: f32,
        mul_im_t0: f32,
        mul_re_t1: f32,
        yr: f32,
        yi: f32,
        yr_nf: f32,
        yi_nf: f32,
    ) {
        if !ctx.trace_stage {
            return;
        }
        let start = ctx.start;
        let end = start.saturating_add(ctx.count);
        if idx < start || idx >= end {
            return;
        }
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].t0={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            t0
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].t1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            t1
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].re={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            re
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].im={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            im
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].mul_re_t0={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            mul_re_t0
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].mul_im_t1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            mul_im_t1
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].mul_im_t0={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            mul_im_t0
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].mul_re_t1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            mul_re_t1
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].yr={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yr
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].yi={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yi
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].yr_nf={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yr_nf
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx[{}].yi_nf={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yi_nf
        );
        if ctx.want_bits {
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].t0=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                t0.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].t1=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                t1.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].re=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                re.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].im=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                im.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].mul_re_t0=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                mul_re_t0.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].mul_im_t1=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                mul_im_t1.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].mul_im_t0=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                mul_im_t0.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].mul_re_t1=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                mul_re_t1.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].yr=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yr.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].yi=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yi.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].yr_nf=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yr_nf.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].pre_rotate.idx_bits[{}].yi_nf=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yi_nf.to_bits()
            );
        }
    }

    pub(crate) fn dump_stage_fft(ctx: &TraceContext, spectrum: &[KissFftCpx]) {
        if !ctx.trace_stage {
            return;
        }
        let len = spectrum.len();
        let start = ctx.start.min(len);
        let end = start.saturating_add(ctx.count).min(len);
        for i in start..end {
            let value = spectrum[i];
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].fft.idx[{}].r={:.9e}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value.r
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].fft.idx[{}].i={:.9e}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                i,
                value.i
            );
            if ctx.want_bits {
                std::println!(
                    "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].fft.idx_bits[{}].r=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.r.to_bits()
                );
                std::println!(
                    "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].fft.idx_bits[{}].i=0x{:08x}",
                    ctx.frame,
                    tag_name(ctx.tag),
                    ctx.call,
                    ctx.channel,
                    ctx.block,
                    i,
                    value.i.to_bits()
                );
            }
        }
    }

    pub(crate) fn dump_stage_scale(ctx: &TraceContext, scale: f32) {
        if !ctx.trace_stage {
            return;
        }
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].scale={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            scale
        );
        if ctx.want_bits {
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].scale_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                scale.to_bits()
            );
        }
    }

    pub(crate) fn dump_stage_post_rotate(
        ctx: &TraceContext,
        idx: usize,
        t0: f32,
        t1: f32,
        fp_r: f32,
        fp_i: f32,
        yr: f32,
        yi: f32,
    ) {
        if !ctx.trace_stage {
            return;
        }
        let start = ctx.start;
        let end = start.saturating_add(ctx.count);
        if idx < start || idx >= end {
            return;
        }
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].t0={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            t0
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].t1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            t1
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].fp.r={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            fp_r
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].fp.i={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            fp_i
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].yr={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yr
        );
        std::println!(
            "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx[{}].yi={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            idx,
            yi
        );
        if ctx.want_bits {
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].t0=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                t0.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].t1=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                t1.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].fp.r=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                fp_r.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].fp.i=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                fp_i.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].yr=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yr.to_bits()
            );
            std::println!(
                "celt_mdct_stage[{}].{}.call[{}].ch[{}].block[{}].post_rotate.idx_bits[{}].yi=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                ctx.call,
                ctx.channel,
                ctx.block,
                idx,
                yi.to_bits()
            );
        }
    }

    pub(crate) fn dump_window_detail(
        ctx: &TraceContext,
        input: &[f32],
        window: &[CeltCoef],
        overlap: usize,
        n2: usize,
        index: usize,
    ) {
        let quarter_overlap = (overlap + 3) >> 2;
        if index >= quarter_overlap {
            return;
        }
        let half_overlap = overlap >> 1;
        let n2_isize = n2 as isize;
        let xp1 = half_overlap as isize + (2 * index) as isize;
        let xp2 = (half_overlap + n2 - 1) as isize - (2 * index) as isize;
        let wp1 = half_overlap as isize + (2 * index) as isize;
        let wp2 = half_overlap as isize - 1 - (2 * index) as isize;

        let a = input[(xp1 + n2_isize) as usize];
        let b = input[xp2 as usize];
        let c = input[xp1 as usize];
        let d = input[(xp2 - n2_isize) as usize];
        let w1 = window[wp1 as usize];
        let w2 = window[wp2 as usize];
        let mul_aw2 = a * w2;
        let mul_bw1 = b * w1;
        let mul_cw1 = c * w1;
        let mul_dw2 = d * w2;
        let re = mul_aw2 + mul_bw1;
        let im = mul_cw1 - mul_dw2;
        let re_fma = fmaf(a, w2, b * w1);
        let im_fma = fmaf(c, w1, -(d * w2));

        std::println!(
            "celt_mdct_win_detail[{}].{}.call[{}].ch[{}].block[{}].i={}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            index
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.a={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            a
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.b={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            b
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.c={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            c
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.d={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            d
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.w1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            w1
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.w2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            w2
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.mul_aw2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_aw2
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.mul_bw1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_bw1
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.mul_cw1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_cw1
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.mul_dw2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_dw2
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.re={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            re
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.im={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            im
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.re_fma={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            re_fma
        );
        std::println!(
            "celt_mdct_win_detail[{}].{}.im_fma={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            im_fma
        );
        if ctx.want_bits {
            std::println!(
                "celt_mdct_win_detail[{}].{}.a_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                a.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.b_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                b.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.c_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                c.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.d_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                d.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.w1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                w1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.w2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                w2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.mul_aw2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_aw2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.mul_bw1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_bw1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.mul_cw1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_cw1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.mul_dw2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_dw2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.re_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                re.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.im_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                im.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.re_fma_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                re_fma.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail[{}].{}.im_fma_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                im_fma.to_bits()
            );
        }
    }

    pub(crate) fn dump_window_detail_tail(
        ctx: &TraceContext,
        input: &[f32],
        window: &[CeltCoef],
        overlap: usize,
        n2: usize,
        n4: usize,
        index: usize,
    ) {
        let quarter_overlap = (overlap + 3) >> 2;
        if index >= quarter_overlap {
            return;
        }
        let half_overlap = overlap >> 1;
        let n2_isize = n2 as isize;
        let base_xp1 = half_overlap as isize + (2 * (n4 - quarter_overlap)) as isize;
        let xp1 = base_xp1 + (2 * index) as isize;
        let base_xp2 =
            half_overlap as isize + n2 as isize - 1 - (2 * (n4 - quarter_overlap)) as isize;
        let xp2 = base_xp2 - (2 * index) as isize;
        let wp1 = (2 * index) as isize;
        let wp2 = overlap as isize - 1 - (2 * index) as isize;

        let a = input[(xp1 - n2_isize) as usize];
        let b = input[xp2 as usize];
        let c = input[xp1 as usize];
        let d = input[(xp2 + n2_isize) as usize];
        let w1 = window[wp1 as usize];
        let w2 = window[wp2 as usize];
        let mul_aw1 = a * w1;
        let mul_bw2 = b * w2;
        let mul_cw2 = c * w2;
        let mul_dw1 = d * w1;
        let re = mul_bw2 - mul_aw1;
        let im = mul_cw2 + mul_dw1;
        let re_fma_bw2 = fmaf(b, w2, -mul_aw1);
        let re_fma_aw1 = fmaf(-a, w1, mul_bw2);

        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.call[{}].ch[{}].block[{}].i={}",
            ctx.frame,
            tag_name(ctx.tag),
            ctx.call,
            ctx.channel,
            ctx.block,
            index
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.a={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            a
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.b={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            b
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.c={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            c
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.d={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            d
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.w1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            w1
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.w2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            w2
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.mul_aw1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_aw1
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.mul_bw2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_bw2
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.mul_cw2={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_cw2
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.mul_dw1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            mul_dw1
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.re={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            re
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.im={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            im
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.re_fma={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            re_fma_bw2
        );
        std::println!(
            "celt_mdct_win_detail_tail[{}].{}.re_fma_aw1={:.9e}",
            ctx.frame,
            tag_name(ctx.tag),
            re_fma_aw1
        );
        if ctx.want_bits {
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.a_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                a.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.b_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                b.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.c_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                c.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.d_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                d.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.w1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                w1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.w2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                w2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.mul_aw1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_aw1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.mul_bw2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_bw2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.mul_cw2_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_cw2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.mul_dw1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                mul_dw1.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.re_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                re.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.im_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                im.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.re_fma_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                re_fma_bw2.to_bits()
            );
            std::println!(
                "celt_mdct_win_detail_tail[{}].{}.re_fma_aw1_bits=0x{:08x}",
                ctx.frame,
                tag_name(ctx.tag),
                re_fma_aw1.to_bits()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use alloc::vec::Vec;
    use core::f32::consts::PI;
    use crate::celt::modes::opus_custom_mode_find_static;
    use libm::fmaf;

    fn c_mul_fma(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
        let r = fmaf(a.r, b.r, -(a.i * b.i));
        let i = fmaf(a.r, b.i, a.i * b.r);
        KissFftCpx::new(r, i)
    }

    fn naive_fft(input: &[KissFftCpx], inverse: bool) -> Vec<KissFftCpx> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = KissFftCpx::default();
            for (n_idx, sample) in input.iter().enumerate() {
                let angle = 2.0 * PI * (k * n_idx) as f32 / n as f32;
                let (sin_term, cos_term) = if inverse {
                    angle.sin_cos()
                } else {
                    let (s, c) = (-angle).sin_cos();
                    (s, c)
                };
                let tw = KissFftCpx::new(cos_term, sin_term);
                let prod = c_mul_fma(*sample, tw);
                sum.r += prod.r;
                sum.i += prod.i;
            }
            out.push(sum);
        }
        out
    }

    fn reference_forward(
        lookup: &MdctLookup,
        input: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Vec<f32> {
        let n = lookup.effective_len(shift);
        let n2 = n >> 1;
        let n4 = n >> 2;
        let twiddles = lookup.twiddles(shift);
        let folded = fold_input(input, window, overlap, n2);
        let mut spectrum = pre_rotate_forward(&folded, twiddles, n4, None);
        let scale = 1.0 / n4 as f32;
        for entry in spectrum.iter_mut() {
            entry.r *= scale;
            entry.i *= scale;
        }
        let freq = naive_fft(&spectrum, false);
        let mut out = vec![0.0f32; stride * n2];
        post_rotate_forward(&freq, twiddles, &mut out, stride, None);
        out
    }

    fn reference_backward(
        lookup: &MdctLookup,
        input: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Vec<f32> {
        let n = lookup.effective_len(shift);
        let n2 = n >> 1;
        let twiddles = lookup.twiddles(shift);
        let pre = pre_rotate_backward(input, twiddles, stride);
        let freq = naive_fft(&pre, true);
        let mut out = vec![0.0f32; overlap.max((overlap >> 1) + n2)];
        post_rotate_backward(&freq, twiddles, &mut out, window, overlap);
        out
    }

    fn make_sine_window(overlap: usize) -> Vec<f32> {
        (0..overlap)
            .map(|i| (PI * (i as f32 + 0.5) / overlap as f32).sin())
            .collect()
    }

    #[test]
    fn forward_matches_reference_for_small_sizes() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);
            let mut input = vec![0.0f32; overlap + n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.37).sin();
            }
            let mut output = vec![0.0f32; n / 2];
            clt_mdct_forward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_forward(&mdct, &input, &window, overlap, 0, 1);
            for (lhs, rhs) in output.iter().zip(reference.iter()) {
                assert!((lhs - rhs).abs() < 1e-4, "{} vs {}", lhs, rhs);
            }
        }
    }

    #[test]
    fn forward_matches_reference_with_stride() {
        let n = 64usize;
        let overlap = n / 2;
        let stride = 4usize;
        let mdct = MdctLookup::new(n, 0);
        let window = make_sine_window(overlap);
        let mut input = vec![0.0f32; overlap + n];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample = (i as f32 * 0.17).sin();
        }
        let mut output = vec![0.0f32; stride * (n / 2)];
        clt_mdct_forward(&mdct, &input, &mut output, &window, overlap, 0, stride);
        let reference = reference_forward(&mdct, &input, &window, overlap, 0, stride);
        for (lhs, rhs) in output.iter().zip(reference.iter()) {
            assert!((lhs - rhs).abs() < 1e-4, "{} vs {}", lhs, rhs);
        }
    }

    #[test]
    fn backward_matches_reference_for_small_sizes() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);
            let mut input = vec![0.0f32; n / 2];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.19).cos();
            }
            let mut output = vec![0.0f32; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_backward(&mdct, &input, &window, overlap, 0, 1);
            for (lhs, rhs) in output.iter().zip(reference.iter()) {
                assert!((lhs - rhs).abs() < 1e-4, "{} vs {}", lhs, rhs);
            }
        }
    }

    /// Tests MDCT for multiple transform sizes.
    ///
    /// This is an adapted port of `test_unit_mdct.c` that validates the
    /// forward and backward MDCT transforms match the naive reference
    /// implementation across various sizes used by Opus.
    #[test]
    fn mdct_multiple_sizes() {
        // Test power-of-2 sizes up to a reasonable limit
        for &n in &[32usize, 64, 128, 256, 512] {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);

            // Tolerance scales with size due to accumulated floating point errors
            let tol = 1e-3 + (n as f32) * 1e-5;

            // Test forward transform
            let mut input = vec![0.0f32; overlap + n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = ((i as f32 * 0.37) + (i as f32 * 0.13).cos()).sin();
            }
            let mut output = vec![0.0f32; n / 2];
            clt_mdct_forward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_forward(&mdct, &input, &window, overlap, 0, 1);
            for (j, (lhs, rhs)) in output.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (lhs - rhs).abs() < tol,
                    "forward n={} bin {}: {} vs {}",
                    n,
                    j,
                    lhs,
                    rhs
                );
            }

            // Test backward transform
            let mut freq = vec![0.0f32; n / 2];
            for (i, sample) in freq.iter_mut().enumerate() {
                *sample = (i as f32 * 0.19).cos();
            }
            let mut time = vec![0.0f32; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward(&mdct, &freq, &mut time, &window, overlap, 0, 1);
            let reference_back = reference_backward(&mdct, &freq, &window, overlap, 0, 1);
            for (j, (lhs, rhs)) in time.iter().zip(reference_back.iter()).enumerate() {
                assert!(
                    (lhs - rhs).abs() < tol,
                    "backward n={} sample {}: {} vs {}",
                    n,
                    j,
                    lhs,
                    rhs
                );
            }
        }
    }

    #[test]
    fn mdct_backward_compare_output() {
        let mode = opus_custom_mode_find_static(48_000, 120)
            .expect("static 48k/120 mode should be available");
        let mdct = &mode.mdct;
        let overlap = mode.overlap;
        let n = mdct.len();
        let n2 = n / 2;
        let window = mode.window;

        let mut input = vec![0.0f32; n2];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample = (i as f32 * 0.19).cos();
        }

        let output_len = overlap.max((overlap >> 1) + n2);
        let mut output = vec![0.0f32; output_len];
        clt_mdct_backward(mdct, &input, &mut output, window, overlap, 0, 1);

        for (i, sample) in output.iter().enumerate() {
            std::println!("mdct_backward_out[{}]={:.9e}", i, sample);
        }
    }
}
