//! Debug and profiling utilities mirroring `silk/debug.{c,h}`.
//!
//! The original C implementation exposes two sets of helpers behind compile-time
//! toggles:
//! - `SILK_TIC_TOC` drives lightweight timers that measure nested sections.
//! - `SILK_DEBUG` provides data dumps for offline inspection.
//!
//! This Rust module ports the same interface while keeping the default build
//! free from side effects.  Both facilities become active only when the matching
//! Cargo features (`silk_tic_toc` and `silk_debug`) are enabled.

#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "silk_tic_toc")]
use core::fmt::Write as _;

#[cfg(any(feature = "silk_tic_toc", feature = "silk_debug"))]
use core::cell::UnsafeCell;
use alloc::string::String;
use alloc::vec::Vec;

pub const SILK_DEBUG: bool = cfg!(feature = "silk_debug");
pub const SILK_TIC_TOC: bool = cfg!(feature = "silk_tic_toc");

pub const NUM_TIMERS_MAX: usize = 50;
pub const NUM_TIMERS_MAX_TAG_LEN: usize = 30;
pub const NUM_STORES_MAX: usize = 100;

type TimerNow = fn() -> u64;

const fn default_timer_now() -> u64 {
    0
}

#[cfg(any(feature = "silk_tic_toc", feature = "silk_debug"))]
struct StaticMut<T> {
    inner: UnsafeCell<T>,
}

#[cfg(any(feature = "silk_tic_toc", feature = "silk_debug"))]
unsafe impl<T> Sync for StaticMut<T> {}

#[cfg(any(feature = "silk_tic_toc", feature = "silk_debug"))]
impl<T> StaticMut<T> {
    const fn new(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(value),
        }
    }

    unsafe fn get(&self) -> *mut T {
        self.inner.get()
    }
}

/// Aggregated timer statistics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimerReport {
    pub tag: String,
    pub count: u32,
    pub min: u64,
    pub max: u64,
    pub sum: u128,
    pub depth: i32,
}

impl TimerReport {
    #[must_use]
    pub fn avg(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        (self.sum / u128::from(self.count)) as u64
    }
}

#[cfg(feature = "silk_tic_toc")]
struct TimerState {
    n_timers: usize,
    depth_ctr: i32,
    tag_lengths: [usize; NUM_TIMERS_MAX],
    tags: [[u8; NUM_TIMERS_MAX_TAG_LEN]; NUM_TIMERS_MAX],
    start: [u64; NUM_TIMERS_MAX],
    count: [u32; NUM_TIMERS_MAX],
    sum: [u128; NUM_TIMERS_MAX],
    min: [u64; NUM_TIMERS_MAX],
    max: [u64; NUM_TIMERS_MAX],
    depth: [i32; NUM_TIMERS_MAX],
}

#[cfg(feature = "silk_tic_toc")]
impl TimerState {
    const fn new() -> Self {
        Self {
            n_timers: 0,
            depth_ctr: 0,
            tag_lengths: [0; NUM_TIMERS_MAX],
            tags: [[0; NUM_TIMERS_MAX_TAG_LEN]; NUM_TIMERS_MAX],
            start: [0; NUM_TIMERS_MAX],
            count: [0; NUM_TIMERS_MAX],
            sum: [0; NUM_TIMERS_MAX],
            min: [u64::MAX; NUM_TIMERS_MAX],
            max: [0; NUM_TIMERS_MAX],
            depth: [0; NUM_TIMERS_MAX],
        }
    }

    fn reset(&mut self) {
        self.n_timers = 0;
        self.depth_ctr = 0;
        self.tag_lengths = [0; NUM_TIMERS_MAX];
        self.tags = [[0; NUM_TIMERS_MAX_TAG_LEN]; NUM_TIMERS_MAX];
        self.start = [0; NUM_TIMERS_MAX];
        self.count = [0; NUM_TIMERS_MAX];
        self.sum = [0; NUM_TIMERS_MAX];
        self.min = [u64::MAX; NUM_TIMERS_MAX];
        self.max = [0; NUM_TIMERS_MAX];
        self.depth = [0; NUM_TIMERS_MAX];
    }

    fn find(&self, tag: &str) -> Option<usize> {
        let tag_bytes = tag.as_bytes();
        (0..self.n_timers).find(|&idx| {
            self.tag_lengths[idx] == tag_bytes.len()
                && self.tags[idx][..self.tag_lengths[idx]] == *tag_bytes
        })
    }

    fn get_or_insert(&mut self, tag: &str) -> usize {
        if let Some(idx) = self.find(tag) {
            return idx;
        }

        let id = self.n_timers;
        assert!(
            id < NUM_TIMERS_MAX,
            "Exceeded maximum number of timers ({NUM_TIMERS_MAX})"
        );
        let tag_bytes = tag.as_bytes();
        assert!(
            tag_bytes.len() <= NUM_TIMERS_MAX_TAG_LEN,
            "Timer tag \"{tag}\" exceeds maximum length ({NUM_TIMERS_MAX_TAG_LEN})"
        );

        self.tags[id][..tag_bytes.len()].copy_from_slice(tag_bytes);
        self.tag_lengths[id] = tag_bytes.len();
        self.count[id] = 0;
        self.sum[id] = 0;
        self.min[id] = u64::MAX;
        self.max[id] = 0;
        self.depth[id] = self.depth_ctr;
        self.n_timers += 1;
        id
    }

    fn tag_to_string(&self, idx: usize) -> String {
        let len = self.tag_lengths[idx];
        String::from_utf8(self.tags[idx][..len].to_vec()).unwrap_or_default()
    }
}

#[cfg(feature = "silk_tic_toc")]
static TIMER_STATE: StaticMut<TimerState> = StaticMut::new(TimerState::new());

#[cfg(feature = "silk_tic_toc")]
static TIMER_SOURCE: StaticMut<TimerNow> = StaticMut::new(default_timer_now);

#[cfg(feature = "silk_tic_toc")]
pub fn set_timer_source(source: TimerNow) -> TimerNow {
    unsafe {
        let ptr = TIMER_SOURCE.get();
        let previous = *ptr;
        *ptr = source;
        previous
    }
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn set_timer_source(_source: TimerNow) -> TimerNow {
    default_timer_now
}

#[cfg(feature = "silk_tic_toc")]
fn now() -> u64 {
    unsafe {
        let ptr = TIMER_SOURCE.get();
        (*ptr)()
    }
}

#[cfg(feature = "silk_tic_toc")]
pub fn reset_timers() {
    unsafe {
        (*TIMER_STATE.get()).reset();
        *TIMER_SOURCE.get() = default_timer_now;
    }
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn reset_timers() {}

#[cfg(feature = "silk_tic_toc")]
pub fn tic(tag: &str) {
    unsafe {
        let state = &mut *TIMER_STATE.get();
        let id = state.get_or_insert(tag);
        state.depth[id] = state.depth_ctr;
        state.start[id] = now();
        state.depth_ctr += 1;
    }
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn tic(_tag: &str) {}

#[cfg(feature = "silk_tic_toc")]
pub fn toc(tag: &str) {
    unsafe {
        let state = &mut *TIMER_STATE.get();
        let Some(id) = state.find(tag) else {
            state.depth_ctr = state.depth_ctr.saturating_sub(1);
            return;
        };

        let elapsed = now().saturating_sub(state.start[id]);
        if elapsed < 100_000_000 {
            state.count[id] = state.count[id].saturating_add(1);
            state.sum[id] = state.sum[id].saturating_add(u128::from(elapsed));
            state.max[id] = state.max[id].max(elapsed);
            state.min[id] = state.min[id].min(elapsed);
        }
        state.depth_ctr = state.depth_ctr.saturating_sub(1);
    }
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn toc(_tag: &str) {}

#[cfg(feature = "silk_tic_toc")]
pub fn timer_snapshot() -> Option<Vec<TimerReport>> {
    unsafe {
        let state = &*TIMER_STATE.get();
        if state.n_timers == 0 {
            return None;
        }

        let mut reports = Vec::with_capacity(state.n_timers);
        for idx in 0..state.n_timers {
            let count = state.count[idx];
            if count == 0 {
                continue;
            }

            reports.push(TimerReport {
                tag: state.tag_to_string(idx),
                count,
                min: if state.min[idx] == u64::MAX {
                    0
                } else {
                    state.min[idx]
                },
                max: state.max[idx],
                sum: state.sum[idx],
                depth: state.depth[idx],
            });
        }

        if reports.is_empty() {
            None
        } else {
            Some(reports)
        }
    }
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn timer_snapshot() -> Option<Vec<TimerReport>> {
    None
}

#[cfg(feature = "silk_tic_toc")]
pub fn timer_table() -> Option<String> {
    let reports = timer_snapshot()?;
    let mut total_avg_sum = 0u128;
    let mut total_avg_weight = 0u64;
    for report in &reports {
        total_avg_sum = total_avg_sum.saturating_add(report.sum);
        total_avg_weight = total_avg_weight.saturating_add(u64::from(report.count));
    }

    let mut output = String::new();
    writeln!(
        output,
        "                                min         avg         max      count"
    )
    .ok()?;

    for report in &reports {
        let indent = match report.depth {
            0 => "",
            1 => " ",
            2 => "  ",
            3 => "   ",
            _ => "    ",
        };

        let avg = report.avg();
        writeln!(
            output,
            "{indent}{:<27}{:8}{:12}{:12}{:10}",
            report.tag,
            report.min,
            avg,
            report.max,
            report.count
        )
        .ok()?;
    }

    writeln!(output, "                                microseconds").ok()?;
    Some(output)
}

#[cfg(not(feature = "silk_tic_toc"))]
pub fn timer_table() -> Option<String> {
    None
}

/// Stored debug payload for a virtual file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredData {
    pub name: String,
    pub data: Vec<u8>,
}

#[cfg(feature = "silk_debug")]
struct DebugStore {
    entries: Vec<StoredData>,
}

#[cfg(feature = "silk_debug")]
impl DebugStore {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

#[cfg(feature = "silk_debug")]
static DEBUG_STORE: StaticMut<Option<DebugStore>> = StaticMut::new(None);

#[cfg(feature = "silk_debug")]
fn debug_store_mut() -> &'static mut DebugStore {
    unsafe {
        let ptr = DEBUG_STORE.get();
        if (*ptr).is_none() {
            *ptr = Some(DebugStore::new());
        }
        (*ptr).as_mut().unwrap()
    }
}

#[cfg(feature = "silk_debug")]
pub fn debug_store_data(file_name: &str, payload: &[u8]) {
    let store = debug_store_mut();
    if store.entries.len() >= NUM_STORES_MAX {
        return;
    }

    let entry = store
        .entries
        .iter_mut()
        .find(|entry| entry.name == file_name);
    match entry {
        Some(existing) => existing.data.extend_from_slice(payload),
        None => store.entries.push(StoredData {
            name: file_name.into(),
            data: payload.to_vec(),
        }),
    }
}

#[cfg(not(feature = "silk_debug"))]
pub fn debug_store_data(_file_name: &str, _payload: &[u8]) {}

#[cfg(feature = "silk_debug")]
pub fn debug_store_snapshot() -> Vec<StoredData> {
    debug_store_mut().entries.clone()
}

#[cfg(not(feature = "silk_debug"))]
pub fn debug_store_snapshot() -> Vec<StoredData> {
    Vec::new()
}

#[cfg(feature = "silk_debug")]
pub fn debug_store_reset() {
    if let Some(store) = unsafe { (*DEBUG_STORE.get()).as_mut() } {
        store.entries.clear();
    }
}

#[cfg(not(feature = "silk_debug"))]
pub fn debug_store_reset() {}

#[cfg(feature = "silk_debug")]
pub fn debug_store_count() -> usize {
    debug_store_mut().entries.len()
}

#[cfg(not(feature = "silk_debug"))]
pub fn debug_store_count() -> usize {
    0
}

#[cfg(all(test, feature = "silk_tic_toc"))]
mod timer_tests {
    use super::*;

    static mut NOW: u64 = 0;

    fn fake_now() -> u64 {
        unsafe { NOW }
    }

    #[test]
    fn timer_accumulates_stats() {
        reset_timers();
        set_timer_source(fake_now);

        unsafe { NOW = 0 };
        tic("LPC");
        unsafe { NOW = 125 };
        toc("LPC");

        unsafe { NOW = 200 };
        tic("LPC");
        unsafe { NOW = 340 };
        toc("LPC");

        let reports = timer_snapshot().expect("Timers should produce data");
        assert_eq!(reports.len(), 1);
        let report = &reports[0];
        assert_eq!(report.tag, "LPC");
        assert_eq!(report.count, 2);
        assert_eq!(report.min, 125);
        assert_eq!(report.max, 140);
        assert_eq!(report.avg(), 132);

        reset_timers();
    }
}

#[cfg(all(test, feature = "silk_debug"))]
mod debug_store_tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn store_accumulates_payloads() {
        debug_store_reset();

        debug_store_data("trace.bin", &[1, 2, 3]);
        debug_store_data("trace.bin", &[4, 5]);
        debug_store_data("other.bin", &[9]);

        let mut entries = debug_store_snapshot();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "other.bin");
        assert_eq!(entries[0].data, vec![9]);
        assert_eq!(entries[1].name, "trace.bin");
        assert_eq!(entries[1].data, vec![1, 2, 3, 4, 5]);

        debug_store_reset();
        assert_eq!(debug_store_count(), 0);
    }
}

