//! Channel layout helpers mirrored from `opus_multistream.c`.
#![cfg_attr(not(test), allow(dead_code))]

use crate::celt::isqrt32;

/// Sentinel used by the reference encoder when auto-selecting the bitrate.
pub(crate) const OPUS_AUTO: i32 = -1000;
/// Maximum bitrate marker mirrored from the public Opus defines.
pub(crate) const OPUS_BITRATE_MAX: i32 = -1;

/// Mirrors the mapping type enum embedded in the multistream encoder state.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MappingType {
    None,
    Surround,
    Ambisonics,
}

/// Internal multistream channel layout description.
///
/// Mirrors the layout prefix embedded in the multistream encoder/decoder
/// states. The `mapping` table uses `255` as a sentinel for channels that are
/// omitted from the encoded streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChannelLayout {
    pub nb_channels: usize,
    pub nb_streams: usize,
    pub nb_coupled_streams: usize,
    pub mapping: [u8; 256],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VorbisLayout {
    nb_streams: usize,
    nb_coupled_streams: usize,
    mapping: [u8; 8],
}

/* Index is nb_channel-1 */
#[allow(dead_code)]
const VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 0,
        mapping: [0, 255, 255, 255, 255, 255, 255, 255],
    }, /* 1: mono */
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 1,
        mapping: [0, 1, 255, 255, 255, 255, 255, 255],
    }, /* 2: stereo */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 1,
        mapping: [0, 2, 1, 255, 255, 255, 255, 255],
    }, /* 3: 1-d surround */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 2,
        mapping: [0, 1, 2, 3, 255, 255, 255, 255],
    }, /* 4: quadraphonic surround */
    VorbisLayout {
        nb_streams: 3,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 255, 255, 255],
    }, /* 5: 5-channel surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 255, 255],
    }, /* 6: 5.1 surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 255],
    }, /* 7: 6.1 surround */
    VorbisLayout {
        nb_streams: 5,
        nb_coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    }, /* 8: 7.1 surround */
];

/// Verifies that the layout only references stream indices that exist.
#[must_use]
pub(crate) fn validate_layout(layout: &ChannelLayout) -> bool {
    let Some(max_channel) = layout.nb_streams.checked_add(layout.nb_coupled_streams) else {
        return false;
    };

    if max_channel > u8::MAX as usize {
        return false;
    }

    if layout.nb_channels > layout.mapping.len() {
        return false;
    }

    layout
        .mapping
        .iter()
        .take(layout.nb_channels)
        .all(|&value| value == u8::MAX || usize::from(value) < max_channel)
}

/// Ensures each stream in the layout has a channel mapping.
#[must_use]
pub(crate) fn validate_encoder_layout(layout: &ChannelLayout) -> bool {
    for stream in 0..layout.nb_streams {
        if stream < layout.nb_coupled_streams {
            if get_left_channel(layout, stream, None).is_none() {
                return false;
            }
            if get_right_channel(layout, stream, None).is_none() {
                return false;
            }
        } else if get_mono_channel(layout, stream, None).is_none() {
            return false;
        }
    }

    true
}

/// Validates the ambisonics channel count and returns the derived stream layout.
#[must_use]
pub(crate) fn validate_ambisonics(channels: usize) -> Option<(usize, usize)> {
    if !(1..=227).contains(&channels) {
        return None;
    }

    let order_plus_one = isqrt32(channels as u32) as usize;
    let acn_channels = order_plus_one.checked_mul(order_plus_one)?;
    let nondiegetic_channels = channels.checked_sub(acn_channels)?;

    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    let streams = acn_channels + usize::from(nondiegetic_channels != 0);
    let coupled_streams = usize::from(nondiegetic_channels != 0);
    Some((streams, coupled_streams))
}

fn surround_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    let nb_streams = layout.nb_streams;
    let nb_coupled = layout.nb_coupled_streams;
    let nb_lfe = usize::from(lfe_stream.is_some());
    if nb_streams == 0 || nb_coupled > nb_streams || nb_streams < nb_coupled + nb_lfe {
        return None;
    }
    if frame_size == 0 || sample_rate <= 0 {
        return None;
    }
    if rates.len() < nb_streams {
        return None;
    }

    let nb_uncoupled = nb_streams - nb_coupled - nb_lfe;
    let nb_normal = 2 * nb_coupled + nb_uncoupled;
    if nb_normal == 0 {
        return None;
    }

    let frame_rate = sample_rate / frame_size as i32;
    let channel_offset = 40 * frame_rate.max(50);
    let bitrate = if bitrate_bps == OPUS_AUTO {
        nb_normal as i32 * (channel_offset + sample_rate + 10_000) + 8000 * nb_lfe as i32
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_normal as i32 * 300_000 + nb_lfe as i32 * 128_000
    } else {
        bitrate_bps
    };

    let lfe_offset = bitrate
        .checked_div(20)
        .map(|value| value.min(3000))
        .and_then(|value| value.checked_add(15 * frame_rate.max(50)))?;
    let stream_offset =
        (((bitrate - channel_offset * nb_normal as i32 - lfe_offset * nb_lfe as i32)
            / nb_normal as i32)
            / 2)
        .clamp(0, 20_000);
    let coupled_ratio = 512;
    let lfe_ratio = 32;

    let total = ((nb_uncoupled as i32) << 8)
        + coupled_ratio * nb_coupled as i32
        + lfe_ratio * nb_lfe as i32;
    if total == 0 {
        return None;
    }
    let channel_rate = 256
        * (bitrate
            - lfe_offset * nb_lfe as i32
            - stream_offset * (nb_coupled as i32 + nb_uncoupled as i32)
            - channel_offset * nb_normal as i32)
        / total;

    for (stream, slot) in rates.iter_mut().take(nb_streams).enumerate() {
        let value = if stream < nb_coupled {
            2 * channel_offset + (stream_offset + ((channel_rate * coupled_ratio) >> 8)).max(0)
        } else if lfe_stream == Some(stream) {
            (lfe_offset + ((channel_rate * lfe_ratio) >> 8)).max(0)
        } else {
            channel_offset + (stream_offset + channel_rate).max(0)
        };
        *slot = value;
    }

    Some(())
}

fn ambisonics_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    if frame_size == 0 || sample_rate <= 0 || layout.nb_streams == 0 {
        return None;
    }
    if rates.len() < layout.nb_streams {
        return None;
    }

    let nb_channels = layout.nb_streams + layout.nb_coupled_streams;
    let total_rate = if bitrate_bps == OPUS_AUTO {
        let term = sample_rate + 60 * sample_rate / frame_size as i32;
        (layout.nb_coupled_streams + layout.nb_streams) as i32 * term
            + layout.nb_streams as i32 * 15_000
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_channels as i32 * 320_000
    } else {
        bitrate_bps
    };

    let per_stream_rate = total_rate / layout.nb_streams as i32;
    for slot in rates.iter_mut().take(layout.nb_streams) {
        *slot = per_stream_rate;
    }

    Some(())
}

/// Computes the bitrate distribution across all streams.
#[must_use]
pub(crate) fn rate_allocation(
    layout: &ChannelLayout,
    mapping_type: MappingType,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<i32> {
    match mapping_type {
        MappingType::Ambisonics => {
            ambisonics_rate_allocation(layout, bitrate_bps, frame_size, sample_rate, rates)?
        }
        _ => surround_rate_allocation(
            layout,
            bitrate_bps,
            lfe_stream,
            frame_size,
            sample_rate,
            rates,
        )?,
    }

    let mut sum = 0i64;
    for rate in rates.iter_mut().take(layout.nb_streams) {
        *rate = (*rate).max(500);
        sum += i64::from(*rate);
    }

    i32::try_from(sum).ok()
}

fn next_index(prev: Option<usize>) -> Option<usize> {
    match prev {
        Some(idx) => idx.checked_add(1),
        None => Some(0),
    }
}

fn find_channel(layout: &ChannelLayout, start: usize, target: usize) -> Option<usize> {
    let limit = layout.nb_channels.min(layout.mapping.len());
    (start..limit).find(|&i| usize::from(layout.mapping[i]) == target)
}

/// Returns the next channel mapped to the left slot of `stream_id`.
#[must_use]
pub(crate) fn get_left_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the right slot of `stream_id`.
#[must_use]
pub(crate) fn get_right_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?.checked_add(1)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the mono stream `stream_id`.
#[must_use]
pub(crate) fn get_mono_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_add(layout.nb_coupled_streams)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_from_mapping(
        nb_channels: usize,
        nb_streams: usize,
        nb_coupled_streams: usize,
        mapping_slice: &[u8],
    ) -> ChannelLayout {
        let mut mapping = [u8::MAX; 256];
        let count = mapping_slice.len().min(mapping.len());
        mapping[..count].copy_from_slice(&mapping_slice[..count]);
        ChannelLayout {
            nb_channels,
            nb_streams,
            nb_coupled_streams,
            mapping,
        }
    }

    #[test]
    fn accepts_valid_layout() {
        let layout = layout_from_mapping(3, 2, 1, &[0, 1, 2]);
        assert!(validate_layout(&layout));
    }

    #[test]
    fn rejects_out_of_range_stream_indices() {
        let layout = layout_from_mapping(1, 1, 0, &[1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn rejects_when_stream_count_exceeds_byte_limit() {
        let layout = layout_from_mapping(2, 200, 60, &[0, 1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn iterates_all_channels_for_a_stream() {
        let layout = layout_from_mapping(4, 3, 1, &[0, 0, 1, 2]);

        assert_eq!(get_left_channel(&layout, 0, None), Some(0));
        assert_eq!(get_left_channel(&layout, 0, Some(0)), Some(1));
        assert_eq!(get_left_channel(&layout, 0, Some(1)), None);

        assert_eq!(get_right_channel(&layout, 0, None), Some(2));
        assert_eq!(get_right_channel(&layout, 0, Some(2)), None);

        assert_eq!(get_mono_channel(&layout, 1, None), Some(3));
        assert_eq!(get_mono_channel(&layout, 1, Some(3)), None);
    }

    #[test]
    fn validate_encoder_layout_rejects_missing_channel() {
        let layout = layout_from_mapping(2, 1, 1, &[0, u8::MAX]);
        assert!(!validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_encoder_layout_accepts_complete_mapping() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        assert!(validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_ambisonics_computes_stream_and_coupling_counts() {
        assert_eq!(validate_ambisonics(1), Some((1, 0)));
        assert_eq!(validate_ambisonics(6), Some((5, 1)));
        assert_eq!(validate_ambisonics(7), None);
        assert_eq!(validate_ambisonics(228), None);
    }

    #[test]
    fn rate_allocation_handles_stereo_surround_defaults() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates = [0; 1];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 120_000);
        assert_eq!(rates, [120_000]);
    }

    #[test]
    fn rate_allocation_accounts_for_lfe_stream() {
        let layout = layout_from_mapping(6, 4, 2, &[0, 4, 1, 2, 3, 5]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            256_000,
            Some(3),
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 255_995);
        assert_eq!(rates, [95_120, 95_120, 57_560, 8_195]);
    }

    #[test]
    fn rate_allocation_splits_evenly_for_ambisonics() {
        let layout = layout_from_mapping(4, 4, 0, &[0, 1, 2, 3]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Ambisonics,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 264_000);
        assert_eq!(rates, [66_000; 4]);
    }

    #[test]
    fn rate_allocation_rejects_insufficient_rate_storage() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates: [i32; 0] = [];

        assert!(
            rate_allocation(
                &layout,
                MappingType::Surround,
                64_000,
                None,
                960,
                48_000,
                &mut rates
            )
            .is_none()
        );
    }
}
