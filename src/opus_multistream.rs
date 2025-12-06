//! Channel layout helpers mirrored from `opus_multistream.c`.
#![cfg_attr(not(test), allow(dead_code))]

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

/// Verifies that the layout only references stream indices that exist.
#[must_use]
pub(crate) fn validate_layout(layout: &ChannelLayout) -> bool {
    let Some(max_channel) = layout
        .nb_streams
        .checked_add(layout.nb_coupled_streams)
    else {
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
}
