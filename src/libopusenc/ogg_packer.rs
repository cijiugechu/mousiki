use alloc::collections::VecDeque;
use alloc::vec::Vec;

use mousiki_ogg::{PacketMetadata, StreamEncoder};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OggPacker {
    stream: StreamEncoder,
    pending_pages: VecDeque<Vec<u8>>,
    muxing_delay: u64,
    current_granule: u64,
    last_granule: u64,
}

impl OggPacker {
    #[must_use]
    pub(crate) fn new(serialno: i32) -> Self {
        Self {
            stream: StreamEncoder::new(serialno),
            pending_pages: VecDeque::new(),
            muxing_delay: 0,
            current_granule: 0,
            last_granule: 0,
        }
    }

    pub(crate) fn set_muxing_delay(&mut self, delay: u64) {
        self.muxing_delay = delay;
    }

    pub(crate) fn commit_packet(
        &mut self,
        packet: &[u8],
        granulepos: u64,
        eos: bool,
    ) -> Result<(), ()> {
        let segments_needed = packet.len() / 255 + 1;
        if self.muxing_delay != 0
            && granulepos.saturating_sub(self.last_granule) > self.muxing_delay
        {
            self.flush_page();
        }
        if self
            .stream
            .pending_segment_count()
            .saturating_add(segments_needed)
            > 255
        {
            self.flush_page();
        }

        if self
            .stream
            .push_packet_data(
                packet,
                PacketMetadata {
                    end_of_stream: eos,
                    granule_position: granulepos as i64,
                    ..PacketMetadata::default()
                },
            )
            .is_err()
        {
            return Err(());
        }
        self.current_granule = granulepos;

        if self.muxing_delay != 0
            && granulepos.saturating_sub(self.last_granule) >= self.muxing_delay
        {
            self.flush_page();
        }

        Ok(())
    }

    pub(crate) fn flush_page(&mut self) -> bool {
        let mut flushed = false;
        while let Some(page) = self.stream.flush_page_bytes_with_fill(i32::MAX) {
            self.pending_pages.push_back(page);
            flushed = true;
        }
        if flushed {
            self.last_granule = self.current_granule;
        }
        flushed
    }

    pub(crate) fn get_next_page(&mut self) -> Option<Vec<u8>> {
        self.pending_pages.pop_front()
    }

    pub(crate) fn chain(&mut self, serialno: i32) {
        self.flush_page();
        self.stream.reset_stream(serialno);
        self.pending_pages.clear();
        self.current_granule = 0;
        self.last_granule = 0;
    }
}
