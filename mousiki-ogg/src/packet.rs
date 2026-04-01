extern crate alloc;

use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PacketMetadata {
    pub beginning_of_stream: bool,
    pub end_of_stream: bool,
    pub granule_position: i64,
    pub sequence_number: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Packet {
    data: Vec<u8>,
    metadata: PacketMetadata,
}

impl Packet {
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            metadata: PacketMetadata::default(),
        }
    }

    #[must_use]
    pub fn with_metadata(data: Vec<u8>, metadata: PacketMetadata) -> Self {
        Self { data, metadata }
    }

    #[must_use]
    pub(crate) fn from_raw_parts(
        data: Vec<u8>,
        beginning_of_stream: bool,
        end_of_stream: bool,
        granule_position: i64,
        sequence_number: i64,
    ) -> Self {
        Self::with_metadata(
            data,
            PacketMetadata {
                beginning_of_stream,
                end_of_stream,
                granule_position,
                sequence_number,
            },
        )
    }

    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    #[must_use]
    pub fn metadata(&self) -> PacketMetadata {
        self.metadata
    }

    #[must_use]
    pub fn is_beginning_of_stream(&self) -> bool {
        self.metadata.beginning_of_stream
    }

    #[must_use]
    pub fn is_end_of_stream(&self) -> bool {
        self.metadata.end_of_stream
    }

    #[must_use]
    pub fn granule_position(&self) -> i64 {
        self.metadata.granule_position
    }

    #[must_use]
    pub fn sequence_number(&self) -> i64 {
        self.metadata.sequence_number
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.data.clear();
        self.metadata = PacketMetadata::default();
    }
}
