use crate::pack::{BitOrder, BitPacker, BitPackerError, BitUnpacker};
use crate::{Packet, Page, StreamState, SyncState};

pub type OggPage = Page;
pub type OggPacket = Packet;
pub type OggStreamState = StreamState;
pub type OggSyncState = SyncState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggPackBuffer {
    order: BitOrder,
    writer: BitPacker,
    reader: Option<BitUnpacker>,
}

impl Default for OggPackBuffer {
    fn default() -> Self {
        Self {
            order: BitOrder::Lsb,
            writer: BitPacker::new(BitOrder::Lsb),
            reader: None,
        }
    }
}

impl OggPackBuffer {
    fn reset_for_write(&mut self, order: BitOrder) {
        self.order = order;
        self.writer = BitPacker::new(order);
        self.reader = None;
    }

    fn reset_for_read(&mut self, order: BitOrder, buf: &[u8]) {
        self.order = order;
        self.writer = BitPacker::new(order);
        self.reader = Some(BitUnpacker::new(order, buf));
    }
}

fn map_pack_err(err: BitPackerError) -> i32 {
    match err {
        BitPackerError::InvalidBitCount | BitPackerError::ReadPastEnd => -1,
    }
}

pub fn oggpack_writeinit(buffer: &mut OggPackBuffer) {
    buffer.reset_for_write(BitOrder::Lsb);
}

pub fn oggpack_b_writeinit(buffer: &mut OggPackBuffer) {
    buffer.reset_for_write(BitOrder::Msb);
}

pub fn oggpack_writecheck(buffer: &OggPackBuffer) -> i32 {
    buffer.writer.writecheck()
}

pub fn oggpack_b_writecheck(buffer: &OggPackBuffer) -> i32 {
    oggpack_writecheck(buffer)
}

pub fn oggpack_writetrunc(buffer: &mut OggPackBuffer, bits: i64) {
    buffer.writer.writetrunc(bits as usize);
}

pub fn oggpack_b_writetrunc(buffer: &mut OggPackBuffer, bits: i64) {
    oggpack_writetrunc(buffer, bits);
}

pub fn oggpack_writealign(buffer: &mut OggPackBuffer) -> i32 {
    buffer.writer.writealign().map_or_else(map_pack_err, |_| 0)
}

pub fn oggpack_b_writealign(buffer: &mut OggPackBuffer) -> i32 {
    oggpack_writealign(buffer)
}

pub fn oggpack_writecopy(buffer: &mut OggPackBuffer, source: &[u8], bits: i64) -> i32 {
    buffer
        .writer
        .writecopy(source, bits as usize)
        .map_or_else(map_pack_err, |_| 0)
}

pub fn oggpack_b_writecopy(buffer: &mut OggPackBuffer, source: &[u8], bits: i64) -> i32 {
    oggpack_writecopy(buffer, source, bits)
}

pub fn oggpack_reset(buffer: &mut OggPackBuffer) {
    buffer.writer.reset();
}

pub fn oggpack_b_reset(buffer: &mut OggPackBuffer) {
    oggpack_reset(buffer);
}

pub fn oggpack_writeclear(buffer: &mut OggPackBuffer) {
    buffer.writer.writeclear();
}

pub fn oggpack_b_writeclear(buffer: &mut OggPackBuffer) {
    oggpack_writeclear(buffer);
}

pub fn oggpack_readinit(buffer: &mut OggPackBuffer, bytes: &[u8]) {
    buffer.reset_for_read(BitOrder::Lsb, bytes);
}

pub fn oggpack_b_readinit(buffer: &mut OggPackBuffer, bytes: &[u8]) {
    buffer.reset_for_read(BitOrder::Msb, bytes);
}

pub fn oggpack_write(buffer: &mut OggPackBuffer, value: u32, bits: i32) -> i32 {
    buffer
        .writer
        .write(value, bits)
        .map_or_else(map_pack_err, |_| 0)
}

pub fn oggpack_b_write(buffer: &mut OggPackBuffer, value: u32, bits: i32) -> i32 {
    oggpack_write(buffer, value, bits)
}

pub fn oggpack_look(buffer: &OggPackBuffer, bits: i32) -> i64 {
    buffer
        .reader
        .as_ref()
        .and_then(|reader| reader.look(bits).ok())
        .map_or(-1, i64::from)
}

pub fn oggpack_b_look(buffer: &OggPackBuffer, bits: i32) -> i64 {
    oggpack_look(buffer, bits)
}

pub fn oggpack_look1(buffer: &OggPackBuffer) -> i64 {
    oggpack_look(buffer, 1)
}

pub fn oggpack_b_look1(buffer: &OggPackBuffer) -> i64 {
    oggpack_look1(buffer)
}

pub fn oggpack_adv(buffer: &mut OggPackBuffer, bits: i32) -> i32 {
    buffer.reader.as_mut().map_or(-1, |reader| {
        reader.adv(bits).map_or_else(map_pack_err, |_| 0)
    })
}

pub fn oggpack_b_adv(buffer: &mut OggPackBuffer, bits: i32) -> i32 {
    oggpack_adv(buffer, bits)
}

pub fn oggpack_adv1(buffer: &mut OggPackBuffer) -> i32 {
    oggpack_adv(buffer, 1)
}

pub fn oggpack_b_adv1(buffer: &mut OggPackBuffer) -> i32 {
    oggpack_adv1(buffer)
}

pub fn oggpack_read(buffer: &mut OggPackBuffer, bits: i32) -> i64 {
    buffer
        .reader
        .as_mut()
        .and_then(|reader| reader.read(bits).ok())
        .map_or(-1, i64::from)
}

pub fn oggpack_b_read(buffer: &mut OggPackBuffer, bits: i32) -> i64 {
    oggpack_read(buffer, bits)
}

pub fn oggpack_read1(buffer: &mut OggPackBuffer) -> i64 {
    oggpack_read(buffer, 1)
}

pub fn oggpack_b_read1(buffer: &mut OggPackBuffer) -> i64 {
    oggpack_read1(buffer)
}

pub fn oggpack_bytes(buffer: &OggPackBuffer) -> i64 {
    buffer
        .reader
        .as_ref()
        .map_or(buffer.writer.bytes() as i64, |reader| reader.bytes() as i64)
}

pub fn oggpack_bits(buffer: &OggPackBuffer) -> i64 {
    buffer
        .reader
        .as_ref()
        .map_or(buffer.writer.bits() as i64, |reader| reader.bits() as i64)
}

pub fn oggpack_b_bytes(buffer: &OggPackBuffer) -> i64 {
    oggpack_bytes(buffer)
}

pub fn oggpack_b_bits(buffer: &OggPackBuffer) -> i64 {
    oggpack_bits(buffer)
}

pub fn oggpack_get_buffer(buffer: &OggPackBuffer) -> &[u8] {
    buffer.writer.buffer()
}

pub fn oggpack_b_get_buffer(buffer: &OggPackBuffer) -> &[u8] {
    oggpack_get_buffer(buffer)
}

pub fn ogg_stream_packetin(state: &mut OggStreamState, packet: &OggPacket) -> i32 {
    state.packet_in(packet)
}

pub fn ogg_stream_iovecin(
    state: &mut OggStreamState,
    buffers: &[&[u8]],
    e_o_s: bool,
    granulepos: i64,
) -> i32 {
    state.iovec_in(buffers, e_o_s, granulepos)
}

pub fn ogg_stream_pageout(state: &mut OggStreamState) -> Option<OggPage> {
    state.page_out()
}

pub fn ogg_stream_pageout_fill(state: &mut OggStreamState, nfill: i32) -> Option<OggPage> {
    state.page_out_fill(nfill)
}

pub fn ogg_stream_flush(state: &mut OggStreamState) -> Option<OggPage> {
    state.flush()
}

pub fn ogg_stream_flush_fill(state: &mut OggStreamState, nfill: i32) -> Option<OggPage> {
    state.flush_fill(nfill)
}

pub fn ogg_sync_check(state: &OggSyncState) -> i32 {
    state.check()
}

pub fn ogg_sync_buffer(state: &mut OggSyncState, size: usize) -> &mut [u8] {
    state.buffer(size)
}

pub fn ogg_sync_wrote(state: &mut OggSyncState, bytes: usize) -> i32 {
    state.wrote(bytes)
}

pub fn ogg_sync_pageseek(state: &mut OggSyncState) -> Result<Option<OggPage>, i64> {
    state.pageseek()
}

pub fn ogg_sync_pageout(state: &mut OggSyncState) -> Result<Option<OggPage>, i32> {
    state.pageout()
}

pub fn ogg_stream_pagein(state: &mut OggStreamState, page: &OggPage) -> i32 {
    state.page_in(page)
}

pub fn ogg_stream_packetout(state: &mut OggStreamState) -> Result<Option<OggPacket>, i32> {
    state.packet_out()
}

pub fn ogg_stream_packetpeek(state: &mut OggStreamState) -> Result<Option<OggPacket>, i32> {
    state.packet_peek()
}

pub fn ogg_stream_check(state: &OggStreamState) -> i32 {
    state.check()
}

pub fn ogg_stream_eos(state: &OggStreamState) -> i32 {
    state.eos()
}

pub fn ogg_page_checksum_set(page: &mut OggPage) {
    page.checksum_set();
}

pub fn ogg_page_version(page: &OggPage) -> i32 {
    page.version()
}

pub fn ogg_page_continued(page: &OggPage) -> i32 {
    page.continued()
}

pub fn ogg_page_bos(page: &OggPage) -> i32 {
    page.bos()
}

pub fn ogg_page_eos(page: &OggPage) -> i32 {
    page.eos()
}

pub fn ogg_page_granulepos(page: &OggPage) -> i64 {
    page.granulepos()
}

pub fn ogg_page_serialno(page: &OggPage) -> i32 {
    page.serialno()
}

pub fn ogg_page_pageno(page: &OggPage) -> i64 {
    page.pageno()
}

pub fn ogg_page_packets(page: &OggPage) -> i32 {
    page.packets()
}

pub fn ogg_packet_clear(packet: &mut OggPacket) {
    packet.clear();
}
