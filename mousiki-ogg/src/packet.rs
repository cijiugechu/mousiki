extern crate alloc;

use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Packet {
    pub packet: Vec<u8>,
    pub b_o_s: i32,
    pub e_o_s: i32,
    pub granulepos: i64,
    pub packetno: i64,
}

impl Packet {
    #[must_use]
    pub fn new(packet: Vec<u8>, b_o_s: bool, e_o_s: bool, granulepos: i64, packetno: i64) -> Self {
        Self {
            packet,
            b_o_s: i32::from(b_o_s),
            e_o_s: i32::from(e_o_s),
            granulepos,
            packetno,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> i64 {
        self.packet.len() as i64
    }

    pub fn clear(&mut self) {
        self.packet.clear();
        self.b_o_s = 0;
        self.e_o_s = 0;
        self.granulepos = 0;
        self.packetno = 0;
    }
}
