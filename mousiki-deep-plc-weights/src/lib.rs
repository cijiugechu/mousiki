#![no_std]

pub const DNN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/weights_blob.bin"));
