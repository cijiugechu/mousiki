#![doc = include_str!("../README.md")]
#![no_std]

#[allow(clippy::all)]
pub mod dred_rdovae_dec_data {
    include!(concat!(env!("OUT_DIR"), "/dred_rdovae_dec_data.rs"));
}

#[allow(clippy::all)]
pub mod dred_stats_data {
    include!(concat!(env!("OUT_DIR"), "/dred_rdovae_stats_data.rs"));
}
