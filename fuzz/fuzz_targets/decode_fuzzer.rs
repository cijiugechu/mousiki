#![no_main]

use libfuzzer_sys::fuzz_target;
use mousiki::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decoder_create, opus_decoder_ctl,
};
use mousiki::packet::{opus_packet_get_bandwidth, opus_packet_get_nb_channels};

const MAX_FRAME_SAMP: usize = 5760;
const MAX_PACKET: usize = 1500;
const SETUP_BYTE_COUNT: usize = 8;
const MAX_DECODES: usize = 12;

fn read_be_u32(bytes: &[u8]) -> Option<u32> {
    let chunk = bytes.get(..4)?;
    Some(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
    ]))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < SETUP_BYTE_COUNT + 1 {
        return;
    }

    let toc = &data[SETUP_BYTE_COUNT..];
    let bandwidth = match opus_packet_get_bandwidth(toc) {
        Ok(bandwidth) => bandwidth,
        Err(_) => return,
    };
    let channels = match opus_packet_get_nb_channels(toc) {
        Ok(channels) => channels,
        Err(_) => return,
    };
    let sample_rate = bandwidth.sample_rate() as i32;

    let mut decoder = match opus_decoder_create(sample_rate, channels as i32) {
        Ok(decoder) => decoder,
        Err(_) => return,
    };

    let pcm_len = MAX_FRAME_SAMP.saturating_mul(channels);
    let mut pcm = [0i16; MAX_FRAME_SAMP * 2];
    let pcm = &mut pcm[..pcm_len];

    let mut i = 0usize;
    let mut num_decodes = 0usize;
    while i + SETUP_BYTE_COUNT < data.len() && num_decodes < MAX_DECODES {
        num_decodes += 1;

        let len = match read_be_u32(&data[i..]) {
            Some(value) => value as usize,
            None => break,
        };
        let packet_offset = match i.checked_add(SETUP_BYTE_COUNT) {
            Some(value) => value,
            None => break,
        };
        let end = match packet_offset.checked_add(len) {
            Some(value) => value,
            None => break,
        };

        if len > MAX_PACKET || end > data.len() {
            break;
        }

        let fec = data[i + 4] & 1 != 0;
        if len == 0 {
            let mut frame_size = 0i32;
            let _ = opus_decoder_ctl(
                &mut decoder,
                OpusDecoderCtlRequest::GetLastPacketDuration(&mut frame_size),
            );
            let frame_size = frame_size.max(0) as usize;
            let _ = opus_decode(&mut decoder, None, 0, pcm, frame_size, fec);
        } else {
            let packet = &data[packet_offset..end];
            let _ = opus_decode(&mut decoder, Some(packet), len, pcm, MAX_FRAME_SAMP, fec);
        }

        i = end;
    }
});
