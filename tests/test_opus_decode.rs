use mousiki::opus::opus_pcm_soft_clip;
use mousiki::opus_decoder::{
    OpusDecodeError, OpusDecoder, OpusDecoderCtlRequest, opus_decode, opus_decoder_create,
    opus_decoder_ctl, opus_decoder_get_nb_samples,
};
use mousiki::packet::opus_packet_get_nb_channels;

const MAX_PACKET: usize = 1500;
const MAX_FRAME_SAMP: usize = 5760;

#[derive(Clone)]
struct FastRand {
    rz: u32,
    rw: u32,
}

impl FastRand {
    fn new(seed: u32) -> Self {
        Self { rz: seed, rw: seed }
    }

    fn next(&mut self) -> u32 {
        self.rz = 36969u32
            .wrapping_mul(self.rz & 0xFFFF)
            .wrapping_add(self.rz >> 16);
        self.rw = 18000u32
            .wrapping_mul(self.rw & 0xFFFF)
            .wrapping_add(self.rw >> 16);
        (self.rz << 16).wrapping_add(self.rw)
    }
}

fn deb2_impl(t: &mut [u8], out: &mut [u8], p: &mut usize, k: usize, x: usize, y: usize) {
    if x > 2 {
        if y < 3 {
            for i in 0..y {
                *p = p.saturating_sub(1);
                out[*p] = t[i + 1];
            }
        }
        return;
    }

    t[x] = t[x - y];
    deb2_impl(t, out, p, k, x + 1, y);
    for i in (t[x - y] + 1)..(k as u8) {
        t[x] = i;
        deb2_impl(t, out, p, k, x + 1, x);
    }
}

fn debruijn2(k: usize) -> Vec<u8> {
    let mut out = vec![0u8; k * k];
    let mut t = vec![0u8; k * 2];
    let mut p = k * k;
    deb2_impl(&mut t, &mut out, &mut p, k, 1, 1);
    out
}

fn seed_from_env() -> u32 {
    std::env::var("SEED")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0xC0DEC0DE)
}

fn no_fuzz() -> bool {
    if std::env::var_os("TEST_OPUS_FUZZ").is_some() {
        return false;
    }
    std::env::var_os("TEST_OPUS_NOFUZZ").is_some() || cfg!(debug_assertions)
}

fn strict_final_range() -> bool {
    std::env::var_os("TEST_OPUS_STRICT_FINAL_RANGE").is_some()
}

fn set_random_complexity(
    decoder: &mut OpusDecoder<'_>,
    rng: &mut FastRand,
) -> Result<(), String> {
    let complexity = (rng.next() % 11) as i32;
    opus_decoder_ctl(decoder, OpusDecoderCtlRequest::SetComplexity(complexity))
        .map_err(|err| format!("set complexity failed: {err:?}"))
}

fn decode_i16(
    decoder: &mut OpusDecoder<'_>,
    data: Option<&[u8]>,
    len: usize,
    pcm: &mut [i16],
    frame_size: usize,
    decode_fec: bool,
    context: &str,
) -> Result<usize, String> {
    opus_decode(decoder, data, len, pcm, frame_size, decode_fec)
        .map_err(|err| format!("{context} failed: {err:?}"))
}

fn last_packet_duration(decoder: &mut OpusDecoder<'_>) -> Result<i32, String> {
    let mut dur = 0i32;
    opus_decoder_ctl(decoder, OpusDecoderCtlRequest::GetLastPacketDuration(&mut dur))
        .map_err(|err| format!("get last packet duration failed: {err:?}"))?;
    Ok(dur)
}

fn final_range(decoder: &mut OpusDecoder<'_>) -> Result<u32, String> {
    let mut range = 0u32;
    opus_decoder_ctl(decoder, OpusDecoderCtlRequest::GetFinalRange(&mut range))
        .map_err(|err| format!("get final range failed: {err:?}"))?;
    Ok(range)
}

fn test_decoder_code0(no_fuzz: bool, rng: &mut FastRand) -> Result<(), String> {
    let fsv = [48_000, 24_000, 16_000, 12_000, 8000];
    let mut decs = Vec::with_capacity(10);
    let strict_range = strict_final_range();

    for t in 0..(fsv.len() * 2) {
        let fs = fsv[t / 2];
        let channels = (t & 1) + 1;
        let dec = opus_decoder_create(fs, channels as i32)
            .map_err(|err| format!("decoder create failed: {err:?}"))?;
        decs.push(dec);
    }

    let mut packet = vec![0u8; MAX_PACKET];
    let guard_offset = 16usize;
    let outbuf_len = MAX_FRAME_SAMP * 2;
    let mut outbuf_int = vec![32749i16; (MAX_FRAME_SAMP + 16) * 2];

    {
        let outbuf = &mut outbuf_int[guard_offset..guard_offset + outbuf_len];

        for t in 0..decs.len() {
            let factor = 48_000 / fsv[t / 2];
            let plc_size = (120 / factor) as usize;
            for &fec in &[false, true] {
                set_random_complexity(&mut decs[t], rng)?;

                let out_samples =
                    decode_i16(&mut decs[t], None, 0, outbuf, plc_size, fec, "plc")?;
                if out_samples != plc_size {
                    return Err("plc returned unexpected sample count".to_string());
                }

                let dur = last_packet_duration(&mut decs[t])?;
                if dur != plc_size as i32 {
                    return Err("plc last packet duration mismatch".to_string());
                }

                let err = opus_decode(&mut decs[t], None, 0, outbuf, plc_size + 2, fec);
                if !matches!(err, Err(OpusDecodeError::BadArgument)) {
                    return Err("non-2.5ms PLC frame size should be rejected".to_string());
                }

                let len_cases = [0usize, 1, 10, rng.next() as usize];
                for &len in &len_cases {
                    let out_samples =
                        decode_i16(&mut decs[t], None, len, outbuf, plc_size, fec, "plc")?;
                    if out_samples != plc_size {
                        return Err("plc returned unexpected sample count".to_string());
                    }
                }

                let dur = last_packet_duration(&mut decs[t])?;
                if dur != plc_size as i32 {
                    return Err("plc last packet duration mismatch".to_string());
                }

                let out_samples = decode_i16(
                    &mut decs[t],
                    Some(&packet[..0]),
                    0,
                    outbuf,
                    plc_size,
                    fec,
                    "zero-length packet",
                )?;
                if out_samples != plc_size {
                    return Err("zero-length packet PLC mismatch".to_string());
                }

                outbuf[0] = 32749;
                let err = opus_decode(&mut decs[t], Some(&packet[..0]), 0, outbuf, 0, fec);
                if !matches!(err, Err(OpusDecodeError::BadArgument)) {
                    return Err("zero frame size should be rejected".to_string());
                }
                if outbuf[0] != 32749 {
                    return Err("zero-length decode modified output buffer".to_string());
                }

                let err = opus_decode(
                    &mut decs[t],
                    Some(&packet),
                    packet.len() + 1,
                    outbuf,
                    MAX_FRAME_SAMP,
                    fec,
                );
                if !matches!(err, Err(OpusDecodeError::BadArgument)) {
                    return Err("oversized packet length should be rejected".to_string());
                }
                let err = opus_decode(
                    &mut decs[t],
                    Some(&packet),
                    usize::MAX,
                    outbuf,
                    MAX_FRAME_SAMP,
                    fec,
                );
                if !matches!(err, Err(OpusDecodeError::BadArgument)) {
                    return Err("oversized packet length should be rejected".to_string());
                }

                opus_decoder_ctl(&mut decs[t], OpusDecoderCtlRequest::ResetState)
                    .map_err(|err| format!("reset decoder failed: {err:?}"))?;
            }
        }

        for i in 0u8..64 {
            let mut expected = [0usize; 10];
            packet[0] = i << 2;
            packet[1] = 255;
            packet[2] = 255;
            let channels = opus_packet_get_nb_channels(&packet[..1])
                .map_err(|err| format!("get nb channels failed: {err:?}"))?;
            if channels != (i as usize & 1) + 1 {
                return Err("channel count mismatch".to_string());
            }

            for (t, dec) in decs.iter().enumerate() {
                expected[t] = opus_decoder_get_nb_samples(dec, &packet, 1)
                    .map_err(|err| format!("get nb samples failed: {err:?}"))?;
                if expected[t] > 2880 {
                    return Err("unexpected nb samples for code 0".to_string());
                }
            }

            for j in 0u16..256 {
                packet[1] = j as u8;
                let mut dec_final_range2 = 0u32;
                for t in 0..decs.len() {
                    set_random_complexity(&mut decs[t], rng)?;
                    let out_samples = decode_i16(
                        &mut decs[t],
                        Some(&packet[..3]),
                        3,
                        outbuf,
                        MAX_FRAME_SAMP,
                        false,
                        "code 0 decode",
                    )?;
                    if out_samples != expected[t] {
                        return Err("code 0 decoded sample count mismatch".to_string());
                    }
                    let dur = last_packet_duration(&mut decs[t])?;
                    if dur != out_samples as i32 {
                        return Err("code 0 last packet duration mismatch".to_string());
                    }
                    let range = final_range(&mut decs[t])?;
                    if t == 0 {
                        dec_final_range2 = range;
                    } else if range != dec_final_range2 {
                        return Err("final range mismatch across decoders".to_string());
                    }
                }
            }

            for t in 0..decs.len() {
                let factor = 48_000 / fsv[t / 2];
                for _ in 0..6 {
                    set_random_complexity(&mut decs[t], rng)?;
                    let out_samples = decode_i16(
                        &mut decs[t],
                        None,
                        0,
                        outbuf,
                        expected[t],
                        false,
                        "plc",
                    )?;
                    if out_samples != expected[t] {
                        return Err("plc returned unexpected sample count".to_string());
                    }
                    let dur = last_packet_duration(&mut decs[t])?;
                    if dur != out_samples as i32 {
                        return Err("plc last packet duration mismatch".to_string());
                    }
                }
                if expected[t] != (120 / factor) as usize {
                    let plc_size = (120 / factor) as usize;
                    let out_samples =
                        decode_i16(&mut decs[t], None, 0, outbuf, plc_size, false, "plc")?;
                    if out_samples != plc_size {
                        return Err("short PLC returned unexpected sample count".to_string());
                    }
                    let dur = last_packet_duration(&mut decs[t])?;
                    if dur != out_samples as i32 {
                        return Err("short PLC last packet duration mismatch".to_string());
                    }
                }

                let err = opus_decode(
                    &mut decs[t],
                    Some(&packet[..2]),
                    2,
                    outbuf,
                    expected[t].saturating_sub(1),
                    false,
                );
                if err.is_ok() {
                    return Err("short frame size should be rejected".to_string());
                }
            }
        }

        if !no_fuzz {
            let cmodes = [16u8, 20, 24, 28];
            let cres = [116290185u32, 2172123586, 2172123586, 2172123586];
            let lres = [3285687739u32, 1481572662, 694350475];
            let lmodes = [0u8, 4, 8];

            let mode = (rng.next() % 4) as usize;
            packet[0] = cmodes[mode] << 3;
            let mut dec_final_acc = 0u32;
            let t = (rng.next() % 10) as usize;
            for i in 0..65536u32 {
                let factor = 48_000 / fsv[t / 2];
                packet[1] = (i >> 8) as u8;
                packet[2] = (i & 255) as u8;
                packet[3] = 255;
                let out_samples = decode_i16(
                    &mut decs[t],
                    Some(&packet[..4]),
                    4,
                    outbuf,
                    MAX_FRAME_SAMP,
                    false,
                    "3-byte prefix",
                )?;
                if out_samples != (120 / factor) as usize {
                    return Err("3-byte prefix sample count mismatch".to_string());
                }
                let range = final_range(&mut decs[t])?;
                dec_final_acc = dec_final_acc.wrapping_add(range);
            }
            if dec_final_acc != cres[mode] {
                return Err("3-byte prefix final range mismatch".to_string());
            }

            let mode = (rng.next() % 3) as usize;
            packet[0] = lmodes[mode] << 3;
            dec_final_acc = 0;
            let t = (rng.next() % 10) as usize;
            for i in 0..65536u32 {
                let factor = 48_000 / fsv[t / 2];
                packet[1] = (i >> 8) as u8;
                packet[2] = (i & 255) as u8;
                packet[3] = 255;
                let out_samples = decode_i16(
                    &mut decs[t],
                    Some(&packet[..4]),
                    4,
                    outbuf,
                    MAX_FRAME_SAMP,
                    false,
                    "3-byte prefix long",
                )?;
                if out_samples != (480 / factor) as usize {
                    return Err("3-byte prefix long sample count mismatch".to_string());
                }
                let range = final_range(&mut decs[t])?;
                dec_final_acc = dec_final_acc.wrapping_add(range);
            }
            if strict_range && dec_final_acc != lres[mode] {
                return Err("3-byte prefix long final range mismatch".to_string());
            }

            let skip = (rng.next() % 7) as usize;
            for i in 0u8..64 {
                packet[0] = i << 2;
                let mut expected = [0usize; 10];
                for (t, dec) in decs.iter().enumerate() {
                    expected[t] = opus_decoder_get_nb_samples(dec, &packet, 1)
                        .map_err(|err| format!("get nb samples failed: {err:?}"))?;
                }
                for j in (2 + skip..1275).step_by(4) {
                    for jj in 0..j {
                        packet[jj + 1] = (rng.next() & 255) as u8;
                    }
                    let mut dec_final_range2 = 0u32;
                    for t in 0..decs.len() {
                        set_random_complexity(&mut decs[t], rng)?;
                        let out_samples = decode_i16(
                            &mut decs[t],
                            Some(&packet[..j + 1]),
                            j + 1,
                            outbuf,
                            MAX_FRAME_SAMP,
                            false,
                            "random packets",
                        )?;
                        if out_samples != expected[t] {
                            return Err("random packet sample count mismatch".to_string());
                        }
                        let range = final_range(&mut decs[t])?;
                        if t == 0 {
                            dec_final_range2 = range;
                        } else if range != dec_final_range2 {
                            return Err("final range mismatch across decoders".to_string());
                        }
                    }
                }
            }

            let modes = debruijn2(64);
            let plen = ((rng.next() % 18 + 3) * 8) as usize + skip + 3;
            let mut decbak = opus_decoder_create(48_000, 1)
                .map_err(|err| format!("decoder create failed: {err:?}"))?;
            for i in 0..4096usize {
                packet[0] = modes[i] << 2;
                let mut expected = [0usize; 10];
                for (t, dec) in decs.iter().enumerate() {
                    expected[t] = opus_decoder_get_nb_samples(dec, &packet, plen)
                        .map_err(|err| format!("get nb samples failed: {err:?}"))?;
                }
                for j in 0..plen {
                    packet[j + 1] = ((rng.next() | rng.next()) & 255) as u8;
                }

                opus_decoder_ctl(&mut decbak, OpusDecoderCtlRequest::ResetState)
                    .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                let out_samples = decode_i16(
                    &mut decbak,
                    Some(&packet[..plen + 1]),
                    plen + 1,
                    outbuf,
                    expected[0],
                    true,
                    "fec decode",
                )?;
                if out_samples != expected[0] {
                    return Err("fec decode sample count mismatch".to_string());
                }

                opus_decoder_ctl(&mut decbak, OpusDecoderCtlRequest::ResetState)
                    .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                let out_samples =
                    decode_i16(&mut decbak, None, 0, outbuf, MAX_FRAME_SAMP, true, "fec plc")?;
                if out_samples < 20 {
                    return Err("fec plc returned too few samples".to_string());
                }

                opus_decoder_ctl(&mut decbak, OpusDecoderCtlRequest::ResetState)
                    .map_err(|err| format!("reset decoder failed: {err:?}"))?;
                let out_samples =
                    decode_i16(&mut decbak, None, 0, outbuf, MAX_FRAME_SAMP, false, "plc")?;
                if out_samples < 20 {
                    return Err("plc returned too few samples".to_string());
                }

                for t in 0..decs.len() {
                    set_random_complexity(&mut decs[t], rng)?;
                    let out_samples = decode_i16(
                        &mut decs[t],
                        Some(&packet[..plen + 1]),
                        plen + 1,
                        outbuf,
                        MAX_FRAME_SAMP,
                        false,
                        "random packets",
                    )?;
                    if out_samples != expected[t] {
                        return Err("random packet sample count mismatch".to_string());
                    }
                    let dur = last_packet_duration(&mut decs[t])?;
                    if dur != out_samples as i32 {
                        return Err("random packet last packet duration mismatch".to_string());
                    }
                }
            }

            let plen = ((rng.next() % 18 + 3) * 8) as usize + skip + 3;
            let t = (rng.next() & 3) as usize;
            for i in 0..4096usize {
                packet[0] = modes[i] << 2;
                let expected = opus_decoder_get_nb_samples(&decs[t], &packet, plen)
                    .map_err(|err| format!("get nb samples failed: {err:?}"))?;
                for _ in 0..10 {
                    set_random_complexity(&mut decs[t], rng)?;
                    for j in 0..plen {
                        packet[j + 1] = ((rng.next() | rng.next()) & 255) as u8;
                    }
                    let out_samples = decode_i16(
                        &mut decs[t],
                        Some(&packet[..plen + 1]),
                        plen + 1,
                        outbuf,
                        MAX_FRAME_SAMP,
                        false,
                        "random packets",
                    )?;
                    if out_samples != expected {
                        return Err("random packet sample count mismatch".to_string());
                    }
                }
            }

            let tmodes = [25u8 << 2];
            let tseeds = [140441u32];
            let tlen = [157usize];
            let tret = [480usize];
            let t = (rng.next() & 1) as usize;
            for i in 0..1usize {
                packet[0] = tmodes[i];
                let mut local_rng = FastRand::new(tseeds[i]);
                for j in 1..tlen[i] {
                    packet[j] = (local_rng.next() & 255) as u8;
                }
                let out_samples = decode_i16(
                    &mut decs[t],
                    Some(&packet[..tlen[i]]),
                    tlen[i],
                    outbuf,
                    MAX_FRAME_SAMP,
                    false,
                    "preselected packets",
                )?;
                if out_samples != tret[i] {
                    return Err("preselected packet sample count mismatch".to_string());
                }
            }
        }
    }

    let prefix_ok = outbuf_int[..guard_offset]
        .iter()
        .all(|&sample| sample == 32749);
    let suffix_ok = outbuf_int[guard_offset + outbuf_len..]
        .iter()
        .all(|&sample| sample == 32749);
    if !prefix_ok || !suffix_ok {
        return Err("guard samples were modified".to_string());
    }

    Ok(())
}

#[test]
fn opus_decode_code0() {
    let seed = seed_from_env();
    let mut rng = FastRand::new(seed);
    test_decoder_code0(no_fuzz(), &mut rng)
        .unwrap_or_else(|err| panic!("{err} (seed={seed})"));
}

#[test]
fn opus_decode_soft_clip() {
    let mut x = vec![0f32; 1024];
    let mut s = [0f32; 8];

    for i in 0..1024 {
        for (j, sample) in x.iter_mut().enumerate() {
            *sample = ((j & 255) as f32) * (1.0 / 32.0) - 4.0;
        }
        opus_pcm_soft_clip(&mut x[i..], 1024 - i, 1, &mut s);
        for &sample in &x[i..] {
            if !(sample <= 1.0 && sample >= -1.0) {
                panic!("soft clip exceeded bounds");
            }
        }
    }

    for i in 1..9usize {
        for (j, sample) in x.iter_mut().enumerate() {
            *sample = ((j & 255) as f32) * (1.0 / 32.0) - 4.0;
        }
        let frame_size = 1024 / i;
        opus_pcm_soft_clip(&mut x, frame_size, i, &mut s);
        for &sample in &x[..frame_size * i] {
            if !(sample <= 1.0 && sample >= -1.0) {
                panic!("soft clip exceeded bounds");
            }
        }
    }

    opus_pcm_soft_clip(&mut x, 0, 1, &mut s);
    opus_pcm_soft_clip(&mut x, 1, 0, &mut s);
    opus_pcm_soft_clip(&mut x, 1, 1, &mut []);
    opus_pcm_soft_clip(&mut x, 1, 9, &mut s);
    opus_pcm_soft_clip(&mut [], 1, 1, &mut s);
}
