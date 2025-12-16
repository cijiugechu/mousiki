use core::convert::TryFrom;

use crate::packet::MAX_FRAMES_PER_PACKET;

/// Errors surfaced by the extension helpers, mirroring the C API codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionError {
    BadArgument,
    BufferTooSmall,
    InvalidPacket,
}

impl ExtensionError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            ExtensionError::BadArgument => -1,
            ExtensionError::BufferTooSmall => -2,
            ExtensionError::InvalidPacket => -4,
        }
    }
}

/// Extension descriptor used by the Opus padding helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpusExtensionData<'a> {
    pub id: u8,
    pub frame: i32,
    pub data: &'a [u8],
    pub len: i32,
}

fn skip_extension_payload(
    data: &[u8],
    mut pos: usize,
    mut len: i32,
    trailing_short_len: usize,
    id_byte: u8,
    header_size: &mut usize,
) -> Result<(usize, i32), ExtensionError> {
    *header_size = 0;

    let id = id_byte >> 1;
    let l_flag = id_byte & 1;

    if (id == 0 && l_flag == 1) || id == 2 {
        return Ok((pos, len));
    }

    if id > 0 && id < 32 {
        let need = usize::from(l_flag);
        if len < need as i32 {
            return Err(ExtensionError::InvalidPacket);
        }
        pos = pos.checked_add(need).ok_or(ExtensionError::InvalidPacket)?;
        len -= need as i32;
        return Ok((pos, len));
    }

    if l_flag == 0 {
        if len < trailing_short_len as i32 {
            return Err(ExtensionError::InvalidPacket);
        }
        let advance = (len as usize)
            .checked_sub(trailing_short_len)
            .ok_or(ExtensionError::InvalidPacket)?;
        pos = pos
            .checked_add(advance)
            .ok_or(ExtensionError::InvalidPacket)?;
        len = trailing_short_len as i32;
    } else {
        let mut bytes = 0usize;
        loop {
            if len < 1 {
                return Err(ExtensionError::InvalidPacket);
            }
            let lacing = *data.get(pos).ok_or(ExtensionError::InvalidPacket)? as usize;
            pos += 1;
            *header_size += 1;
            len -= 1;
            bytes = bytes
                .checked_add(lacing)
                .ok_or(ExtensionError::InvalidPacket)?;
            len = len
                .checked_sub(lacing as i32)
                .ok_or(ExtensionError::InvalidPacket)?;
            if lacing != 255 {
                break;
            }
        }
        if bytes > data.len().saturating_sub(pos) {
            return Err(ExtensionError::InvalidPacket);
        }
        pos += bytes;
    }

    Ok((pos, len))
}

fn skip_extension(
    data: &[u8],
    pos: usize,
    len: i32,
    header_size: &mut usize,
) -> Result<(usize, i32), ExtensionError> {
    if len == 0 {
        *header_size = 0;
        return Ok((pos, 0));
    }
    if len < 1 {
        return Err(ExtensionError::InvalidPacket);
    }
    let id_byte = *data.get(pos).ok_or(ExtensionError::InvalidPacket)?;
    let (pos, remaining) = skip_extension_payload(data, pos + 1, len - 1, 0, id_byte, header_size)?;
    *header_size += 1;
    Ok((pos, remaining))
}

pub struct OpusExtensionIterator<'a> {
    data: &'a [u8],
    curr_pos: usize,
    repeat_start: usize,
    last_long: Option<usize>,
    src_pos: usize,
    curr_len: i32,
    repeat_len: i32,
    src_len: i32,
    trailing_short_len: usize,
    nb_frames: usize,
    frame_max: usize,
    curr_frame: usize,
    repeat_frame: usize,
    repeat_l: u8,
}

impl<'a> OpusExtensionIterator<'a> {
    pub fn new(data: &'a [u8], nb_frames: usize) -> Self {
        assert!(nb_frames <= MAX_FRAMES_PER_PACKET);
        let len = i32::try_from(data.len()).expect("extension padding length fits in i32");
        Self {
            data,
            curr_pos: 0,
            repeat_start: 0,
            last_long: None,
            src_pos: 0,
            curr_len: len,
            repeat_len: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
        }
    }

    pub fn reset(&mut self) {
        self.curr_pos = 0;
        self.repeat_start = 0;
        self.last_long = None;
        self.src_pos = 0;
        self.curr_len =
            i32::try_from(self.data.len()).expect("extension padding length fits in i32");
        self.repeat_len = 0;
        self.src_len = 0;
        self.trailing_short_len = 0;
        self.curr_frame = 0;
        self.repeat_frame = 0;
        self.repeat_l = 0;
    }

    pub fn set_frame_max(&mut self, frame_max: usize) {
        self.frame_max = frame_max;
    }

    fn next_repeat(&mut self) -> Result<Option<OpusExtensionData<'a>>, ExtensionError> {
        debug_assert!(self.repeat_frame > 0);
        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let mut header_size = 0usize;
                let repeat_id_byte = *self
                    .data
                    .get(self.src_pos)
                    .ok_or(ExtensionError::InvalidPacket)?;
                let (new_src_pos, new_src_len) =
                    skip_extension(self.data, self.src_pos, self.src_len, &mut header_size)?;
                self.src_pos = new_src_pos;
                self.src_len = new_src_len;
                if repeat_id_byte <= 3 {
                    continue;
                }
                let mut adjusted_id_byte = repeat_id_byte;
                if self.repeat_l == 0
                    && self.repeat_frame + 1 >= self.nb_frames
                    && Some(self.src_pos) == self.last_long
                {
                    adjusted_id_byte &= !1;
                }
                let curr_start = self.curr_pos;
                let (new_curr_pos, new_curr_len) = skip_extension_payload(
                    self.data,
                    self.curr_pos,
                    self.curr_len,
                    self.trailing_short_len,
                    adjusted_id_byte,
                    &mut header_size,
                )?;
                self.curr_pos = new_curr_pos;
                self.curr_len = new_curr_len;
                if self.curr_len < 0 {
                    return Err(ExtensionError::InvalidPacket);
                }
                if self.repeat_frame >= self.frame_max {
                    continue;
                }
                let payload_start = curr_start
                    .checked_add(header_size)
                    .ok_or(ExtensionError::InvalidPacket)?;
                if payload_start > self.curr_pos {
                    return Err(ExtensionError::InvalidPacket);
                }
                let payload_len = self.curr_pos - payload_start;
                let len = i32::try_from(payload_len).map_err(|_| ExtensionError::InvalidPacket)?;
                let frame =
                    i32::try_from(self.repeat_frame).map_err(|_| ExtensionError::InvalidPacket)?;
                return Ok(Some(OpusExtensionData {
                    id: adjusted_id_byte >> 1,
                    frame,
                    data: &self.data[payload_start..self.curr_pos],
                    len,
                }));
            }
            self.src_pos = self.repeat_start;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }
        self.repeat_start = self.curr_pos;
        self.last_long = None;
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        Ok(None)
    }

    pub fn next_extension(&mut self) -> Result<Option<OpusExtensionData<'a>>, ExtensionError> {
        if self.curr_len < 0 {
            return Err(ExtensionError::InvalidPacket);
        }

        if self.repeat_frame > 0 && let Some(ext) = self.next_repeat()? {
            return Ok(Some(ext));
        }

        if self.curr_frame >= self.frame_max {
            return Ok(None);
        }

        while self.curr_len > 0 {
            let curr_data0 = self.curr_pos;
            let id_byte = *self
                .data
                .get(curr_data0)
                .ok_or(ExtensionError::InvalidPacket)?;
            let id = id_byte >> 1;
            let l_flag = id_byte & 1;
            let mut header_size = 0usize;
            let (new_pos, new_len) =
                skip_extension(self.data, self.curr_pos, self.curr_len, &mut header_size)?;
            self.curr_pos = new_pos;
            self.curr_len = new_len;
            if self.curr_len < 0 {
                return Err(ExtensionError::InvalidPacket);
            }

            if id == 1 {
                if l_flag == 0 {
                    self.curr_frame += 1;
                } else {
                    let incr = *self
                        .data
                        .get(curr_data0 + 1)
                        .ok_or(ExtensionError::InvalidPacket)?;
                    if incr == 0 {
                        continue;
                    }
                    self.curr_frame = self
                        .curr_frame
                        .checked_add(incr as usize)
                        .ok_or(ExtensionError::InvalidPacket)?;
                }
                if self.curr_frame >= self.nb_frames {
                    self.curr_len = -1;
                    return Err(ExtensionError::InvalidPacket);
                }
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_start = self.curr_pos;
                self.last_long = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                self.repeat_l = l_flag;
                self.repeat_frame = self.curr_frame + 1;
                let repeat_len = curr_data0
                    .checked_sub(self.repeat_start)
                    .ok_or(ExtensionError::InvalidPacket)?;
                self.repeat_len =
                    i32::try_from(repeat_len).map_err(|_| ExtensionError::InvalidPacket)?;
                self.src_pos = self.repeat_start;
                self.src_len = self.repeat_len;
                if let Some(ext) = self.next_repeat()? {
                    return Ok(Some(ext));
                }
            } else if id > 2 {
                if id >= 32 {
                    self.last_long = Some(self.curr_pos);
                    self.trailing_short_len = 0;
                } else {
                    self.trailing_short_len = self
                        .trailing_short_len
                        .checked_add(l_flag as usize)
                        .ok_or(ExtensionError::InvalidPacket)?;
                }
                if self.curr_frame >= self.frame_max {
                    continue;
                }
                let data_start = curr_data0
                    .checked_add(header_size)
                    .ok_or(ExtensionError::InvalidPacket)?;
                if data_start > self.curr_pos {
                    return Err(ExtensionError::InvalidPacket);
                }
                let payload_len = self.curr_pos - data_start;
                let len = i32::try_from(payload_len).map_err(|_| ExtensionError::InvalidPacket)?;
                let frame =
                    i32::try_from(self.curr_frame).map_err(|_| ExtensionError::InvalidPacket)?;
                return Ok(Some(OpusExtensionData {
                    id,
                    frame,
                    data: &self.data[data_start..self.curr_pos],
                    len,
                }));
            }
        }

        Ok(None)
    }

    pub fn find(&mut self, id: u8) -> Result<Option<OpusExtensionData<'a>>, ExtensionError> {
        loop {
            let Some(ext) = self.next_extension()? else {
                return Ok(None);
            };
            if ext.id == id {
                return Ok(Some(ext));
            }
        }
    }
}

pub fn opus_packet_extensions_count(
    data: &[u8],
    len: usize,
    nb_frames: usize,
) -> Result<usize, ExtensionError> {
    assert!(len <= data.len());
    let bounded = &data[..len];
    let mut iter = OpusExtensionIterator::new(bounded, nb_frames);
    let mut count = 0usize;
    loop {
        match iter.next_extension() {
            Ok(Some(_)) => count += 1,
            Ok(None) | Err(_) => break Ok(count),
        }
    }
}

pub fn opus_packet_extensions_count_ext(
    data: &[u8],
    len: usize,
    nb_frame_exts: &mut [i32],
    nb_frames: usize,
) -> Result<usize, ExtensionError> {
    assert!(len <= data.len());
    assert!(nb_frame_exts.len() >= nb_frames);
    let bounded = &data[..len];
    let mut iter = OpusExtensionIterator::new(bounded, nb_frames);
    for slot in nb_frame_exts.iter_mut().take(nb_frames) {
        *slot = 0;
    }
    let mut count = 0usize;
    loop {
        match iter.next_extension() {
            Ok(Some(ext)) => {
                nb_frame_exts[ext.frame as usize] += 1;
                count += 1;
            }
            Ok(None) | Err(_) => break Ok(count),
        }
    }
}

pub fn opus_packet_extensions_parse<'a>(
    data: &'a [u8],
    len: usize,
    nb_frames: usize,
    extensions: &mut [OpusExtensionData<'a>],
) -> Result<usize, ExtensionError> {
    assert!(len <= data.len());
    let bounded = &data[..len];
    let mut iter = OpusExtensionIterator::new(bounded, nb_frames);
    let mut count = 0usize;
    while let Some(ext) = iter.next_extension()? {
        if count == extensions.len() {
            return Err(ExtensionError::BufferTooSmall);
        }
        extensions[count] = ext;
        count += 1;
    }
    Ok(count)
}

pub fn opus_packet_extensions_parse_ext<'a>(
    data: &'a [u8],
    len: usize,
    nb_frames: usize,
    extensions: &mut [OpusExtensionData<'a>],
    nb_frame_exts: &[i32],
) -> Result<usize, ExtensionError> {
    assert!(len <= data.len());
    assert!(nb_frames <= MAX_FRAMES_PER_PACKET);
    assert!(nb_frame_exts.len() >= nb_frames);
    let bounded = &data[..len];
    let mut nb_frames_cum = [0usize; MAX_FRAMES_PER_PACKET + 1];
    let mut prev_total = 0usize;
    for (dst, &val) in nb_frames_cum
        .iter_mut()
        .take(nb_frames)
        .zip(nb_frame_exts.iter())
    {
        let count = usize::try_from(val).map_err(|_| ExtensionError::BadArgument)?;
        *dst = prev_total;
        prev_total = prev_total
            .checked_add(count)
            .ok_or(ExtensionError::InvalidPacket)?;
    }
    nb_frames_cum[nb_frames] = prev_total;

    let mut iter = OpusExtensionIterator::new(bounded, nb_frames);
    let mut count = 0usize;
    while let Some(ext) = iter.next_extension()? {
        let frame = ext.frame as usize;
        let idx = nb_frames_cum[frame];
        if idx >= extensions.len() {
            return Err(ExtensionError::BufferTooSmall);
        }
        debug_assert!(idx < nb_frames_cum[frame + 1]);
        extensions[idx] = ext;
        nb_frames_cum[frame] = idx + 1;
        count += 1;
    }
    Ok(count)
}

fn write_extension_payload(
    mut data: Option<&mut [u8]>,
    len: usize,
    pos: usize,
    ext: &OpusExtensionData<'_>,
    last: bool,
) -> Result<usize, ExtensionError> {
    if ext.id < 32 {
        debug_assert!(ext.len >= 0 && ext.len <= 1);
        let need = ext.len as usize;
        if len.saturating_sub(pos) < need {
            return Err(ExtensionError::BufferTooSmall);
        }
        if let Some(buf) = data.as_mut().filter(|_| need > 0) {
            buf[pos] = ext
                .data
                .first()
                .copied()
                .ok_or(ExtensionError::BadArgument)?;
        }
        Ok(pos + need)
    } else {
        let ext_len = usize::try_from(ext.len).map_err(|_| ExtensionError::BadArgument)?;
        let mut length_bytes = 1 + ext_len / 255;
        if last {
            length_bytes = 0;
        }
        let available = len.checked_sub(pos).ok_or(ExtensionError::BufferTooSmall)?;
        if available < length_bytes + ext_len {
            return Err(ExtensionError::BufferTooSmall);
        }
        if let Some(buf) = data.as_mut().filter(|_| !last) {
            for byte in buf.iter_mut().skip(pos).take(ext_len / 255) {
                *byte = 255;
            }
            if length_bytes > 0 {
                buf[pos + length_bytes - 1] = (ext_len % 255) as u8;
            }
        }
        if let Some(buf) = data.as_mut() {
            buf[pos + length_bytes..pos + length_bytes + ext_len]
                .copy_from_slice(ext.data.get(..ext_len).ok_or(ExtensionError::BadArgument)?);
        }
        Ok(pos + length_bytes + ext_len)
    }
}

#[allow(clippy::needless_option_as_deref)]
fn write_extension(
    mut data: Option<&mut [u8]>,
    len: usize,
    pos: usize,
    ext: &OpusExtensionData<'_>,
    last: bool,
) -> Result<usize, ExtensionError> {
    if len.saturating_sub(pos) < 1 {
        return Err(ExtensionError::BufferTooSmall);
    }
    let mut pos = pos;
    let l_flag = if ext.id < 32 {
        ext.len as u8
    } else {
        u8::from(!last)
    };
    if let Some(buf) = data.as_mut() {
        buf[pos] = (ext.id << 1) + l_flag;
    }
    pos += 1;
    write_extension_payload(data, len, pos, ext, last)
}

#[allow(clippy::needless_option_as_deref)]
pub fn opus_packet_extensions_generate(
    mut data: Option<&mut [u8]>,
    len: usize,
    extensions: &[OpusExtensionData<'_>],
    nb_frames: usize,
    pad: bool,
) -> Result<usize, ExtensionError> {
    if nb_frames > MAX_FRAMES_PER_PACKET {
        return Err(ExtensionError::BadArgument);
    }
    if data.as_ref().is_some_and(|buf| len > buf.len()) {
        return Err(ExtensionError::BadArgument);
    }

    let nb_extensions = extensions.len();
    let mut frame_min_idx = [nb_extensions; MAX_FRAMES_PER_PACKET];
    let mut frame_max_idx = [0usize; MAX_FRAMES_PER_PACKET];

    for frame in 0..nb_frames {
        frame_min_idx[frame] = nb_extensions;
        frame_max_idx[frame] = 0;
    }

    for (i, ext) in extensions.iter().enumerate() {
        if !(3..=127).contains(&ext.id) {
            return Err(ExtensionError::BadArgument);
        }
        if ext.frame < 0 || ext.frame as usize >= nb_frames {
            return Err(ExtensionError::BadArgument);
        }
        if ext.id < 32 {
            if ext.len < 0 || ext.len > 1 {
                return Err(ExtensionError::BadArgument);
            }
        } else if ext.len < 0 {
            return Err(ExtensionError::BadArgument);
        }
        if ext.len > 0 && ext.data.len() < ext.len as usize {
            return Err(ExtensionError::BadArgument);
        }

        let frame = ext.frame as usize;
        frame_min_idx[frame] = frame_min_idx[frame].min(i);
        frame_max_idx[frame] = frame_max_idx[frame].max(i + 1);
    }

    let mut frame_repeat_idx = frame_min_idx;
    let mut curr_frame = 0usize;
    let mut pos = 0usize;
    let mut written = 0usize;

    for f in 0..nb_frames {
        let mut last_long_idx: Option<usize> = None;
        let mut _trailing_short_len = 0usize;
        let mut repeat_count = 0usize;

        if f + 1 < nb_frames {
            for i in frame_min_idx[f]..frame_max_idx[f] {
                if extensions[i].frame as usize == f {
                    let mut g = f + 1;
                    while g < nb_frames {
                        if frame_repeat_idx[g] >= frame_max_idx[g] {
                            break;
                        }
                        debug_assert_eq!(extensions[frame_repeat_idx[g]].frame as usize, g);
                        if extensions[frame_repeat_idx[g]].id != extensions[i].id {
                            break;
                        }
                        if extensions[frame_repeat_idx[g]].id < 32
                            && extensions[frame_repeat_idx[g]].len != extensions[i].len
                        {
                            break;
                        }
                        g += 1;
                    }
                    if g < nb_frames {
                        break;
                    }
                    if extensions[i].id >= 32 {
                        last_long_idx = Some(frame_repeat_idx[nb_frames - 1]);
                        _trailing_short_len = 0;
                    } else {
                        _trailing_short_len += extensions[i].len as usize;
                    }
                    for g in f + 1..nb_frames {
                        let mut j = frame_repeat_idx[g].saturating_add(1);
                        while j < frame_max_idx[g] && extensions[j].frame as usize != g {
                            j += 1;
                        }
                        frame_repeat_idx[g] = j;
                    }
                    repeat_count += 1;
                    frame_repeat_idx[f] = i;
                }
            }
        }

        for i in frame_min_idx[f]..frame_max_idx[f] {
            if extensions[i].frame as usize == f {
                if f != curr_frame {
                    let diff = f - curr_frame;
                    if len.saturating_sub(pos) < 2 {
                        return Err(ExtensionError::BufferTooSmall);
                    }
                    if diff == 1 {
                        if let Some(buf) = data.as_deref_mut() {
                            buf[pos] = 0x02;
                        }
                        pos += 1;
                    } else {
                        if let Some(buf) = data.as_deref_mut() {
                            buf[pos] = 0x03;
                        }
                        pos += 1;
                        if len.saturating_sub(pos) < 1 {
                            return Err(ExtensionError::BufferTooSmall);
                        }
                        if let Some(buf) = data.as_deref_mut() {
                            buf[pos] = diff as u8;
                        }
                        pos += 1;
                    }
                    curr_frame = f;
                }

                pos = write_extension(
                    data.as_deref_mut(),
                    len,
                    pos,
                    &extensions[i],
                    written + 1 == nb_extensions,
                )?;
                written += 1;

                if repeat_count > 0 && frame_repeat_idx[f] == i {
                    let nb_repeated = repeat_count * (nb_frames - (f + 1));
                    let last = written + nb_repeated == nb_extensions
                        || (last_long_idx.is_none() && i + 1 >= frame_max_idx[f]);
                    if len.saturating_sub(pos) < 1 {
                        return Err(ExtensionError::BufferTooSmall);
                    }
                    if let Some(buf) = data.as_deref_mut() {
                        buf[pos] = 0x04 + u8::from(!last);
                    }
                    pos += 1;
                    for g in f + 1..nb_frames {
                        let mut j = frame_min_idx[g];
                        while j < frame_repeat_idx[g] {
                            if extensions[j].frame as usize == g {
                                pos = write_extension_payload(
                                    data.as_deref_mut(),
                                    len,
                                    pos,
                                    &extensions[j],
                                    last && Some(j) == last_long_idx,
                                )?;
                                written += 1;
                            }
                            j += 1;
                        }
                        frame_min_idx[g] = frame_repeat_idx[g];
                    }
                    if last {
                        curr_frame += 1;
                    }
                }
            }
        }
    }

    debug_assert_eq!(written, nb_extensions);

    if pad && pos < len {
        let padding = len - pos;
        if let Some(buf) = data.as_deref_mut() {
            buf.copy_within(0..pos, padding);
            for byte in buf.iter_mut().take(padding) {
                *byte = 0x01;
            }
        }
        pos += padding;
    }

    Ok(pos)
}
