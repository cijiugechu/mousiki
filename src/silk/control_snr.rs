//! Port of `silk_control_SNR` from `silk/control_SNR.c`.
//!
//! The helper maps the requested target bitrate to an SNR tuning parameter
//! using empirically derived lookup tables.  The SILK encoder later uses this
//! Q7 value when choosing gains for the residual quantiser.

use crate::silk::MAX_NB_SUBFR;
use crate::silk::encoder::EncoderChannelState;
use crate::silk::errors::SilkError;

/// Target-rate lookup table for narrowband (fs = 8 kHz) divided by 21.
const TARGET_RATE_NB_DIV21: [u8; 107] = [
    0, 15, 39, 52, 61, 68, 74, 79, 84, 88, 92, 95, 99, 102, 105, 108, 111, 114, 117, 119, 122, 124,
    126, 129, 131, 133, 135, 137, 139, 142, 143, 145, 147, 149, 151, 153, 155, 157, 158, 160, 162,
    163, 165, 167, 168, 170, 171, 173, 174, 176, 177, 179, 180, 182, 183, 185, 186, 187, 189, 190,
    192, 193, 194, 196, 197, 199, 200, 201, 203, 204, 205, 207, 208, 209, 211, 212, 213, 215, 216,
    217, 219, 220, 221, 223, 224, 225, 227, 228, 230, 231, 232, 234, 235, 236, 238, 239, 241, 242,
    243, 245, 246, 248, 249, 250, 252, 253, 255,
];

/// Target-rate lookup table for medium-band (fs = 12 kHz) divided by 21.
const TARGET_RATE_MB_DIV21: [u8; 155] = [
    0, 0, 28, 43, 52, 59, 65, 70, 74, 78, 81, 85, 87, 90, 93, 95, 98, 100, 102, 105, 107, 109, 111,
    113, 115, 116, 118, 120, 122, 123, 125, 127, 128, 130, 131, 133, 134, 136, 137, 138, 140, 141,
    143, 144, 145, 147, 148, 149, 151, 152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165,
    166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
    186, 187, 188, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 236, 237,
    238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
];

/// Target-rate lookup table for wideband (fs >= 16 kHz) divided by 21.
const TARGET_RATE_WB_DIV21: [u8; 191] = [
    0, 0, 0, 8, 29, 41, 49, 56, 62, 66, 70, 74, 77, 80, 83, 86, 88, 91, 93, 95, 97, 99, 101, 103,
    105, 107, 108, 110, 112, 113, 115, 116, 118, 119, 121, 122, 123, 125, 126, 127, 129, 130, 131,
    132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 156, 157, 158, 159, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
    171, 172, 173, 174, 175, 176, 177, 177, 178, 179, 180, 181, 181, 182, 183, 184, 185, 185, 186,
    187, 188, 189, 189, 190, 191, 192, 192, 193, 194, 195, 195, 196, 197, 198, 198, 199, 200, 200,
    201, 202, 203, 203, 204, 205, 206, 206, 207, 208, 209, 209, 210, 211, 211, 212, 213, 214, 214,
    215, 216, 216, 217, 218, 219, 219, 220, 221, 221, 222, 223, 224, 224, 225, 226, 226, 227, 228,
    229, 229, 230, 231, 232, 232, 233, 234, 234, 235, 236, 237, 237, 238, 239, 240, 240, 241, 242,
    243, 243, 244, 245, 246, 246, 247, 248, 249, 249, 250, 251, 252, 253, 255,
];

/// Mirrors `silk_control_SNR`.
pub fn control_snr(
    channel: &mut EncoderChannelState,
    target_rate_bps: i32,
) -> Result<(), SilkError> {
    let (fs_khz, nb_subfr) = {
        let common = channel.common();
        (common.fs_khz, common.nb_subfr)
    };

    let mut adjusted_rate = target_rate_bps;
    if nb_subfr == MAX_NB_SUBFR / 2 {
        adjusted_rate -= 2000 + fs_khz / 16;
    }

    let (table, bound) = match fs_khz {
        8 => (&TARGET_RATE_NB_DIV21[..], TARGET_RATE_NB_DIV21.len()),
        12 => (&TARGET_RATE_MB_DIV21[..], TARGET_RATE_MB_DIV21.len()),
        _ => (&TARGET_RATE_WB_DIV21[..], TARGET_RATE_WB_DIV21.len()),
    };

    let mut id = (adjusted_rate + 200) / 400;
    id = (id - 10).min(bound as i32 - 1);
    let snr_db_q7 = if id <= 0 {
        0
    } else {
        i32::from(table[id as usize]) * 21
    };

    let common = channel.common_mut();
    common.target_rate_bps = target_rate_bps;
    common.snr_db_q7 = snr_db_q7;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::control_snr;
    use crate::silk::MAX_NB_SUBFR;
    use crate::silk::encoder::EncoderChannelState;

    #[test]
    fn follows_nb_lookup_table() {
        let mut channel = EncoderChannelState::default();
        {
            let common = channel.common_mut();
            common.fs_khz = 8;
            common.nb_subfr = MAX_NB_SUBFR;
        }
        control_snr(&mut channel, 8_000).unwrap();
        assert_eq!(channel.common().snr_db_q7, 92 * 21);
    }

    #[test]
    fn accounts_for_two_subframe_mode() {
        let mut channel = EncoderChannelState::default();
        {
            let common = channel.common_mut();
            common.fs_khz = 12;
            common.nb_subfr = MAX_NB_SUBFR / 2;
        }
        control_snr(&mut channel, 12_000).unwrap();
        assert_eq!(channel.common().target_rate_bps, 12_000);
        assert_eq!(channel.common().snr_db_q7, 95 * 21);
    }

    #[test]
    fn clamps_low_rates_to_zero() {
        let mut channel = EncoderChannelState::default();
        control_snr(&mut channel, 1_000).unwrap();
        assert_eq!(channel.common().snr_db_q7, 0);
    }

    #[test]
    fn saturates_to_highest_table_entry() {
        let mut channel = EncoderChannelState::default();
        control_snr(&mut channel, 1_000_000).unwrap();
        assert_eq!(channel.common().snr_db_q7, 255 * 21);
    }
}
