//! Static PVQ combinatorics table matching the reference non-small-footprint CELT path.
//!
//! The reference decoder indexes `CELT_PVQ_U_ROW[n][k]` directly with the real
//! `k` column, so each row pointer is intentionally biased into the flattened
//! storage. We mirror that layout here and build the table at compile time,
//! which keeps the Rust decode path aligned with the C implementation's
//! zero-allocation fast path without hand-maintaining a 1,488-entry literal.

const MAX_PVQ_DIMENSION: usize = 208;
const MAX_PVQ_ROW: usize = 14;

pub(super) const CELT_PVQ_U_ROW_OFFSETS: [usize; 15] = [
    0, 208, 415, 621, 826, 1030, 1233, 1336, 1389, 1421, 1441, 1455, 1464, 1470, 1473,
];

const PVQ_U_ROW_LENGTHS: [usize; 15] = [
    209, 208, 207, 206, 205, 204, 104, 54, 33, 21, 15, 10, 7, 4, 1,
];

const fn build_pvq_u_data() -> [u32; 1488] {
    let mut table = [[0u64; MAX_PVQ_DIMENSION + 1]; MAX_PVQ_ROW + 1];
    table[0][0] = 1;

    let mut n = 1usize;
    while n <= MAX_PVQ_ROW {
        let mut k = 1usize;
        while k <= MAX_PVQ_DIMENSION {
            table[n][k] = table[n - 1][k]
                .saturating_add(table[n][k - 1])
                .saturating_add(table[n - 1][k - 1]);
            k += 1;
        }
        n += 1;
    }

    let mut data = [0u32; 1488];
    let mut out_index = 0usize;
    let mut row = 0usize;
    while row < PVQ_U_ROW_LENGTHS.len() {
        let end_col = row + PVQ_U_ROW_LENGTHS[row];
        let mut col = row;
        while col < end_col {
            data[out_index] = table[row][col] as u32;
            out_index += 1;
            col += 1;
        }
        row += 1;
    }

    data
}

pub(super) const CELT_PVQ_U_ROW_LENGTHS: [usize; 15] = PVQ_U_ROW_LENGTHS;
pub(super) const CELT_PVQ_U_DATA: [u32; 1488] = build_pvq_u_data();
