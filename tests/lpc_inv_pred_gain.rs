use mousiki::silk::lpc_inv_pred_gain::{SILK_MAX_ORDER_LPC, lpc_inverse_pred_gain};

// The C harness in `silk/tests/test_unit_LPC_inv_pred_gain.c` runs 10,000
// iterations; we scale that back to keep the Rust test suite quick while still
// exercising a wide spread of random predictors.
const LOOP_COUNT: usize = 250;

fn lcg_rand(seed: &mut u32) -> i16 {
    const RAND_MULTIPLIER: u32 = 1_103_515_245;
    const RAND_INCREMENT: u32 = 12_345;
    *seed = seed
        .wrapping_mul(RAND_MULTIPLIER)
        .wrapping_add(RAND_INCREMENT);
    // Mirror the positive 31-bit output of C's rand() and keep the low 16 bits.
    ((*seed >> 1) & 0x7FFF_FFFF) as i16
}

fn check_stability(coeffs: &[i16]) -> bool {
    let mut sum_a = 0i32;
    let mut sum_abs_a = 0i32;

    for &coeff in coeffs {
        sum_a += i32::from(coeff);
        sum_abs_a += i32::from(coeff).abs();
    }

    if sum_a >= 4096 {
        return false;
    }
    if sum_abs_a < 4096 {
        return true;
    }

    let mut y = [0.0f64; SILK_MAX_ORDER_LPC];
    y[0] = 1.0;

    for i in 0..10_000 {
        let mut sum = 0.0;
        for (j, &coeff) in coeffs.iter().enumerate() {
            sum += y[j] * f64::from(coeff);
        }
        for j in (1..coeffs.len()).rev() {
            y[j] = y[j - 1];
        }
        y[0] = sum / 4096.0;

        if !(y[0] < 10_000.0 && y[0] > -10_000.0) {
            return false;
        }

        if i & 7 == 0 {
            let amp: f64 = y[..coeffs.len()].iter().map(|v| v.abs()).sum();
            if amp < 0.00001 {
                return true;
            }
        }
    }
    true
}

#[test]
fn lpc_inverse_pred_gain_rejects_unstable_filters() {
    let mut seed = 0u32;

    for iteration in 0..LOOP_COUNT {
        for order in (2..=SILK_MAX_ORDER_LPC).step_by(2) {
            for shift in 0..16 {
                let mut coeffs = [0i16; SILK_MAX_ORDER_LPC];
                for coeff in coeffs.iter_mut().take(order) {
                    *coeff = lcg_rand(&mut seed) >> shift;
                }

                let gain = lpc_inverse_pred_gain(&coeffs[..order]);
                if gain != 0 && !check_stability(&coeffs[..order]) {
                    panic!(
                        "Unstable predictor survived (iter {iteration}, order {order}, shift {shift}, seed {seed:08x})"
                    );
                }
            }
        }
    }
}
