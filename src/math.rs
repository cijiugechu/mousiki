/// Computes the integer logarithm base 2 of a value
/// This is equivalent to floor(log2(x))
pub(crate) fn ilog(x: isize) -> isize {
    if x <= 0 {
        return 0;
    }
    64 - x.leading_zeros() as isize
}
