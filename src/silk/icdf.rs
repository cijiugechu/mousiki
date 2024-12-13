#[derive(Debug, Clone, Copy)]
pub struct ICDFContext {
    pub total: u32,
    /// cumulative distribution table
    pub dist_table: &'static [usize],
}

#[macro_export]
macro_rules! icdf {
  ($total: expr; $($dist: expr), +) => {
      ICDFContext {
          total: $total,
          dist_table: &[$($dist), +]
      }
  };
}
