extern crate std;

use core::fmt;
use std::io::{self, Write};

#[inline]
pub(crate) fn emit_stdout(args: fmt::Arguments<'_>) {
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_fmt(args);
    let _ = stdout.write_all(b"\n");
}
