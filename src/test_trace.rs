#[cfg(test)]
extern crate std;

#[cfg(test)]
use core::fmt;
#[cfg(test)]
use std::io::{self, Write};

#[cfg(test)]
#[inline]
pub(crate) fn emit_stdout(args: fmt::Arguments<'_>) {
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_fmt(args);
    let _ = stdout.write_all(b"\n");
}

#[allow(unused_macros)]
macro_rules! trace_println {
    ($($arg:tt)*) => {{
        #[cfg(test)]
        {
            $crate::test_trace::emit_stdout(format_args!($($arg)*));
        }
        #[cfg(not(test))]
        {
            let _ = format_args!($($arg)*);
        }
    }};
}

#[allow(unused_imports)]
pub(crate) use trace_println;
