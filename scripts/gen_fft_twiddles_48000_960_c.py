#!/usr/bin/env python3
import re
from pathlib import Path

SOURCE = Path("opus-c/celt/static_modes_float.h")
DEST = Path("ctests/fft_twiddles_48000_960.h")

text = SOURCE.read_text(encoding="utf-8")
match = re.search(
    r"fft_twiddles48000_960\\[480\\]\\s*=\\s*\\{(.*?)\\};",
    text,
    re.S,
)
if not match:
    raise SystemExit("fft_twiddles48000_960 table not found")

body = match.group(1).strip()

lines = [
    "/*",
    " * Source: opus-c/celt/static_modes_float.h (fft_twiddles48000_960).",
    " * Regenerate with: python3 scripts/gen_fft_twiddles_48000_960_c.py",
    " */",
    "#ifndef CTESTS_FFT_TWIDDLES_48000_960_H",
    "#define CTESTS_FFT_TWIDDLES_48000_960_H",
    "",
    "static const kiss_twiddle_cpx fft_twiddles48000_960[480] = {",
    body,
    "};",
    "",
    "#endif",
]

DEST.write_text("\n".join(lines), encoding="utf-8")
