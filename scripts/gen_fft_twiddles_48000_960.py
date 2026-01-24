#!/usr/bin/env python3
import re
from pathlib import Path

SOURCE = Path("opus-c/celt/static_modes_float.h")
DEST = Path("src/celt/fft_twiddles_48000_960.rs")

text = SOURCE.read_text(encoding="utf-8")
match = re.search(
    r"fft_twiddles48000_960\[480\]\s*=\s*\{(.*?)\};",
    text,
    re.S,
)
if not match:
    raise SystemExit("fft_twiddles48000_960 table not found")

body = match.group(1)
pairs = re.findall(r"\{([^}]+)\}", body)
values = []
for pair in pairs:
    parts = [p.strip() for p in pair.split(",") if p.strip()]
    if len(parts) < 2:
        continue
    r, i = parts[0], parts[1]
    if not r.endswith("f") or not i.endswith("f"):
        raise SystemExit(f"unexpected literal suffix in: {pair}")
    values.append((r[:-1], i[:-1]))

if len(values) != 480:
    raise SystemExit(f"expected 480 twiddles, got {len(values)}")

lines = []
lines.append("// Source: opus-c/celt/static_modes_float.h (fft_twiddles48000_960).")
lines.append("// Regenerate with: python3 scripts/gen_fft_twiddles_48000_960.py")
lines.append("// (copies the C static table verbatim).")
lines.append("use super::mini_kfft::KissFftCpx;")
lines.append("")
lines.append("pub(crate) const FFT_TWIDDLES_48000_960: [KissFftCpx; 480] = [")
for r, i in values:
    lines.append(f"    KissFftCpx::new({r}_f32, {i}_f32),")
lines.append("];")
lines.append("")

DEST.write_text("\n".join(lines), encoding="utf-8")
