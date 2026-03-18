#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench_codec_compare.sh --input INPUT.pcm [options]

Options:
  --input PATH                 Raw 16-bit little-endian PCM input
  --sample-rate HZ             Default: 48000
  --channels N                 Default: 2
  --frame-size SAMPLES         Default: 960
  --application NAME           voip|audio|restricted-lowdelay (default: audio)
  --bitrate BPS                Default: 64000
  --complexity N               Default: 10
  --bitrate-mode MODE          vbr|cvbr|cbr (default: cvbr)
  --warmup N                   Default: 3
  --measure N                  Default: 10
  --max-frames N               Limit corpus and benchmark length
  --ctests-build-dir DIR       Default: ctests/build-bench
  --help                       Show this help

This script:
1. Builds the Rust benchmark binary.
2. Builds the C benchmark target without modifying opus-c.
3. Generates a C-encoded packet corpus.
4. Runs Rust and C encode benchmarks on the same PCM input.
5. Runs Rust and C decode benchmarks on the same packet corpus.
6. Prints a single CSV table.
USAGE
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
ctests_build_dir="${repo_root}/ctests/build-bench"

input=""
sample_rate=48000
channels=2
frame_size=960
application="audio"
bitrate=64000
complexity=10
bitrate_mode="cvbr"
warmup=3
measure=10
max_frames=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="$2"
      shift 2
      ;;
    --sample-rate)
      sample_rate="$2"
      shift 2
      ;;
    --channels)
      channels="$2"
      shift 2
      ;;
    --frame-size)
      frame_size="$2"
      shift 2
      ;;
    --application)
      application="$2"
      shift 2
      ;;
    --bitrate)
      bitrate="$2"
      shift 2
      ;;
    --complexity)
      complexity="$2"
      shift 2
      ;;
    --bitrate-mode)
      bitrate_mode="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --measure)
      measure="$2"
      shift 2
      ;;
    --max-frames)
      max_frames="$2"
      shift 2
      ;;
    --ctests-build-dir)
      ctests_build_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${input}" ]]; then
  echo "missing --input" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${input}" ]]; then
  echo "input file not found: ${input}" >&2
  exit 1
fi

rust_bin="${repo_root}/target/release/codec_bench"
c_bin="${ctests_build_dir}/opus_codec_bench"
output_dir="${repo_root}/target/codec-bench"
mkdir -p "${output_dir}"
packet_corpus="${output_dir}/decode-corpus-${sample_rate}-${channels}ch-${frame_size}-${application}-${bitrate}.opusbench"

common_args=(
  --sample-rate "${sample_rate}"
  --channels "${channels}"
  --frame-size "${frame_size}"
  --application "${application}"
  --bitrate "${bitrate}"
  --complexity "${complexity}"
  --bitrate-mode "${bitrate_mode}"
)

if [[ -n "${max_frames}" ]]; then
  common_args+=(--max-frames "${max_frames}")
fi

decode_args=()
if [[ -n "${max_frames}" ]]; then
  decode_args+=(--max-frames "${max_frames}")
fi

echo "Building Rust benchmark binary..." >&2
cargo build --release --bin codec_bench --manifest-path "${repo_root}/Cargo.toml" >&2

echo "Building C benchmark target..." >&2
cmake -S "${repo_root}/ctests" -B "${ctests_build_dir}" \
  -DOPUS_CTESTS_FIXED_POINT=OFF \
  -DOPUS_CTESTS_ENABLE_FLOAT_API=ON >&2
cmake --build "${ctests_build_dir}" --target opus_codec_bench >&2

echo "Generating C packet corpus..." >&2
"${c_bin}" packets \
  --input "${input}" \
  --output "${packet_corpus}" \
  "${common_args[@]}" >&2

csv_header="implementation,operation,sample_rate,channels,frame_size,application,bitrate,complexity,bitrate_mode,frames,warmup_iters,measure_iters,median_ns_per_frame,p95_ns_per_frame,median_packets_per_sec,median_realtime_x"
echo "${csv_header}"

"${rust_bin}" encode \
  --input "${input}" \
  "${common_args[@]}" \
  --warmup "${warmup}" \
  --measure "${measure}" \
  --format csv \
  --no-header

"${c_bin}" encode \
  --input "${input}" \
  "${common_args[@]}" \
  --warmup "${warmup}" \
  --measure "${measure}" \
  --format csv \
  --no-header

"${rust_bin}" decode \
  --packets "${packet_corpus}" \
  --warmup "${warmup}" \
  --measure "${measure}" \
  --format csv \
  --no-header \
  "${decode_args[@]}"

"${c_bin}" decode \
  --packets "${packet_corpus}" \
  --warmup "${warmup}" \
  --measure "${measure}" \
  --format csv \
  --no-header \
  "${decode_args[@]}"
