#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run.sh [options] [-- ctest-args...]

Options:
  -b, --build-dir DIR   Build directory (default: ctests/build)
  -j, --jobs N          Parallel build jobs
  --clean               Remove build directory before configuring
  --cmake-arg ARG       Extra argument passed to cmake (repeatable)
  --ctest-arg ARG       Extra argument passed to ctest (repeatable)
  -h, --help            Show this help

Examples:
  ./run.sh
  ./run.sh -j 8
  ./run.sh --clean
  ./run.sh --cmake-arg -DOPUS_CUSTOM_MODES=ON
  ./run.sh -- --verbose
USAGE
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="${script_dir}/build"
clean=0
jobs=""
cmake_args=()
ctest_args=("--output-on-failure")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-dir)
      build_dir="$2"
      shift 2
      ;;
    -j|--jobs)
      jobs="$2"
      shift 2
      ;;
    --clean)
      clean=1
      shift
      ;;
    --cmake-arg)
      cmake_args+=("$2")
      shift 2
      ;;
    --ctest-arg)
      ctest_args+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        ctest_args+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found in PATH" >&2
  exit 1
fi
if ! command -v ctest >/dev/null 2>&1; then
  echo "ctest not found in PATH" >&2
  exit 1
fi

if [[ $clean -eq 1 && -d "$build_dir" ]]; then
  if [[ -z "$build_dir" || "$build_dir" == "/" ]]; then
    echo "Refusing to remove build dir: '$build_dir'" >&2
    exit 1
  fi
  rm -rf -- "$build_dir"
fi

cmake -S "$script_dir" -B "$build_dir" "${cmake_args[@]}"

if [[ -n "$jobs" ]]; then
  cmake --build "$build_dir" --parallel "$jobs"
else
  cmake --build "$build_dir"
fi

ctest --test-dir "$build_dir" "${ctest_args[@]}"
