#!/bin/sh
set -eu

usage() {
    cat <<'EOF'
usage: fetch_dred_vectors.sh [--url <url>] [--dest <dir>] [--sha256 <hash>]

Downloads and extracts the external DRED vector bundle into a local directory.
Defaults:
  --url     $DRED_VECTORS_URL
  --dest    $DRED_VECTORS_PATH or testdata/dred_vectors
  --sha256  $DRED_VECTORS_SHA256
EOF
}

url=""
dest=""
sha=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --url)
            shift
            url="${1:-}"
            ;;
        --dest)
            shift
            dest="${1:-}"
            ;;
        --sha256)
            shift
            sha="${1:-}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "$url" ]; then
    url="${DRED_VECTORS_URL:-}"
fi
if [ -z "$dest" ]; then
    dest="${DRED_VECTORS_PATH:-testdata/dred_vectors}"
fi
if [ -z "$sha" ]; then
    sha="${DRED_VECTORS_SHA256:-}"
fi

if [ -z "$url" ]; then
    echo "Missing DRED vector URL (set --url or DRED_VECTORS_URL)." >&2
    usage >&2
    exit 2
fi

if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "Destination is not empty: $dest" >&2
    exit 2
fi

mkdir -p "$dest"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive_name="$(basename "$url")"
archive_path="$tmpdir/$archive_name"

if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$archive_path"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" "$url"
else
    echo "Missing downloader: install curl or wget." >&2
    exit 1
fi

if [ -n "$sha" ]; then
    if command -v shasum >/dev/null 2>&1; then
        echo "$sha  $archive_path" | shasum -a 256 -c -
    elif command -v sha256sum >/dev/null 2>&1; then
        echo "$sha  $archive_path" | sha256sum -c -
    else
        echo "Missing sha256 checker: install shasum or sha256sum." >&2
        exit 1
    fi
fi

extract_dir="$tmpdir/extract"
mkdir -p "$extract_dir"

case "$archive_name" in
    *.tar.gz|*.tgz)
        tar -xzf "$archive_path" -C "$extract_dir"
        ;;
    *.tar.xz)
        tar -xJf "$archive_path" -C "$extract_dir"
        ;;
    *.zip)
        unzip -q "$archive_path" -d "$extract_dir"
        ;;
    *)
        echo "Unknown archive format: $archive_name" >&2
        exit 1
        ;;
esac

if [ -f "$extract_dir/vector1_dred.bit" ]; then
    src="$extract_dir"
else
    subdir="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
    if [ -n "$subdir" ] && [ -f "$subdir/vector1_dred.bit" ]; then
        src="$subdir"
    else
        echo "Could not locate vector1_dred.bit in extracted data." >&2
        exit 1
    fi
fi

cp -R "$src"/. "$dest"/

echo "DRED vectors installed in $dest"
