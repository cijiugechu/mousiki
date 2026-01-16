use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

const MODEL_SHA256: &str = "4ec556dd87e63c17c4a805c40685ef3fe1fad7c8b26b123f2ede553b50158cb1";
const MODEL_TARBALL: &str =
    "opus_data-4ec556dd87e63c17c4a805c40685ef3fe1fad7c8b26b123f2ede553b50158cb1.tar.gz";
const DEFAULT_URL: &str = "https://media.xiph.org/opus/models/opus_data-4ec556dd87e63c17c4a805c40685ef3fe1fad7c8b26b123f2ede553b50158cb1.tar.gz";

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=DRED_WEIGHTS_PATH");
    println!("cargo:rerun-if-env-changed=DRED_WEIGHTS_URL");
    println!("cargo:rerun-if-env-changed=DRED_WEIGHTS_SHA256");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let work_dir = out_dir.join("dred_weights");
    fs::create_dir_all(&work_dir)?;

    let source_root = match env::var("DRED_WEIGHTS_PATH") {
        Ok(path) => {
            println!("cargo:rerun-if-changed={}", path);
            prepare_from_path(&work_dir, PathBuf::from(path))?
        }
        Err(_) => {
            if env::var_os("CARGO_FEATURE_FETCH").is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "DRED_WEIGHTS_PATH not set and fetch feature disabled. \
Set DRED_WEIGHTS_PATH to a directory or tarball containing \
dred_rdovae_dec_data.c, dred_rdovae_stats_data.c, dred_rdovae_enc_data.c, \
and pitchdnn_data.c, or enable the mousiki-dred-weights `fetch` feature \
(use `dred_fetch` in the main crate).",
                ));
            }
            download_and_extract(&work_dir)?
        }
    };

    let dec_path = resolve_source_file(&source_root, "dred_rdovae_dec_data.c")?;
    let stats_path = resolve_source_file(&source_root, "dred_rdovae_stats_data.c")?;
    let enc_path = resolve_source_file(&source_root, "dred_rdovae_enc_data.c")?;
    let pitch_path = resolve_source_file(&source_root, "pitchdnn_data.c")?;

    println!("cargo:rerun-if-changed={}", dec_path.display());
    println!("cargo:rerun-if-changed={}", stats_path.display());
    println!("cargo:rerun-if-changed={}", enc_path.display());
    println!("cargo:rerun-if-changed={}", pitch_path.display());

    generate_rust(&dec_path, &out_dir.join("dred_rdovae_dec_data.rs"))?;
    generate_rust(&stats_path, &out_dir.join("dred_rdovae_stats_data.rs"))?;
    generate_rust(&enc_path, &out_dir.join("dred_rdovae_enc_data.rs"))?;
    generate_rust(&pitch_path, &out_dir.join("pitchdnn_data.rs"))?;

    Ok(())
}

fn prepare_from_path(work_dir: &Path, path: PathBuf) -> io::Result<PathBuf> {
    if path.is_dir() {
        return Ok(path);
    }

    if path.is_file() {
        return extract_tarball(work_dir, &path, None).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("Failed to extract {}: {err}", path.display()),
            )
        });
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "DRED_WEIGHTS_PATH not found: {}. Provide a directory containing \
dred_rdovae_dec_data.c, dred_rdovae_stats_data.c, dred_rdovae_enc_data.c, and \
pitchdnn_data.c, or a tarball from Xiph.",
            path.display()
        ),
    ))
}

fn download_and_extract(work_dir: &Path) -> io::Result<PathBuf> {
    let url = env::var("DRED_WEIGHTS_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let expected_sha =
        env::var("DRED_WEIGHTS_SHA256").unwrap_or_else(|_| MODEL_SHA256.to_string());

    let tar_path = work_dir.join(MODEL_TARBALL);
    let extract_root = work_dir.join("extracted");
    let stamp_path = work_dir.join("model.stamp");

    if stamp_matches(&stamp_path, &expected_sha)
        && extracted_files_exist(&extract_root)
    {
        return Ok(extract_root);
    }

    if !tar_path.exists() || sha256_path(&tar_path)? != expected_sha {
        download_model(&url, &tar_path)?;
        let actual_sha = sha256_path(&tar_path)?;
        if actual_sha != expected_sha {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Downloaded model checksum mismatch: expected {expected_sha}, got {actual_sha}. \
Delete the cached tarball at {} or override DRED_WEIGHTS_SHA256 / DRED_WEIGHTS_URL.",
                    tar_path.display()
                ),
            ));
        }
    }

    let extracted = extract_tarball(work_dir, &tar_path, Some(&expected_sha)).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Failed to extract {}: {err}", tar_path.display()),
        )
    })?;
    Ok(extracted)
}

fn download_model(url: &str, dest: &Path) -> io::Result<()> {
    let agent = ureq::Agent::config_builder().build().new_agent();
    let mut response = agent
        .get(url)
        .header("User-Agent", "mousiki-dred-weights")
        .call()
        .map_err(|err| {
            io::Error::other(format!(
                "Failed to download {url}: {err}. Check proxy env \
(ALL_PROXY/HTTPS_PROXY/HTTP_PROXY) or set DRED_WEIGHTS_PATH."
            ))
        })?;

    let mut reader = response.body_mut().as_reader();
    let mut file = File::create(dest)?;
    io::copy(&mut reader, &mut file)?;
    Ok(())
}

fn extract_tarball(work_dir: &Path, tar_path: &Path, stamp: Option<&str>) -> io::Result<PathBuf> {
    let extract_root = work_dir.join("extracted");
    if extract_root.exists() {
        fs::remove_dir_all(&extract_root)?;
    }
    fs::create_dir_all(&extract_root)?;

    let tar_gz = File::open(tar_path)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive.unpack(&extract_root)?;

    if let Some(expected_sha) = stamp {
        let stamp_path = work_dir.join("model.stamp");
        fs::write(&stamp_path, expected_sha)?;
    }

    Ok(extract_root)
}

fn stamp_matches(stamp_path: &Path, expected_sha: &str) -> bool {
    if let Ok(existing) = fs::read_to_string(stamp_path) {
        return existing.trim() == expected_sha;
    }
    false
}

fn extracted_files_exist(root: &Path) -> bool {
    let dec = root.join("dnn").join("dred_rdovae_dec_data.c");
    let stats = root.join("dnn").join("dred_rdovae_stats_data.c");
    let enc = root.join("dnn").join("dred_rdovae_enc_data.c");
    let pitch = root.join("dnn").join("pitchdnn_data.c");
    dec.exists() && stats.exists() && enc.exists() && pitch.exists()
}

fn resolve_source_file(root: &Path, name: &str) -> io::Result<PathBuf> {
    let direct = root.join(name);
    if direct.exists() {
        return Ok(direct);
    }
    let dnn = root.join("dnn").join(name);
    if dnn.exists() {
        return Ok(dnn);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "Missing {name} under {}. Expected {} or {}. \
Check DRED_WEIGHTS_PATH or re-download the model.",
            root.display(),
            root.join(name).display(),
            root.join("dnn").join(name).display(),
        ),
    ))
}

fn sha256_path(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn strip_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut line_comment = false;
    let mut block_comment = false;

    while let Some(c) = chars.next() {
        if line_comment {
            if c == '\n' {
                line_comment = false;
                output.push(c);
            }
            continue;
        }

        if block_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment = false;
            }
            continue;
        }

        if c == '/' {
            match chars.peek() {
                Some('/') => {
                    chars.next();
                    line_comment = true;
                    continue;
                }
                Some('*') => {
                    chars.next();
                    block_comment = true;
                    continue;
                }
                _ => {}
            }
        }

        output.push(c);
    }

    output
}

fn parse_header(header: &str) -> io::Result<(String, String, usize)> {
    let header = header.replace(['\n', '\r'], " ");
    let open = header.find('[').ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Missing '[' in array header")
    })?;
    let close = header[open + 1..]
        .find(']')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing ']' in array header"))?;
    let len_str = header[open + 1..open + 1 + close].trim();
    let len = len_str.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid array length: {len_str}"),
        )
    })?;
    let before = header[..open].trim();
    let tokens: Vec<&str> = before.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid array header: {before}"),
        ));
    }
    let name = tokens[tokens.len() - 1].to_string();
    let c_type = tokens[..tokens.len() - 1].join(" ");
    Ok((c_type, name, len))
}

fn map_c_type(c_type: &str) -> io::Result<&'static str> {
    match c_type {
        "float" => Ok("f32"),
        "double" => Ok("f64"),
        "int" => Ok("i32"),
        "unsigned int" => Ok("u32"),
        "opus_int8" | "int8_t" | "signed char" | "char" => Ok("i8"),
        "opus_uint8" | "uint8_t" | "unsigned char" => Ok("u8"),
        "opus_int16" | "int16_t" | "short" => Ok("i16"),
        "opus_uint16" | "uint16_t" | "unsigned short" => Ok("u16"),
        "opus_int32" | "int32_t" | "long" => Ok("i32"),
        "opus_uint32" | "uint32_t" | "unsigned long" => Ok("u32"),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported C type: {other}"),
        )),
    }
}

fn clean_value(raw: &str, rust_type: &str) -> String {
    let mut value = raw.trim().to_string();
    if value.is_empty() {
        return value;
    }

    if matches!(rust_type, "f32" | "f64") {
        if value.ends_with('f') || value.ends_with('F') {
            value.pop();
        }
    } else if matches!(rust_type, "u8" | "u16" | "u32" | "u64") {
        if value.ends_with('u') || value.ends_with('U') {
            value.pop();
        }
    } else if value.ends_with('l') || value.ends_with('L') {
        value.pop();
    }

    value
}

fn write_array(
    out: &mut BufWriter<File>,
    rust_name: &str,
    rust_type: &str,
    len: usize,
    values_str: &str,
) -> io::Result<()> {
    writeln!(out, "#[rustfmt::skip]")?;
    writeln!(out, "pub const {rust_name}: [{rust_type}; {len}] = [")?;

    let mut count = 0usize;
    for raw in values_str.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = clean_value(trimmed, rust_type);
        write!(out, "{value}")?;
        count += 1;
        if count < len {
            if count.is_multiple_of(8) {
                writeln!(out, ",")?;
            } else {
                write!(out, ", ")?;
            }
        }
    }

    if count != len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Value count mismatch for {rust_name}: {count} != {len}"),
        ));
    }

    writeln!(out, "\n];\n")?;
    Ok(())
}

fn generate_rust(source_path: &Path, out_path: &Path) -> io::Result<()> {
    let content = fs::read_to_string(source_path)?;
    let content = strip_comments(&content);

    let mut out = BufWriter::new(File::create(out_path)?);
    writeln!(
        out,
        "// Auto-generated from {}",
        source_path.display()
    )?;

    let marker = "const ";
    let mut cursor = 0usize;
    let mut total = 0usize;

    while let Some(pos) = content[cursor..].find(marker) {
        let start = cursor + pos + marker.len();
        let after = &content[start..];
        let brace_pos = match after.find('{') {
            Some(value) => value,
            None => {
                cursor = start;
                continue;
            }
        };
        let header = &after[..brace_pos];
        if !header.contains('[') {
            cursor = start + brace_pos + 1;
            continue;
        }
        let end_pos = after[brace_pos + 1..]
            .find("};")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Missing array terminator in {}. The file may be truncated \
or incompatible; try re-downloading or setting DRED_WEIGHTS_PATH.",
                        source_path.display()
                    ),
                )
            })?;
        let values_str = &after[brace_pos + 1..brace_pos + 1 + end_pos];

        let (c_type, name, len) = match parse_header(header) {
            Ok(value) => value,
            Err(_) => {
                cursor = start + brace_pos + 1;
                continue;
            }
        };
        let rust_type = match map_c_type(&c_type) {
            Ok(value) => value,
            Err(_) => {
                cursor = start + brace_pos + 1;
                continue;
            }
        };
        let rust_name = name.to_ascii_uppercase();

        write_array(&mut out, &rust_name, rust_type, len, values_str).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("Failed to parse {}: {err}", source_path.display()),
            )
        })?;
        total += 1;

        cursor = start + brace_pos + 1 + end_pos + 2;
    }

    if total == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "No arrays found in {}. The model file may be incompatible; \
ensure it contains dred_rdovae_* tables or re-download the model.",
                source_path.display()
            ),
        ));
    }

    Ok(())
}
