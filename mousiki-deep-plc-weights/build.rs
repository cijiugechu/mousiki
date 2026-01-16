use std::collections::HashMap;
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

const WEIGHT_BLOCK_SIZE: usize = 64;
const WEIGHT_BLOB_VERSION: i32 = 0;
const WEIGHT_NAME_LEN: usize = 44;

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=DNN_WEIGHTS_PATH");
    println!("cargo:rerun-if-env-changed=DNN_WEIGHTS_URL");
    println!("cargo:rerun-if-env-changed=DNN_WEIGHTS_SHA256");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let work_dir = out_dir.join("dnn_weights");
    fs::create_dir_all(&work_dir)?;

    let source_root = match env::var("DNN_WEIGHTS_PATH") {
        Ok(path) => {
            println!("cargo:rerun-if-changed={}", path);
            prepare_from_path(&work_dir, PathBuf::from(path))?
        }
        Err(_) => {
            if env::var_os("CARGO_FEATURE_FETCH").is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "DNN_WEIGHTS_PATH not set and fetch feature disabled. \
Set DNN_WEIGHTS_PATH to a directory or tarball containing pitchdnn_data.c, \
fargan_data.c, plc_data.c, dred_rdovae_enc_data.c, and dred_rdovae_dec_data.c, \
or enable the mousiki-deep-plc-weights `fetch` feature.",
                ));
            }
            download_and_extract(&work_dir)?
        }
    };

    let pitch_path = resolve_source_file(&source_root, "pitchdnn_data.c")?;
    let fargan_path = resolve_source_file(&source_root, "fargan_data.c")?;
    let plc_path = resolve_source_file(&source_root, "plc_data.c")?;
    let enc_path = resolve_source_file(&source_root, "dred_rdovae_enc_data.c")?;
    let dec_path = resolve_source_file(&source_root, "dred_rdovae_dec_data.c")?;

    println!("cargo:rerun-if-changed={}", pitch_path.display());
    println!("cargo:rerun-if-changed={}", fargan_path.display());
    println!("cargo:rerun-if-changed={}", plc_path.display());
    println!("cargo:rerun-if-changed={}", enc_path.display());
    println!("cargo:rerun-if-changed={}", dec_path.display());

    let out_path = out_dir.join("weights_blob.bin");
    let mut out = BufWriter::new(File::create(&out_path)?);

    write_weight_list(&mut out, &pitch_path, "pitchdnn_arrays")?;
    write_weight_list(&mut out, &fargan_path, "fargan_arrays")?;
    write_weight_list(&mut out, &plc_path, "plcmodel_arrays")?;
    write_weight_list(&mut out, &enc_path, "rdovaeenc_arrays")?;
    write_weight_list(&mut out, &dec_path, "rdovaedec_arrays")?;

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
            "DNN_WEIGHTS_PATH not found: {}. Provide a directory containing \
pitchdnn_data.c, fargan_data.c, plc_data.c, dred_rdovae_enc_data.c, and \
dred_rdovae_dec_data.c, or a tarball from Xiph.",
            path.display()
        ),
    ))
}

fn download_and_extract(work_dir: &Path) -> io::Result<PathBuf> {
    let url = env::var("DNN_WEIGHTS_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let expected_sha =
        env::var("DNN_WEIGHTS_SHA256").unwrap_or_else(|_| MODEL_SHA256.to_string());

    let tar_path = work_dir.join(MODEL_TARBALL);
    let extract_root = work_dir.join("extracted");
    let stamp_path = work_dir.join("model.stamp");

    if stamp_matches(&stamp_path, &expected_sha) && extracted_files_exist(&extract_root) {
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
Delete the cached tarball at {} or override DNN_WEIGHTS_SHA256 / DNN_WEIGHTS_URL.",
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
        .header("User-Agent", "mousiki-deep-plc-weights")
        .call()
        .map_err(|err| {
            io::Error::other(format!(
                "Failed to download {url}: {err}. Check proxy env \
(ALL_PROXY/HTTPS_PROXY/HTTP_PROXY) or set DNN_WEIGHTS_PATH."
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
    let pitch = root.join("dnn").join("pitchdnn_data.c");
    let fargan = root.join("dnn").join("fargan_data.c");
    let plc = root.join("dnn").join("plc_data.c");
    let enc = root.join("dnn").join("dred_rdovae_enc_data.c");
    let dec = root.join("dnn").join("dred_rdovae_dec_data.c");
    pitch.exists() && fargan.exists() && plc.exists() && enc.exists() && dec.exists()
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
Check DNN_WEIGHTS_PATH or re-download the model.",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayType {
    F32,
    F64,
    I32,
    U32,
    I8,
    U8,
    I16,
    U16,
    I64,
    U64,
}

impl ArrayType {
    fn rust_name(self) -> &'static str {
        match self {
            ArrayType::F32 => "f32",
            ArrayType::F64 => "f64",
            ArrayType::I32 => "i32",
            ArrayType::U32 => "u32",
            ArrayType::I8 => "i8",
            ArrayType::U8 => "u8",
            ArrayType::I16 => "i16",
            ArrayType::U16 => "u16",
            ArrayType::I64 => "i64",
            ArrayType::U64 => "u64",
        }
    }

    fn size_bytes(self) -> usize {
        match self {
            ArrayType::F32 => 4,
            ArrayType::F64 => 8,
            ArrayType::I32 => 4,
            ArrayType::U32 => 4,
            ArrayType::I8 => 1,
            ArrayType::U8 => 1,
            ArrayType::I16 => 2,
            ArrayType::U16 => 2,
            ArrayType::I64 => 8,
            ArrayType::U64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
struct ArrayData {
    array_type: ArrayType,
    bytes: Vec<u8>,
}

fn map_c_type(c_type: &str) -> io::Result<ArrayType> {
    match c_type {
        "float" => Ok(ArrayType::F32),
        "double" => Ok(ArrayType::F64),
        "int" => Ok(ArrayType::I32),
        "unsigned int" => Ok(ArrayType::U32),
        "opus_int8" | "int8_t" | "signed char" | "char" => Ok(ArrayType::I8),
        "opus_uint8" | "uint8_t" | "unsigned char" => Ok(ArrayType::U8),
        "opus_int16" | "int16_t" | "short" => Ok(ArrayType::I16),
        "opus_uint16" | "uint16_t" | "unsigned short" => Ok(ArrayType::U16),
        "opus_int32" | "int32_t" | "long" => Ok(ArrayType::I32),
        "opus_uint32" | "uint32_t" | "unsigned long" => Ok(ArrayType::U32),
        "int64_t" | "opus_int64" => Ok(ArrayType::I64),
        "uint64_t" | "opus_uint64" => Ok(ArrayType::U64),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported C type: {other}"),
        )),
    }
}

fn c_type_size(c_type: &str) -> Option<usize> {
    map_c_type(c_type).ok().map(|kind| kind.size_bytes())
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
    }

    while matches!(value.chars().last(), Some('u' | 'U' | 'l' | 'L')) {
        value.pop();
    }

    value
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

fn parse_int_signed(value: &str) -> io::Result<i64> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix("-0x").or_else(|| value.strip_prefix("-0X")) {
        let parsed = i64::from_str_radix(hex, 16).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid int: {value}"))
        })?;
        return Ok(-parsed);
    }
    if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        return i64::from_str_radix(hex, 16).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid int: {value}"))
        });
    }
    value.parse::<i64>().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Invalid int: {value}"))
    })
}

fn parse_int_unsigned(value: &str) -> io::Result<u64> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid int: {value}"))
        });
    }
    value.parse::<u64>().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Invalid int: {value}"))
    })
}

fn parse_values(values_str: &str, array_type: ArrayType, len: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(len * array_type.size_bytes());
    let mut count = 0usize;

    for raw in values_str.split(',') {
        let mut trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        trimmed = trimmed.trim_matches(['{', '}']);
        let trimmed = trimmed.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value = clean_value(trimmed, array_type.rust_name());
        if value.is_empty() {
            continue;
        }

        match array_type {
            ArrayType::F32 => {
                let parsed = value.parse::<f32>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid f32: {value}"))
                })?;
                bytes.extend_from_slice(&parsed.to_le_bytes());
            }
            ArrayType::F64 => {
                let parsed = value.parse::<f64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid f64: {value}"))
                })?;
                bytes.extend_from_slice(&parsed.to_le_bytes());
            }
            ArrayType::I8 => {
                let parsed = parse_int_signed(&value)?;
                let value = i8::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid i8: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::U8 => {
                let parsed = parse_int_unsigned(&value)?;
                let value = u8::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid u8: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::I16 => {
                let parsed = parse_int_signed(&value)?;
                let value = i16::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid i16: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::U16 => {
                let parsed = parse_int_unsigned(&value)?;
                let value = u16::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid u16: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::I32 => {
                let parsed = parse_int_signed(&value)?;
                let value = i32::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid i32: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::U32 => {
                let parsed = parse_int_unsigned(&value)?;
                let value = u32::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid u32: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::I64 => {
                let parsed = parse_int_signed(&value)?;
                let value = i64::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid i64: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ArrayType::U64 => {
                let parsed = parse_int_unsigned(&value)?;
                let value = u64::try_from(parsed).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid u64: {value}"))
                })?;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        count += 1;
    }

    if count != len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Value count mismatch: {count} != {len}"),
        ));
    }

    Ok(bytes)
}

fn parse_arrays(content: &str) -> io::Result<HashMap<String, ArrayData>> {
    let mut arrays = HashMap::new();
    let marker = "const ";
    let mut cursor = 0usize;

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
                    "Missing array terminator; the file may be truncated or incompatible.",
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
        let array_type = match map_c_type(&c_type) {
            Ok(value) => value,
            Err(_) => {
                cursor = start + brace_pos + 1;
                continue;
            }
        };
        let bytes = parse_values(values_str, array_type, len).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("Failed to parse array {name}: {err}"),
            )
        })?;

        arrays.insert(
            name,
            ArrayData {
                array_type,
                bytes,
            },
        );

        cursor = start + brace_pos + 1 + end_pos + 2;
    }

    if arrays.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No arrays found; the model file may be incompatible.",
        ));
    }

    Ok(arrays)
}

#[derive(Debug)]
struct WeightEntry {
    name: String,
    type_id: i32,
    size_expr: String,
    data_name: String,
}

fn parse_weight_type(value: &str) -> io::Result<i32> {
    match value.trim() {
        "WEIGHT_TYPE_float" => Ok(0),
        "WEIGHT_TYPE_int" => Ok(1),
        "WEIGHT_TYPE_qweight" => Ok(2),
        "WEIGHT_TYPE_int8" => Ok(3),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown weight type: {other}"),
        )),
    }
}

fn parse_weight_list(content: &str, list_name: &str) -> io::Result<Vec<WeightEntry>> {
    let needle = format!("WeightArray {list_name}");
    let start = content
        .find(&needle)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "WeightArray list missing"))?;
    let after = &content[start + needle.len()..];
    let brace_pos = after.find('{').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Missing '{{' for {list_name}"),
        )
    })?;
    let list_start = start + needle.len() + brace_pos + 1;

    let mut depth = 1usize;
    let mut end = None;
    for (idx, ch) in content[list_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(list_start + idx);
                    break;
                }
            }
            _ => {}
        }
    }
    let list_end = end.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unterminated list for {list_name}"),
        )
    })?;

    let list_body = &content[list_start..list_end];
    let entries = extract_entries(list_body);
    let mut results = Vec::new();

    for entry in entries {
        let fields = split_fields(entry);
        if fields.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid WeightArray entry: {entry}"),
            ));
        }
        let raw_name = fields[0].trim();
        if raw_name == "NULL" || raw_name == "0" {
            break;
        }
        let name = parse_c_string(raw_name)?;
        let type_id = parse_weight_type(fields[1])?;
        let size_expr = fields[2].trim().to_string();
        let data_name = normalize_array_ref(fields[3]);
        results.push(WeightEntry {
            name,
            type_id,
            size_expr,
            data_name,
        });
    }

    if results.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("No entries found in {list_name}"),
        ));
    }

    Ok(results)
}

fn extract_entries(list_body: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut depth = 0usize;
    let mut start = None;

    for (idx, ch) in list_body.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(idx + 1);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                }
                if depth == 0 {
                    if let Some(start_idx) = start.take() {
                        let entry = list_body[start_idx..idx].trim();
                        if !entry.is_empty() {
                            entries.push(entry);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    entries
}

fn split_fields(entry: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = 0usize;
    let mut in_string = false;
    let mut paren_depth = 0usize;
    let mut prev_escape = false;

    for (idx, ch) in entry.char_indices() {
        match ch {
            '"' if !prev_escape => {
                in_string = !in_string;
            }
            '(' if !in_string => {
                paren_depth += 1;
            }
            ')' if !in_string => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                }
            }
            ',' if !in_string && paren_depth == 0 => {
                fields.push(entry[start..idx].trim());
                start = idx + 1;
            }
            _ => {}
        }
        prev_escape = ch == '\\' && !prev_escape;
    }

    if start < entry.len() {
        fields.push(entry[start..].trim());
    }

    fields
}

fn parse_c_string(value: &str) -> io::Result<String> {
    let trimmed = value.trim();
    if let Some(stripped) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Ok(stripped.to_string());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Expected string literal, got {trimmed}"),
    ))
}

fn normalize_array_ref(value: &str) -> String {
    let mut current = value.trim();

    loop {
        let trimmed = current.trim_start();
        if !trimmed.starts_with('(') {
            current = trimmed;
            break;
        }
        if let Some(close) = trimmed.find(')') {
            current = &trimmed[close + 1..];
        } else {
            current = trimmed;
            break;
        }
    }

    let mut current = current.trim();
    if let Some(stripped) = current.strip_prefix('&') {
        current = stripped.trim();
    }

    if let Some(idx) = current.find('[') {
        current = &current[..idx];
    }

    current.trim_matches(|c: char| c == '(' || c == ')').trim().to_string()
}

fn parse_size_expr(expr: &str, arrays: &HashMap<String, ArrayData>) -> io::Result<usize> {
    let mut result = 1usize;
    for part in expr.split('*') {
        let part = part.trim();
        if part.starts_with("sizeof") {
            let open = part.find('(').ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Invalid sizeof: {part}"))
            })?;
            let close = part.rfind(')').ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Invalid sizeof: {part}"))
            })?;
            let inner = part[open + 1..close].trim();
            let inner = inner.split('[').next().unwrap_or(inner).trim();
            if let Some(array) = arrays.get(inner) {
                result = result
                    .checked_mul(array.bytes.len())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Size overflow"))?;
                continue;
            }
            if let Some(size) = c_type_size(inner) {
                result = result
                    .checked_mul(size)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Size overflow"))?;
                continue;
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown sizeof target: {inner}"),
            ));
        }

        let value = part.parse::<usize>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid size: {part}"))
        })?;
        result = result
            .checked_mul(value)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Size overflow"))?;
    }

    Ok(result)
}

fn write_weight_list(out: &mut BufWriter<File>, path: &Path, list_name: &str) -> io::Result<()> {
    let content = fs::read_to_string(path)?;
    let content = strip_comments(&content);
    let arrays = parse_arrays(&content)?;
    let weights = parse_weight_list(&content, list_name)?;

    for entry in weights {
        let array = arrays.get(&entry.data_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Missing data array {}", entry.data_name),
            )
        })?;

        let expected_size = parse_size_expr(&entry.size_expr, &arrays)?;
        if expected_size != array.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Size mismatch for {}: list expects {}, array has {}",
                    entry.name,
                    expected_size,
                    array.bytes.len()
                ),
            ));
        }

        write_weight_entry(out, &entry, array)?;
    }

    Ok(())
}

fn write_weight_entry(out: &mut BufWriter<File>, entry: &WeightEntry, array: &ArrayData) -> io::Result<()> {
    let size = array.bytes.len();
    if size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Empty weight array: {}", entry.name),
        ));
    }
    let block_size = ((size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;

    let size_i32 = i32::try_from(size).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Weight too large: {}", entry.name),
        )
    })?;
    let block_i32 = i32::try_from(block_size).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Weight too large: {}", entry.name),
        )
    })?;

    let mut header = [0u8; WEIGHT_BLOCK_SIZE];
    header[0..4].copy_from_slice(b"DNNw");
    header[4..8].copy_from_slice(&WEIGHT_BLOB_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&entry.type_id.to_le_bytes());
    header[12..16].copy_from_slice(&size_i32.to_le_bytes());
    header[16..20].copy_from_slice(&block_i32.to_le_bytes());

    let name_bytes = entry.name.as_bytes();
    let copy_len = name_bytes.len().min(WEIGHT_NAME_LEN - 1);
    if name_bytes.len() >= WEIGHT_NAME_LEN {
        eprintln!("[mousiki-deep-plc-weights] warning: name {} truncated", entry.name);
    }
    header[20..20 + copy_len].copy_from_slice(&name_bytes[..copy_len]);

    out.write_all(&header)?;
    out.write_all(&array.bytes)?;

    if block_size > size {
        let pad = vec![0u8; block_size - size];
        out.write_all(&pad)?;
    }

    Ok(())
}
