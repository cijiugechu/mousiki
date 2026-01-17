#![cfg(feature = "deep_plc")]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn resolve_vector_root(base: &Path) -> Option<PathBuf> {
    if base.join("vector1_dred.bit").exists() {
        return Some(base.to_path_buf());
    }

    let mut subdirs = base.read_dir().ok()?.filter_map(|entry| {
        let entry = entry.ok()?;
        let file_type = entry.file_type().ok()?;
        if file_type.is_dir() {
            Some(entry.path())
        } else {
            None
        }
    });

    let first = subdirs.next()?;
    if subdirs.next().is_some() {
        return None;
    }
    if first.join("vector1_dred.bit").exists() {
        Some(first)
    } else {
        None
    }
}

fn find_vectors_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("DRED_VECTORS_PATH") {
        return Some(PathBuf::from(path));
    }

    let default_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("dred_vectors");
    if default_path.is_dir() {
        Some(default_path)
    } else {
        None
    }
}

#[test]
fn dred_vectors_match_reference() {
    let Some(base_path) = find_vectors_path() else {
        eprintln!("DRED vectors not found; set DRED_VECTORS_PATH to enable.");
        return;
    };

    let Some(vector_root) = resolve_vector_root(&base_path) else {
        eprintln!("DRED vectors not found under {}.", base_path.display());
        return;
    };

    let Ok(dred_vectors) = env::var("CARGO_BIN_EXE_dred_vectors") else {
        eprintln!("dred_vectors binary not available; skipping.");
        return;
    };

    let mut cmd = Command::new(dred_vectors);
    if !cfg!(feature = "deep_plc_weights") {
        let Some(blob) = env::var_os("DNN_BLOB") else {
            eprintln!("Missing DNN_BLOB; skipping dred_vectors.");
            return;
        };
        cmd.arg("--dnn-blob").arg(blob);
    }
    cmd.arg(vector_root);

    let status = cmd.status().expect("run dred_vectors");
    assert!(status.success(), "dred_vectors failed with {status}");
}
