// Shared helpers for integration tests (`tests/*.rs`).
//
// This file is included from two places so the logic has a single definition:
// - Integration tests: `mod common;` then `common::test_output_path(...)`.
// - Library unit tests: `include!("../tests/common/mod.rs")` inside the
//   `#[cfg(test)]` `test_utils` module in `src/lib.rs`.
//
// `tests/common/mod.rs` is never compiled as its own test target, so nothing
// here ends up in the shipped library binary.

/// Returns a standardized test output path under `tests/output/{category}/`,
/// creating the directory if needed. The path is relative to the package root,
/// which `cargo test` uses as the working directory on all platforms
/// (macOS / Linux / Windows).
///
/// # Arguments
/// * `category` - The subdirectory name (e.g., "encode_video", "transcode")
/// * `filename` - The output filename
pub fn test_output_path(category: &str, filename: &str) -> std::path::PathBuf {
    let output_dir = std::path::PathBuf::from("tests/output").join(category);
    std::fs::create_dir_all(&output_dir).ok();
    output_dir.join(filename)
}

/// Removes a test output file if it exists.
/// Only some test crates use this, so suppress dead-code warnings in the rest.
#[allow(dead_code)]
pub fn remove_test_output(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}
