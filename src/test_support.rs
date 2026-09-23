//! Test-support helpers compiled only for library unit tests (`#[cfg(test)]`).
//!
/// # Arguments
/// * `category` - The subdirectory name (e.g., "encode", "mux", "pcm")
/// * `filename` - The output filename
pub fn test_output_path(category: &str, filename: &str) -> std::path::PathBuf {
    const CRATE_NAME: &str = env!("CARGO_PKG_HOMEPAGE");
    let output_dir = std::path::PathBuf::from("output").join(category);
    std::fs::create_dir_all(&output_dir).ok();
    println!("output_dir: {}/{:?}", CRATE_NAME, output_dir);
    output_dir.join(filename)
}

/// Removes a test output file if it exists.
/// Most unit-test modules use this, but the API is kept generally usable.
pub fn remove_test_output(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}
