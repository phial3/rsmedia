// Shared helpers for *integration* tests (`tests/*.rs`). This module is
// intentionally independent of the library crate (`src/`): integration tests
// `mod common;` and use `common::test_output_path(...)` / the frame generators.

static OUTPUT_ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();

fn output_root() -> &'static tempfile::TempDir {
    OUTPUT_ROOT.get_or_init(|| {
        tempfile::Builder::new()
            .prefix("rsmedia-test-output-")
            .tempdir()
            .expect("failed to create test output tempdir")
    })
}

/// Returns a standardized test output path under `{tempdir}/{category}/`,
/// creating the directory if needed. The directory is private to the running
/// test binary and is automatically removed when the process exits.
///
/// # Arguments
/// * `category` - The subdirectory name (e.g., "encode", "transcode")
/// * `filename` - The output filename. May be empty to obtain the directory
///   itself (useful when a helper writes multiple files into a target dir).
pub fn test_output_path(category: &str, filename: &str) -> std::path::PathBuf {
    let output_dir = output_root().path().join(category);
    std::fs::create_dir_all(&output_dir)
        .unwrap_or_else(|e| panic!("failed to create test output directory {output_dir:?}: {e}"));
    output_dir.join(filename)
}

/// Removes a test output file if it exists.
/// Only some test crates use this, so suppress dead-code warnings in the rest.
#[allow(dead_code)]
pub fn remove_test_output(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Synthetic frame generators (used by integration tests that
// drive the high-level MediaFrame API, e.g. tests/codec_matrix.rs).
// ---------------------------------------------------------------------------

/// Generates a video test frame (RGB24): horizontal red / vertical green
/// gradients plus a phase-shifted blue channel, so round-trips exercise every
/// plane.
#[allow(dead_code)]
pub fn gradient_video_frame(width: usize, height: usize, phase: f32) -> rsmedia::MediaFrame<u8> {
    let mut frame =
        rsmedia::MediaFrame::<u8>::new_video_frame(width, height, rsmedia::PixelFormat::RGB24)
            .expect("video frame allocation");
    let samples = frame
        .data
        .as_packed_mut()
        .expect("RGB24 frames are interleaved");
    for y in 0..height {
        for x in 0..width {
            samples[[y, x, 0]] = ((x as f32 / width as f32) * 255.0) as u8;
            samples[[y, x, 1]] = ((y as f32 / height as f32) * 255.0) as u8;
            samples[[y, x, 2]] = (phase * 255.0) as u8;
        }
    }
    frame
}

/// Generates one stereo sine-wave audio frame (FLTP, f32 samples).
/// The encoder's native sample format is negotiated and converted
/// automatically, so this single layout feeds every audio codec.
#[allow(dead_code)]
pub fn sine_audio_frame(
    freq: f32,
    channels: u32,
    nb_samples: u32,
    sample_rate: u32,
) -> rsmedia::MediaFrame<f32> {
    let mut frame = rsmedia::MediaFrame::<f32>::new_audio_frame(
        rsmedia::SampleFormat::FLTP,
        channels,
        nb_samples,
        sample_rate,
    )
    .expect("audio frame allocation");
    let planes = frame.data.as_planes_mut().expect("FLTP frames are planar");
    for plane in planes.iter_mut() {
        for i in 0..nb_samples as usize {
            let t = i as f32 / sample_rate as f32;
            // FLTP is planar: one `(1, nb_samples)` plane per channel.
            plane[[0, i]] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        }
    }
    frame
}
