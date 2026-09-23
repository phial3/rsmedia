//! Still-image round trip: one frame through a `Muxer`, read back by the
//! demuxer and the decoder.
//!
//! Until this file existed the crate's image coverage was a PNG *sequence*
//! (`src/io.rs`) and a single JPEG read (`assets/cat.jpg`): `bmp`, `tiff` and
//! `webp` had no test at all, and no test ever wrote one image through
//! [`Muxer`] and read it back. Two halves fix that:
//!
//! * [`test_single_image_roundtrip`] — write one frame to `out.<ext>` and read
//!   it back, asserting geometry, non-blank pixels and a **one**-frame count.
//!   The muxer comes from the extension (`image2` for the still formats, `gif`
//!   for `.gif`), exactly as a caller would get it.
//! * [`reference_decoding::test_decode_reference_images`] — decode images
//!   produced by the `image` crate instead, so the **decoding** half is covered
//!   even when this FFmpeg build lacks the matching encoder (`libwebp` is
//!   missing from several distro builds, Homebrew included). Gated on the
//!   `image` feature, like the other image-crate tests.
//!
//! Rows whose encoder is not in this build report `SKIP <ext>` — pre-checked
//! with `find_encoder_by_name`, never by matching an error variant (the same
//! rule the other format matrices follow). At least one row must round-trip,
//! so an environment where nothing works cannot pass silently.
//!
//! Writing a *second* frame to a plain image filename is not covered here: the
//! `image2` muxer refuses it (`Cannot write more than one file with the same
//! name …`), and a sequence needs a `%03d` pattern — see
//! `src/io.rs::test_write_image_sequence` for that half.
//!
//! Requires the `ndarray` feature (frames come from the high-level API).

mod common;

use std::path::Path;

use rsmedia::{DecoderBuilder, EncoderBuilder, MediaType, Muxer, Reader, Result, StreamReader};
use rsmpeg::avcodec::AVCodec;
use rsmpeg::ffi;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const FPS: f32 = 25.0;

/// One row: an extension a caller would use, plus the encoder FFmpeg lists for
/// it. `.jpg` and `.jpeg` share `mjpeg`; both are exercised because the
/// extension is what picks the muxer here.
struct ImageSpec {
    ext: &'static str,
    codec: &'static str,
}

const IMAGES: &[ImageSpec] = &[
    ImageSpec {
        ext: "png",
        codec: "png",
    },
    ImageSpec {
        ext: "jpg",
        codec: "mjpeg",
    },
    ImageSpec {
        ext: "jpeg",
        codec: "mjpeg",
    },
    ImageSpec {
        ext: "bmp",
        codec: "bmp",
    },
    ImageSpec {
        ext: "tiff",
        codec: "tiff",
    },
    ImageSpec {
        ext: "gif",
        codec: "gif",
    },
    // WebP has no native FFmpeg encoder — it is always `libwebp`, an external
    // library this build may well have been compiled without.
    ImageSpec {
        ext: "webp",
        codec: "libwebp",
    },
];

/// The codec a still-image encoder writes, so the round trip can assert that
/// the file really carries it (`codecpar().codec_id`).
fn codec_id(encoder: &str) -> ffi::AVCodecID {
    match encoder {
        "png" => ffi::AV_CODEC_ID_PNG,
        "mjpeg" => ffi::AV_CODEC_ID_MJPEG,
        "bmp" => ffi::AV_CODEC_ID_BMP,
        "tiff" => ffi::AV_CODEC_ID_TIFF,
        "gif" => ffi::AV_CODEC_ID_GIF,
        "libwebp" => ffi::AV_CODEC_ID_WEBP,
        other => panic!("no codec id mapped for encoder `{other}`"),
    }
}

/// 本构建是否提供该编码器（缺它 ⇒ 该行跳过，先探测而不是靠错误变体判断）。
fn encoder_available(name: &str) -> bool {
    std::ffi::CString::new(name)
        .ok()
        .is_some_and(|name| AVCodec::find_encoder_by_name(&name).is_some())
}

/// Writes one frame to `path`, with the muxer and the encoder for `spec`.
fn write_single_image(path: &Path, spec: &ImageSpec) -> Result<()> {
    common::remove_test_output(path);

    let mut muxer = Muxer::new(path)?;
    let encoder = EncoderBuilder::new_video(WIDTH, HEIGHT)
        .with_fps(FPS)
        .with_codec_name(spec.codec.to_string())
        .build()?;
    let index = muxer.add_encoder(encoder)?;

    let frame = common::gradient_video_frame(WIDTH, HEIGHT, 0.5);
    muxer.mux(frame.to_avframe()?, index)?;
    muxer.finish()
}

/// Reads `path` back and checks it is one non-blank frame of the right size.
fn verify_single_image(path: &Path, spec: &ImageSpec) -> Result<()> {
    let bytes = std::fs::metadata(path)?.len();
    assert!(bytes > 0, "{}: the file is empty", spec.ext);

    let reader = StreamReader::new(path)?;
    let streams = reader.input().streams();
    assert_eq!(streams.len(), 1, "{}: stream count", spec.ext);
    assert!(
        streams[0].codecpar().codec_type().is_video(),
        "{}: a still image is one video stream",
        spec.ext
    );
    assert_eq!(
        streams[0].codecpar().codec_id,
        codec_id(spec.codec),
        "{}: the file carries the wrong codec",
        spec.ext
    );

    let mut reader = StreamReader::new(path)?;
    let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
    let mut frames = 0;
    while let Some(frame) = decoder.decode_frame(&mut reader)? {
        assert_eq!(
            (frame.width, frame.height),
            (WIDTH, HEIGHT),
            "{}: frame {frames} geometry",
            spec.ext
        );
        // Layout-agnostic: plane 0 is the luma plane for planar formats, the
        // packed array for interleaved ones and the palette indices for GIF.
        let plane = frame.data.plane(0).expect("a decoded image has a plane");
        let min = *plane.iter().min().expect("non-empty frame");
        let max = *plane.iter().max().expect("non-empty frame");
        assert!(
            max - min > 32,
            "{}: frame {frames} looks blank: {min}..{max}",
            spec.ext
        );
        frames += 1;
    }
    assert_eq!(
        frames, 1,
        "{}: a still image holds exactly one frame",
        spec.ext
    );

    Ok(())
}

/// Writes each still format and reads it straight back.
///
/// An encoder missing from this FFmpeg build skips its row; anything else
/// fails, so a broken image path cannot hide behind a skip.
#[test]
fn test_single_image_roundtrip() -> Result<()> {
    let mut passed = Vec::new();
    let mut skipped = Vec::new();
    let mut failed = Vec::new();

    for spec in IMAGES {
        if !encoder_available(spec.codec) {
            println!(
                "SKIP {}: {} is not in this FFmpeg build",
                spec.ext, spec.codec
            );
            skipped.push(spec.ext);
            continue;
        }

        let path = common::test_output_path("image_formats", &format!("roundtrip.{}", spec.ext));
        let result =
            write_single_image(&path, spec).and_then(|()| verify_single_image(&path, spec));
        common::remove_test_output(&path);

        match result {
            Ok(()) => {
                println!("{} ok", spec.ext);
                passed.push(spec.ext);
            }
            Err(error) => {
                println!("FAIL {}: {error:#}", spec.ext);
                failed.push((spec.ext, format!("{error:#}")));
            }
        }
    }

    println!(
        "images: {} passed {passed:?}, {} skipped {skipped:?}, {} failed",
        passed.len(),
        skipped.len(),
        failed.len()
    );
    assert!(failed.is_empty(), "image formats failed: {failed:#?}");
    assert!(
        !passed.is_empty(),
        "no still image round-tripped at all — check the FFmpeg build"
    );
    Ok(())
}

/// WebP encoding needs the external `libwebp` library: a build compiled without
/// it must report `Unsupported` — "this build cannot do it" — and **not**
/// `InvalidConfig`, so a caller (or a test matrix) skips instead of believing
/// it misconfigured the call. On a build that has `libwebp` the row above
/// covers the working path instead.
#[test]
fn test_webp_without_libwebp_reports_unsupported() -> Result<()> {
    if encoder_available("libwebp") {
        println!("SKIP: libwebp is in this FFmpeg build, see test_single_image_roundtrip");
        return Ok(());
    }

    let spec = ImageSpec {
        ext: "webp",
        codec: "libwebp",
    };
    let path = common::test_output_path("image_formats", "no_libwebp.webp");
    let err = write_single_image(&path, &spec).expect_err("writing webp without libwebp must fail");
    common::remove_test_output(&path);

    assert!(
        err.is_unsupported(),
        "a missing encoder is a build capability gap: {err:?}"
    );
    assert!(!err.is_invalid_config(), "{err:?}");
    assert!(
        err.to_string().contains("libwebp"),
        "the message must name the missing encoder: {err}"
    );
    Ok(())
}

/// The decoding half, independent of which encoders this build has.
///
/// The reference files come from the `image` crate rather than from FFmpeg, so
/// a build without `libwebp` still proves that rsmedia can *read* WebP. Gated
/// on the `image` feature (the crate the other image-crate tests use).
#[cfg(feature = "image")]
mod reference_decoding {
    use super::{DecoderBuilder, HEIGHT, MediaType, Reader, Result, StreamReader, WIDTH, ffi};
    use crate::common;
    use image::{ImageFormat, RgbImage};

    /// The formats to decode, with the extension FFmpeg guesses the demuxer from.
    const REFERENCES: &[(&str, ImageFormat, ffi::AVCodecID)] = &[
        ("png", ImageFormat::Png, ffi::AV_CODEC_ID_PNG),
        ("jpg", ImageFormat::Jpeg, ffi::AV_CODEC_ID_MJPEG),
        ("bmp", ImageFormat::Bmp, ffi::AV_CODEC_ID_BMP),
        ("tiff", ImageFormat::Tiff, ffi::AV_CODEC_ID_TIFF),
        ("gif", ImageFormat::Gif, ffi::AV_CODEC_ID_GIF),
        ("webp", ImageFormat::WebP, ffi::AV_CODEC_ID_WEBP),
    ];

    /// A gradient plain enough for any of the formats above to store.
    fn reference_image() -> RgbImage {
        RgbImage::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
            image::Rgb([
                (x * 255 / WIDTH as u32) as u8,
                (y * 255 / HEIGHT as u32) as u8,
                0x40,
            ])
        })
    }

    #[test]
    fn test_decode_reference_images() -> Result<()> {
        let image = reference_image();
        let mut decoded_formats = Vec::new();

        for (ext, format, codec_id) in REFERENCES {
            let path = common::test_output_path("image_formats", &format!("ref.{ext}"));
            common::remove_test_output(&path);
            image
                .save_with_format(&path, *format)
                .unwrap_or_else(|e| panic!("failed to write the {ext} reference image: {e}"));

            let reader = StreamReader::new(&path)?;
            let streams = reader.input().streams();
            assert_eq!(streams.len(), 1, "{ext}: stream count");
            assert!(streams[0].codecpar().codec_type().is_video());
            assert_eq!(
                streams[0].codecpar().codec_id,
                *codec_id,
                "{ext}: demuxer did not report the expected codec"
            );

            // 解码器都是 FFmpeg 内置的，缺席就是构建坏了 —— 不给跳过口子，
            // `build_from_reader` 会以 `Unsupported` 明确失败。

            let mut reader = StreamReader::new(&path)?;
            let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
            let mut frames = 0;
            while let Some(frame) = decoder.decode_frame(&mut reader)? {
                assert_eq!(
                    (frame.width, frame.height),
                    (WIDTH, HEIGHT),
                    "{ext}: frame {frames} geometry"
                );
                let plane = frame.data.plane(0).expect("a decoded image has a plane");
                let min = *plane.iter().min().expect("non-empty frame");
                let max = *plane.iter().max().expect("non-empty frame");
                assert!(max - min > 32, "{ext}: frame {frames} looks blank");
                frames += 1;
            }
            assert_eq!(frames, 1, "{ext}: a still image holds exactly one frame");

            println!("{ext} decoded");
            decoded_formats.push(*ext);
            common::remove_test_output(&path);
        }

        // 每张真值图都必须真的解码过：这张表里的解码器都是 FFmpeg 内置的，
        // 没有可跳过的行（`webp` 解码器就与 `libwebp` 编码器无关）。
        assert_eq!(
            decoded_formats,
            REFERENCES
                .iter()
                .map(|(ext, _, _)| *ext)
                .collect::<Vec<_>>(),
            "some reference images were not decoded"
        );
        Ok(())
    }
}
