//! RIIR: https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/vaapi_encode.c
mod common;
use anyhow::{Context, Result};
use common::test_output_path;
use rsmpeg::{
    avcodec::{AVCodec, AVCodecContext},
    avutil::{AVFrame, AVHWDeviceContext, ra},
    error::RsmpegError,
    ffi::{
        AV_HWDEVICE_TYPE_CUDA, AV_HWDEVICE_TYPE_VAAPI, AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NV12, AV_PIX_FMT_VAAPI, AV_PIX_FMT_VIDEOTOOLBOX,
        AVHWDeviceType, AVPixelFormat,
    },
};
use std::{
    ffi::CStr,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    slice,
};

fn set_hwframe_ctx(
    avctx: &mut AVCodecContext,
    hw_device_ctx: &AVHWDeviceContext,
    width: i32,
    height: i32,
    hw_format: AVPixelFormat,
    sw_format: AVPixelFormat,
) -> Result<()> {
    let mut hw_frames_ref = hw_device_ctx.hwframe_ctx_alloc();
    hw_frames_ref.data().format = hw_format;
    hw_frames_ref.data().sw_format = sw_format;
    hw_frames_ref.data().width = width;
    hw_frames_ref.data().height = height;
    hw_frames_ref.data().initial_pool_size = 20;

    hw_frames_ref
        .init()
        .context("Failed to initialize VAAPI frame context")?;

    avctx.set_hw_frames_ctx(hw_frames_ref);

    Ok(())
}

/// Writes `frames` frames of synthetic NV12 (luma ramp + neutral chroma) as raw
/// planes — the exact layout [`hw_encode`] reads back.
///
/// The upstream example consumes a checked-in `bear.yuv`, which this repo does
/// not ship; the resulting `ENOENT` made all three hardware tests unrunnable
/// even on a machine that has the device. Generating the input instead keeps
/// the tests faithful to the example while depending on nothing but a device.
fn write_nv12_input(path: &Path, width: i32, height: i32, frames: usize) -> Result<()> {
    let (w, h) = (width as usize, height as usize);
    let mut file = File::create(path).context("Fail to create input file")?;
    let uv = vec![128u8; w * h / 2];
    for frame in 0..frames {
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                // Diagonal ramp that shifts per frame, so no two frames are equal.
                y[row * w + col] = ((col + row + frame * 8) % 256) as u8;
            }
        }
        file.write_all(&y).context("Write Y failed.")?;
        file.write_all(&uv).context("Write UV failed.")?;
    }
    Ok(())
}

/// The output of a hardware encode must actually be there: without this the tests
/// only prove the calls did not error — the "writes it, never checks it" pattern
/// that let the missing input asset go unnoticed.
fn assert_output_non_empty(path: &Path) {
    let len = std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("no output at {path:?}: {e}"))
        .len();
    assert!(len > 0, "encoder produced an empty file: {path:?}");
}

fn encode_write(
    avctx: &mut AVCodecContext,
    frame: Option<&AVFrame>,
    fout: &mut File,
) -> Result<()> {
    avctx.send_frame(frame).context("Send frame failed")?;
    loop {
        let mut packet = match avctx.receive_packet() {
            Ok(packet) => packet,
            Err(RsmpegError::EncoderDrainError) | Err(RsmpegError::EncoderFlushedError) => {
                break;
            }
            Err(e) => Err(e).context("Receive packet failed.")?,
        };
        packet.set_stream_index(0);
        let data = unsafe { slice::from_raw_parts(packet.data, packet.size as usize) };
        fout.write_all(data).context("Write output frame failed.")?;
    }
    Ok(())
}

struct HwEncodeConfig<'a> {
    input: &'a Path,
    output: &'a std::path::Path,
    width: i32,
    height: i32,
    encode_codec: &'a CStr,
    device_type: AVHWDeviceType,
    hw_format: AVPixelFormat,
    sw_format: AVPixelFormat,
}

fn hw_encode(config: &HwEncodeConfig<'_>) -> Result<()> {
    let size = config.width as usize * config.height as usize;

    let mut fin = File::open(config.input).context("Fail to open input file")?;
    let mut fout = File::create(config.output).context("Fail to open output file")?;

    let hw_device_ctx = AVHWDeviceContext::create(config.device_type, None, None, 0)
        .context("Failed to create a VAAPI device")?;

    let codec =
        AVCodec::find_encoder_by_name(config.encode_codec).context("Could not find encoder.")?;

    let mut avctx = AVCodecContext::new(&codec);

    avctx.set_width(config.width);
    avctx.set_height(config.height);
    avctx.set_time_base(ra(1, 25));
    avctx.set_framerate(ra(25, 1));
    avctx.set_sample_aspect_ratio(ra(1, 1));
    avctx.set_pix_fmt(config.hw_format);

    set_hwframe_ctx(
        &mut avctx,
        &hw_device_ctx,
        config.width,
        config.height,
        config.hw_format,
        config.sw_format,
    )
    .context("Failed to set hwframe context.")?;

    avctx
        .open(None)
        .context("Cannot open video encoder codec")?;

    loop {
        let mut sw_frame = AVFrame::new();

        // read data into software frame, and transfer them into hw frame
        sw_frame.set_width(config.width);
        sw_frame.set_height(config.height);
        sw_frame.set_format(config.sw_format);
        sw_frame.get_buffer(0).context("Get buffer failed.")?;

        let y = unsafe { slice::from_raw_parts_mut(sw_frame.data_mut()[0], size) };
        match fin.read_exact(y) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            e @ Err(_) => e.context("Read Y failed.")?,
        }
        let uv = unsafe { slice::from_raw_parts_mut(sw_frame.data_mut()[1], size / 2) };
        match fin.read_exact(uv) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            e @ Err(_) => e.context("Read UV failed.")?,
        }

        let mut hw_frame = AVFrame::new();
        avctx
            .hw_frames_ctx_mut()
            .unwrap()
            .get_buffer(&mut hw_frame)
            .context("Get buffer failed")?;
        hw_frame
            .hwframe_transfer_data(&sw_frame)
            .context("Error while transferring frame data to surface.")?;

        encode_write(&mut avctx, Some(&hw_frame), &mut fout).context("Failed to encode.")?;
    }

    encode_write(&mut avctx, None, &mut fout).context("Failed to encode.")?;
    Ok(())
}

#[test]
#[ignore = "requires a real VAAPI device (CI runners have none)"]
fn vaapi_encode_test_vaapi() {
    let (width, height) = (320, 180);
    let input = test_output_path("vaapi_encode", "input.yuv");
    write_nv12_input(&input, width, height, 8).unwrap();
    let output_path = test_output_path("vaapi_encode", "vaapi_encode_test_vaapi.h264");
    hw_encode(&HwEncodeConfig {
        input: &input,
        output: &output_path,
        width,
        height,
        encode_codec: c"h264_vaapi",
        device_type: AV_HWDEVICE_TYPE_VAAPI,
        hw_format: AV_PIX_FMT_VAAPI,
        sw_format: AV_PIX_FMT_NV12,
    })
    .unwrap();
    assert_output_non_empty(&output_path);
}

/// You should test this with nvenc enabled in compilation(e.g. utils/linux_ffmpeg.rs) https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENC
///
/// I use this rather than vaapi test(since I don't have a vaapi compatible device).
///
/// They are almost the same, the only differences:
///
/// - device_type:  AV_HWDEVICE_TYPE_CUDA for nvenc,    AV_HWDEVICE_TYPE_VAAPI for vaapi
/// - hw_format:    AV_PIX_FMT_CUDA for nvenc,          AV_PIX_FMT_VAAPI for vaapi
#[test]
#[ignore = "requires an NVIDIA GPU (CI runners have none)"]
fn nvenc_encode_test_nvenc() {
    let (width, height) = (320, 180);
    let input = test_output_path("nvenc_encode", "input.yuv");
    write_nv12_input(&input, width, height, 8).unwrap();
    let output_path = test_output_path("nvenc_encode", "nvenc_encode_test_nvenc.h264");
    hw_encode(&HwEncodeConfig {
        input: &input,
        output: &output_path,
        width,
        height,
        encode_codec: c"h264_nvenc",
        device_type: AV_HWDEVICE_TYPE_CUDA,
        hw_format: AV_PIX_FMT_CUDA,
        sw_format: AV_PIX_FMT_NV12,
    })
    .unwrap();
    assert_output_non_empty(&output_path);
}

#[test]
#[ignore = "requires a machine with working VideoToolbox (CI runners have none)"]
fn toolbox_encode_test_videotoolbox() {
    let (width, height) = (320, 180);
    let input = test_output_path("toolbox_encode", "input.yuv");
    write_nv12_input(&input, width, height, 8).unwrap();
    let output_path = test_output_path("toolbox_encode", "toolbox_encode_test_h264.h264");
    hw_encode(&HwEncodeConfig {
        input: &input,
        output: &output_path,
        width,
        height,
        encode_codec: c"h264_videotoolbox",
        device_type: AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        hw_format: AV_PIX_FMT_VIDEOTOOLBOX,
        sw_format: AV_PIX_FMT_NV12,
    })
    .unwrap();
    assert_output_non_empty(&output_path);
}
