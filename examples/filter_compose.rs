//! 多输入滤镜图（合成）示例：把多路画面 / 声音合成一路，并直接编码落盘。
//!
//! 自包含：不需要任何外部素材，`cargo run --example filter_compose` 即可运行。
//! 五个高频形态各跑一遍，并用**像素 / 采样值**校验合成结果，而不是只打印 spec：
//!
//! | 形态 | 输入 → 输出 | 校验点 |
//! |---|---|---|
//! | `hstack` | 2 路视频 → 1 路宽屏 | 左右半屏亮度，并编码写入 `/tmp/filter_compose.mp4` |
//! | `vstack` | 2 路视频 → 1 路竖屏 | 上下半屏亮度 |
//! | `overlay` | 基底 + 叠加层 → 1 路 | 叠加区域内 / 外的亮度 |
//! | `concat` | 2 段 → 首尾相接的 1 路 | 输出帧数与两段的先后顺序 |
//! | `amix` | 2 路音频 → 混音 1 路 | 每个采样点的混音值 |
//!
//! 需要自由组装更多路输入 / 输出时用 `FilterGraphBuilder::new()`：
//! `add_input_with(标签, 端点)` → `add_node(FilterNode::new(滤镜).with_inputs([标签..]).with_label(..))`
//! → `add_output(上游标签, 端点)` → `build()`。端点声明每一路进出的尺寸 / 格式 / 时间基；
//! 标签把节点接起来（多输入滤镜的每个输入都要显式绑定标签，顺序即 pad 顺序）。

use anyhow::{Context, Result};

use rsmedia::ffmpeg::ffi::AVRational;
use rsmedia::filter::{AudioEndpoint, FilterGraphBuilder, VideoEndpoint};
use rsmedia::{EncoderBuilder, MediaFrame, Muxer, PixelFormat, SampleFormat};

/// 合成用的单路画面尺寸：每一路输入都是 160x120（YUV420P 要求宽高为偶数）。
const W: u32 = 160;
const H: u32 = 120;
/// 帧率，同时也是视频端点的时间基（1/25）。
const FPS: i32 = 25;
/// 输入音频：48kHz 单声道，每帧 1024 个采样点。
const SAMPLE_RATE: i32 = 48000;
const SAMPLES_PER_FRAME: usize = 1024;
/// 混音两路的常量采样值。`amix` 默认按路数归一化，故期望值 = (A + B) / 2。
const MIX_A: f32 = 0.5;
const MIX_B: f32 = -0.25;

const OUTPUT_PATH: &str = "/tmp/filter_compose.mp4";

fn main() -> Result<()> {
    rsmedia::init()?;

    let frames = compose_hstack_to_file()?;
    compose_vstack()?;
    compose_overlay()?;
    compose_concat()?;
    compose_amix()?;

    println!("\n五个合成场景全部校验通过；hstack 的结果已编码为 {OUTPUT_PATH}（{frames} 帧）");
    Ok(())
}

/// `hstack`：两路 160x120 并排成 320x120，逐帧校验左右半屏后编码落盘。
///
/// 这是「合成 + 转码」一条链：滤镜图输出的是普通 `AVFrame`，可直接交给 `Muxer`。
fn compose_hstack_to_file() -> Result<usize> {
    println!("== 1) hstack：两路画面并排（320x120），并编码为 mp4 ==");

    let input = video_endpoint(W, H);
    let mut graph = FilterGraphBuilder::hstack(&[input, input], video_endpoint(W * 2, H))
        .context("failed to build hstack graph")?;

    let encoder = EncoderBuilder::new_video(W * 2, H)
        .with_fps(FPS as f32)
        .build()?;
    let enc_tb = encoder.time_base();
    let mut muxer = Muxer::new(OUTPUT_PATH)?;
    let stream = muxer.add_encoder(encoder)?;

    let mut written = 0usize;
    for index in 0..FPS as i64 {
        // 图输入 0 = 左半（暗），图输入 1 = 右半（亮）。
        let left = solid_frame(W, H, 40, index)?;
        let right = solid_frame(W, H, 200, index)?;
        graph.push_frame_to(0, Some(left.to_avframe()?))?;
        graph.push_frame_to(1, Some(right.to_avframe()?))?;

        // 两路各凑齐一帧，hstack 就吐一帧合成结果。
        let frame = graph
            .receive_frame_from(0)?
            .context("hstack should emit one frame per input pair")?;
        verify_split(&MediaFrame::<u8>::from_avframe(&frame)?, 40, 200)?;

        // 图的输出时间基与编码器一致（都是 1/25），pts 直接用帧序号。
        let mut frame = frame;
        frame.set_pts(index);
        frame.set_time_base(enc_tb);
        muxer.mux(frame, stream)?;
        written += 1;
    }

    // 两路 EOF 之后图里可能还有残余帧（hstack 无延迟，通常为 0）。
    for mut frame in graph.drain_output(0)? {
        frame.set_pts(written as i64);
        frame.set_time_base(enc_tb);
        muxer.mux(frame, stream)?;
        written += 1;
    }
    muxer.finish()?;

    println!("   已写入 {OUTPUT_PATH}：{written} 帧 320x120，左右半屏亮度校验通过");
    Ok(written)
}

/// `vstack`：两路 160x120 叠成 160x240，校验上下半屏。
fn compose_vstack() -> Result<()> {
    println!("== 2) vstack：两路画面上下堆叠（160x240）==");

    let input = video_endpoint(W, H);
    let mut graph = FilterGraphBuilder::vstack(&[input, input], video_endpoint(W, H * 2))
        .context("failed to build vstack graph")?;

    let top_frame = solid_frame(W, H, 30, 0)?;
    let bottom_frame = solid_frame(W, H, 220, 0)?;
    graph.push_frame_to(0, Some(top_frame.to_avframe()?))?;
    graph.push_frame_to(1, Some(bottom_frame.to_avframe()?))?;

    let frame = graph
        .receive_frame_from(0)?
        .context("vstack should emit a frame once both inputs have one")?;
    let composed = MediaFrame::<u8>::from_avframe(&frame)?;
    anyhow::ensure!(
        (composed.width, composed.height) == (W, H * 2),
        "unexpected vstack output size: {}x{}",
        composed.width,
        composed.height
    );

    let planes = composed
        .data
        .as_planes()
        .context("YUV420P frames are planar")?;
    let top = planes[0][[H as usize / 2, W as usize / 2]];
    let bottom = planes[0][[(H + H / 2) as usize, W as usize / 2]];
    anyhow::ensure!(
        top == 30 && bottom == 220,
        "vstack mismatch: top half = {top}, bottom half = {bottom}"
    );
    println!("   上半屏 = {top}、下半屏 = {bottom}，校验通过");
    Ok(())
}

/// `overlay`：把 40x30 的叠加层贴到基底 (100, 20) 处（画中画 / 水印）。
fn compose_overlay() -> Result<()> {
    println!("== 3) overlay：小图叠到大图上（画中画 / 水印）==");

    let (overlay_w, overlay_h) = (40u32, 30u32);
    let (x, y) = (100usize, 20usize);
    let mut graph = FilterGraphBuilder::overlay(
        video_endpoint(W, H),
        video_endpoint(overlay_w, overlay_h),
        &x.to_string(),
        &y.to_string(),
        video_endpoint(W, H),
    )
    .context("failed to build overlay graph")?;

    let base = solid_frame(W, H, 20, 0)?;
    let over = solid_frame(overlay_w, overlay_h, 240, 0)?;
    graph.push_frame_to(0, Some(base.to_avframe()?))?;
    graph.push_frame_to(1, Some(over.to_avframe()?))?;

    let frame = graph
        .receive_frame_from(0)?
        .context("overlay should emit a frame once both inputs have one")?;
    let composed = MediaFrame::<u8>::from_avframe(&frame)?;
    let planes = composed
        .data
        .as_planes()
        .context("YUV420P frames are planar")?;

    let inside = planes[0][[y + overlay_h as usize / 2, x + overlay_w as usize / 2]];
    let outside = planes[0][[y / 2, x / 2]];
    anyhow::ensure!(
        inside == 240 && outside == 20,
        "overlay mismatch: inside = {inside}, outside = {outside}"
    );
    println!("   叠加区中心 = {inside}、基底区 = {outside}，校验通过");
    Ok(())
}

/// `concat`：两段各 5 帧首尾相接，校验输出帧数与两段的先后。
fn compose_concat() -> Result<()> {
    println!("== 4) concat：两段视频首尾相接 ==");

    const SEGMENT_FRAMES: i64 = 5;
    let input = video_endpoint(W, H);
    let mut graph = FilterGraphBuilder::concat(&[input, input], video_endpoint(W, H))
        .context("failed to build concat graph")?;

    for index in 0..SEGMENT_FRAMES {
        let first = solid_frame(W, H, 20, index)?;
        let second = solid_frame(W, H, 220, index)?;
        graph.push_frame_to(0, Some(first.to_avframe()?))?;
        graph.push_frame_to(1, Some(second.to_avframe()?))?;
    }

    // 输入 0 的帧先出；推完 EOF 后由 drain 收尾接上输入 1 的帧。
    let mut lumas = Vec::new();
    while let Some(frame) = graph.receive_frame_from(0)? {
        lumas.push(center_luma(&MediaFrame::<u8>::from_avframe(&frame)?)?);
    }
    for frame in graph.drain_output(0)? {
        lumas.push(center_luma(&MediaFrame::<u8>::from_avframe(&frame)?)?);
    }

    let mut expected = vec![20u8; SEGMENT_FRAMES as usize];
    expected.extend(vec![220u8; SEGMENT_FRAMES as usize]);
    anyhow::ensure!(
        lumas == expected,
        "concat must keep the input order: got {lumas:?}, expected {expected:?}"
    );
    println!(
        "   输出 {} 帧：前 {} 帧来自输入 0、后 {} 帧来自输入 1，校验通过",
        lumas.len(),
        SEGMENT_FRAMES,
        SEGMENT_FRAMES
    );
    Ok(())
}

/// `amix`：两路音频混音（`amix` 默认按路数归一化：输出 = (A + B) / 2）。
fn compose_amix() -> Result<()> {
    println!("== 5) amix：两路音频混音 ==");

    const INPUT_FRAMES: usize = 4;
    let endpoint = audio_endpoint();
    let mut graph = FilterGraphBuilder::amix(&[endpoint, endpoint], "longest", endpoint)
        .context("failed to build amix graph")?;

    for index in 0..INPUT_FRAMES as i64 {
        let a = tone_frame(MIX_A, index)?;
        let b = tone_frame(MIX_B, index)?;
        graph.push_frame_to(0, Some(a.to_avframe()?))?;
        graph.push_frame_to(1, Some(b.to_avframe()?))?;
    }

    let mut frames = Vec::new();
    while let Some(frame) = graph.receive_frame_from(0)? {
        frames.push(frame);
    }
    frames.extend(graph.drain_output(0)?);

    let expected = (MIX_A + MIX_B) / 2.0;
    let mut samples = 0usize;
    let mut worst: f32 = 0.0;
    for frame in &frames {
        let media = MediaFrame::<f32>::from_avframe(frame)?;
        let planes = media.data.as_planes().context("FLTP frames are planar")?;
        for &value in planes[0].iter() {
            worst = worst.max((value - expected).abs());
            samples += 1;
        }
    }

    anyhow::ensure!(
        samples == INPUT_FRAMES * SAMPLES_PER_FRAME,
        "amix should emit every sample of the longest input, got {samples}"
    );
    anyhow::ensure!(
        worst <= 1e-4,
        "amix sample value mismatch: expected {expected}, max deviation {worst:e}"
    );
    println!("   输出 {samples} 个采样点，混音值 {expected}（最大偏差 {worst:e}），校验通过");
    Ok(())
}

/// 这一批场景统一的视频端点：YUV420P、25fps、时间基 1/25。
fn video_endpoint(width: u32, height: u32) -> VideoEndpoint {
    VideoEndpoint::new(
        width as i32,
        height as i32,
        PixelFormat::YUV420P,
        AVRational { num: 1, den: FPS },
        AVRational { num: FPS, den: 1 },
    )
}

/// 统一的音频端点：48kHz 单声道 FLTP，时间基 1/48000。
fn audio_endpoint() -> AudioEndpoint {
    AudioEndpoint::new(
        1,
        SAMPLE_RATE,
        SampleFormat::FLTP,
        AVRational {
            num: 1,
            den: SAMPLE_RATE,
        },
    )
}

/// 生成一张 YUV420P 纯色帧：亮度面全填 `luma`，两个色度面填中性 128。
fn solid_frame(width: u32, height: u32, luma: u8, pts: i64) -> Result<MediaFrame<u8>> {
    let mut frame = MediaFrame::<u8>::new_video_frame(width, height, PixelFormat::YUV420P)?;
    {
        let planes = frame
            .data
            .as_planes_mut()
            .context("YUV420P frames are planar")?;
        planes[0].fill(luma);
        planes[1].fill(128);
        planes[2].fill(128);
    }
    frame.set_pts(pts);
    frame.set_time_base(AVRational { num: 1, den: FPS });
    Ok(frame)
}

/// 生成一帧 48kHz 单声道 FLTP 音频，所有采样点填同一个值。
fn tone_frame(value: f32, pts: i64) -> Result<MediaFrame<f32>> {
    let nb_samples = SAMPLES_PER_FRAME as u32;
    let mut frame =
        MediaFrame::<f32>::new_audio_frame(SampleFormat::FLTP, 1, nb_samples, SAMPLE_RATE as u32)?;
    {
        let planes = frame
            .data
            .as_planes_mut()
            .context("FLTP frames are planar")?;
        planes[0].fill(value);
    }
    frame.set_pts(pts);
    Ok(frame)
}

/// 校验 320x120 的合成帧：左半亮度是 `left`、右半是 `right`。
fn verify_split(frame: &MediaFrame<u8>, left: u8, right: u8) -> Result<()> {
    let planes = frame
        .data
        .as_planes()
        .context("YUV420P frames are planar")?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    for (x, expected) in [(10usize, left), (width - 10, right)] {
        for y in [10usize, height / 2, height - 10] {
            let got = planes[0][[y, x]];
            anyhow::ensure!(
                got == expected,
                "hstack pixel ({x},{y}) = {got}, expected {expected}"
            );
        }
    }
    Ok(())
}

/// 读一张 YUV420P 帧中心点的亮度值。
fn center_luma(frame: &MediaFrame<u8>) -> Result<u8> {
    let planes = frame
        .data
        .as_planes()
        .context("YUV420P frames are planar")?;
    Ok(planes[0][[frame.height as usize / 2, frame.width as usize / 2]])
}
