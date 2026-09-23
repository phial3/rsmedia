//! 证件照 / 单图处理示例：一条完整的"图片进、图片出"管线。
//!
//! 覆盖证件照处理最常见的四类需求，全部用 rsmedia 现有类型组合，不引入新类型：
//!
//! | 需求 | 实现 | 默认 |
//! |---|---|---|
//! | 指定像素尺寸（一寸 295x413） | `crop` + `scale` 滤镜（先按目标比例裁，再缩放到精确尺寸） | 开启 |
//! | 指定文件大小（min~max KB） | mjpeg `qscale` 二分搜索，`BufferWriter` 内存内量字节 | 开启 |
//! | 换背景色 | 多输入滤镜图：`colorkey` 抠掉原底色 -> `overlay` 到新底色 | **关闭**（见下） |
//! | 美颜（磨皮/提亮/锐化） | `smartblur` + `lutyuv` + `unsharp`（单输入滤镜链） | **关闭**（见下） |
//!
//! 换底与美颜默认关闭的原因（实测数据）：
//! - `colorkey` 只认**单一均匀色**。真实照片背景往往不均（阴影、渐变、杂物），
//!   且米色/浅灰墙面与肤色在色度上接近——similarity 调大抠穿人脸、调小抠不净，
//!   两头都坏。仅纯色底（摄影棚蓝/红/白背景布）照片适合开启。
//! - 美颜链会真实损失画质：smartblur 抹平细节 + lutyuv 整体提亮 + unsharp
//!   再锐化，实测对源图亮度 PSNR 仅 ~29dB。证件照多数场景要求"忠实原图"。
//!
//! 真人发丝级抠图、人脸自动定位不在 FFmpeg 能力内，需要外部 AI 模型
//! 产出 alpha 蒙版后接进本示例的 overlay 那一步（rsmedia 负责合成，AI 库负责 mask）。
//!
//! 用法：所有参数都是 `main()` 顶部的变量，按需修改后运行：
//! `cargo run --release --example id_photo`

use anyhow::{Context, Result};

use rsmedia::ffmpeg::ffi::AVRational;
use rsmedia::filter::{Filter, FilterGraphBuilder, FilterNode, VideoEndpoint, video};
use rsmedia::io::BufferWriter;
use rsmedia::{
    DecoderBuilder, EncoderBuilder, MediaFrame, MediaType, Options, PixelFormat, StreamReader,
};

/// 合成模式源图尺寸：故意用非 295:413 比例的底图，验证裁剪。
const SRC_W: usize = 600;
const SRC_H: usize = 800;
/// 合成图的蓝底颜色
const SYNTH_BG_RGB: (u8, u8, u8) = (215, 139, 67);

/// 换底参数：(抠像色, 新底色, colorkey 相似度)。
type BgSwap = ((u8, u8, u8), (u8, u8, u8), f32);

fn main() -> Result<()> {
    rsmedia::init()?;

    // ==================== 可调参数（全部在这里，改这些变量即可） ====================

    // 输入照片路径；None = 用程序内合成的蓝底人像自测全流程。
    // 真实照片示例：let source: Option<&str> = Some("/tmp/20260923-110717.jpg");
    let source: Option<&str> = None;

    // 输出路径。
    let out_path = "/tmp/rsmedia_id_photo.jpg";

    // 目标像素尺寸：一寸 295x413；蓝底报名照常用 358x441。
    let (target_w, target_h) = (295usize, 413usize);

    // 文件大小区间（KB）。min 设 0 表示不设下限。
    let (min_kb, max_kb) = (0u64, 50u64);

    // 换底：None = 不换底（默认，忠实原图）。
    // Some((抠像色, 新底色, 相似度))：仅当源图是**均匀纯色背景**时启用，
    // 否则 colorkey 会抠穿人脸或留下原底残色（见文件头注释）。
    // 例：抠蓝底换白底 -> Some(((67,139,215), (255,255,255), 0.15))
    let background_swap: Option<BgSwap> = None;

    // 美颜（磨皮 + 轻微提亮 + 锐化）：默认 false，会损失画质（实测 ~29dB）。
    let beautify = false;

    // ==================== 管线 ====================

    let photo = match source {
        Some(path) => load_photo(path)?,
        None => {
            println!("source 为 None，使用合成的蓝底人像（{SRC_W}x{SRC_H}）");
            synthetic_id_photo()
        }
    };

    // 1) 换底（可选）：抠掉原底色 -> 垫上新底色。
    let photo = match background_swap {
        Some((key_rgb, new_bg, similarity)) => swap_background(&photo, key_rgb, similarity, new_bg)
            .context("step 1: swap background")?,
        None => {
            println!("跳过换底（background_swap = None）");
            photo
        }
    };
    // 2) 按目标比例居中裁剪 + 缩放到精确像素（+ 可选美颜）。
    let sized = crop_scale(&photo, target_w, target_h, beautify).context("step 2: crop/scale")?;
    // 3) 压到目标文件大小区间 [min_kb, max_kb]（二分 mjpeg qscale）。
    let bytes =
        encode_to_size(&sized, min_kb * 1024, max_kb * 1024).context("step 3: encode to size")?;

    std::fs::write(out_path, &bytes).context("failed to write output")?;
    println!(
        "已写出 {out_path}：{target_w}x{target_h}，{} KB（要求 {min_kb}~{max_kb} KB）",
        bytes.len() / 1024
    );

    // 合成模式顺带校验：尺寸正确；开了换底则校验新底色生效且人像未被盖住。
    if source.is_none() {
        verify(&sized, target_w, target_h, background_swap.is_some())?;
        println!("校验通过");
    }
    Ok(())
}

/// 从磁盘读一张照片（jpg/png 由 FFmpeg 自动探测），解码第一帧为 RGB24。
fn load_photo(path: &str) -> Result<MediaFrame<u8>> {
    let mut reader = StreamReader::new(path)?;
    let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
        .with_pix_fmt(PixelFormat::RGB24)
        .build_from_reader(&reader)?;
    decoder
        .decode_frame(&mut reader)?
        .context("no image frame decoded")
}

/// 合成一张"证件照"：蓝底 + 中间一个深色人像轮廓（头 + 肩）。
fn synthetic_id_photo() -> MediaFrame<u8> {
    let mut frame =
        MediaFrame::<u8>::new_video_frame(SRC_W, SRC_H, PixelFormat::RGB24).expect("rgb frame");
    {
        let samples = frame.data.as_packed_mut().expect("RGB24 is interleaved");
        for y in 0..SRC_H {
            for x in 0..SRC_W {
                // 头：以 (300, 300) 为心、半径 120 的圆；肩：y>480 的梯形。
                let head = ((x as i32 - 300).pow(2) + (y as i32 - 300).pow(2)) < 120 * 120;
                let shoulder = y > 480 && (x as i32 - 300).abs() * 2 < (y - 480) as i32 * 3 + 200;
                let [r, g, b] = if head || shoulder {
                    [60, 50, 45]
                } else {
                    [SYNTH_BG_RGB.0, SYNTH_BG_RGB.1, SYNTH_BG_RGB.2]
                };
                samples[[y, x, 0]] = r;
                samples[[y, x, 1]] = g;
                samples[[y, x, 2]] = b;
            }
        }
    }
    frame.set_pts(0);
    frame
}

/// 抠掉 `key_rgb` 底色、垫上 `new_bg` 新底色：`colorkey` 把原底色抠成透明，
/// `overlay` 到纯色新底上。
///
/// 两路输入尺寸相同（新底 = 照片尺寸），故新底用同尺寸的纯色帧。
///
/// 仅适用于**均匀纯色背景**的照片：米色/浅灰墙面与肤色色度接近，similarity
/// 稍大就会连人脸一起抠穿（实测 0.25 时人脸大面积变新底色）。
fn swap_background(
    photo: &MediaFrame<u8>,
    key_rgb: (u8, u8, u8),
    similarity: f32,
    new_bg: (u8, u8, u8),
) -> Result<MediaFrame<u8>> {
    let (w, h) = (photo.width as i32, photo.height as i32);
    let endpoint = rgb_endpoint(w, h);

    let mut builder = FilterGraphBuilder::new();
    builder.add_input_with("photo", endpoint);
    builder.add_input_with("bg", endpoint);
    // colorkey 作用于照片那一路，输出打标签供 overlay 引用。
    builder.add_node(
        FilterNode::new(video::colorkey(
            &format!("0x{:02X}{:02X}{:02X}", key_rgb.0, key_rgb.1, key_rgb.2),
            similarity,
            0.1,
        ))
        .with_inputs(["photo"])
        .with_label("keyed"),
    );
    // overlay 输入 0 = 主画面（新底色），输入 1 = 叠加层（抠像后的照片，
    // 原底色处已透明，透出白底；人像处不透明，盖在白底上）。
    builder.add_node(FilterNode::new(video::overlay("0", "0", None)).with_inputs(["bg", "keyed"]));
    builder.add_output_tail(endpoint);
    let mut graph = builder
        .build()
        .context("failed to build background-swap graph")?;

    let mut bg = MediaFrame::<u8>::new_video_frame(photo.width, photo.height, PixelFormat::RGB24)?;
    {
        let samples = bg.data.as_packed_mut().context("RGB24 is interleaved")?;
        for y in 0..bg.height {
            for x in 0..bg.width {
                samples[[y, x, 0]] = new_bg.0;
                samples[[y, x, 1]] = new_bg.1;
                samples[[y, x, 2]] = new_bg.2;
            }
        }
    }
    bg.set_pts(0);

    graph.push_frame_to(0, Some(photo.to_avframe()?))?;
    graph.push_frame_to(1, Some(bg.to_avframe()?))?;
    let frame = graph
        .receive_frame_from(0)?
        .context("overlay should emit once both inputs have a frame")?;
    Ok(MediaFrame::<u8>::from_avframe(&frame)?)
}

/// 按目标宽高比居中裁剪，缩放到精确像素；`beautify` 为 true 时追加美颜链。
///
/// 单输入线性滤镜链，直接产出内存帧（不落盘、不二次编码）。
fn crop_scale(
    photo: &MediaFrame<u8>,
    tw: usize,
    th: usize,
    beautify: bool,
) -> Result<MediaFrame<u8>> {
    let target_ratio = tw as f64 / th as f64;
    let (w, h) = (photo.width, photo.height);
    // 居中裁剪框：源更宽则裁宽，更高则裁高。
    let (cw, ch) = if (w as f64 / h as f64) > target_ratio {
        ((h as f64 * target_ratio) as u32, h as u32)
    } else {
        (w as u32, (w as f64 / target_ratio) as u32)
    };
    let cw = cw.min(w as u32);
    let ch = ch.min(h as u32);
    let cx = ((w - cw as usize) / 2) as i32;
    let cy = ((h - ch as usize) / 2) as i32;

    let endpoint = rgb_endpoint(w as i32, h as i32);
    let out_endpoint = rgb_endpoint(tw as i32, th as i32);
    let mut chain: Vec<Filter> = vec![
        video::crop(cx, cy, cw, ch),
        video::scale(tw as u32, th as u32, Some("lanczos")),
    ];
    if beautify {
        // 磨皮 + 轻微提亮 + 锐化，证件照标准美颜三连（有画质损失，默认关）。
        chain.push(video::smartblur(0.1, 3.0));
        chain.push(video::lutyuv(Some("val+8"), None, None));
        chain.push(video::unsharp());
    }

    let mut builder = FilterGraphBuilder::new();
    builder.add_input_with("src", endpoint);
    for filter in chain {
        builder.add_node(filter);
    }
    builder.add_output_tail(out_endpoint);
    let mut graph = builder
        .build()
        .context("failed to build crop/scale graph")?;

    graph.push_frame_to(0, Some(photo.to_avframe()?))?;
    let frame = graph
        .receive_frame_from(0)?
        .context("filter chain should emit the processed frame")?;
    Ok(MediaFrame::<u8>::from_avframe(&frame)?)
}

/// 二分搜索 mjpeg `qscale`（2=最好 .. 31=最差），找到**不超过 `max_bytes`
/// 的最大体积**（即最高质量）的编码，返回其 JPEG 字节；若结果小于
/// `min_bytes`（体积下限）则报错。
///
/// 体积随 qscale 单调递减。先二分出最小的达标 qscale，再向更高质量（更小
/// qscale）回退到体积上限边缘。每次尝试重建编码器（FFmpeg 编码器 flush 后
/// 不可复用），画面不变、只变质量。
fn encode_to_size(frame: &MediaFrame<u8>, min_bytes: u64, max_bytes: u64) -> Result<Vec<u8>> {
    let largest = encode_jpeg(frame, 31)?;
    anyhow::ensure!(
        largest.len() as u64 <= max_bytes,
        "cannot fit {max_bytes} bytes even at qscale 31 ({} bytes)",
        largest.len()
    );
    let smallest = encode_jpeg(frame, 2)?;
    if smallest.len() as u64 <= max_bytes {
        anyhow::ensure!(
            smallest.len() as u64 >= min_bytes,
            "even the highest quality (qscale 2, {} bytes) is below the {min_bytes}-byte minimum",
            smallest.len()
        );
        return Ok(smallest); // 最高质量就已达标，无需降质
    }

    // 不变量：lo 超体积、hi 达标。二分收敛到最小的达标 qscale。
    let (mut lo, mut hi) = (2u32, 31u32);
    let mut best = largest;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let candidate = encode_jpeg(frame, mid)?;
        if candidate.len() as u64 <= max_bytes {
            best = candidate;
            hi = mid;
        } else {
            lo = mid;
        }
    }
    anyhow::ensure!(
        best.len() as u64 >= min_bytes,
        "no qscale lands in [{min_bytes}, {max_bytes}] bytes: closest below cap is {} bytes",
        best.len()
    );
    Ok(best)
}

/// 把一帧编码为 JPEG 字节（mjpeg + image2pipe，全程内存内）。
///
/// 用 `YUVJ444P`：无 chroma 下采样，既保留色彩细节，又允许 295x413 这类奇数
/// 尺寸（`YUVJ420P` 要求宽高为偶数）。
///
/// 固定量化质量用 `qmin = qmax = N` 钉死（mjpeg 有效范围 2~31，N 越大质量越
/// 低、体积越小），等价于 CLI 的 `-qmin/-qmax`。
fn encode_jpeg(frame: &MediaFrame<u8>, qscale: u32) -> Result<Vec<u8>> {
    let mut options = Options::new();
    options.insert("qmin", qscale.to_string());
    options.insert("qmax", qscale.to_string());

    let encoder = EncoderBuilder::new_video(frame.width, frame.height)
        .with_codec_name("mjpeg".to_string())
        .with_pix_fmt(PixelFormat::YUVJ444P)
        .with_fps(25.0)
        .with_options(options)
        .build()
        .context("failed to build mjpeg encoder")?;
    let enc_tb = encoder.time_base();

    // image2 muxer 会按编号自己打开磁盘文件；写进内存 AVIO 要用 image2pipe。
    let writer = BufferWriter::new("image2pipe")?;
    let mut muxer = rsmedia::Muxer::new_from_writer(writer);
    let stream = muxer.add_encoder(encoder)?;

    let mut av = frame.to_avframe()?;
    av.set_pts(0);
    av.set_time_base(enc_tb);
    muxer.mux(av, stream)?;
    muxer.finish()?;
    Ok(muxer.into_writer().into_bytes())
}

/// 证件照统一用的 RGB24 视频端点（25fps，时间基 1/25）。
fn rgb_endpoint(width: i32, height: i32) -> VideoEndpoint {
    VideoEndpoint::new(
        width,
        height,
        PixelFormat::RGB24,
        AVRational { num: 1, den: 25 },
        AVRational { num: 25, den: 1 },
    )
}

/// 校验合成模式的结果：尺寸正确；开了换底则再校验新底色生效且人像未被盖住。
fn verify(frame: &MediaFrame<u8>, tw: usize, th: usize, swapped: bool) -> Result<()> {
    anyhow::ensure!(
        (frame.width, frame.height) == (tw, th),
        "unexpected size {}x{}",
        frame.width,
        frame.height
    );
    if !swapped {
        return Ok(());
    }
    let samples = frame.data.as_packed().context("RGB24 is interleaved")?;
    // 左上角应是新底色（白）。
    let corner = [samples[[0, 0, 0]], samples[[0, 0, 1]], samples[[0, 0, 2]]];
    anyhow::ensure!(
        corner.iter().all(|&v| v > 240),
        "background not replaced: corner = {corner:?}"
    );
    // 合成头像是深色：源 (300,300) 经裁剪缩放后落在 (148,155) 附近，
    // 该点应仍是深色（证明人像盖在白底上，而不是白底整体盖住人像）。
    let face = [
        samples[[155, 148, 0]],
        samples[[155, 148, 1]],
        samples[[155, 148, 2]],
    ];
    anyhow::ensure!(
        face.iter().all(|&v| v < 120),
        "subject lost under the new background: face pixel = {face:?}"
    );
    Ok(())
}
