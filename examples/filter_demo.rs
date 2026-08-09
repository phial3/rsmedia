//! 滤镜（Filter）使用示例
//!
//! 演示两件事：
//! 1. 在编码器中构建一条滤镜链（缩放 + 裁剪 + 降噪 + 画质增强 + 时间水印）。
//! 2. 打印出 `filter::video` / `filter::audio` 提供的常用滤镜 API 目录，方便查阅用法。
//!
//! 运行：`cargo run --example filter_demo`

use rsmedia::{
    colors, filter,
    frame::MediaFrame,
    time::{self, Time},
    EncoderBuilder, PixelFormat,
};

use std::path::Path;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    rsmedia::init().unwrap();

    // ---- 1) 构建一条视频滤镜链并编码成文件 ----
    let width = 640;
    let height = 640;

    // 典型滤镜链：缩放 -> 裁剪 -> 视频降噪 -> 饱和度增强
    // 注意：`drawtext`（文字/时间水印）需要 FFmpeg 编译时开启 libfreetype，
    // 本地精简版可能缺失，故此处仅用于打印目录，未放入实际编码链。
    let filters = vec![
        filter::video::scale(1280, 720, Some("bicubic")),
        filter::video::crop(20, 20, width, height),
        filter::video::hqdn3d(3.0, 2.0), // 视频降噪
        filter::video::saturation(1.3),  // 画质增强：饱和度
    ];

    let output_path = Path::new("/tmp/filter_rainbow.mp4");

    let mut encoder = EncoderBuilder::new_video(width as usize, height as usize)
        .with_filters(filters)
        .build_wrapped(output_path)
        .expect("failed to create encoder");

    let duration: Time = Time::from_nth_of_a_second(24);
    let mut position = Time::zero();

    for i in 0..128 {
        let mut frame = rainbow_frame(width as usize, height as usize, i as f32 / 128.0);
        frame.set_pts(
            position
                .aligned_with_rational(encoder.time_base())
                .into_value()
                .unwrap(),
        );
        encoder.encode(frame)?;
        position = position.aligned_with(duration).add();
    }
    encoder.finish()?;
    println!("encoded -> {}", output_path.display());

    // ---- 2) 打印滤镜 API 目录（仅展示 spec，不实际运行）----
    print_filter_catalog();

    Ok(())
}

/// 打印 `filter` 模块提供的滤镜 API 及其生成的 FFmpeg spec。
fn print_filter_catalog() {
    println!("\n===== video filters =====");
    let v = vec![
        (
            "scale(1280,720,None)",
            filter::video::scale(1280, 720, None).spec(),
        ),
        (
            "crop(x,y,w,h)",
            filter::video::crop(10, 10, 640, 360).spec(),
        ),
        ("fade_in(30)", filter::video::fade_in(30).spec()),
        ("fade_out(30,30)", filter::video::fade_out(30, 30).spec()),
        ("unsharp()", filter::video::unsharp().spec()),
        ("blur(2.0)", filter::video::blur(2.0).spec()),
        (
            "eq(brightness,contrast)",
            filter::video::eq(0.1, 1.2).spec(),
        ),
        ("fps(30)", filter::video::fps(30.0).spec()),
        (
            "yadif(\"send_frame\")",
            filter::video::yadif("send_frame").spec(),
        ),
        (
            "pad(w,h,x,y,color)",
            filter::video::pad(1920, 1080, 0, 0, "black").spec(),
        ),
        ("setdar(16,9)", filter::video::setdar(16, 9).spec()),
        ("setsar(1,1)", filter::video::setsar(1, 1).spec()),
        ("hue(30)", filter::video::hue(30).spec()),
        ("negate()", filter::video::negate().spec()),
        ("noise(10)", filter::video::noise(10).spec()),
        ("hqdn3d(3.0,2.0)", filter::video::hqdn3d(3.0, 2.0).spec()),
        ("nlmeans(1.5)", filter::video::nlmeans(1.5).spec()),
        ("gamma(1.2)", filter::video::gamma(1.2).spec()),
        ("saturation(1.3)", filter::video::saturation(1.3).spec()),
        ("vibrance(0.4)", filter::video::vibrance(0.4).spec()),
        ("deblock()", filter::video::deblock().spec()),
        ("DrawText::new(...).time_text(\"%{localtime}\")", {
            filter::video::DrawText::new("", 10, 10, 24, "white")
                .time_text("%{localtime}")
                .build()
                .spec()
        }),
        ("transpose(1)", filter::video::transpose(1).spec()),
        (
            "delogo(0,0,100,50)",
            filter::video::delogo(0, 0, 100, 50).spec(),
        ),
    ];
    for (name, spec) in v {
        println!("  {name}\n    -> {spec}");
    }

    println!("\n===== audio filters =====");
    let a = vec![
        ("volume(1.5)", filter::audio::volume(1.5).spec()),
        ("loudnorm(-16.0)", filter::audio::loudnorm(-16.0).spec()),
        ("highpass(80)", filter::audio::highpass(80).spec()),
        ("lowpass(4000)", filter::audio::lowpass(4000).spec()),
        ("atempo(1.25)", filter::audio::atempo(1.25).spec()),
        (
            "fft_denoise(20,-50)",
            filter::audio::fft_denoise(20, -50).spec(),
        ),
        ("denoise(15.0)", filter::audio::denoise(15.0).spec()),
        ("compressor(3.0,Some(30),Some(200))", {
            filter::audio::compressor(3.0, Some(30.0), Some(200.0))
                .unwrap()
                .spec()
        }),
    ];
    for (name, spec) in a {
        println!("  {name}\n    -> {spec}");
    }
}

fn rainbow_frame(width: usize, height: usize, p: f32) -> MediaFrame<u8> {
    let rgb = colors::hsv_to_rgb(p * 360.0, 100.0, 100.0);
    let mut frame = MediaFrame::<u8>::new_video_frame(
        width,
        height,
        PixelFormat::RGB24,
        time::new_rational(1, 24),
    )
    .unwrap();
    for y in 0..height {
        for x in 0..width {
            frame.data[[y, x, 0]] = rgb[0];
            frame.data[[y, x, 1]] = rgb[1];
            frame.data[[y, x, 2]] = rgb[2];
        }
    }
    frame
}
