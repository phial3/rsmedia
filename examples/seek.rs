//! 演示解码器的定位（seek）能力。
//!
//! 覆盖的 API（直接作用于 `Reader + Seekable`，裸 `Decoder` 同步解码）：
//! - `Seekable::seek_to_frame` —— 定位到指定帧号
//! - `Seekable::seek_to_timestamp` —— 定位到指定时间戳（毫秒）
//! - `Seekable::seek_to_start` —— 回到开头

use rsmedia::io::Seekable;
use rsmedia::{Decoder, DecoderBuilder, MediaType, Reader, StreamReader};

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    rsmedia::init()?;

    let source = Path::new("/tmp/test.mp4");
    let mut reader = StreamReader::new(source)?;
    let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
    let stream_index = decoder.stream_index();

    // 1. 定位到第 3 帧（按帧号 + AVSEEK_FLAG_FRAME 定位）
    reader.seek_to_frame(stream_index, 3, rsmpeg::ffi::AVSEEK_FLAG_FRAME as i32)?;
    println!(
        "seek_to_frame(3) -> pts={}",
        next_pts(&mut decoder, &mut reader)?
    );

    // 2. 定位到 10 秒处（按时间戳 seek，落到最近关键帧）
    reader.seek_to_timestamp(10_000)?;
    println!(
        "seek_to_timestamp(10_000ms) -> pts={}",
        next_pts(&mut decoder, &mut reader)?
    );

    // 3. 回到开头
    reader.seek_to_start()?;
    println!(
        "seek_to_start() -> pts={}",
        next_pts(&mut decoder, &mut reader)?
    );

    Ok(())
}

/// 逐步读取，直到解码出下一帧并返回其 PTS。
fn next_pts<R>(decoder: &mut Decoder, reader: &mut R) -> Result<i64>
where
    R: Reader,
{
    for _ in 0..100 {
        if let Some(frame) = decoder.decode_frame(reader)? {
            return Ok(frame.pts);
        }
    }
    Err(anyhow::anyhow!("no frame after seek"))
}
