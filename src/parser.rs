//! 裸流帧切分（`AVCodecParser`）：把无容器封装的压缩码流切成帧。
//!
//! `.h264` / `.h265` 裸文件、自定义协议送来的分段码流没有容器索引，
//! 解码前必须先按访问单元（帧）切分——`AVCodecParser` 就是 FFmpeg 内置的
//! 这一能力。rsmedia 将其包装为 [`PacketParser`]：喂数据、出帧边界。
//!
//! # 用法
//!
//! ```no_run
//! use rsmedia::parser::PacketParser;
//!
//! # fn main() -> rsmedia::error::Result<()> {
//! // h264 的 codec_id（可从 StreamInfo::codec_id 取得）
//! let codec_id = 27u32; // AV_CODEC_ID_H264
//! let mut parser = PacketParser::new(codec_id)?;
//!
//! # let chunk: Vec<u8> = todo!("一段裸码流（可为任意分段）");
//! for frame in parser.parse(&chunk)? {
//!     // `frame` 是一个完整访问单元的裸字节（Annex B）
//!     let _ = frame;
//! }
//! // 收尾：冲刷缓冲中的最后一帧
//! let last = parser.flush()?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, RsmediaError};
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParserContext, AVPacket};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

/// 裸码流帧切分器：把无容器的压缩码流按访问单元（帧）切分。
///
/// 内部持有 FFmpeg 的 `AVCodecParserContext` 与一个配套的
/// `AVCodecContext`（不打开、仅作为 parse 的上下文）。
pub struct PacketParser {
    parser: AVCodecParserContext,
    codec_ctx: AVCodecContext,
}

impl PacketParser {
    /// 为指定 codec 创建切分器（`codec_id` 即 `AV_CODEC_ID_*` 的原始值，
    /// 与 [`StreamInfo::codec_id`](crate::stream::StreamInfo::codec_id) 同源）。
    pub fn new(codec_id: u32) -> Result<Self> {
        let parser = AVCodecParserContext::init(codec_id as ffi::AVCodecID).ok_or_else(|| {
            RsmediaError::invalid_config(format!("no bitstream parser for codec id {codec_id}"))
        })?;
        let codec = AVCodec::find_decoder(codec_id as _).ok_or_else(|| {
            RsmediaError::invalid_config(format!("no decoder for codec id {codec_id}"))
        })?;
        Ok(Self {
            parser,
            codec_ctx: AVCodecContext::new(&codec),
        })
    }

    /// 送入一段码流（任意分段），返回本段中切出的完整帧字节（Annex B）。
    ///
    /// 内部按 FFmpeg 语义缓冲：一次 `parse` 可能产出零、一或多个帧；
    /// 分段边界不需要与帧边界对齐。
    pub fn parse(&mut self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        assert!(!data.is_empty(), "use `flush` to drain the parser");
        let mut out = Vec::new();
        let mut packet = AVPacket::new();
        let (ready, _) = self
            .parser
            .parse_packet(&mut self.codec_ctx, &mut packet, data)?;
        if ready {
            out.push(packet_bytes(&packet));
        }
        Ok(out)
    }

    /// 冲刷切分器，取出缓冲中的最后一帧（如有）。
    ///
    /// 对应 `av_parser_parse2` 的 `buf == NULL` 语义；rsmpeg 的
    /// [`parse_packet`](Self::parse) 无法表达 NULL，故在此直呼 ffi。
    pub fn flush(&mut self) -> Result<Option<Vec<u8>>> {
        let mut out_data: *mut u8 = std::ptr::null_mut();
        let mut out_size: i32 = 0;
        let used = unsafe {
            ffi::av_parser_parse2(
                self.parser.as_mut_ptr(),
                self.codec_ctx.as_mut_ptr(),
                &mut out_data,
                &mut out_size,
                std::ptr::null(),
                0,
                ffi::AV_NOPTS_VALUE,
                ffi::AV_NOPTS_VALUE,
                0,
            )
        };
        if used < 0 {
            return Err(RsmpegError::AVError(used).into());
        }
        if out_size > 0 {
            Ok(Some(
                unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size as usize) }
                    .to_vec(),
            ))
        } else {
            Ok(None)
        }
    }
}

/// 取出 packet 携带的裸字节拷贝。
fn packet_bytes(packet: &AVPacket) -> Vec<u8> {
    if packet.size <= 0 || packet.data.is_null() {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(packet.data as *const u8, packet.size as usize) }.to_vec()
}
