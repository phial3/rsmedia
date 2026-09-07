//! 统一媒体格式定义：采样格式与帧格式。
//!
//! 本模块集中存放与媒体格式相关的通用类型：
//! - [`SampleFormat`]：音频采样格式（对应 FFmpeg `AV_SAMPLE_FMT_*`）；
//! - [`FrameFormat`]：帧格式的统一表示（视频=像素格式 / 音频=采样格式），
//!   同时用作滤镜图的输入格式声明（见 [`crate::filter::Filter::with_input_format`]）。

use crate::pixel::PixelFormat;
use crate::strutils;

use rsmpeg::avutil;
use rsmpeg::ffi;

/// 输出格式旗标（对应 FFmpeg `AVFMT_*`）。
#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum AvFormatFlags {
    NO_FILE = ffi::AVFMT_NOFILE,
    NEED_NUMBER = ffi::AVFMT_NEEDNUMBER,
    SHOW_IDS = ffi::AVFMT_SHOW_IDS,
    GLOBAL_HEADER = ffi::AVFMT_GLOBALHEADER,
    NO_TIMESTAMPS = ffi::AVFMT_NOTIMESTAMPS,
    GENERIC_INDEX = ffi::AVFMT_GENERIC_INDEX,
    TS_DISCONT = ffi::AVFMT_TS_DISCONT,
    VARIABLE_FPS = ffi::AVFMT_VARIABLE_FPS,
    NO_DIMENSIONS = ffi::AVFMT_NODIMENSIONS,
    NO_STREAMS = ffi::AVFMT_NOSTREAMS,
    NO_BINSEARCH = ffi::AVFMT_NOBINSEARCH,
    NO_GENSEARCH = ffi::AVFMT_NOGENSEARCH,
    NO_BYTE_SEEK = ffi::AVFMT_NO_BYTE_SEEK,
    #[cfg(not(any(feature = "ffmpeg8", feature = "ffmpeg9")))]
    ALLOW_FLUSH = ffi::AVFMT_ALLOW_FLUSH,
    TS_NONSTRICT = ffi::AVFMT_TS_NONSTRICT,
    TS_NEGATIVE = ffi::AVFMT_TS_NEGATIVE,
    SEEK_TO_PTS = ffi::AVFMT_SEEK_TO_PTS,
}

/// 音频采样格式（对应 FFmpeg `AV_SAMPLE_FMT_*`）。
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum SampleFormat {
    /// < none
    NONE = ffi::AV_SAMPLE_FMT_NONE,
    /// < unsigned 8 bits
    U8 = ffi::AV_SAMPLE_FMT_U8,
    /// < signed 16 bits
    S16 = ffi::AV_SAMPLE_FMT_S16,
    /// < signed 32 bits
    S32 = ffi::AV_SAMPLE_FMT_S32,
    /// < float
    FLT = ffi::AV_SAMPLE_FMT_FLT,
    /// < double
    DBL = ffi::AV_SAMPLE_FMT_DBL,
    /// < unsigned 8 bits, planar
    U8P = ffi::AV_SAMPLE_FMT_U8P,
    /// < signed 16 bits, planar
    S16P = ffi::AV_SAMPLE_FMT_S16P,
    /// < signed 32 bits, planar
    S32P = ffi::AV_SAMPLE_FMT_S32P,
    /// < float, planar
    FLTP = ffi::AV_SAMPLE_FMT_FLTP,
    /// < double, planar
    DBLP = ffi::AV_SAMPLE_FMT_DBLP,
    /// < signed 64 bits
    S64 = ffi::AV_SAMPLE_FMT_S64,
    /// < signed 64 bits, planar
    S64P = ffi::AV_SAMPLE_FMT_S64P,
}

impl SampleFormat {
    pub fn is_planar(&self) -> bool {
        avutil::sample_fmt_is_planar(*self as _)
    }

    pub fn get_bytes_per_sample(&self) -> Option<usize> {
        avutil::get_bytes_per_sample(*self as _)
    }

    pub fn get_sample_fmt_name(&self) -> String {
        avutil::get_sample_fmt_name(*self as _).map_or("Unknown".to_string(), |s| {
            strutils::cstr_to_string(s).unwrap()
        })
    }

    pub fn get_packed_sample_fmt(&self) -> Option<SampleFormat> {
        avutil::get_packed_sample_fmt(*self as _).map(SampleFormat::from)
    }

    pub fn get_planar_sample_fmt(&self) -> Option<SampleFormat> {
        avutil::get_planar_sample_fmt(*self as _).map(SampleFormat::from)
    }
}

impl From<ffi::AVSampleFormat> for SampleFormat {
    fn from(item: ffi::AVSampleFormat) -> Self {
        match item {
            ffi::AV_SAMPLE_FMT_NONE => SampleFormat::NONE,
            ffi::AV_SAMPLE_FMT_U8 => SampleFormat::U8,
            ffi::AV_SAMPLE_FMT_S16 => SampleFormat::S16,
            ffi::AV_SAMPLE_FMT_S32 => SampleFormat::S32,
            ffi::AV_SAMPLE_FMT_FLT => SampleFormat::FLT,
            ffi::AV_SAMPLE_FMT_DBL => SampleFormat::DBL,
            ffi::AV_SAMPLE_FMT_U8P => SampleFormat::U8P,
            ffi::AV_SAMPLE_FMT_S16P => SampleFormat::S16P,
            ffi::AV_SAMPLE_FMT_S32P => SampleFormat::S32P,
            ffi::AV_SAMPLE_FMT_FLTP => SampleFormat::FLTP,
            ffi::AV_SAMPLE_FMT_DBLP => SampleFormat::DBLP,
            ffi::AV_SAMPLE_FMT_S64 => SampleFormat::S64,
            ffi::AV_SAMPLE_FMT_S64P => SampleFormat::S64P,
            // 遇到未知/版本差异的格式时回退为 NONE 而非 panic，避免库内部直接崩溃
            _ => SampleFormat::NONE,
        }
    }
}

/// 帧格式的统一表示：视频帧为像素格式，音频帧为采样格式。
///
/// 同时用作滤镜图的**输入格式声明**（[`crate::filter::Filter::with_input_format`], [`crate::frame::MediaFrame`]）：
/// 声明进图帧必须满足的格式（未声明时默认=编码器/解码器协商格式）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    /// 视频帧的像素格式（如 [`PixelFormat::RGB24`]、[`PixelFormat::YUV420P`]）。
    Pixel(PixelFormat),
    /// 音频帧的采样格式（如 [`SampleFormat::FLTP`]）。
    Sample(SampleFormat),
}

impl FrameFormat {
    /// 取回 FFmpeg 原生格式值（`AV_PIX_FMT_*` 或 `AV_SAMPLE_FMT_*` 的数字表示）。
    #[inline]
    pub fn as_raw(&self) -> i32 {
        match self {
            FrameFormat::Pixel(p) => (*p).into(),
            FrameFormat::Sample(s) => *s as i32,
        }
    }

    /// 提取视频像素格式（音频格式返回 `None`）。
    pub fn into_pixel(self) -> Option<PixelFormat> {
        match self {
            Self::Pixel(fmt) => Some(fmt),
            Self::Sample(_) => None,
        }
    }

    /// 提取音频采样格式（视频格式返回 `None`）。
    pub fn into_sample(self) -> Option<SampleFormat> {
        match self {
            Self::Sample(fmt) => Some(fmt),
            Self::Pixel(_) => None,
        }
    }
}

impl From<PixelFormat> for FrameFormat {
    fn from(fmt: PixelFormat) -> Self {
        Self::Pixel(fmt)
    }
}

impl From<SampleFormat> for FrameFormat {
    fn from(fmt: SampleFormat) -> Self {
        Self::Sample(fmt)
    }
}
