//! Turn a container (file, URL, capture device) into decoded frames.
//!
//! A [`Decoder`] owns one open stream of the input: [`DecoderBuilder::build`]
//! and [`DecoderBuilder::build_from_reader`] pick the *best* stream of the
//! requested [`MediaType`] for you, so one decoder yields one video, audio or
//! subtitle track. Want several tracks? Build one decoder per stream.
//!
//! # The decode family
//!
//! | Call | Yields |
//! |---|---|
//! | [`decode::<T>()`](Decoder::decode) | [`MediaFrame<T>`] — element type `T` must match the sample width |
//! | [`decode_frame`](Decoder::decode_frame) | `MediaFrame<u8>` — the common 8-bit video path |
//! | [`decode_raw`](Decoder::decode_raw) | raw FFmpeg [`AVFrame`], no conversion |
//! | [`decode_subtitle_segment`](Decoder::decode_subtitle_segment) | timed text segments |
//!
//! Each call decodes **one** frame and reads from the [`Reader`] you pass in,
//! which keeps seeking and stream ownership on the caller's side. When the
//! reader is exhausted the decoder still holds buffered frames (reordering,
//! decoder delay): drain them with [`Decoder::drain`] / [`Decoder::drain_raw`],
//! which keep returning `Some(frame)` until the codec is empty, then `None`.
//!
//! # What happens to a frame on the way out
//!
//! Hardware frames are downloaded to system memory, frames are converted to the
//! builder's unified output format ([`with_pix_fmt`](DecoderBuilder::with_pix_fmt)
//! / [`with_sample_fmt`](DecoderBuilder::with_sample_fmt), defaulting to
//! `YUV420P` for video), resized if [`with_resize`](DecoderBuilder::with_resize)
//! asked for it, and pushed through the filter graph from
//! [`with_filters`](DecoderBuilder::with_filters). The output is therefore
//! always plain, storable sample data.
//!
//! # Caveats
//!
//! - **Capture devices** (webcam / mic / screen) additionally need
//!   [`init()`](crate::init()) once, plus the demuxer name via
//!   [`with_input_format`](DecoderBuilder::with_input_format) and its private
//!   options via [`with_input_options`](DecoderBuilder::with_input_options).
//! - A decoder is [`Send`] but **not** [`Sync`]: drive it from one thread.
//! - After the stream is drained, further input fails fast; call
//!   [`reset`](Decoder::reset) to reuse the decoder on a new segment.
//!
//! # Example
//!
//! ```no_run
//! use rsmedia::{DecoderBuilder, MediaType, StreamReader};
//! # fn main() -> rsmedia::error::Result<()> {
//! let mut reader = StreamReader::new("/tmp/test.mp4")?;
//! let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
//! while let Some(frame) = decoder.decode_frame(&mut reader)? {
//!     println!("{}x{} pts={:?}", frame.width, frame.height, frame.pts);
//! }
//! # Ok(())
//! # }
//! ```

use crate::codec::{AVCodecFlag, AVCodecFlag2, ThreadType};
use crate::error::{Context, Result, RsmediaError};
use crate::filter::{AudioParams, Filter, FilterGraph, FilterParams, VideoParams};
use crate::flags::FlagSet;
use crate::fmt::{FrameFormat, SampleFormat};
use crate::frame::{ElementType, MediaFrame};
use crate::hwaccel::{HWContext, HWDeviceConfig};
use crate::io::{Reader, StreamReader, StreamReaderBuilder};
use crate::location::Location;
use crate::options::AVOptions;
use crate::pixel::PixelFormat;
use crate::resample;
use crate::resize::Resize;
use crate::scale::{ScaleAlgorithm, ScaleQuality, Scaler, VideoSpec};
use crate::state::ProcessState;
use crate::stream::{MediaType, StreamInfo};
use crate::strutils;
use crate::subtitle::SubtitleSegment;
use crate::time::{Rational, Rescale, Time};

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket, AVSubtitle};
use rsmpeg::avformat::AVStream;
use rsmpeg::avutil::{self, AVChannelLayoutRef, AVFrame};
use rsmpeg::ffi;

use std::collections::VecDeque;
use std::sync::Arc;

ffi_enum_from!(
    /// 帧丢弃粒度（`AVCodecContext.skip_frame`，`AVDiscard`）。
    ///
    /// 只影响**解码器是否把解出的帧交出来**，不省去比特流解析/解码本身
    /// （`skip_frame` 之上还有 `AV_CODEC_FLAG2_SKIP_MANUAL`、`avcodec_send_packet`
    /// 层面的跳过，本类型不涉及）。用于只要关键帧的场景：生成缩略图/预览、
    /// 快速粗剪、按关键帧建索引等。
    ///
    /// # Panics
    ///
    /// [`From<AVDiscard>`](ffi::AVDiscard) 对表外取值 panic（编程错误信号）；
    /// 值来自 FFmpeg 内部时改用 [`from_ffi_checked`](Self::from_ffi_checked)。
    #[allow(non_camel_case_types)]
    SkipFrame => ffi::AVDiscard,
    repr = i32,
    fallback = panic {
        /// 不丢弃任何帧（FFmpeg 默认行为之外的最宽松档，`AVDISCARD_NONE`）。
        NONE => ffi::AVDISCARD_NONE;
        /// FFmpeg 默认：只丢弃 AVI 里长度为 0 的"无用包"这类帧。
        DEFAULT => ffi::AVDISCARD_DEFAULT;
        /// 丢弃非参考帧（`AVDISCARD_NONREF`）。
        NONREF => ffi::AVDISCARD_NONREF;
        /// 丢弃双向预测帧（B 帧，`AVDISCARD_BIDIR`）。
        BIDIR => ffi::AVDISCARD_BIDIR;
        /// 丢弃所有非帧内编码的帧（`AVDISCARD_NONINTRA`）。
        NONINTRA => ffi::AVDISCARD_NONINTRA;
        /// 丢弃所有非关键帧（`AVDISCARD_NONKEY`）：只剩 I 帧。
        NONKEY => ffi::AVDISCARD_NONKEY;
        /// 丢弃全部帧（`AVDISCARD_ALL`）——解码器只推进状态、不产出帧。
        ALL => ffi::AVDISCARD_ALL;
    }
);

ffi_enum!(
    /// 解码错误识别力度（`AVCodecContext.err_recognition`，`AV_EF_*` 位）。
    ///
    /// 决定解码器**把什么当错误**，以及发现后是"带伤继续"还是直接失败：
    /// 默认是 `0`（一个位都不置，FFmpeg 的 `err_detect` 选项默认值），损坏的码流
    /// 会被掩盖成错帧/糊帧继续输出；要"宁可失败也不出错帧"就置上
    /// [`EXPLODE`](Self::EXPLODE)（通常连同 [`BUFFER`](Self::BUFFER) /
    /// [`CRCCHECK`](Self::CRCCHECK) 一起），解码 API 会以 `Err` 报告而不是静默继续。
    ///
    /// 位可组合（`ErrRecognition::BUFFER | ErrRecognition::EXPLODE`，结果为
    /// [`FlagSet<ErrRecognition>`](crate::FlagSet)，可直接传给
    /// [`DecoderBuilder::with_err_recognition`]）；
    /// [`IGNORE_ERR`](Self::IGNORE_ERR) 与
    /// [`EXPLODE`](Self::EXPLODE) 语义相反，FFmpeg 按位判断，同时置位时行为由
    /// FFmpeg 内部顺序决定，调用方不应同时给出。
    #[allow(non_camel_case_types)]
    ErrRecognition, u32 {
        /// 校验 CRC 之类的校验和（`AV_EF_CRCCHECK`）。**默认并不开启**：
        /// `err_recognition` 的默认值是 0，要校验就得显式置这一位。
        CRCCHECK => ffi::AV_EF_CRCCHECK;
        /// 把码流层（比特流语法）的异常当错误（`AV_EF_BITSTREAM`）。
        BITSTREAM => ffi::AV_EF_BITSTREAM;
        /// 把不完整/越界的缓冲当错误（`AV_EF_BUFFER`）。
        BUFFER => ffi::AV_EF_BUFFER;
        /// 发现错误立即**失败**而不是尽力掩盖（`AV_EF_EXPLODE`）。
        EXPLODE => ffi::AV_EF_EXPLODE;
        /// 忽略可忽略的错误，继续解码（`AV_EF_IGNORE_ERR`）。
        IGNORE_ERR => ffi::AV_EF_IGNORE_ERR;
        /// "谨慎"档：更多一致性检查（`AV_EF_CAREFUL`，慢）。
        CAREFUL => ffi::AV_EF_CAREFUL;
        /// "严格合规"档：只接受完全符合标准的内容（`AV_EF_COMPLIANT`，更慢）。
        COMPLIANT => ffi::AV_EF_COMPLIANT;
        /// "激进"档：为找错误而做的检查（`AV_EF_AGGRESSIVE`，最慢）。
        AGGRESSIVE => ffi::AV_EF_AGGRESSIVE;
    }
);

/// Builds a [`Decoder`].
#[derive(Debug)]
pub struct DecoderBuilder {
    /// `None` = 未显式设置，构建时取 `AVCodecFlag::LOW_DELAY`。存的是调用方显式给出的
    /// `AV_CODEC_FLAG_*` 位集。
    flags: Option<FlagSet<AVCodecFlag>>,
    /// `AVCodecContext.flags2`（`AV_CODEC_FLAG2_*`）。`None` = FFmpeg 默认。
    flags2: Option<FlagSet<AVCodecFlag2>>,
    /// `AVCodecContext.thread_type`（`FF_THREAD_*`）。`None` = FFmpeg 默认。
    thread_type: Option<FlagSet<ThreadType>>,
    /// `None` = 未显式设置
    thread_count: Option<i32>,
    media_type: MediaType,
    codec_name: Option<String>,
    /// 显式绑定的**容器流索引**（`build_from_reader` 生效）。`None` 时自动选"最佳流"。
    stream_index: Option<usize>,
    codec_opts: Option<AVOptions>,
    filters: Option<Vec<Filter>>,
    hw_device_config: Option<HWDeviceConfig>,
    /// 硬件帧池的预分配表面数（`None` = [`crate::hwaccel::DEFAULT_HW_POOL_SIZE`]）。
    hw_pool_size: Option<i32>,
    /// 缩放核选择（互斥，只取一个算法位）
    scale_algorithm: ScaleAlgorithm,
    /// 缩放质量位（可多位，见 [`ScaleQuality`]）。
    scale_quality: FlagSet<ScaleQuality>,
    /// 是否用 `AVBufferPool` 池化缩放输出的帧缓冲（默认关闭）。
    scale_pool: bool,
    /// 视频像素宽*高设置
    resize: Option<Resize>,
    /// 解码输出目标像素格式（仅视频），默认 [`PixelFormat::YUV420P`]。
    pix_fmt: Option<PixelFormat>,
    /// 解码输出目标采样格式（仅音频）。`None` 表示保留编解码器原生格式。
    sample_fmt: Option<SampleFormat>,
    /// 帧丢弃粒度（`AVCodecContext.skip_frame`）。`None` = FFmpeg 默认
    /// （[`SkipFrame::DEFAULT`]：只丢弃 AVI 里长度为 0 的"无用包"这类帧，
    /// 不是 [`SkipFrame::NONE`]）。
    skip_frame: Option<SkipFrame>,
    /// 错误识别位集（`AVCodecContext.err_recognition`）。`None` = FFmpeg 默认，
    /// 即 `0`（**所有位都不置**，连 [`CRCCHECK`](ErrRecognition::CRCCHECK) 也没开）。
    err_recognition: Option<FlagSet<ErrRecognition>>,
    /// 输入 demuxer 名（`avformat_open_input` 的 `iformat`）。`None` = 由
    /// libavformat 自行探测。
    ///
    /// 只被 [`build`](Self::build) 使用；[`build_from_reader`](Self::build_from_reader)
    /// 的输入由调用方自己打开，格式已在那一侧定下。
    input_format: Option<String>,
    /// 传给 `avformat_open_input` 的输入参数（`framerate` / `video_size` /
    /// `pixel_format` 等 demuxer 私有键）。
    ///
    /// 与 `codec_opts` 严格分开：后者写进 `avcodec_open2`，管的是**编解码器**
    /// 参数（`preset`/`tune`…）。采集设备的端点参数只被 demuxer 认，两处互不通用。
    input_opts: Option<AVOptions>,
}

impl DecoderBuilder {
    /// create a new decoder builder with specified media type.
    ///
    /// # Arguments
    ///
    /// `media_type` - The media type of the decoder.
    pub fn new(media_type: MediaType) -> Self {
        Self {
            media_type,
            filters: None,
            codec_name: None,
            stream_index: None,
            codec_opts: None,
            hw_device_config: None,
            hw_pool_size: None,
            thread_count: None,
            flags: None,
            flags2: None,
            thread_type: None,
            scale_algorithm: ScaleAlgorithm::default(),
            scale_quality: ScaleQuality::default_mask(),
            scale_pool: false,
            resize: None,
            pix_fmt: None,
            sample_fmt: None,
            skip_frame: None,
            err_recognition: None,
            input_format: None,
            input_opts: None,
        }
    }

    // 与 EncoderBuilder 共有的那批 setter：定义与文档在 `macros.rs` 的宏里，
    // 改一次两端同时生效（见 `impl_codec_builder_setters` 的说明）。
    impl_codec_builder_setters!();

    /// Bind the decoder to one **specific container stream index** instead of
    /// the media type's "best" stream.
    ///
    /// 默认（不调用本方法）行为不变：`build_from_reader` 用
    /// `Reader::find_best_stream(media_type)` 选流。设置后 `build_from_reader`
    /// 直接绑定该容器索引的流：
    /// - 流的媒体类型必须与 builder 一致，否则 [`RsmediaError::InvalidConfig`]；
    /// - 未显式给定 [`with_codec_name`](Self::with_codec_name) 时，从流的
    ///   `codecpar.codec_id` 反查本构建可用的解码器名；
    /// - codecpar / 时间基 / 帧率全部取自立点流，不做"最佳流"替代。
    ///
    /// 多流代理解码（[`Demuxer`](crate::mux::Demuxer)）必须用它：同一容器可能
    /// 有两条同类型流（如 44.1 kHz 与 8 kHz 各一条 AAC），"最佳流"的参数只对
    /// 其中一条正确 —— 绑定错流时第二流按错误参数解码，产出静默错位的帧。
    pub fn with_stream_index(mut self, index: usize) -> Self {
        self.stream_index = Some(index);
        self
    }

    /// Set the resize strategy applied to decoded video frames.
    ///
    /// Controls the output dimensions: [`Resize::Exact`] forces an exact size,
    /// [`Resize::Fit`]/[`Resize::FitEven`] keep the aspect ratio while fitting
    /// within the given bounds. When `None`, frames keep their source size.
    pub fn with_resize(mut self, resize: Resize) -> Self {
        self.resize = Some(resize);
        self
    }

    /// Set the output pixel format of decoded video frames.
    ///
    /// 默认 [`PixelFormat::YUV420P`]。可指定任何能表示为数据平面的像素格式
    /// （见 [`PixelFormat::data_layout`]）：
    /// - packed 8bit：GRAY8 / YUYV422 / UYVY422 / RGB24 / BGR24 / RGBA 族
    ///   —— 交错为单个数组；
    /// - planar / 半平面 / 9..16bit：YUV420P / YUV422P / YUV444P / NV12 /
    ///   GBRP / YUV420P10LE 等 —— 每平面一个数组。
    ///
    /// 解码输出 [`MediaFrame::data`](crate::frame::MediaFrame::data) 的变体与
    /// 形状随目标格式而变。位流 / 调色板 / 硬件格式无法用数据平面表达，
    /// 构建时即报错。
    ///
    /// 源格式与目标不一致时由 swscale 自动转换（如 NV12 → RGBA）。
    ///
    /// 仅对视频解码器有效；其他媒体类型构建时返回错误（fail-fast）。
    ///
    /// 注意：解码输出帧的元素类型 `T` 必须与格式的每样本字节数一致 ——
    /// 8bit 格式用 `u8`，9..16bit 格式用 `u16`（见 [`PixelFormat::bytes_per_component`]）。
    pub fn with_pix_fmt(mut self, pix_fmt: PixelFormat) -> Self {
        self.pix_fmt = Some(pix_fmt);
        self
    }

    /// Set the output sample format of decoded audio frames.
    ///
    /// 默认保留编解码器**原生**采样格式（AAC/AC-3 为 `FLTP`、MP2 为 `S16P`、
    /// PCM 为各自的 `S16LE`/`S32LE`…）。指定本项后，解码输出统一重采样到目标
    /// 格式，于是 `decode::<T>` 的元素类型不再需要随源文件而变 ——
    /// 例如统一到 [`SampleFormat::FLTP`] 后，任何音频文件都能用 `decode::<f32>()`
    /// 读取，不必先查 `codecpar().format`。
    ///
    /// 采样率与声道布局**不变**，只换采样格式（同一率下等长转换）。源格式已与
    /// 目标相同时不做任何转换，因此该选项在无需转换时零开销。
    ///
    /// 解码输出 `MediaFrame::data` 的交错/平面变体随之变化（见
    /// [`SampleFormat::data_layout`]）：平面格式（`FLTP` 等）每声道一个
    /// `(1, nb_samples)` 平面，交错格式（`FLT` 等）为单个
    /// `(1, nb_samples, nb_channels)` 数组。
    ///
    /// 仅对音频解码器有效；其他媒体类型构建时返回错误（fail-fast）。
    ///
    /// 注意：元素类型 `T` 的字节宽度必须与目标格式的每样本字节数一致
    /// （见 [`SampleFormat::get_bytes_per_sample`]）——`FLTP` 用 `f32`、
    /// `S16P` 用 `i16`、`S32P` 用 `i32`。
    pub fn with_sample_fmt(mut self, sample_fmt: SampleFormat) -> Self {
        self.sample_fmt = Some(sample_fmt);
        self
    }

    /// 设置帧丢弃粒度（`AVCodecContext.skip_frame`）：让解码器只交出部分帧。
    ///
    /// 典型用法是 [`SkipFrame::NONKEY`] —— 只要关键帧，用来出缩略图、建索引、
    /// 快速预览而不必解码全部内容。被丢弃的帧**不会**出现在
    /// [`decode`](Decoder::decode) 的结果里：调用方看到的帧数就是保留下来的帧数
    /// （帧的 pts 仍是原时间戳，所以能按时间对齐回原流）。丢弃发生在解码器内部，
    /// 越靠后的档位（[`NONINTRA`](SkipFrame::NONINTRA)、
    /// [`NONKEY`](SkipFrame::NONKEY)）省下的工作量越多，但**都不会**省掉比特流
    /// 解析本身。
    ///
    /// 关键帧判定由解码器按码流给出的 `AV_PKT_FLAG_KEY` 决定，与容器无关；
    /// [`ALL`](SkipFrame::ALL) 会让解码器只推进状态、不产出任何帧。
    ///
    /// 注意：这是**解码端**的丢弃策略，与"让编码器在某帧强制插关键帧"
    /// （[`MediaFrame::force_key_frame`](crate::MediaFrame::force_key_frame)）
    /// 是两端配合的关系 —— 没有关键帧的码流上使用 `NONKEY` 只会得到空结果。
    ///
    /// ```no_run
    /// use rsmedia::{DecoderBuilder, MediaType, SkipFrame};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// let mut reader = rsmedia::StreamReader::new("assets/mp4.mp4")?;
    /// let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
    ///     .with_skip_frame(SkipFrame::NONKEY)
    ///     .build_from_reader(&reader)?;
    /// // 每次调用都返回一个关键帧，非关键帧在解码器内部被丢弃。
    /// while let Some(frame) = decoder.decode_frame(&mut reader)? {
    ///     println!("key frame at pts {}", frame.pts);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_skip_frame(mut self, skip_frame: SkipFrame) -> Self {
        self.skip_frame = Some(skip_frame);
        self
    }

    /// 设置错误识别掩码（`AVCodecContext.err_recognition`）。
    ///
    /// 默认一个位都不置（即**宽松容错**）：损坏的码流会被
    /// 解码器尽力掩盖（`error_concealment`）成错帧继续输出，调用方拿到的是
    /// "看起来正常"的画面。需要"宁可失败也不出错帧"时，把
    /// [`ErrRecognition::EXPLODE`] 加进来，解码 API 就会以 `Err` 报告损坏，
    /// 由调用方决定重试/跳过/中止。
    ///
    /// 可传单个位，也可传 `|` 组合出的位集（见 [`ErrRecognition`]）——两者是同一个参数。
    ///
    /// ```no_run
    /// use rsmedia::{DecoderBuilder, MediaType, ErrRecognition};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// let strict = ErrRecognition::BUFFER | ErrRecognition::EXPLODE;
    /// let mut reader = rsmedia::StreamReader::new("assets/mp4.mp4")?;
    /// let _decoder = DecoderBuilder::new(MediaType::VIDEO)
    ///     .with_err_recognition(strict)
    ///     .build_from_reader(&reader)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_err_recognition(
        mut self,
        err_recognition: impl Into<FlagSet<ErrRecognition>>,
    ) -> Self {
        self.err_recognition = Some(err_recognition.into());
        self
    }

    /// 指定输入的 demuxer 名（`avformat_open_input` 的 `iformat`）。
    ///
    /// 默认（不调用本方法）由 libavformat 按 `source` 的内容自行探测容器格式。
    /// 采集设备（webcam / 麦克风 / 屏幕）没有可探测的头，必须显式声明它。
    /// 常用取值（`-f` 后的名字，跨平台）：
    ///
    /// | 平台 | 视频 | 音频 |
    /// |---|---|---|
    /// | macOS | `avfoundation` | `avfoundation` |
    /// | Windows | `dshow` | `dshow` |
    /// | Linux | `v4l2` | `alsa` / `pulse` |
    ///
    /// 屏幕录制另有 `x11grab`（X11）/ `gdigrab`（Windows）/ `fbdev`（framebuffer），
    /// 采集卡有 `decklink` / `vfwcap`。实际可用项取决于该 FFmpeg 构建编译进了哪些
    /// 设备，用 `ffmpeg -devices` 可列出当前构建支持的（名字不在其中时
    /// `avformat_open_input` 直接失败）。
    ///
    /// 此时 `source` 是**设备的索引对**而非文件名，例如 macOS 上
    /// `avfoundation` 的 `"0:none"` = 0 号视频设备 + 不采集音频，
    /// `"0:1"` = 同时采集 0 号视频与 1 号音频（一个设备 URL 出音视频双流，
    /// 天然同步，省掉跨设备对齐）。
    ///
    /// 采集端点通常还要用 [`with_input_options`](Self::with_input_options) 补
    /// `framerate` / `video_size` / `pixel_format` 等**设备私有**参数（这些键由
    /// demuxer 解释，`avfoundation` / `dshow` / `v4l2` 都认）—— 注意它与
    /// [`with_options`](Self::with_options) 不是一回事，后者写的是**编解码器**参数
    /// （`avcodec_open2`），设备端点参数只被 demuxer 认。
    ///
    /// 仅作用于 [`build`](Self::build)；[`build_from_reader`](Self::build_from_reader)
    /// 的输入由你自己打开，格式已在那一侧确定，本设置会被忽略。
    ///
    /// 名称含 NUL 字节时在构建期报错，而不是 panic。
    ///
    /// # 前置条件：必须先 [`init()`](crate::init())
    ///
    /// 采集设备的 demuxer 是由 `avdevice_register_all()` 注册进来的，而这一步在
    /// [`init()`](crate::init()) / [`init_with()`](crate::init_with()) 里、且只在 once-guard
    /// 内执行一次。**没调用过 `init()` 就构建，会以 `Cannot open input file. (-2)`
    /// 失败** —— 错误信息完全看不出是缺初始化，很容易误判成 URL 或格式名写错。
    ///
    /// ```no_run
    /// use rsmedia::{DecoderBuilder, MediaType, StreamReaderBuilder};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// rsmedia::init()?;                 // 注册 avfoundation / dshow / v4l2 等设备
    /// let mut reader = StreamReaderBuilder::new("0:none")
    ///     .with_format("avfoundation")  // 解码用的 reader，格式与构建期一致
    ///     .build()?;
    /// let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
    ///     .with_input_format("avfoundation")
    ///     .build_from_reader(&reader)?;
    /// # let _ = decoder.decode_frame(&mut reader);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// 普通文件输入不需要这一前置条件（容器 demuxer 是编译期静态注册的）。
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rsmedia::{DecoderBuilder, MediaType, AVOptions};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// let mut opts = AVOptions::new();
    /// opts.set("framerate", "30");
    /// opts.set("video_size", "1280x720");
    ///
    /// let _decoder = DecoderBuilder::new(MediaType::VIDEO)
    ///     .with_input_format("avfoundation")
    ///     .with_input_options(opts)
    ///     .build("0:none")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_input_format(mut self, format: impl Into<String>) -> Self {
        self.input_format = Some(format.into());
        self
    }

    /// 设置传给 `avformat_open_input` 的输入参数（demuxer 私有键）。
    ///
    /// 采集设备类输入几乎总需要它：光有 demuxer 名只能打开端点，帧率、分辨率、
    /// 像素格式仍要由设备侧协商，指定哪个组合才可用是平台与设备相关的。例如
    /// `avfoundation` 的 `framerate` / `video_size`、`v4l2` 的
    /// `input_format` / `framerate`、`dshow` 的 `video_size`。
    ///
    /// **与 [`with_options`](Self::with_options) 的区别**（容易混淆）：
    ///
    /// | 方法 | 写入哪一步 | 管什么 |
    /// |---|---|---|
    /// | `with_input_options` | `avformat_open_input` | demuxer / 协议 / 设备端点参数 |
    /// | `with_options` | `avcodec_open2` | 编解码器私有参数（`preset`/`tune`…） |
    ///
    /// 两者互不通用：把 `framerate` 交给 `with_options` 会被 codec 层忽略，
    /// 反之把 `preset` 交给本方法也一样无效。
    ///
    /// 仅作用于 [`build`](Self::build)；[`build_from_reader`](Self::build_from_reader)
    /// 的输入由你自己打开，参数已在那一侧传过，本设置会被忽略。
    pub fn with_input_options(mut self, options: AVOptions) -> Self {
        self.input_opts = Some(options);
        self
    }

    fn setup_codec_context(&self, decoder: &mut AVCodecContext, input: &AVStream) -> Result<()> {
        let media_type = self.media_type;
        if media_type as ffi::AVMediaType != decoder.codec_type {
            return Err(RsmediaError::invalid_config(format!(
                "decoder was built for media type {media_type:?}, but the codec of the given \
                 stream is {:?}",
                decoder.codec_type
            )));
        }

        decoder.apply_codecpar(&input.codecpar())?;
        decoder.set_flags(
            self.flags
                .unwrap_or_else(|| AVCodecFlag::LOW_DELAY.into())
                .bits() as i32,
        );
        if let Some(flags2) = self.flags2 {
            crate::codec::set_flags2(decoder, flags2.bits() as i32);
        }
        if let Some(thread_type) = self.thread_type {
            crate::codec::set_thread_type(decoder, thread_type.bits() as i32);
        }
        if let Some(cnt) = self.thread_count {
            crate::codec::set_thread_count(decoder, cnt);
        }
        decoder.set_time_base(input.time_base);
        decoder.set_pkt_timebase(input.time_base);
        if let Some(framerate) = input.guess_framerate() {
            decoder.set_framerate(framerate);
        }

        // 稳定性策略：rsmpeg 未生成 skip_frame / err_recognition 访问器，直接写字段
        // （encode.rs 写 rc_max_rate 同例）。两项都必须在 `avcodec_open2` 之前生效，
        // 否则不会进入解码器初始化。
        //
        // SAFETY: `decoder` 在此处独占（`&mut`），`deref_mut` 只在块内存活。
        unsafe {
            let ctx_raw = decoder.deref_mut();
            if let Some(skip_frame) = self.skip_frame {
                ctx_raw.skip_frame = skip_frame.into();
            }
            if let Some(err_recognition) = self.err_recognition {
                ctx_raw.err_recognition = err_recognition.bits() as i32;
            }
        }

        Ok(())
    }

    /// 构建一个**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 适合需要精细控制 reader 生命周期的高级场景：这里只为**探测流信息**临时
    /// 打开一次 reader，构建结束即释放；解码时需每帧传入你自己的 reader
    /// （`decoder.decode(&mut reader)`），seek 也由你显式操作该 reader。
    ///
    /// [`with_input_format`](Self::with_input_format) /
    /// [`with_input_options`](Self::with_input_options) 只作用于这里那个临时
    /// reader。用 [`build_from_reader`](Self::build_from_reader) 时它们被忽略 ——
    /// 那条路径的输入由你打开，格式与参数已在那一侧定下。
    pub fn build(self, source: impl Into<Location>) -> Result<Decoder> {
        let reader = self.open_reader(source)?;
        self.build_from_reader(&reader)
    }

    /// 按 `with_input_format` / `with_input_options` 打开构建期用的 reader。
    ///
    /// 两者都未设置时走 [`StreamReader::new`]（内容探测）。
    fn open_reader(&self, source: impl Into<Location>) -> Result<StreamReader> {
        if self.input_format.is_none() && self.input_opts.is_none() {
            return StreamReader::new(source);
        }
        // 只有调用方显式要求时才付这个代价：builder 的其余部分与 reader 无关
        let mut builder = StreamReaderBuilder::new(source);
        if let Some(format) = self.input_format.as_deref() {
            builder = builder.with_format(format);
        }
        if let Some(options) = self.input_opts.clone() {
            builder = builder.with_options(options);
        }
        builder.build()
    }

    /// 输出像素格式：仅视频有效。
    ///
    /// 任何能表示为数据平面的格式都接受（布局由描述符推导，见
    /// `PixelFormat::data_layout`）；位流 / 调色板 / 硬件格式在构建期快速失败，
    /// 而不是拖到运行时。解码输出经 swscale 统一转换到目标格式。非视频类型配置
    /// 了 `pix_fmt` 视为调用方错误，快速失败而非静默忽略。
    fn resolve_output_pix_fmt(&self) -> Result<PixelFormat> {
        match (self.media_type, self.pix_fmt) {
            (MediaType::VIDEO, Some(fmt)) => {
                if !fmt.is_plane_storable() {
                    return Err(RsmediaError::invalid_config(format!(
                        "with_pix_fmt({fmt:?}): the format cannot be stored as sample planes \
                         (bitstream, paletted and hardware formats are not supported)"
                    )));
                }
                Ok(fmt)
            }
            (_, None) => Ok(PixelFormat::YUV420P),
            // 属于调用方错误（把只对视频生效的 setter 用在了别的类型上）
            (media_type, Some(fmt)) => Err(RsmediaError::invalid_config(format!(
                "with_pix_fmt({fmt:?}) is only valid for {} decoders, got media type: {media_type:?}",
                MediaType::VIDEO.get_media_name()
            ))),
        }
    }

    /// 输出采样格式：仅音频有效。
    ///
    /// `None` = 保留编解码器原生格式（默认），代价为零；指定后解码帧在进滤镜图
    /// 之前统一转换到目标格式。非音频类型配置了 `sample_fmt` 视为调用方错误，
    /// 快速失败而非静默忽略。
    fn resolve_output_sample_fmt(&self) -> Result<Option<SampleFormat>> {
        match (self.media_type, self.sample_fmt) {
            (MediaType::AUDIO, None) => Ok(None),
            (MediaType::AUDIO, Some(fmt)) if fmt != SampleFormat::NONE => Ok(Some(fmt)),
            (MediaType::AUDIO, Some(fmt)) => Err(RsmediaError::invalid_config(format!(
                "with_sample_fmt({fmt:?}): not a usable output sample format"
            ))),
            (media_type, Some(fmt)) => Err(RsmediaError::invalid_config(format!(
                "with_sample_fmt({fmt:?}) is only valid for {} decoders, got media type: {media_type:?}",
                MediaType::AUDIO.get_media_name()
            ))),
            (_, None) => Ok(None),
        }
    }

    /// 初始化硬件解码上下文。
    ///
    /// # Errors
    ///
    /// 设备创建失败、`setup_decoder_frames` 失败，或编解码器选中的硬件像素格式
    /// rsmedia 不建模时返回错误。
    fn init_hw_context(
        &self,
        mut cfg: HWDeviceConfig,
        codec: &AVCodec,
        ctx: &mut AVCodecContext,
        width: u32,
        height: u32,
    ) -> Result<Arc<HWContext>> {
        // 以 `avcodec_get_hw_config` 的声明为准回写配置：hw frames 按这个字段
        // 分配，配置里的值若与编解码器声明不符，会得到错误的输出格式。
        // 未知格式（绑定/FFmpeg 版本不匹配）报错而不是 panic。
        let hw_pixel = cfg
            .device_type
            .find_hw_pixel_format_with_codec(codec)
            .ok_or_else(|| {
                // 名字来自 FFmpeg，可能是非 UTF-8；它只用于文案，
                // 拿不到合法字符串并不改变"不支持"这一结论。
                let codec_name =
                    strutils::cstr_to_string(codec.name()).unwrap_or_else(|_| "unknown".to_owned());
                RsmediaError::unsupported(format!(
                    "hardware decoding is not available for codec: {codec_name}"
                ))
            })?;
        cfg.hw_pixel_format = PixelFormat::from_ffi_checked(hw_pixel).ok_or_else(|| {
            RsmediaError::unsupported(format!(
                "codec {} selected hardware pixel format {hw_pixel}, which rsmedia does not model",
                strutils::cstr_to_string_lossy(codec.name())
            ))
        })?;

        tracing::info!(
            "Video decoder with HW acceleration codec: {:?}, hw_pixel: {:?}, config: {:#?}",
            codec.name(),
            cfg.hw_pixel_format,
            cfg
        );

        // 注意：setup_decoder_frames 可能会改变 ctx.pix_fmt
        HWContext::new(cfg)
            .and_then(|hw| {
                hw.setup_decoder_frames(
                    ctx,
                    width,
                    height,
                    self.hw_pool_size
                        .unwrap_or(crate::hwaccel::DEFAULT_HW_POOL_SIZE),
                )?;
                Ok(hw)
            })
            .context("Hardware acceleration context initialization failed")
    }

    /// 由解码器状态推导滤镜图的输入/输出参数。
    ///
    /// sink 端格式 = 解码输出格式：图内负责把链的输出转回来，`MediaFrame` 因此
    /// 拿到的仍是 `with_pix_fmt` / `with_sample_fmt` 承诺的格式。
    /// 收集并校验滤镜链声明的图输入格式（[`Filter::with_input_format`]）。
    ///
    /// - 多条滤镜各自声明时必须**一致**（同一格式）：这个声明的意义是"帧进图前
    ///   先转成它、图内 buffer 源按它声明"，链上各条各说各话没有可行实现；
    /// - 声明必须与滤镜的媒体类型同族（视频滤镜配采样格式同样不可实现）。
    ///
    /// 老行为取"链上第一条声明"且不校验：后面的声明被静默忽略、冲突根本不报
    /// —— 静默建出与声明不符的图，属于"构造即可触发"的缺陷，现在 loud fail。
    fn resolve_filter_input_format(&self) -> Result<Option<FrameFormat>> {
        let Some(filters) = self.filters.as_ref() else {
            return Ok(None);
        };
        let mut declared: Option<FrameFormat> = None;
        for filter in filters {
            let Some(fmt) = filter.input_format() else {
                continue;
            };
            if let Some(prev) = declared {
                if prev != fmt {
                    return Err(RsmediaError::invalid_config(format!(
                        "filters declare conflicting input formats: {prev:?} vs {fmt:?} — \
                         all `with_input_format` declarations in one chain must agree"
                    )));
                }
            } else {
                declared = Some(fmt);
            }
            let media_type = filter.media_type();
            let same_family = matches!(
                (media_type, fmt),
                (MediaType::VIDEO, FrameFormat::Pixel { .. })
                    | (MediaType::AUDIO, FrameFormat::Sample { .. })
            );
            if !same_family {
                return Err(RsmediaError::invalid_config(format!(
                    "filter for {media_type:?} declares input format {fmt:?} — a cross-type \
                     format declaration has no workable implementation"
                )));
            }
        }
        Ok(declared)
    }

    fn build_filter_params(
        &self,
        ctx: &AVCodecContext,
        width: u32,
        height: u32,
        output_pix_fmt: PixelFormat,
        output_sample_fmt: Option<SampleFormat>,
        filter_input_format: Option<FrameFormat>,
    ) -> Result<FilterParams> {
        match self.media_type {
            MediaType::VIDEO => Ok(FilterParams::Video(VideoParams {
                width,
                height,
                src_format: filter_input_format
                    .and_then(FrameFormat::into_pixel)
                    .unwrap_or(output_pix_fmt),
                format: output_pix_fmt,
                time_base: ctx.time_base.into(),
                frame_rate: ctx.framerate.into(),
                pixel_aspect: ctx.sample_aspect_ratio.into(),
                // 图输入（源侧）色彩描述符：解码上下文在 `apply_codecpar` 时已把
                // 流的色彩元数据拷进 AVCodecContext。进图帧携带同样的元数据，
                // 源侧如实声明后 7.1+ 的图能自动插转，避免 full/limited range
                // 错位直进编码器。
                color_space: ctx.colorspace,
                color_range: ctx.color_range,
            })),
            MediaType::AUDIO => {
                // sink 格式 = 解码输出采样格式（未指定则保留原生）。
                let format = output_sample_fmt.unwrap_or(SampleFormat::from(ctx.sample_fmt));
                Ok(FilterParams::Audio(AudioParams {
                    nb_channels: ctx.ch_layout.nb_channels,
                    sample_rate: ctx.sample_rate,
                    format,
                    src_format: filter_input_format
                        .and_then(FrameFormat::into_sample)
                        .unwrap_or(format),
                    time_base: ctx.time_base.into(),
                }))
            }
            _ => Err(RsmediaError::invalid_config(format!(
                "a {:?} filter cannot be used on this stream",
                self.media_type
            ))),
        }
    }

    /// 用给定的 reader 构建**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 高级/内部场景使用（如 mux 多流共享同一 reader）。解码时需每帧传入
    /// reader，且不支持 seek。
    ///
    /// 输入流绑定规则（见 [`with_stream_index`](Self::with_stream_index)）：
    /// - 未设置 → 绑定 `find_best_stream(media_type)` 为该媒体类型选出的
    ///   "最佳流"（独立构建场景的默认语义）；
    /// - 设置后 → 绑定指定的容器流索引，流媒体类型必须与本 builder 一致
    ///   （不一致 `InvalidConfig`）。**同类型多流容器必须走这条路**："最佳流"
    ///   的 codecpar/时间基只对其中一条流正确。
    ///
    /// codec 解析规则（见 [`with_codec_name`](Self::with_codec_name)）：调用方
    /// 指定的名字最优先（异种 decoder 场景，如 hwaccel 变体 `h264_cuvid`，只能
    /// 按名查）；未指定时**直取 `AVCodec` 句柄** —— 从绑定的流
    /// `codecpar.codec_id` 一次查找直达，不做"流 → 名字 → codec"的 round-trip。
    pub fn build_from_reader<R: Reader>(mut self, reader: &R) -> Result<Decoder> {
        let media_type = self.media_type;

        // 绑定输入流
        let (stream_index, input_stream) = match self.stream_index {
            Some(index) => {
                let input_stream = reader.stream_at(index)?;
                let stream_type = MediaType::from_ffi_checked(input_stream.codecpar().codec_type)
                    .unwrap_or(MediaType::UNKNOWN);
                if stream_type != media_type {
                    return Err(RsmediaError::invalid_config(format!(
                        "decoder builder is for {media_type:?}, but stream {index} is \
                         {stream_type:?} — bind a matching stream, or leave \
                         with_stream_index unset to pick the best one by media type"
                    )));
                }
                (index, input_stream)
            }
            None => {
                // `find_best_stream` 也回传 codec 名，这里同样取之不用：
                // 解码器句柄在 ② 里由绑定流的 codecpar 统一直取。
                let (index, _codec_name) = reader.find_best_stream(media_type)?;
                (index, reader.stream_at(index)?)
            }
        };

        // codec 句柄单点解析：
        // - 指定的名字最优先
        // - 未点名时直接从**绑定流**的 `codecpar.codec_id` 取 `AVCodec` 句柄
        let codec = match &self.codec_name {
            Some(name) => {
                let name_c = strutils::str_to_cstring(name)?;
                AVCodec::find_decoder_by_name(&name_c).ok_or_else(|| {
                    RsmediaError::unsupported(format!(
                        "decoder '{name}' is not available in this FFmpeg build"
                    ))
                })?
            }
            None => {
                let codec_id = input_stream.codecpar().codec_id;
                AVCodec::find_decoder(codec_id).ok_or_else(|| {
                    RsmediaError::unsupported(format!(
                        "no decoder for codec_id {codec_id:#x} of stream {stream_index} \
                         is available in this FFmpeg build"
                    ))
                })?
            }
        };

        let duration = Time::new(Some(input_stream.duration), input_stream.time_base.into());
        let nb_frames = input_stream.nb_frames;
        let frame_rate = (
            avutil::av_q2d(input_stream.r_frame_rate) as f32,
            avutil::av_q2d(input_stream.avg_frame_rate) as f32,
        );

        let mut decode_ctx = AVCodecContext::new(&codec);
        self.setup_codec_context(&mut decode_ctx, input_stream)?;

        // video
        let init_width = decode_ctx.width as u32;
        let init_height = decode_ctx.height as u32;

        let hw_context = if media_type == MediaType::VIDEO {
            match self.hw_device_config.take() {
                Some(cfg) => Some(self.init_hw_context(
                    cfg,
                    &codec,
                    &mut decode_ctx,
                    init_width,
                    init_height,
                )?),
                None => None,
            }
        } else {
            None
        };

        // `take()` 而非直接 `and_then`：后者会部分移出 `self.codec_opts`，之后就
        // 不能再整体借用 `self`（后面的 `resolve_output_*` 需要 `&self`）。
        let dict = self.codec_opts.take().and_then(|opts| opts.into_dict());
        decode_ctx
            .open(dict)
            .context("Failed to open decoder for stream")?;

        let stream_info = StreamInfo::from_stream(input_stream)?;
        tracing::info!("{stream_info}");

        let output_pix_fmt = self.resolve_output_pix_fmt()?;
        let output_sample_fmt = self.resolve_output_sample_fmt()?;

        // 滤镜链声明的图输入格式（见 [`Filter::with_input_format`]）：帧在进图
        // 之前会被转成它，图内 buffer 源按同一格式声明，两者必须一致（声明与
        // 实喂不符会被 av_buffersrc 拒绝）。未声明时图输入 = 解码输出格式，
        // 零额外转换，与从前完全一致。多条声明冲突 / 跨类型声明在这里直接
        // 报 `InvalidConfig`（见 [`Self::resolve_filter_input_format`]）。
        let filter_input_format = self.resolve_filter_input_format()?;

        let filter_graph = match self.filters.take() {
            Some(filters) => {
                // 滤镜图输入端点按 resize 折算后的尺寸声明：帧在进图**之前**会被
                // swscale 缩放到 resize 的目标尺寸（`receive_normalized_frame` 步骤 2），
                // 而 buffersrc 的端点参数在 build 时定格 —— 若仍声明 codecpar 尺寸，
                // 实喂帧与声明不符，`av_buffersrc` 会以 EINVAL 直接拒绝。
                let (filter_w, filter_h) = match self.resize {
                    Some(resize) => resize.compute_for((init_width, init_height)).ok_or_else(
                        || {
                            RsmediaError::invalid_config(format!(
                                "the resize configuration {resize:?} yields no valid size for a \
                                 {init_width}x{init_height} stream"
                            ))
                        },
                    )?,
                    None => (init_width, init_height),
                };
                let params = self.build_filter_params(
                    &decode_ctx,
                    filter_w,
                    filter_h,
                    output_pix_fmt,
                    output_sample_fmt,
                    filter_input_format,
                )?;
                // 滤镜链的媒体类型与可用性校验都在 `init` 内（缺失滤镜 → `InvalidConfig`）
                let graph = FilterGraph::build(&params, filters.as_slice())?;
                // 参数随图一起留下：重启流水线时必须重建一张新图。
                Some(DecodeFilterChain {
                    graph,
                    params,
                    filters,
                })
            }
            None => None,
        };

        Ok(Decoder {
            media_type,
            stream_index,
            duration,
            nb_frames,
            frame_rate,
            hw_context,
            filter_graph,
            context: decode_ctx,
            state: ProcessState::Normal,
            scaler: Scaler::new_with_options(self.scale_algorithm, self.scale_quality)
                .with_buffer_pool(self.scale_pool),
            resize: self.resize,
            output_pix_fmt,
            output_sample_fmt,
            filter_input_format,
            resampler: None,
            pending_frames: VecDeque::new(),
        })
    }
}

/// The decode pipeline's filter graph **together with what it was built from**.
///
/// The graph and its inputs are one unit on purpose: the pipeline can only be
/// restarted by rebuilding the graph (see `FilterGraph::rebuild`), which needs
/// those exact parameters, so keeping them apart would let them drift. The same
/// reasoning as `ProcessState`: one fact, one place.
struct DecodeFilterChain {
    graph: FilterGraph,
    params: FilterParams,
    filters: Vec<Filter>,
}

/// Decode video files and streams.
///
/// # Example
///
/// ```ignore
/// let mut reader = StreamReader::new("video.mp4").unwrap();
/// let mut decoder = Decoder::new_video("video.mp4").unwrap();
/// while let Some(frame) = decoder.decode(&mut reader).unwrap() {
///     println!("Got frame!");
/// }
/// ```
pub struct Decoder {
    context: AVCodecContext,
    filter_graph: Option<DecodeFilterChain>,
    hw_context: Option<Arc<HWContext>>,
    /// (r_frame_rate, avg_frame_rate)
    frame_rate: (f32, f32),
    nb_frames: i64,
    duration: Time,
    stream_index: usize,
    media_type: MediaType,
    state: ProcessState,
    /// 视频像素缩放器（改尺寸 + 换格式）。与 [`Self::resampler`] 不同，它**不是**
    /// `Option`：`Scaler::new` 不需要输入规格（swscale 从帧属性推导源格式），而
    /// `Resampler` 的输入规格只有第一帧才知道、因此惰性建在 `resample_if_needed` 里。
    scaler: Scaler,
    resize: Option<Resize>,
    /// 解码输出目标像素格式（仅视频）
    output_pix_fmt: PixelFormat,
    /// 解码输出目标采样格式（仅音频）；`None` = 保留编解码器原生格式
    output_sample_fmt: Option<SampleFormat>,
    /// 滤镜链声明的**进图**格式（见 [`Filter::with_input_format`]）；`None` =
    /// 图输入就是解码输出格式（零额外转换）。有值时进图前的帧会被转成它。
    filter_input_format: Option<FrameFormat>,
    /// 音频输出格式重采样器（与视频的 `scaler` 一一对应；跨帧复用同一个 [`Resampler`]，
    /// 避免逐帧重建丢掉重采样延迟）。
    ///
    /// `None` = 还没遇到过需要转换的帧：上下文的输入格式要到第一帧才知道，因此它由
    /// [`Resampler::new`] 在第一次转换时建立（见 [`resample::resample_if_needed`]）。
    ///
    /// **不需要在流末排空**：它的目标布局与采样率都取自帧本身、只换采样格式（见
    /// [`Self::convert_decoded_audio`]），上下文不做重采样因而没有延迟线 —— 与需要
    /// 排空的那个（输出采样率取自编码器的 `Encoder` 侧重采样器）不同。
    resampler: Option<resample::Resampler>,
    /// 解码器已经吐出、但还没交出去的帧（**未经归一化**：还没做 HW 下载、缩放与滤镜）
    ///
    /// 一个包可以解出**多帧**（H.264 场编码、MPEG-2 field picture…），而
    /// [`decode_raw_packet`](Self::decode_raw_packet) 一次只交一帧。当解码器输入
    /// 缓冲已满（`avcodec_send_packet` 返回 EAGAIN）而输出队列里还压着帧时，必须先
    /// 把这些帧取出来腾地方，否则新包送不进去。取出来的帧暂存在这里，按 FIFO 交
    /// 给调用方，保证交付顺序与解码顺序一致（`decode.` 侧的 `pending_packets` 是
    /// 同一件事的镜像）。
    pending_frames: VecDeque<AVFrame>,
}

impl Decoder {
    /// Create a decoder to decode the specified source.
    ///
    /// # Arguments
    ///
    /// * `source` - The source to open (path, URL, or capture device).
    #[inline]
    pub fn new_video(source: impl Into<Location>) -> Result<Decoder> {
        DecoderBuilder::new(MediaType::VIDEO).build(source)
    }

    /// Create a decoder to decode the audio stream of the specified source.
    ///
    /// # Arguments
    ///
    /// * `source` - The source to open (path, URL, or capture device).
    #[inline]
    pub fn new_audio(source: impl Into<Location>) -> Result<Decoder> {
        DecoderBuilder::new(MediaType::AUDIO).build(source)
    }

    /// Create a decoder to decode the subtitle stream of the specified source.
    ///
    /// 字幕解码走独立的 [`decode_subtitle_segment`](Self::decode_subtitle_segment)
    /// 通道（输出 [`SubtitleSegment`]，与音视频的 AVFrame 通道并行）。
    ///
    /// # Arguments
    ///
    /// * `source` - The source to open (path, URL, or capture device).
    #[inline]
    pub fn new_subtitle(source: impl Into<Location>) -> Result<Decoder> {
        DecoderBuilder::new(MediaType::SUBTITLE).build(source)
    }

    /// Width of the decoder's input video frame, in pixels.
    #[inline(always)]
    pub fn width(&self) -> i32 {
        self.context.width
    }

    /// Height of the decoder's input video frame, in pixels.
    #[inline(always)]
    pub fn height(&self) -> i32 {
        self.context.height
    }

    /// `AVCodecContext.flags` 位集（`AV_CODEC_FLAG_*`，取值见 [`AVCodecFlag`]）。
    ///
    /// 读的是 `avcodec_open2` **之后**的实际值（未设置时即 rsmedia 的解码默认值
    /// `LOW_DELAY`），编解码器自行调整过的位同样会反映出来。查询用
    /// [`FlagSet::contains`]，需要原始整数时用 [`FlagSet::bits`]。
    #[inline]
    pub fn flags(&self) -> FlagSet<AVCodecFlag> {
        FlagSet::from_bits(self.context.flags as u32)
    }

    /// `AVCodecContext.flags2` 位集（`AV_CODEC_FLAG2_*`，取值见
    /// [`AVCodecFlag2`]）。
    #[inline]
    pub fn flags2(&self) -> FlagSet<AVCodecFlag2> {
        FlagSet::from_bits(self.context.flags2 as u32)
    }

    /// `AVCodecContext.thread_type` 位集（`FF_THREAD_*`，取值见
    /// [`ThreadType`]）。
    #[inline]
    pub fn thread_type(&self) -> FlagSet<ThreadType> {
        FlagSet::from_bits(self.context.thread_type as u32)
    }

    /// `AVCodecContext.thread_count`（0 = 自动）。
    ///
    /// 读的是 `avcodec_open2` **之后**的实际值。未调
    /// [`DecoderBuilder::with_thread_count`] 时 rsmedia 不写该字段，值由 FFmpeg
    /// 决定；显式设置过则原样交给 FFmpeg —— `0` 会被 `avcodec_open2` 改写成自动
    /// 推导出的线程数，非 0 值（含负数）原样保留。取值是否合法由 FFmpeg 判断，
    /// rsmedia 不做取舍。
    #[inline]
    pub fn thread_count(&self) -> i32 {
        self.context.thread_count
    }

    /// The pixel format of the frames this decoder **yields**.
    ///
    /// With [`DecoderBuilder::with_pix_fmt`] that is the configured output format
    /// (decoding converts to it); without it, video still comes out as
    /// [`PixelFormat::YUV420P`]. The codec's own internal format is not what this
    /// reports — read it from the container's stream parameters if needed.
    #[inline]
    pub fn pix_fmt(&self) -> PixelFormat {
        self.output_pix_fmt
    }

    /// Audio samples per second.
    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.context.sample_rate
    }

    /// The sample format of the frames this decoder **yields**, which is the
    /// element type `T` [`decode`](Self::decode) must be called with.
    ///
    /// Audio keeps the codec's native format unless
    /// [`DecoderBuilder::with_sample_fmt`] unifies the output, in which case this
    /// reports that target format — so the value always matches the frames that
    /// actually arrive.
    #[inline]
    pub fn sample_fmt(&self) -> SampleFormat {
        self.output_sample_fmt
            .unwrap_or_else(|| SampleFormat::from(self.context.sample_fmt))
    }

    #[inline]
    pub fn ch_layout(&self) -> AVChannelLayoutRef<'_> {
        self.context.ch_layout()
    }

    /// Get the decoders input duration
    #[inline(always)]
    pub fn duration(&self) -> Time {
        self.duration
    }

    /// Get decoder time base.
    #[inline(always)]
    pub fn time_base(&self) -> Rational {
        self.duration.time_base()
    }

    /// Number of frames in the input stream (`AVStream.nb_frames`).
    ///
    /// `0` when the container does not state a count.
    #[inline(always)]
    pub fn nb_frames(&self) -> i64 {
        self.nb_frames
    }

    /// The input stream's frame rates, as `(r_frame_rate, avg_frame_rate)`.
    ///
    /// Two rates, not one: `r_frame_rate` is the lowest rate that can represent
    /// all timestamps exactly, `avg_frame_rate` the average over the stream.
    #[inline(always)]
    pub fn frame_rates(&self) -> (f32, f32) {
        self.frame_rate
    }

    #[inline(always)]
    pub fn media_type(&self) -> MediaType {
        self.media_type
    }

    /// Index of the input stream this decoder reads: the best stream of the
    /// builder's [`MediaType`], chosen at build time. Callers feeding packets
    /// themselves must filter the reader's packets by this index.
    #[inline]
    pub fn stream_index(&self) -> usize {
        self.stream_index
    }

    /// Check if decoder is in draining mode.
    pub fn is_drained(&self) -> bool {
        self.state.is_drained()
    }

    /// Whether the decoder itself has reached EOF.
    ///
    /// This is **not** the "may I stop?" predicate: with a filter graph attached
    /// the graph may still hold buffered frames after the decoder is done (a
    /// delayed filter such as `framerate`), so stopping here would drop them. Use
    /// [`is_finished`](Self::is_finished) for that.
    pub fn is_flushed(&self) -> bool {
        self.state.is_flushed()
    }

    /// Whether the decode pipeline is fully done: the decoder reached EOF **and**
    /// the filter graph (when present) has flushed its buffered frames.
    ///
    /// This is the predicate a caller loop should stop on. Stopping at
    /// [`is_flushed`](Self::is_flushed) instead would cut off frames a delayed
    /// filter still has to emit; calling `decode`/`decode_raw` after *this* is
    /// true is what returns "cannot decode after flushed".
    pub fn is_finished(&self) -> bool {
        self.is_flushed()
            && match &self.filter_graph {
                Some(chain) => chain.graph.is_flushed(),
                None => true,
            }
    }

    /// 阶段守卫：解码器只在 `Normal` 阶段接受输入。
    ///
    /// EOS 送出之后（`Drained`/`Flushed`）再送包，FFmpeg 只会回
    /// "Decoder is already flushed"/EINVAL，调用方看不出该怎么办；这里统一
    /// 快速失败，并提示唯一的出路是 [`reset`](Self::reset)。
    fn ensure_normal(&self) -> Result<()> {
        if self.state.is_normal() {
            Ok(())
        } else {
            Err(RsmediaError::invalid_config(format!(
                "Decoder cannot decode after drained/flushed (state: {:?}). Call reset().",
                self.state
            )))
        }
    }

    /// Decode a single frame.
    ///
    /// `T` must match the width of the samples being decoded: the byte size of
    /// the frame's format. For video that is the output pixel format's
    /// [`bytes_per_component`](PixelFormat::bytes_per_component) — 8-bit formats
    /// take `u8`, 9..16-bit ones `u16`. For audio it is the decoded sample
    /// format: the codec's native one unless
    /// [`with_sample_fmt`](DecoderBuilder::with_sample_fmt) unifies the output
    /// (e.g. to `FLTP`, which every audio codec can be read as with `decode::<f32>()`).
    /// A mismatch is rejected with the format and both widths in the error.
    ///
    /// # Return value
    ///
    /// The decoded frame, or [`None`] at end of stream. The frame carries its
    /// own timestamp (`pts` + time base); there is no separate tuple.
    ///
    /// # Errors
    ///
    /// An error once the whole pipeline is finished — decoder at EOF and filter
    /// graph flushed; call [`reset`](Self::reset) before decoding again.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rsmedia::{DecoderBuilder, MediaType, StreamReader};
    /// # fn main() -> rsmedia::error::Result<()> {
    /// # let mut reader = StreamReader::new("/tmp/test.mp4")?;
    /// # let mut decoder = DecoderBuilder::new(MediaType::AUDIO).build_from_reader(&reader)?;
    /// while let Some(frame) = decoder.decode::<f32>(&mut reader)? {
    ///     // Do something with frame...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode<T>(&mut self, reader: &mut impl Reader) -> Result<Option<MediaFrame<T>>>
    where
        T: ElementType,
    {
        decode_stream(
            self,
            reader,
            Decoder::decode_packet::<T>,
            Decoder::drain::<T>,
        )
    }

    /// Decode a single frame as a `MediaFrame<u8>`.
    ///
    /// Convenience for `decode::<u8>()` which is the common video path
    /// (8-bit formats such as YUV420P/RGB24). For audio the element type must
    /// match the decoded sample format size: with the codec's native format by
    /// default, or with the format [`with_sample_fmt`](DecoderBuilder::with_sample_fmt)
    /// unifies the output to — e.g. `decode::<f32>()` for `FLTP`/`FLT`. Use
    /// [`decode_raw`](Self::decode_raw) to avoid the typed conversion entirely.
    ///
    /// # Return value
    ///
    /// The decoded frame, or [`None`] at end of stream.
    ///
    /// # Errors
    ///
    /// An error once the whole pipeline is finished (call [`reset`](Self::reset)
    /// to reuse the decoder), or when `u8` does not match the frame format's
    /// sample width (e.g. 9..16-bit video).
    pub fn decode_frame(&mut self, reader: &mut impl Reader) -> Result<Option<MediaFrame<u8>>> {
        self.decode::<u8>(reader)
    }

    /// Decode a single frame and return the raw ffmpeg `AvFrame`.
    ///
    /// The frame comes out after hardware-frame download and filtering, with no
    /// element-type conversion — reach for it when the typed [`MediaFrame`]
    /// conversion is unwanted (raw access, custom handling).
    ///
    /// # Arguments
    ///
    /// * `reader` - A [`Reader`] to read the source from.
    ///
    /// # Return value
    ///
    /// The decoded frame as [`AVFrame`], or [`None`] at end of stream.
    ///
    /// # Errors
    ///
    /// An error once the whole pipeline is finished — call [`reset`](Self::reset)
    /// to decode again.
    pub fn decode_raw<R>(&mut self, reader: &mut R) -> Result<Option<AVFrame>>
    where
        R: Reader,
    {
        decode_stream(self, reader, Decoder::decode_raw_packet, Decoder::drain_raw)
    }

    /// 解码单个 packet 为字幕（仅字幕解码器，低层手动解码 API）。
    ///
    /// 字幕解码走 rsmpeg 的 [`AVCodecContext::decode_subtitle`]（同步 API，
    /// 与音视频的 send_packet/receive_packet 帧通道并行）：每个 packet 直接
    /// 产出 0 或 1 条字幕，无需排空帧队列。传 [`None`] 表示 flush（排空带
    /// `AV_CODEC_CAP_DELAY` 的解码器；无缓冲的解码器直接返回 [`None`]）。
    ///
    /// 返回的 [`AVSubtitle`] 可通过
    /// [`SubtitleSegment::from_avsubtitle`](crate::subtitle::SubtitleSegment::from_avsubtitle)
    /// 转换为纯文本段落，或经 `rect_iter` 自行处理。
    pub fn decode_subtitle_packet(
        &mut self,
        packet: Option<&mut AVPacket>,
    ) -> Result<Option<AVSubtitle>> {
        // 只有送包才需要阶段守卫；`None`（flush）在已排空的解码器上重复调用
        // 也必须保持可用（`drain_subtitle` 依赖这一点）。
        if packet.is_some() {
            self.ensure_normal()?;
        }
        self.context
            .decode_subtitle(packet)
            .context("Failed to decode subtitle packet")
    }

    /// 解码下一条字幕段落（仅字幕解码器，推荐入口）。
    ///
    /// 驱动「读 packet → 解码 → EOF 排空」状态机：跳过非目标流的 packet，
    /// 目标流 packet 经 [`Self::decode_subtitle_packet`] 解码；reader 耗尽后
    /// 以空包 flush 字幕解码器（处理 `AV_CODEC_CAP_DELAY` 的尾部缓冲）。
    /// 无文本 rect 的字幕（如位图字幕）会被跳过。
    ///
    /// # Return value
    ///
    /// The decoded [`SubtitleSegment`], or [`None`] at end of stream.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut decoder = Decoder::new_subtitle("video.mp4").unwrap();
    /// while let Some(segment) = decoder.decode_subtitle_segment(&mut reader)? {
    ///     println!("{}-{}ms: {}", segment.start_ms, segment.end_ms, segment.text);
    /// }
    /// ```
    pub fn decode_subtitle_segment<R>(&mut self, reader: &mut R) -> Result<Option<SubtitleSegment>>
    where
        R: Reader,
    {
        if self.media_type != MediaType::SUBTITLE {
            return Err(RsmediaError::unsupported(format!(
                "decode_subtitle_segment requires a subtitle decoder, got media type: {:?}",
                self.media_type
            )));
        }
        // 与音视频解码共用同一套驱动（`decode_stream`）：读包 → 解码 → EOF 排空的
        // 骨架、阶段守卫、排空上限都只有一份，字幕不再手抄一遍状态机。
        decode_stream(
            self,
            reader,
            Decoder::decode_packet_subtitle,
            Decoder::drain_subtitle,
        )
    }

    /// 解码一个包并取出字幕段落（[`decode_stream`] 的 `on_packet`）。
    fn decode_packet_subtitle(&mut self, packet: &AVPacket) -> Result<Option<SubtitleSegment>> {
        // `avcodec_decode_subtitle2` 只读包数据，但 rsmpeg 的包装签名要求 `&mut`：
        // 用一个共享同一缓冲的别名（`av_packet_ref`，不拷贝负载）满足它。
        let mut alias = AVPacket::new();
        // SAFETY: 两个 `AVPacket` 都由 rsmpeg 管理生命周期；`av_packet_ref` 失败时
        // `alias` 保持空包状态，成功时与 `packet` 共享引用计数缓冲，全程无写入别名。
        let ret = unsafe { ffi::av_packet_ref(alias.as_mut_ptr(), packet.as_ptr()) };
        if ret < 0 {
            return Err(RsmediaError::FFmpeg(rsmpeg::error::RsmpegError::from(ret)));
        }
        Ok(self
            .decode_subtitle_packet(Some(&mut alias))?
            .and_then(|subtitle| SubtitleSegment::from_avsubtitle(&subtitle)))
    }

    /// EOF 后继续取字幕（[`decode_stream`] 的 `on_drain`）。
    ///
    /// 字幕解码器没有帧队列，flush 的方式是「反复喂空包直到不再返回字幕」
    /// （见 `avcodec_decode_subtitle2` 文档）；`AV_CODEC_CAP_DELAY` 的解码器可能
    /// 仍缓存若干条。循环有上限，避免损坏的流让公开 API 转不出来。
    fn drain_subtitle(&mut self) -> Result<Option<SubtitleSegment>> {
        for _ in 0..crate::MAX_DRAIN_OUTPUTS {
            match self.decode_subtitle_packet(None)? {
                Some(subtitle) => {
                    if let Some(segment) = SubtitleSegment::from_avsubtitle(&subtitle) {
                        return Ok(Some(segment));
                    }
                }
                None => {
                    self.state = ProcessState::Flushed;
                    tracing::debug!("Subtitle decoder flushed. EOF reached.");
                    return Ok(None);
                }
            }
        }
        Err(RsmediaError::msg(format!(
            "Subtitle decoder keeps returning subtitles after EOF ({} iterations); giving up",
            crate::MAX_DRAIN_OUTPUTS
        )))
    }

    /// Decode one [`AVPacket`] into a typed [`MediaFrame<T>`].
    ///
    /// Thin wrapper over [`decode_raw_packet`](Self::decode_raw_packet): same
    /// packet-in / frame-out contract, same multi-frame queuing and error
    /// behavior — the only addition is the element-type conversion (`T` must
    /// match the frame format's sample width, see [`decode`](Self::decode)).
    ///
    /// # Return value
    ///
    /// The decoded frame as [`MediaFrame<T>`] if the decoder has one available, [`None`] if not.
    pub fn decode_packet<T>(&mut self, packet: &AVPacket) -> Result<Option<MediaFrame<T>>>
    where
        T: ElementType,
    {
        self.decode_raw_packet(packet)?
            .map(|raw_frame| MediaFrame::<T>::from_avframe(&raw_frame))
            .transpose()
    }

    /// Decode one [`AVPacket`].
    ///
    /// Feeds the packet to the decoder and returns a frame if there is one available. The caller
    /// should keep feeding packets until the decoder returns a frame.
    ///
    /// A packet may decode into **several** frames (H.264 field coding, MPEG-2 field
    /// pictures). Anything beyond the frame returned here is queued internally and handed
    /// out by the following calls, in decode order, so no frame is lost when the caller
    /// moves on to the next packet.
    ///
    /// # Errors
    ///
    /// Returns an error once the decoder has been flushed (`reset` is required before
    /// decoding again) or when the decoder itself fails. A full decoder input buffer is
    /// **not** an error: the frames that are ready are taken out first and the packet is
    /// sent again.
    ///
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    pub fn decode_raw_packet(&mut self, packet: &AVPacket) -> Result<Option<AVFrame>> {
        // 与 `decode`/`decode_raw` 同一阶段守卫：`drain_raw` 之后解码器已收到 EOS，
        // 再送包会被 FFmpeg 拒绝（EINVAL），必须在 `reset()` 之后才能复用。
        self.ensure_normal()?;
        self.send_packet_with_retry(Some(packet))?;
        self.receive_normalized_frame()
    }

    /// 把一个包（`None` = EOS）送进解码器；解码器输入缓冲满（EAGAIN）时先腾地方再重试。
    ///
    /// 这是解码器**唯一**的送包入口，正常送包与送 EOS 都走这里。
    ///
    /// 与编码侧的 `send_frame_with_retry` 是同一件事：`avcodec_send_packet` 返回
    /// EAGAIN 表示"输入缓冲满了，先把已就绪的输出取走"。一个包能解出多帧的流
    /// （H.264 场编码、MPEG-2 field picture）很容易走到这里，而 rsmpeg 把 EAGAIN
    /// 映射成 [`RsmpegError::DecoderFullError`]，不重试就会把"还没收完帧"报成失败。
    ///
    /// 取出的帧进 [`pending_frames`](Self::pending_frames)（FIFO），由
    /// `receive_normalized_frame` 按顺序交付，因此重试不会改变输出顺序，也不丢帧。
    ///
    /// 时间戳换算不在这里做：进入解码器的包带参流时间基，解码器按 `pkt_timebase`
    /// 解释（见 `DecoderBuilder::setup_codec_context`）。
    fn send_packet_with_retry(&mut self, packet: Option<&AVPacket>) -> Result<()> {
        let mut retries = 0usize;
        loop {
            match self.context.send_packet(packet) {
                Ok(()) => return Ok(()),
                Err(rsmpeg::error::RsmpegError::DecoderFullError) => {
                    retries += 1;
                    if retries > crate::MAX_EAGAIN_RETRIES {
                        return Err(RsmediaError::msg(format!(
                            "Decoder keeps returning EAGAIN after {} retries (eof: {}); aborting",
                            crate::MAX_EAGAIN_RETRIES,
                            packet.is_none()
                        )));
                    }
                    tracing::debug!(
                        "Decoder input buffer full (EAGAIN), draining ready frames first."
                    );
                    self.drain_decoder_frames()?;
                }
                Err(e) => return Err(RsmediaError::FFmpeg(e)),
            }
        }
    }

    /// 把解码器此刻已经就绪的帧全部取进 [`pending_frames`](Self::pending_frames)。
    ///
    /// 只搬运、不归一化：归一化（HW 下载 / 缩放 / 滤镜）严格按交付顺序逐帧做，
    /// 否则滤镜图会看到乱序的输入。
    fn drain_decoder_frames(&mut self) -> Result<()> {
        loop {
            match self.decoder_receive_frame() {
                Ok(Some(frame)) => self.pending_frames.push_back(frame),
                // EAGAIN / EOF：此刻再没有可取的帧。
                Ok(None) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }

    /// Drain one frame from the decoder.
    ///
    /// The first call sends end-of-stream and puts the decoder in draining mode;
    /// afterwards the normal decode path returns an error until
    /// [`reset`](Self::reset) is called.
    ///
    /// # Return value
    ///
    /// The decoded frame as [`MediaFrame<T>`] if the decoder has one available, [`None`] if not.
    ///
    /// # Errors
    ///
    /// Errors if the decoder itself fails while flushing.
    pub fn drain<T>(&mut self) -> Result<Option<MediaFrame<T>>>
    where
        T: ElementType,
    {
        self.drain_raw()?
            .map(|raw_frame| MediaFrame::<T>::from_avframe(&raw_frame))
            .transpose()
    }

    /// Drain one frame from the decoder.
    ///
    /// The first call sends end-of-stream and puts the decoder in draining mode;
    /// afterwards the normal decode path returns an error until
    /// [`reset`](Self::reset) is called.
    ///
    /// Frames a previous [`decode_raw_packet`](Self::decode_raw_packet) left queued are
    /// handed out **before** end-of-stream is sent, so draining never strands them.
    ///
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    ///
    /// 作为低层手动解码 API 的一部分公开：配合 [`decode_raw_packet`](Self::decode_raw_packet)
    /// 使用，可逐 packet 送入解码器并排空缓冲帧。需要 [`MediaFrame`] 的高级调用请使用
    /// [`drain`](Self::drain)。
    pub fn drain_raw(&mut self) -> Result<Option<AVFrame>> {
        // 队列里还有帧时先不送 EOS：那些帧属于"正常读阶段"的产物，交完再进
        // draining 模式，状态机与交付顺序都不会被 EOS 抢跑。
        if self.pending_frames.is_empty() && self.state.is_normal() {
            // 送 EOS 同样是"送一个包"：输出队列还压着帧时 FFmpeg 一样会回 EAGAIN，
            // 因此走同一个重试入口。
            self.send_packet_with_retry(None)?;
            // 已发送 EOS，进入 draining 模式。此后 EAGAIN 表示"仍在 drain"，
            // 而非 read 阶段缺包，因此在此处显式置位。
            self.state = ProcessState::Drained;
        }
        self.receive_normalized_frame()
    }

    /// Restarts the whole decode pipeline so the decoder can be used again after
    /// draining (or after a seek).
    ///
    /// This is [`flush_buffers`](Self::flush_buffers) — codec buffers *and* the
    /// filter graph — plus the phase reset. Restoring only the codec would leave a
    /// filtered decoder in a state that reports "ready" while every frame is
    /// rejected by the graph.
    pub fn reset(&mut self) -> Result<()> {
        self.flush_buffers()?;
        self.state = ProcessState::Normal;
        Ok(())
    }

    /// Discards the decoder's buffered frames and resets it to a clean state.
    ///
    /// 冲刷解码器与整条流水线的中间缓冲，回到可继续解码的状态。
    ///
    /// This is `avcodec_flush_buffers`: after a seek the decoder still holds
    /// frames from the old position, and this drops them so decoding restarts at
    /// the new one. It is deliberately **not** named `flush` — on an
    /// [`Encoder`](crate::encode::Encoder), `flush` *drains* everything that is
    /// still buffered towards the writer, the opposite direction.
    ///
    /// The filter graph is part of the pipeline and holds frames of its own
    /// (a delay filter such as `fps` keeps one, and anything queued behind it
    /// stays inside the graph). `avcodec_flush_buffers` knows nothing about it, so
    /// this rebuilds the graph as well; without that, frames from **before the
    /// seek would be emitted after it.
    ///
    /// `pending_frames` is part of the pipeline for the same
    /// reason: those frames were already decoded, so dropping the codec's buffers
    /// without dropping them would deliver pre-seek pictures after the seek.
    ///
    /// 同时把解码**阶段**重置回 `Normal`：本 API 的调用时点就是 "seek 之后要
    /// 重新喂包"，可上一轮若已 `drain` 到过 `Drained`（EOS 已送出），FFmpeg 只会
    /// 回 "Decoder is already flushed"/EINVAL —— 缓冲清空了，阶段却还锁在
    /// "禁止送包"上，调用方只会拿到一句让人摸不着头脑的报错。既然两边缓冲与
    /// 滤镜图都已重建，把阶段一并带回起点才是完整语义（与 [`reset`](Self::reset)
    /// 的差别只剩"是否显式声明这是重启"），已到达 EOF 的调用无需 reset 即可重来。
    pub fn flush_buffers(&mut self) -> Result<()> {
        // SAFETY: `self.context` is a live decoder context we exclusively own (`&mut self`);
        // `avcodec_flush_buffers` only resets its internal buffers.
        unsafe {
            ffi::avcodec_flush_buffers(self.context.as_mut_ptr());
        }
        self.pending_frames.clear();
        self.state = ProcessState::Normal;
        self.rebuild_filter_graph()
    }

    /// Rebuilds the filter graph (if any) so it forgets every buffered frame.
    ///
    /// See [`FilterGraph::rebuild`] for why a rebuild is the only way to do this.
    /// The graph's build parameters were kept at construction for exactly this
    /// call, so a rebuilt graph is identical to the original one.
    fn rebuild_filter_graph(&mut self) -> Result<()> {
        let Some(chain) = self.filter_graph.as_mut() else {
            return Ok(());
        };
        let DecodeFilterChain {
            graph,
            params,
            filters,
        } = chain;
        graph
            .rebuild(params, filters)
            .context("Failed to rebuild the filter graph")
    }

    /// Pulls one frame out of the decoder and returns it in a uniform shape.
    ///
    /// This is the single place where a decoded frame is normalised, in this
    /// order: download hardware frames to system memory, convert the video pixel
    /// format / resize through swscale, convert the audio sample format through
    /// swresample, and finally run the result through the filter graph (when one
    /// is configured). Callers above see only the resulting software frame.
    fn receive_normalized_frame(&mut self) -> Result<Option<AVFrame>> {
        // 0. 解码器在重试路径上腾出来的帧（一个包解出多帧时）优先交付，保证顺序。
        let decoded_frame = match self.pending_frames.pop_front() {
            Some(frame) => frame,
            None => match self.decoder_receive_frame() {
                Ok(Some(f)) => f,
                Ok(None) => {
                    // 解码器当前无帧可出。按状态区分是"仍需更多输入"还是"已到 EOF"：
                    // - Normal / Drained：读阶段或 drain 阶段的 EAGAIN，需要继续喂包，
                    //   此时绝不能刷新 filter（否则会给 buffersrc 发 EOF，后续真实帧
                    //   提交会得到 AVERROR_EOF）。
                    // - Flushed：解码器到达 EOF，此时驱动 filter graph 冲刷内部缓冲帧
                    //   （如 fps/setpts 等带延迟滤镜）。逐帧调用 `process_frame(None)`，
                    //   每帧返回一帧，直到 graph 进入 Flushed 状态。
                    return match self.state {
                        ProcessState::Normal | ProcessState::Drained => Ok(None),
                        ProcessState::Flushed => {
                            if let Some(chain) = self.filter_graph.as_mut()
                                && !chain.graph.is_flushed()
                            {
                                match chain.graph.process_frame(None)? {
                                    Some(frame) => {
                                        // 冲刷出来的缓冲帧同样归一化（见下）。
                                        let graph_tb = Rational::from(frame.time_base);
                                        return Ok(Some(
                                            self.normalize_frame_time_base(frame, Some(graph_tb)),
                                        ));
                                    }
                                    None => {
                                        // 理论上到这里 graph 应已 Flushed；个别 FFmpeg
                                        // 版本在图内还有排队帧时可能尚未置 EOF，断言式
                                        // 检查会在 debug 构建误伤（D3）。宽松处理：本拍
                                        // 返回 None，外层循环（is_finished 谓词）会继续
                                        // 拉取直到排空，不丢帧。
                                        tracing::debug!(
                                            "graph produced no frame, flushed={}",
                                            chain.graph.is_flushed()
                                        );
                                    }
                                }
                            }
                            Ok(None)
                        }
                    };
                }
                Err(e) => return Err(e),
            },
        };

        // 1. 处理硬件加速帧下载,
        let sw_frame = match &self.hw_context {
            Some(hw_ctx) if hw_ctx.is_hw_frame(&decoded_frame) => {
                // hw_frame -> sw_frame
                hw_ctx
                    .hw_download(&decoded_frame)
                    .context("Failed HW frame download")?
            }
            _ => {
                // 已经是 CPU 帧或无 HWaccel
                decoded_frame
            }
        };

        // 2. 统一视频输出格式（如 YUV420P / RGB24，由 `with_pix_fmt` 配置）
        // 例如：
        // 无硬件加速，默认解码格式 YUV420P
        // 存在硬件加速帧，则转换 NV12 -> 目标格式
        // 注意：这里无论是否有 Filter，都会统一转成目标格式（因为
        // `MediaFrame` 视频目前主要支持 YUV420P/RGB24）。因此即使源是
        // yuv444p / nv12 / 10-bit，解码输出也会被转成目标格式，不会颜色错乱。
        // 有滤镜图时目标格式是「滤镜声明的输入格式，否则 `with_pix_fmt` 的值」
        // （见 `build_from_reader`）：图内 buffer 源按同一格式声明，sink 再转回
        // 解码输出格式，因此对外的格式承诺不变。
        let raw_frame = match self.media_type {
            MediaType::VIDEO => {
                let target_sw_pix_fmt = self
                    .filter_input_format
                    .and_then(FrameFormat::into_pixel)
                    .unwrap_or(self.output_pix_fmt);
                // 计算目标尺寸：无 resize 时保持源尺寸，有 resize 时按策略计算
                let (out_w, out_h) = match self.resize {
                    Some(resize) => resize
                        .compute_for((sw_frame.width as u32, sw_frame.height as u32))
                        .ok_or_else(|| {
                            let (w, h) = (sw_frame.width, sw_frame.height);
                            // `with_resize` 里配置的策略算不出可用尺寸 ⇒ 调用方改配置即可。
                            RsmediaError::invalid_config(format!(
                                "the resize configuration {resize:?} yields no valid size for a {w}x{h} frame"
                            ))
                        })?,
                    None => (sw_frame.width as u32, sw_frame.height as u32),
                };
                self.scaler.scale_if_needed(
                    sw_frame,
                    VideoSpec::new(out_w as i32, out_h as i32, target_sw_pix_fmt),
                )?
            }
            MediaType::AUDIO => match self
                .filter_input_format
                .and_then(FrameFormat::into_sample)
                .or(self.output_sample_fmt)
            {
                // 统一音频输出格式（由 `with_sample_fmt` 配置，或滤镜声明的输入
                // 格式）。与视频侧一样在进滤镜图之前完成，图内因此按目标格式声明
                // 输入（见 build_from_reader）。默认（未指定目标）时不进这里；
                // 「目标 == 原生」与空帧两种情况由 `resample_if_needed` 判掉，零开销。
                Some(target) => self
                    .convert_decoded_audio(sw_frame, target.into())
                    .context("Failed to convert decoded audio to the output sample format")?,
                _ => sw_frame,
            },
            _ => {
                // do nothing
                sw_frame
            }
        };

        // 3. 应用 Filter Graph
        if let Some(chain) = self.filter_graph.as_mut() {
            // filter process
            match chain.graph.process_frame(Some(raw_frame))? {
                Some(filtered_frame) => {
                    let graph_tb = Rational::from(filtered_frame.time_base);
                    Ok(Some(
                        self.normalize_frame_time_base(filtered_frame, Some(graph_tb)),
                    ))
                }
                // `process_frame` 返回 `None` 有两种来源：图还需要更多输入（`EAGAIN`，状态
                // 仍是 `Normal`——上一帧已被图吸收但还没攒够帧），或已经排空完毕
                // （EOF）。两种都让外层循环继续驱动解码器，无需区分。
                None => {
                    tracing::debug!(
                        "Filter graph produced no frame (drained: {}, flushed: {})",
                        chain.graph.is_drained(),
                        chain.graph.is_flushed()
                    );
                    Ok(None)
                }
            }
        } else {
            // 无滤镜图：pts 已是流时间基单位（`pkt_timebase` 即绑定流的时间基），只缺
            // time_base 标注 —— 解码器从不填 `AVFrame.time_base`，不补上则下游
            // [`MediaFrame::from_avframe`](crate::MediaFrame::from_avframe) 拿到
            // "数值有单位却没声明单位"，视频帧每帧刷一次警告。
            Ok(Some(self.normalize_frame_time_base(raw_frame, None)))
        }
    }

    /// 把本解码器产出的帧统一到**绑定流的时间基**（`pkt_timebase` 同款）。
    ///
    /// 解码直出的帧 pts 都在流时间基里（送包前 `set_pkt_timebase` 声明了单位），
    /// 但解码器从不填 `AVFrame.time_base`；滤镜图输出的帧恰恰相反：`time_base`
    /// 有效（buffersink 填的输出链时间基，帧率被 `fps` 改写后是 1/out_fps），
    /// 但那不是流时间基。
    ///
    /// `src_tb = None` 表示"解码直出，仅需补 time_base 标注"；`Some` 则表示
    /// "出自滤镜图输出链"，pts/duration 一并重换算。离开本解码器的帧一律
    /// time_base = 流时间基、pts/duration 与之同单位 —— Encoder 的 assign_pts、
    /// MediaFrame::from_avframe 与 mux 的透传链路因此拿到的是自洽的三元组。
    ///
    /// 流时间基无效（极端裸流容器未标 tb）时保持原样返回，不猜单位。
    fn normalize_frame_time_base(&self, mut frame: AVFrame, src_tb: Option<Rational>) -> AVFrame {
        let dst_tb = self.time_base();
        if dst_tb.num() <= 0 {
            return frame;
        }
        match src_tb {
            // 解码直出：pts/duration 已在流时间基，只补标注。
            None => frame.set_time_base(dst_tb.into()),
            Some(src) if src.num() > 0 && src != dst_tb => {
                if frame.pts != ffi::AV_NOPTS_VALUE {
                    frame.set_pts(frame.pts.rescale(src, dst_tb));
                }
                if frame.duration != ffi::AV_NOPTS_VALUE {
                    let duration = frame.duration;
                    // SAFETY: `frame` 由本函数局部独占；写 duration 只改元数据
                    // （rsmpeg 未暴露 duration setter，走 `UnsafeDerefMut`）。
                    unsafe {
                        frame.deref_mut().duration =
                            avutil::av_rescale_q(duration, src.into(), dst_tb.into());
                    }
                }
                frame.set_time_base(dst_tb.into());
            }
            // 图输出 tb 恰等于流 tb（未改帧率的常见情形）或不有效：只补标注。
            Some(_) => frame.set_time_base(dst_tb.into()),
        }
        frame
    }

    /// 把解码出的音频帧转成目标采样格式（声道布局与采样率不变）。
    ///
    /// 目标格式 = 帧自己的布局与采样率 + 由配置决定的采样格式；重采样器跨帧复用，
    /// 因此重采样延迟缓冲里的尾巴不会每帧被丢掉（逐帧新建上下文就会）。
    fn convert_decoded_audio(
        &mut self,
        frame: AVFrame,
        out_sample_fmt: ffi::AVSampleFormat,
    ) -> Result<AVFrame> {
        let out_spec = resample::AudioSpec::from_frame(&frame).with_sample_fmt(out_sample_fmt);
        resample::resample_if_needed(&mut self.resampler, frame, out_spec)
    }

    /// Pull a decoded frame from the decoder.
    ///
    /// `EAGAIN` ("no frame right now, keep feeding packets") and EOF both surface
    /// as `Ok(None)`; EOF additionally moves the phase to `Flushed`. Any other
    /// decoder error propagates.
    fn decoder_receive_frame(&mut self) -> Result<Option<AVFrame>> {
        match self.context.receive_frame() {
            Ok(frame) => Ok(Some(frame)),
            Err(rsmpeg::error::RsmpegError::DecoderDrainError) => {
                // EAGAIN：此刻无帧可出。
                // - read 阶段：表示"该包暂未解出帧，需继续喂包"，此时不应置 Drained，
                //   否则会使后续 drain_raw 误判已进入 draining 而跳过 EOS 发送（见 drain_raw）。
                // - drain 阶段：Drained 已在 drain_raw 中置位，这里保持即可。
                tracing::debug!("Decoder drained. try send new packet again.");
                Ok(None)
            }
            Err(rsmpeg::error::RsmpegError::DecoderFlushedError) => {
                tracing::debug!("Decoder flushed. EOF reached.");
                self.state = ProcessState::Flushed;
                Ok(None)
            }
            Err(e) => {
                tracing::warn!("Failed to receive frame from decoder: {e}");
                Err(RsmediaError::FFmpeg(e))
            }
        }
    }
}

/// 驱动“读 packet → 解码 → EOF 排空”的通用状态机，供 [`Decoder::decode`] 与
/// [`Decoder::decode_raw`] 复用。`on_packet` 处理单个输入包，`on_drain` 处理 EOF
/// 阶段的排空；二者共用同一套“忽略非目标流 / 报错即返回 / 排空到底”的语义。
fn decode_stream<O, OP, OD>(
    decoder: &mut Decoder,
    reader: &mut impl Reader,
    mut on_packet: OP,
    mut on_drain: OD,
) -> Result<Option<O>>
where
    OP: FnMut(&mut Decoder, &AVPacket) -> Result<Option<O>>,
    OD: FnMut(&mut Decoder) -> Result<Option<O>>,
{
    // 入口守卫只看**整条流水线**是否已结束：解码器先到 EOF 时，滤镜图里可能
    // 还压着缓冲帧（如 `aresample`/`fps` 这类带延迟的滤镜），调用方必须能继续
    // 调用把它们取出来。真正的"不能送包"由 `decode_raw_packet` 的
    // `ensure_normal` 负责——两者管的是不同的事。
    if decoder.is_finished() {
        return Err(RsmediaError::invalid_config(
            "Decoder cannot decode after flushed. Call reset().",
        ));
    }

    let mut read_exhausted = false;
    let mut drained_iterations = 0usize;
    Ok(loop {
        if !read_exhausted {
            match reader.read_packet() {
                Ok(Some((stream_index, packet))) => {
                    if stream_index != decoder.stream_index() {
                        // 跳过其它流
                        tracing::trace!("skip stream index: {}, {:?}", stream_index, packet);
                        continue;
                    }
                    if let Some(out) = on_packet(decoder, &packet)? {
                        break Some(out);
                    }
                }
                Ok(None) => {
                    tracing::debug!("No more packets, Reader exhausted.");
                    read_exhausted = true;
                    continue;
                }
                Err(e) => {
                    tracing::error!("Error reading packet: {e}");
                    return Err(e);
                }
            }
        } else {
            match on_drain(decoder) {
                Ok(Some(out)) => break Some(out),
                Ok(None) => {
                    // None 可能来自 Drained（EAGAIN，解码器仍有缓冲帧待产出）或
                    // Flushed（EOF）。若是 Drained 需继续 drain，否则会丢失尾部帧
                    // （多见于含 B 帧的码流）。
                    if decoder.is_drained() {
                        // 有上限的继续排空：解码器若一直回 EAGAIN 而从不报 EOF，
                        // 说明状态机已坏（或码流异常），必须报错而不是当作正常
                        // 结束 —— 否则调用方会把截断的输出当成完整结果。
                        if drained_iterations >= crate::MAX_EAGAIN_RETRIES {
                            return Err(RsmediaError::msg(format!(
                                "Decoder keeps returning EAGAIN after EOF, aborting after {} \
                                 iterations",
                                crate::MAX_EAGAIN_RETRIES
                            )));
                        }
                        drained_iterations += 1;
                        tracing::debug!("Decoder drained, keep draining.");
                        continue;
                    }
                    tracing::debug!("Decoder flushed. EOF reached.");
                    break None;
                }
                Err(e) => {
                    tracing::error!("Error to drain decoder: {e}");
                    return Err(e);
                }
            }
        }
    })
}

/// Important note: Do not forget to drain the decoder after the reader is exhausted. It may still
/// contain frames. Run `drain_raw()` or `drain()` in a loop until no more frames are produced.
impl Drop for Decoder {
    fn drop(&mut self) {
        // 字幕解码走同步的 avcodec_decode_subtitle2，无 send/receive 帧队列
        // 需要排空（且字幕解码器不支持 send_packet flush）
        if self.media_type == MediaType::SUBTITLE {
            return;
        }

        // 1. 先排空解码器里残留的帧。顺序不能颠倒：滤镜图在步骤 2 才冲刷，
        //    若先冲滤镜，解码器排出的帧既进不了图、也不会再从图里流出来。
        //    - `Normal`：先送 EOS 进入 draining；
        //    - `Drained`：EOS 已送过，直接排空（重复送 NULL 只会拿到
        //      AVERROR_EOF 并打出无意义的告警）；
        //    - `Flushed`：已无帧可排，整个步骤跳过。
        if !self.state.is_flushed() {
            let eos_sent = if self.state.is_normal() {
                match self.send_packet_with_retry(None) {
                    Ok(()) => {
                        // 与 `drain` 一致：EOS 送出即进入 draining。不置位的话下面
                        // 的 `Ok(None)`（EAGAIN）会被当成"已到流末尾"立刻退出排空，
                        // 而带帧级多线程的解码器在 EOS 之后完全可以先回一次 EAGAIN
                        // 再吐帧 —— 尾部帧就是这么丢的。
                        self.state = ProcessState::Drained;
                        true
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to send flush packet to decoder during Decoder drop: {e}"
                        );
                        false
                    }
                }
            } else {
                true
            };

            if eos_sent {
                // 兜底上限见 `MAX_EAGAIN_RETRIES`。
                let mut iterations = 0usize;
                loop {
                    if iterations >= crate::MAX_EAGAIN_RETRIES {
                        tracing::warn!(
                            "Decoder drain exceeded {} iterations, forcing EOF.",
                            crate::MAX_EAGAIN_RETRIES
                        );
                        break;
                    }
                    iterations += 1;
                    match self.decoder_receive_frame() {
                        Ok(Some(_frame)) => {
                            // If receive a frame, we continue to drain the queue.
                            tracing::debug!("continue draining decoder queue.");
                        }
                        Ok(None) => {
                            if self.is_drained() {
                                // If we need more, we continue to drain the queue.
                                tracing::debug!("Decoder draining. continue...");
                                continue;
                            } else {
                                tracing::debug!("Decoder flushed. EOF reached.");
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to drain decoder: {e}");
                            break;
                        }
                    }
                }
            }
        }

        // 2. Flush Filter Graph if exists.
        if let Some(chain) = self.filter_graph.as_mut() {
            match chain.graph.flush() {
                Ok(frames) => {
                    if !frames.is_empty() {
                        tracing::warn!(
                            "{} frames dropped during Decoder drop filter flush.",
                            frames.len()
                        );
                    }
                    tracing::debug!("Filter graph flushed during Decoder drop.");
                }
                Err(e) => tracing::error!("Failed to flush filter graph during Decoder drop: {e}"),
            }
        }
    }
}

/// SAFETY:
/// - `Decoder` 内含 `AVCodecContext` / filter graph，FFmpeg 不保证其线程安全，
///   `&Decoder` 跨线程共享（`Sync`）无法成立，故不实现 `Sync`。
/// - `Send`（move 到另一线程独占使用）是安全的：所有资源随对象移动，
///   无线程局部句柄。
unsafe impl Send for Decoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter;
    use std::collections::HashSet;

    /// `with_input_format` 让 `build()` 能打开**没有可探测头部**的输入
    /// （采集设备 / 虚拟源）；不指定格式时，这类输入会以
    /// `Cannot open input file. (-2)` 失败。
    ///
    /// 用 `lavfi` 虚拟源代替真实摄像头：CI 与沙箱里没有设备，也不该有权限依赖。
    /// `lavfi` 与真实设备 demuxer 走的是同一条 `avformat_open_input(url, iformat)`
    /// 路径，因此这里验证的正是设备输入要用的那条。
    #[test]
    fn test_input_format_enables_device_style_input() -> Result<()> {
        // 采集 demuxer 由 avdevice_register_all 注册 —— 见 with_input_format 的
        // 「前置条件」小节。缺这一步会得到毫无线索的 -2。
        crate::init()?;

        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_input_format("lavfi")
            .build("testsrc=size=320x240:rate=30")?;
        assert_eq!(decoder.width(), 320);
        assert_eq!(decoder.height(), 240);

        // 反证：不指定格式时同一个源打不开（容器 demuxer 无法探测 lavfi 图）。
        assert!(
            DecoderBuilder::new(MediaType::VIDEO)
                .build("testsrc=size=320x240:rate=30")
                .is_err(),
            "lavfi input must not open without an explicit format"
        );
        Ok(())
    }

    /// `with_input_options` 的字典走的是 `avformat_open_input`，与
    /// `with_options`（`avcodec_open2`）严格分开；两者互不通用。
    ///
    /// 这里断言的是**副作用**而非"参数被 FFmpeg 校验"：FFmpeg 对不认识的
    /// 输入键是**容忍**的（`StreamReaderBuilder::with_options` 同样如此，实测
    /// 如此），所以"传错键会报错"不是有效断言。能观测的差别是 lavfi 图的
    /// `size`/`rate` 写在 URL 里、参数键影响不到它 —— 因此这里只保证
    /// 设置了参数后构建仍成功（不因参数本身失败），把"字典确实传下去了"
    /// 交给设备类输入在真实设备上的集成测试。
    #[test]
    fn test_input_options_do_not_break_build() -> Result<()> {
        crate::init()?;

        // 真实设备会用的键。lavfi 忽略它们，但参数本身不应导致构建失败。
        let mut opts = AVOptions::new();
        opts.set("framerate", "30");
        opts.set("video_size", "1280x720");
        DecoderBuilder::new(MediaType::VIDEO)
            .with_input_format("lavfi")
            .with_input_options(opts)
            .build("testsrc=size=320x240:rate=30")?;

        // 未知的输入键同样被 FFmpeg 容忍（与 StreamReaderBuilder 一致）。
        let mut bogus = AVOptions::new();
        bogus.set("definitely_not_a_real_option", "1");
        DecoderBuilder::new(MediaType::VIDEO)
            .with_input_format("lavfi")
            .with_input_options(bogus)
            .build("testsrc=size=320x240:rate=30")?;
        Ok(())
    }

    /// 未知格式名以错误返回，不 panic —— 格式名来自调用方，属于输入错误。
    #[test]
    fn test_unknown_input_format_errors_cleanly() {
        let result = DecoderBuilder::new(MediaType::VIDEO)
            .with_input_format("no_such_demuxer")
            .build("testsrc");
        assert!(result.is_err());
    }

    /// 未设置任何输入格式/参数时，`build()` 走内容探测打开输入。
    #[test]
    fn test_default_input_path_unchanged() -> Result<()> {
        let decoder = DecoderBuilder::new(MediaType::VIDEO).build("assets/mp4.mp4")?;
        assert!(decoder.width() > 0 && decoder.height() > 0);
        Ok(())
    }

    #[test]
    fn test_decode_video() -> Result<()> {
        let filters = vec![filter::video::scale(1280, 720, None)?];

        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(filters)
            .build_from_reader(&reader)?;

        loop {
            match decoder.decode_raw(&mut reader) {
                Ok(Some(frame)) => {
                    println!("video frame: {:?}, timebase:{:?}", frame, frame.time_base);
                }
                Ok(None) => {
                    println!("No more frames, decoder flushed");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error decoding frame: {}", e);
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// D1 回归：解码直出的帧自带**流时间基**标注。
    ///
    /// 解码器从不填 `AVFrame.time_base`，老代码把 0/0 的帧原样交出去 —— 下游
    /// （MediaFrame::from_avframe / Encoder::assign_pts）拿到"数值有单位却没
    /// 声明单位"的帧，视频帧每帧刷一次警告、音频帧被误推断成 1/sample_rate。
    #[test]
    fn test_decode_output_frames_carry_stream_time_base() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let stream_tb = decoder.time_base();
        assert!(stream_tb.num() > 0, "fixture 视频流应有有效时间基");

        for i in 0..5 {
            let Some(frame) = decoder.decode_raw(&mut reader)? else {
                assert!(i > 0, "fixture 视频流应至少解出一帧");
                break;
            };
            assert_eq!(
                Rational::from(frame.time_base),
                stream_tb,
                "解码输出帧必须声明流时间基"
            );
            assert!(frame.pts != ffi::AV_NOPTS_VALUE, "解码输出帧的 pts 应有效");
        }
        Ok(())
    }

    /// D1 回归（滤镜图输出）：buffersink 输出链的时间基被归一化回流时间基。
    ///
    /// 图输出帧的 time_base（buffersink 填写）在 scale（不改帧率）后与流相同 ——
    /// 归一化是一次恒等换算，pts 与无滤镜直出逐帧相等、标注统一为流时间基。
    #[test]
    fn test_filtered_decode_frames_carry_stream_time_base() -> Result<()> {
        let mut raw_reader = StreamReader::new("assets/mp4.mp4")?;
        let mut raw_decoder =
            DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&raw_reader)?;
        let stream_tb = raw_decoder.time_base();

        let mut filtered_reader = StreamReader::new("assets/mp4.mp4")?;
        let filters = vec![filter::video::scale(640, 360, None)?];
        let mut filtered = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(filters)
            .build_from_reader(&filtered_reader)?;

        for i in 0..5 {
            let Some(raw) = raw_decoder.decode_raw(&mut raw_reader)? else {
                assert!(i > 0, "fixture 视频流应至少解出一帧");
                break;
            };
            let Some(out) = filtered.decode_raw(&mut filtered_reader)? else {
                panic!("滤镜图（scale 无延迟）不应比直出少出帧 {i}");
            };

            assert_eq!(
                Rational::from(out.time_base),
                stream_tb,
                "滤镜图输出帧也必须声明流时间基"
            );
            assert_eq!(
                out.pts, raw.pts,
                "恒等换算后 pts 应与直出逐帧相等（帧 {i}）"
            );
        }
        Ok(())
    }

    /// D2 回归：`flush_buffers` 也会重置解码阶段 —— 推到 EOF 后（甚至 drain
    /// 过）seek 校验后直接 flush_buffers，无需 `reset()` 即可回到可继续解码态。
    #[test]
    fn test_flush_buffers_resets_phase_after_eof() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;

        while decoder.decode_raw(&mut reader)?.is_some() {}
        assert!(decoder.is_flushed(), "预热：测试先推到 EOF");

        decoder.flush_buffers()?;
        assert!(
            !decoder.is_flushed(),
            "flush_buffers() 必须把阶段带回 Normal（老代码锁在 Flushed）"
        );

        // 阶段复位后重新驱动：reader 已耗尽 → 走 drain → 再次 EOF，全程不该报
        // "cannot send packets after flushed"。
        assert!(decoder.decode_raw(&mut reader)?.is_none());
        assert!(decoder.is_flushed());
        Ok(())
    }

    /// Graph2 回归：`with_resize` + `with_filters` 组合时，buffer 源端点按
    /// **resize 后**的尺寸声明。
    ///
    /// 帧进图之前已被 swscale 缩到 resize 目标尺寸，而 buffersrc 的端点参数在
    /// build 时定格 —— 老代码仍声明 codecpar 尺寸，实喂帧与声明不符，
    /// `av_buffersrc` 以 EINVAL 拒绝（构造即触发）。图内 scale 再缩到 160x90。
    #[test]
    fn test_resize_with_filter_graph_declares_resized_size() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let filters = vec![filter::video::scale(160, 90, None)?];
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 180))
            .with_filters(filters)
            .build_from_reader(&reader)?;

        let first = decoder
            .decode_raw(&mut reader)?
            .expect("fixture 视频流应至少解出一帧");
        assert_eq!(
            (first.width, first.height),
            (160, 90),
            "最终输出应为图内 scale 的目标尺寸"
        );
        Ok(())
    }

    /// Graph3 回归：链内 `with_input_format` 冲突必须 loud fail（InvalidConfig），
    /// 不能像老代码那样静默取第一条、把其余当不存在。
    #[test]
    fn test_conflicting_input_formats_are_rejected() -> Result<()> {
        let filters = vec![
            filter::video::scale(320, 240, None)?.with_input_format(PixelFormat::YUV420P),
            filter::video::scale(320, 240, None)?.with_input_format(PixelFormat::RGB24),
        ];
        let err = match DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(filters)
            .build_from_reader(&StreamReader::new("assets/mp4.mp4")?)
        {
            Ok(_) => panic!("conflicting declarations must fail to build"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("conflicting input formats"),
            "错误信息应当指明冲突：{err:#}"
        );
        Ok(())
    }

    /// Graph3 回归：跨类型声明（视频滤镜配采样格式）没有可行实现，直接拒绝。
    #[test]
    fn test_cross_type_input_format_is_rejected() -> Result<()> {
        let filters =
            vec![filter::video::scale(320, 240, None)?.with_input_format(SampleFormat::FLTP)];
        let err = match DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(filters)
            .build_from_reader(&StreamReader::new("assets/mp4.mp4")?)
        {
            Ok(_) => panic!("cross-type declaration must fail to build"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("cross-type"),
            "错误信息应当指明跨类型：{err:#}"
        );
        Ok(())
    }

    /// `with_flags`/`with_flags2`/`with_thread_type` 的位集参数落到 `AVCodecContext`
    /// 的对应字段（单个标志与 `|` 组合都原样保留），读数走
    /// [`Decoder`] 的 getter（与外部调用方看到的一致）。
    #[test]
    fn test_builder_codec_flags_reach_decoder_context() -> Result<()> {
        use crate::codec::{AVCodecFlag, AVCodecFlag2, ThreadType};

        let reader = StreamReader::new("assets/mp4.mp4")?;
        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_flags(AVCodecFlag::OUTPUT_CORRUPT | AVCodecFlag::BITEXACT)
            .with_flags2(AVCodecFlag2::FAST)
            .with_thread_type(ThreadType::FRAME | ThreadType::SLICE)
            .build_from_reader(&reader)?;

        let want = AVCodecFlag::OUTPUT_CORRUPT.as_raw() | AVCodecFlag::BITEXACT.as_raw();
        assert_eq!(
            decoder.flags().bits() & want,
            want,
            "combined flags must not be truncated to a single bit"
        );
        assert_eq!(decoder.flags2(), AVCodecFlag2::FAST.into());
        assert!(decoder.flags2().contains(AVCodecFlag2::FAST));
        assert_eq!(decoder.thread_type(), ThreadType::FRAME | ThreadType::SLICE);
        // 没有 with_thread_count ⇒ rsmedia 不写该字段，值由 FFmpeg 决定（实测 1）。
        assert!(
            decoder.thread_count() >= 1,
            "未显式设置时应由 FFmpeg 自己定线程数，实际 {}",
            decoder.thread_count()
        );
        Ok(())
    }

    /// 未设置 `with_flags` 时落到 rsmedia 的解码默认值 `LOW_DELAY`；
    /// `thread_count` 则是**用户显式设置才用用户的，没设置就交给 FFmpeg**。
    ///
    /// ⚠️ "未设置"与"显式 0"在解码侧**不等价**，别再合并成一个断言：不写字段时
    /// 上下文保持 `avcodec_alloc_context3` 给的 `1`；而显式写 `0` 是 FFmpeg 的
    /// "自动推导"，`avcodec_open2` 会把它改成实际线程数（本机 8 核 ⇒ 9）。
    /// 所以默认值随 FFmpeg 版本/机器变，**不能**锁成具体数字
    #[test]
    fn test_builder_codec_flags_default_and_thread_count() -> Result<()> {
        use crate::codec::AVCodecFlag;

        let reader = StreamReader::new("assets/mp4.mp4")?;
        let default = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        assert!(
            default.flags().contains(AVCodecFlag::LOW_DELAY),
            "unset flags must fall back to the decoder default LOW_DELAY"
        );

        let explicit = DecoderBuilder::new(MediaType::VIDEO)
            .with_thread_count(2)
            .build_from_reader(&reader)?;
        assert_eq!(explicit.thread_count(), 2, "显式设置的线程数必须原样生效");

        // 多核机器上"未设置"一定不等于 CPU 数；
        // 单核时两者恰好都是 1，这条断言没有判别力，直接跳过。
        let cpus = num_cpus::get() as i32;
        if cpus > 1 {
            assert_ne!(
                default.thread_count(),
                cpus,
                "未设置 thread_count 时不应由 rsmedia 填入 CPU 数（旧行为）"
            );
        }

        // 显式 0：原样交给 FFmpeg 做自动推导，值由 FFmpeg 定，只断言它推导出了
        // 一个合法的正线程数（不锁具体数字，跨机器/跨版本会变）。
        let auto = DecoderBuilder::new(MediaType::VIDEO)
            .with_thread_count(0)
            .build_from_reader(&reader)?;
        assert!(
            auto.thread_count() >= 1,
            "explicit 0 must be left to FFmpeg's auto-detection, got {}",
            auto.thread_count()
        );
        Ok(())
    }

    #[test]
    fn test_decode_audio() -> Result<()> {
        let filters = vec![
            filter::audio::resample(2, 48000, SampleFormat::FLTP),
            filter::audio::volume(1.5)?,
        ];

        let mut reader = StreamReader::new("assets/wav.wav")?;
        let mut decoder = DecoderBuilder::new(MediaType::AUDIO)
            .with_filters(filters)
            .build_from_reader(&reader)?;

        loop {
            match decoder.decode_raw(&mut reader) {
                Ok(Some(frame)) => {
                    println!("audio frame: {:?}, timebase:{:?}", frame, frame.time_base);
                }
                Ok(None) => {
                    println!("No more frames, decoder flushed");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error decoding frame: {}", e);
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// `with_sample_fmt` 统一输出：无论编解码器原生格式是什么，解码帧都转换到
    /// 目标格式，`decode::<T>` 的元素类型因此不必随源文件而变。
    /// `assets/wav.wav` 的帧还带 `AV_CHANNEL_ORDER_UNSPEC` 布局（WAV 无声道
    /// 掩码），顺带回归重采样器的 `AVERROR_INPUT/OUTPUT_CHANGED`。
    #[test]
    fn test_decode_audio_with_sample_fmt_unifies_output() -> Result<()> {
        let audio_path = "assets/wav.wav";
        // 统一到 FLTP：decode::<f32> 全程可用，且每帧的格式都是目标格式。
        let mut reader = StreamReader::new(audio_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::AUDIO)
            .with_sample_fmt(SampleFormat::FLTP)
            .build_from_reader(&reader)?;
        let mut unified_samples = 0u64;
        while let Some(frame) = decoder.decode::<f32>(&mut reader)? {
            assert_eq!(
                frame.format().into_sample(),
                Some(SampleFormat::FLTP),
                "decoded frame was not converted to the requested sample format"
            );
            unified_samples += frame.nb_samples as u64;
        }
        assert!(unified_samples > 0, "no audio decoded");

        // 原生格式（默认）：pcm_s16le 解出 S16，样本总量一致——只换格式不变样本数。
        let mut reader = StreamReader::new(audio_path)?;
        let native_format = reader
            .input()
            .streams()
            .iter()
            .find(|stream| stream.codecpar().codec_type().is_audio())
            .map(|stream| SampleFormat::from(stream.codecpar().format))
            .expect("audio stream");
        let mut decoder = DecoderBuilder::new(MediaType::AUDIO).build_from_reader(&reader)?;
        let mut native_samples = 0u64;
        while let Some(frame) = decoder.decode::<i16>(&mut reader)? {
            native_samples += frame.nb_samples as u64;
        }
        assert_eq!(native_format, SampleFormat::S16);
        assert_eq!(
            native_samples, unified_samples,
            "format conversion changed the decoded sample count"
        );

        Ok(())
    }

    /// `with_sample_fmt` 只对音频解码器有效，`NONE` 不是可用目标：构建时快速失败。
    #[test]
    fn test_decode_builder_sample_fmt_validation() -> Result<()> {
        let reader = StreamReader::new("assets/mp4.mp4")?;
        let video = DecoderBuilder::new(MediaType::VIDEO)
            .with_sample_fmt(SampleFormat::FLTP)
            .build_from_reader(&reader);
        assert!(video.is_err(), "with_sample_fmt must be rejected for video");

        let reader = StreamReader::new("assets/wav.wav")?;
        let none = DecoderBuilder::new(MediaType::AUDIO)
            .with_sample_fmt(SampleFormat::NONE)
            .build_from_reader(&reader);
        assert!(none.is_err(), "SampleFormat::NONE is not a valid target");

        Ok(())
    }

    #[test]
    fn test_decode_video_with_resize() -> Result<()> {
        use crate::Resize;
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .build_from_reader(&reader)?;

        let mut frames = 0usize;
        while let Some(frame) = decoder.decode_raw(&mut reader)? {
            assert_eq!(frame.width, 320);
            assert_eq!(frame.height, 240);
            frames += 1;
        }
        assert!(frames > 0, "expected at least one decoded frame");

        Ok(())
    }

    /// 验证 `with_pix_fmt(RGB24)`：解码输出应为 RGB24；同时 MediaFrame 路径
    /// 可正常转换。
    #[test]
    fn test_decode_video_with_pix_fmt_rgb24() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_pix_fmt(PixelFormat::RGB24)
            .build_from_reader(&reader)?;

        let mut frames = 0usize;
        while let Some(frame) = decoder.decode_raw(&mut reader)? {
            assert_eq!(frame.format, i32::from(PixelFormat::RGB24));
            frames += 1;
        }
        assert!(frames > 0, "expected at least one decoded frame");

        Ok(())
    }

    /// `with_pix_fmt` 只拒绝无法表示为数据平面的格式（位流 / 调色板 / 硬件），
    /// 且在构建时以 `InvalidConfig` 返回错误（而不是 panic，也不是笼统的 `Other`）——
    /// 换一个格式就能成功，所以这是调用方需要改的配置。
    #[test]
    fn test_decode_video_with_pix_fmt_not_storable() {
        for fmt in [
            PixelFormat::MONOWHITE, // 位流：分量不足一字节
            PixelFormat::PAL8,      // 调色板格式：样本指向独立调色板
            PixelFormat::VAAPI,     // 硬件格式：没有主机端样本
        ] {
            let reader = StreamReader::new("assets/mp4.mp4").unwrap();
            let err = match DecoderBuilder::new(MediaType::VIDEO)
                .with_pix_fmt(fmt)
                .build_from_reader(&reader)
            {
                Ok(_) => panic!("{fmt:?} must be rejected"),
                Err(e) => e,
            };
            assert!(
                err.is_invalid_config(),
                "{fmt:?} must be reported as invalid configuration: {err}"
            );
        }
    }

    /// 平面 / 半平面格式同样可以作为解码输出（每平面一个数组），不限 packed 8bit。
    #[test]
    fn test_decode_video_with_planar_pix_fmt_accepted() {
        for fmt in [PixelFormat::NV12, PixelFormat::YUV422P, PixelFormat::GBRP] {
            let reader = StreamReader::new("assets/mp4.mp4").unwrap();
            let result = DecoderBuilder::new(MediaType::VIDEO)
                .with_pix_fmt(fmt)
                .build_from_reader(&reader);
            assert!(result.is_ok(), "{fmt:?} should be accepted");
        }
    }

    /// `with_pix_fmt` 对音频解码器应快速失败，而非静默忽略。
    #[test]
    fn test_decode_audio_with_pix_fmt_fails() {
        let reader = StreamReader::new("assets/mp4.mp4").unwrap();
        let result = DecoderBuilder::new(MediaType::AUDIO)
            .with_pix_fmt(PixelFormat::YUV420P)
            .build_from_reader(&reader);
        assert!(result.is_err());
    }

    /// 验证 `with_resize` 与 `scale` filter 两种缩放方式结果一致，且同时使用时
    /// 按「先 resize 后 filter」的顺序叠加，不冲突。
    #[test]
    fn test_resize_vs_filter_scale() -> Result<()> {
        use crate::Resize;

        let video_path = "assets/mp4.mp4";

        // A) 仅 with_resize
        eprintln!("[A] with_resize only");
        let mut reader_a = StreamReader::new(video_path)?;
        let mut dec_a = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .build_from_reader(&reader_a)?;
        let mut a_dims = HashSet::new();
        while let Some(f) = dec_a.decode_raw(&mut reader_a)? {
            a_dims.insert((f.width, f.height));
        }

        // B) 仅 scale filter
        eprintln!("[B] filter only");
        let mut reader_b = StreamReader::new(video_path)?;
        let mut dec_b = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(vec![filter::video::scale(320, 240, None)?])
            .build_from_reader(&reader_b)?;
        let mut b_dims = HashSet::new();
        while let Some(f) = dec_b.decode_raw(&mut reader_b)? {
            b_dims.insert((f.width, f.height));
        }

        // A 与 B 应得到完全相同的尺寸集合
        assert_eq!(
            a_dims, b_dims,
            "with_resize and filter scale produced different dimensions"
        );
        assert_eq!(a_dims.len(), 1, "expected a single uniform output size");

        // C) 同时使用：resize(320x240) -> filter scale(640x480)，输出应为 filter 尺寸
        eprintln!("[C] resize + filter");
        let mut reader_c = StreamReader::new(video_path)?;
        let mut dec_c = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .with_filters(vec![filter::video::scale(640, 480, None)?])
            .build_from_reader(&reader_c)?;
        let mut c_dims = HashSet::new();
        while let Some(f) = dec_c.decode_raw(&mut reader_c)? {
            c_dims.insert((f.width, f.height));
        }
        assert_eq!(
            c_dims,
            HashSet::from([(640i32, 480i32)]),
            "resize+filter should compose to the filter size"
        );

        Ok(())
    }

    /// builder 的 `with_buffer_pool` 必须进入解码器持有的 [`Scaler`]：
    /// 默认关闭，显式开启后为真（真实 build 路径）。
    #[test]
    fn test_builder_scale_pool_reaches_decoder_scaler() -> Result<()> {
        let video_path = "assets/mp4.mp4";

        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .build_from_reader(&StreamReader::new(video_path)?)?;
        assert!(!decoder.scaler.pool_enabled(), "池化默认关闭");

        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_buffer_pool(true)
            .build_from_reader(&StreamReader::new(video_path)?)?;
        assert!(
            decoder.scaler.pool_enabled(),
            "with_buffer_pool 应进入 Scaler"
        );
        Ok(())
    }

    /// 解码到尾：`is_flushed` 只说解码器自己到 EOF，`is_finished` 才是整条流水线
    /// （含滤镜图）结束。此后继续解码必须报错，而不是静默返回 `None`。
    #[test]
    fn test_decode_reaches_finished_state() -> Result<()> {
        let video_path = "assets/mp4.mp4";
        let mut reader = StreamReader::new(video_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;

        let mut decoded = 0usize;
        while decoder.decode::<u8>(&mut reader)?.is_some() {
            decoded += 1;
            assert!(
                !decoder.is_finished(),
                "reported finished after {decoded} frames, with more still arriving"
            );
        }

        assert!(decoded > 0, "decoded nothing from {}", video_path);
        assert!(decoder.is_flushed(), "EOF must leave the decoder flushed");
        assert!(
            decoder.is_finished(),
            "decoder and its (absent) filter graph must both be finished at EOF"
        );
        assert!(
            decoder.decode::<u8>(&mut reader).is_err(),
            "decoding after the end must error, not silently return None"
        );

        // `reset` 之后可以重新解码（用于 seek 后的复用）。
        decoder.reset()?;
        assert!(!decoder.is_flushed());
        assert!(!decoder.is_finished());
        Ok(())
    }

    /// 带滤镜图的解码器读到 EOF 之后，`reset()` 必须能把**整条流水线**（解码器
    /// + 滤镜图）一起复位。
    ///
    /// 只复位解码器是不够的：滤镜图没有"回退"，一旦见过 EOF 就永久停在 EOF，
    /// 之后每一帧提交都会得到 `AVERROR_EOF` —— 而 `is_finished()` 却会报告
    /// "可以继续"，正是"状态在说谎"。
    #[test]
    fn test_reset_restarts_the_filtered_pipeline() -> Result<()> {
        let video_path = "assets/mp4.mp4";
        let mut reader = StreamReader::new(video_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(vec![crate::filter::video::fps(10.0)?])
            .build_from_reader(&reader)?;

        let mut first_pass = 0usize;
        while decoder.decode::<u8>(&mut reader)?.is_some() {
            first_pass += 1;
        }
        assert!(first_pass > 0, "the filtered decode produced nothing");
        assert!(decoder.is_finished(), "EOF must finish the whole pipeline");

        decoder.reset()?;
        assert!(!decoder.is_flushed() && !decoder.is_finished());

        // 重新读同一个文件：必须能正常出帧，而不是 AVERROR_EOF。
        let mut reader = StreamReader::new(video_path)?;
        let mut second_pass = 0usize;
        while decoder.decode::<u8>(&mut reader)?.is_some() {
            second_pass += 1;
        }
        assert_eq!(
            second_pass, first_pass,
            "a restarted pipeline must decode the same stream again"
        );
        Ok(())
    }

    /// `flush_buffers()`（seek 后使用）必须把**滤镜图里**的缓冲帧一起丢掉。
    ///
    /// `avcodec_flush_buffers` 只管编解码器；带缓冲的滤镜（如 `fps`）会把 seek
    /// 之前的帧留在图里，下一次解码就会把它们当新帧吐出来 —— 于是"seek 到 3s"
    /// 之后拿到的是 0.5s 的画面。这里用一个 GOP 已知的自造文件验证。
    ///
    /// 断言的是**不变量**而不是精确落点：默认的 BACKWARD seek 落在**目标位置或其
    /// 之前的最后一个关键帧**上，所以落点在 `[目标 - 一个 GOP, 目标]` 之间，具体
    /// 取决于容器时间基的取整（实测 seek 2s 在 macOS 上落在 2.0s，在 Linux/ffmpeg 7.1
    /// 上落在 1.6s —— 两者都合法）。而泄漏帧来自 seek 之前读到的位置（约 0.5s），
    /// 与合法区间相隔一个 GOP 以上，所以下面用"目标减一个 GOP"当界限即可区分两者。
    #[test]
    fn test_flush_buffers_drops_frames_buffered_by_the_filter_graph() -> Result<()> {
        const FPS: f64 = 25.0;
        const FILTER_FPS: f64 = 10.0;
        const FRAMES: i64 = 100;
        const GOP_FRAMES: i32 = 10;
        // 4s 的素材、seek 到 3s：seek 前的读取位置（~0.2s）与目标相隔很远，
        // 泄漏帧与合法落点因此有很宽的间隔。
        const SEEK_MS: i64 = 3_000;
        let path = crate::test_support::test_output_path("decode", "test_flush_filter.mp4");

        {
            let mut muxer = crate::Muxer::new(&path)?;
            let encoder = crate::EncoderBuilder::new_video(64, 64)
                .with_fps(FPS as f32)
                .with_gop_size(GOP_FRAMES)
                .build()?;
            let index = muxer.add_encoder(encoder)?;
            for frame_index in 0..FRAMES {
                let mut frame = AVFrame::new();
                frame.set_width(64);
                frame.set_height(64);
                frame.set_format(i32::from(PixelFormat::YUV420P));
                frame
                    .alloc_buffer()
                    .context("Failed to allocate frame buffer")?;
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        let mut reader = StreamReader::new(&path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(vec![crate::filter::video::fps(FILTER_FPS)?])
            .build_from_reader(&reader)?;

        // 从头解几帧，让滤镜图里真正开始有缓冲
        let mut decoded_before_seek = 0i64;
        for _ in 0..5 {
            if decoder.decode_raw(&mut reader)?.is_some() {
                decoded_before_seek += 1;
            }
        }
        assert!(
            decoded_before_seek > 0,
            "the filter graph must have produced frames before the seek for this test to mean anything"
        );

        use crate::io::Seekable;
        reader.seek_to_timestamp(SEEK_MS)?;
        decoder.flush_buffers()?;

        let first = decoder
            .decode_raw(&mut reader)?
            .ok_or_else(|| RsmediaError::msg("no frame decoded after the seek"))?;

        // 滤镜图输出帧的 pts 以 `1/fps` 为时间基（解码器不填 AVFrame.time_base）。
        // 落点最多比目标早一个 GOP；再放宽一帧输出网格的取整。泄漏帧（seek 前的位置，
        // 约 0.5s → pts≈5）远在界限之下，所以这个界限仍能抓住回归。
        let seek_secs = SEEK_MS as f64 / 1000.0;
        let gop_secs = f64::from(GOP_FRAMES) / FPS;
        let earliest_pts = ((seek_secs - gop_secs) * FILTER_FPS) as i64 - 1;
        assert!(
            first.pts >= earliest_pts,
            "the first frame after seeking to {seek_secs}s is at graph pts {} \
             (a legitimate BACKWARD seek lands between {} and {}, one GOP early at worst): \
             the filter graph is still holding pre-seek frames",
            first.pts,
            earliest_pts,
            (seek_secs * FILTER_FPS) as i64
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// flush 之后再走低层入口（`decode_raw_packet`）也要拿到同一个清晰的错误。
    #[test]
    fn test_decode_raw_packet_rejects_a_finished_decoder() -> Result<()> {
        let video_path = "assets/mp4.mp4";
        let mut reader = StreamReader::new(video_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        while decoder.decode_raw(&mut reader)?.is_some() {}
        assert!(decoder.is_finished());

        let mut source = StreamReader::new(video_path)?;
        if let Some((_index, packet)) = source.read_packet()? {
            let err = match decoder.decode_raw_packet(&packet) {
                Ok(_) => panic!("a finished decoder must reject a packet"),
                Err(e) => e,
            };
            assert!(err.is_invalid_config(), "{err}");
            assert!(err.to_string().contains("reset()"), "{err}");
        }
        Ok(())
    }

    /// 解码器输入缓冲满（`avcodec_send_packet` 返回 EAGAIN）时必须先取走已就绪的帧
    /// 再重试，而不是把 `DecoderFullError` 抛给调用方。
    ///
    /// 触发条件是"连续送包而不取帧"——正是 [`decode_raw_packet`](Self::decode_raw_packet)
    /// 返回 `None`（解码器还在缓冲，没到出帧的时候）时调用方的下一步动作。对
    /// `assets/mp4.mp4` 实测第 3 次发送必然 EAGAIN。
    #[test]
    fn test_send_packet_with_retry_recovers_from_a_full_decoder() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");
        let reader = StreamReader::new(video_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut source = StreamReader::new(video_path)?;

        // 连送 4 个包，中途一次也不取帧。
        let mut packets = 0usize;
        while packets < 4 {
            let Some((index, packet)) = source.read_packet()? else {
                break;
            };
            if index != decoder.stream_index() {
                continue;
            }
            decoder
                .send_packet_with_retry(Some(&packet))
                .unwrap_or_else(|e| {
                    panic!("send #{} must survive a full decoder: {e}", packets + 1)
                });
            packets += 1;
        }
        assert_eq!(packets, 4, "the asset must have at least 4 packets");

        // 重试路径取出的帧进了队列，按 FIFO 交付：先到的帧先出来。
        let mut manual = Vec::new();
        while let Some(frame) = decoder.receive_normalized_frame()? {
            manual.push(frame.pts);
        }
        assert!(
            !manual.is_empty(),
            "the buffered frames must still be delivered"
        );

        // 与常规整段解码的前若干帧逐一相同：重试没有丢帧，也没打乱顺序。
        let expected = decode_all_pts(video_path, false)?;
        assert!(expected.len() >= manual.len(), "sanity: prefix must fit");
        assert_eq!(
            manual,
            expected[..manual.len()],
            "frames surfaced by the retry path must be the same prefix, in the same order"
        );
        Ok(())
    }

    /// 手动逐包解码与常规 `decode_frame` 通道必须给出**逐帧相同**的结果。
    ///
    /// 这条锁的是 `receive_normalized_frame` 的交付顺序：
    /// 队列与解码器两处来源合起来仍要按解码顺序输出，既不丢帧也不乱序。
    #[test]
    fn test_decode_raw_packet_recovers_from_a_full_decoder() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");

        let reader = StreamReader::new(video_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut source = StreamReader::new(video_path)?;

        // 逐包手动解码，全程不得报错（送包不重试就会在第 3 个包上失败）。
        let mut manual = Vec::new();
        while let Some((index, packet)) = source.read_packet()? {
            if index != decoder.stream_index() {
                continue;
            }
            if let Some(frame) = decoder.decode_raw_packet(&packet)? {
                manual.push(frame.pts);
            }
        }
        while let Some(frame) = decoder.drain_raw()? {
            manual.push(frame.pts);
        }

        // 与常规 `decode_frame` 路径逐帧一致：既没丢帧，也没乱序。
        let expected = decode_all_pts(video_path, false)?;
        assert!(
            expected.len() > 2,
            "the asset must have several frames, got {}",
            expected.len()
        );
        assert_eq!(
            manual, expected,
            "manual packet decoding must deliver exactly the same frames in the same order"
        );
        Ok(())
    }

    /// 造一段可预测的视频：`noise` 为 `true` 时逐像素填噪声（损坏实验用，
    /// 噪声让码流对字节翻转更敏感），否则填纯黑（静态画面）。
    ///
    /// ⚠️ "留空"必须**显式填**而不能省略：`AVFrame::alloc_buffer()` 走
    /// `av_frame_get_buffer` → `av_buffer_alloc`（`av_malloc`），**不清零**。把未初始化
    /// 的像素送进编码器，每次跑出来的码流都不一样 —— 依赖"同一份输入必得同一份输出"
    /// 的损坏实验会间歇性失败。
    fn write_test_clip(path: &std::path::Path, frames: i64, gop: i32, noise: bool) -> Result<()> {
        let mut muxer = crate::Muxer::new(path)?;
        let encoder = crate::EncoderBuilder::new_video(160, 120)
            .with_fps(25.0)
            .with_gop_size(gop)
            .with_bit_rate(600_000)
            .build()?;
        let index = muxer.add_encoder(encoder)?;
        for i in 0..frames {
            let mut frame = AVFrame::new();
            frame.set_width(160);
            frame.set_height(120);
            frame.set_format(i32::from(PixelFormat::YUV420P));
            frame
                .alloc_buffer()
                .context("Failed to allocate frame buffer")?;
            // 先整帧填黑，保证"留空"是确定的黑而不是脏内存。用 `fill_black` 而不
            // 是 `fill_color`：后者底层 `av_image_fill_color` 自 FFmpeg 7.0 才有，
            // 这里必须在所有受支持的版本上都能跑。
            crate::imgutils::fill_black(&mut frame)?;
            if noise {
                // 线性同余发生器：同样的序号得到同样的画面，损坏实验因此可复现。
                let mut state = (i as u32 + 1) | 1;
                let y = frame.data[0];
                let stride = frame.linesize[0] as usize;
                // SAFETY: `alloc_buffer` 已按 160x120 的 YUV420P 分配好平面；
                // 这里只写 Y 平面的有效行/列（不含行末对齐填充）。
                unsafe {
                    for row in 0..120usize {
                        for col in 0..160usize {
                            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            *y.add(row * stride + col) = (state >> 16) as u8;
                        }
                    }
                }
            }
            frame.set_pts(i);
            muxer.mux(frame, index)?;
        }
        muxer.finish()
    }

    /// 解出全部帧的 pts；遇到错误时返回 `Err`（附带已解出的帧数）。
    ///
    /// 固定单线程解码：**损坏码流在多个解码线程下没有可复现的结果**——错误恢复
    /// 走哪条路径取决于线程调度，同一份损坏文件可能这次被容错掩盖、下次直接
    /// 报错。这个测试的全部意义就是"同一份损坏必得同一个结论"，所以线程数必须
    /// 钉住（编码侧本来就是逐字节确定的，见 `test_err_recognition` 的文件哈希）。
    fn decode_all_pts(path: &std::path::Path, strict: bool) -> Result<Vec<i64>> {
        let mut reader = StreamReader::new(path)?;
        let builder = DecoderBuilder::new(MediaType::VIDEO).with_thread_count(1);
        let builder = if strict {
            builder.with_err_recognition(ErrRecognition::BUFFER | ErrRecognition::EXPLODE)
        } else {
            builder
        };
        let mut decoder = builder.build_from_reader(&reader)?;
        let mut pts = Vec::new();
        while let Some(frame) = decoder.decode_frame(&mut reader)? {
            pts.push(frame.pts);
        }
        Ok(pts)
    }

    /// `with_skip_frame(SkipFrame::NONKEY)`：解码器只交出关键帧，帧数从 50 降到
    /// 关键帧数，且**交出来的正是默认解码里被标记为关键帧的那些**（pts 逐一相同）。
    #[test]
    fn test_skip_frame_nonkey_yields_only_key_frames() -> Result<()> {
        const FRAMES: i64 = 50;
        const GOP: i32 = 25;
        let path = crate::test_support::test_output_path("decode", "test_skip_frame.mp4");
        write_test_clip(&path, FRAMES, GOP, false)?;

        let mut reader = StreamReader::new(&path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut all = Vec::new();
        let mut key_pts = Vec::new();
        while let Some(frame) = decoder.decode_frame(&mut reader)? {
            if frame.key_frame {
                key_pts.push(frame.pts);
            }
            all.push(frame.pts);
        }
        assert_eq!(all.len() as i64, FRAMES, "默认解码应给出全部帧");
        // 断言的依据是"两个通道对**同一批**帧的判断一致"，不是关键帧恰好落在哪个
        // 序号上：编码器何时插关键帧属于编码器策略（GOP 之外还有场景切换判定），
        // 不由这里负责。但要确认关键帧确实存在且不是全部，否则下面的相等断言
        // 会因为两边都退化成"什么都没跳过"而变得没有意义。
        assert!(
            !key_pts.is_empty() && key_pts.len() < all.len(),
            "默认解码应给出部分（而非全部/零个）关键帧，实测 {key_pts:?}"
        );

        let mut reader = StreamReader::new(&path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_skip_frame(SkipFrame::NONKEY)
            .build_from_reader(&reader)?;
        let mut kept = Vec::new();
        while let Some(frame) = decoder.decode_frame(&mut reader)? {
            kept.push(frame.pts);
        }
        assert_eq!(
            kept, key_pts,
            "NONKEY 必须只交出关键帧，且 pts 与默认解码的关键帧一一对应"
        );
        // `ALL` 是更极端的档：什么都不交出来（但仍然把码流走完）。
        let mut reader = StreamReader::new(&path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_skip_frame(SkipFrame::ALL)
            .build_from_reader(&reader)?;
        let mut none = 0usize;
        while decoder.decode_frame(&mut reader)?.is_some() {
            none += 1;
        }
        assert_eq!(none, 0, "ALL 不应交出任何帧");
        Ok(())
    }

    /// `with_err_recognition(EXPLODE)`：损坏的码流必须以 `Err` 报告，而不是被
    /// 解码器的容错（错误掩盖）悄悄糊过去。
    ///
    /// 同一份单字节翻转的产物解两遍：默认配置容错继续、完整解出全部帧；加上
    /// `BUFFER | EXPLODE` 后必须解不动。翻的是容器中段的负载字节（moov 与容器
    /// 结构不动），因此失败必然来自解码器而非解封装。
    #[test]
    fn test_err_recognition_explode_surfaces_corruption() -> Result<()> {
        const FRAMES: i64 = 50;
        const GOP: i32 = 25;
        let path = crate::test_support::test_output_path("decode", "test_err_recognition.mp4");
        write_test_clip(&path, FRAMES, GOP, true)?;
        let clean = std::fs::read(&path)?;

        let mut strict_failures = 0usize;
        for percent in [20usize, 40, 60, 80] {
            let mut bytes = clean.clone();
            let at = clean.len() * percent / 100;
            bytes[at] ^= 0xFF;
            let corrupt = crate::test_support::test_output_path(
                "decode",
                &format!("test_err_recognition_{percent}.mp4"),
            );
            std::fs::write(&corrupt, &bytes)?;

            let tolerant = decode_all_pts(&corrupt, false)?;
            assert_eq!(
                tolerant.len() as i64,
                FRAMES,
                "{percent}% 处的单字节损坏应被默认容错掩盖，而不是中断解码"
            );

            if decode_all_pts(&corrupt, true).is_err() {
                strict_failures += 1;
            }
        }
        assert!(
            strict_failures > 0,
            "EXPLODE 必须至少把一处被默认容错掩盖的损坏报成错误"
        );
        Ok(())
    }
}
