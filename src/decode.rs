use crate::codec::AVCodecFlag;
use crate::error::{Context, Result, RsmediaError};
use crate::filter::{AudioParams, Filter, FilterGraph, FilterParams, VideoParams};
use crate::fmt::FrameFormat;
use crate::frame::{ElementType, MediaFrame};
use crate::hwaccel::{HWContext, HWDeviceConfig};
use crate::io::Reader;
use crate::options::Options;
use crate::resample;
use crate::resize::Resize;
use crate::scale::{ScaleAlgorithm, ScaleQuality, Scaler};
use crate::state::ProcessState;
use crate::stream::StreamInfo;
use crate::strutils;
use crate::subtitle::SubtitleSegment;
use crate::{Location, MediaType, PixelFormat, SampleFormat, StreamReader, Time};

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket, AVSubtitle};
use rsmpeg::avformat::AVStream;
use rsmpeg::avutil::{self, AVChannelLayoutRef, AVFrame};
use rsmpeg::ffi;

use std::sync::Arc;

ffi_enum_wrap_from!(
    /// 帧丢弃粒度（`AVCodecContext.skip_frame`，`AVDiscard`）。
    ///
    /// 只影响**解码器是否把解出的帧交出来**，不省去比特流解析/解码本身
    /// （`skip_frame` 之上还有 `AV_CODEC_FLAG2_SKIP_MANUAL`、`avcodec_send_packet`
    /// 层面的跳过，本类型不涉及）。用于只要关键帧的场景：生成缩略图/预览、
    /// 快速粗剪、按关键帧建索引等。
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
    /// 默认（仅 [`CRCCHECK`](Self::CRCCHECK)）是宽松容错，损坏的码流会被掩盖
    /// 成错帧/糊帧继续输出；要"宁可失败也不出错帧"就加上
    /// [`EXPLODE`](Self::EXPLODE)，解码 API 会以 `Err` 报告而不是静默继续。
    ///
    /// 位可组合（`ErrRecognition::BUFFER | ErrRecognition::EXPLODE`，结果为原始
    /// `i32` 掩码）；[`IGNORE_ERR`](Self::IGNORE_ERR) 与
    /// [`EXPLODE`](Self::EXPLODE) 语义相反，FFmpeg 按位判断，同时置位时行为由
    /// FFmpeg 内部顺序决定，调用方不应同时给出。
    #[allow(non_camel_case_types)]
    ErrRecognition, u32 {
        /// 校验 CRC 之类的校验和（默认开启，`AV_EF_CRCCHECK`）。
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
    /// `None` = 未显式设置，构建时取 [`AVCodecFlag::LOW_DELAY`]。
    flags: Option<AVCodecFlag>,
    /// `None` = 未显式设置，构建时取 [`num_cpus::get`]；`Some(n)` 表示调用方指定过
    /// —— 该"显式"信息被 [`Self::owned_option_keys`] 用来判定配置冲突。
    thread_count: Option<usize>,
    media_type: MediaType,
    codec_name: Option<String>,
    codec_opts: Option<Options>,
    filters: Option<Vec<Filter>>,
    hw_device_config: Option<HWDeviceConfig>,
    /// 缩放核选择（互斥，只取一个算法位）
    scale_algorithm: ScaleAlgorithm,
    /// 缩放质量位（可多位，见 [`ScaleQuality`]）；构建 `Scaler` 时由 [`ScaleQuality::mask`] 合成为掩码
    scale_quality: Vec<ScaleQuality>,
    /// 是否用 `AVBufferPool` 池化缩放输出的帧缓冲（默认关闭）。
    scale_pool: bool,
    resize: Option<Resize>,
    /// 解码输出目标像素格式（仅视频），默认 [`PixelFormat::YUV420P`]。
    pix_fmt: Option<PixelFormat>,
    /// 解码输出目标采样格式（仅音频）。`None` 表示保留编解码器原生格式。
    sample_fmt: Option<SampleFormat>,
    /// 帧丢弃粒度（`AVCodecContext.skip_frame`）。`None` = FFmpeg 默认（不丢弃）。
    skip_frame: Option<SkipFrame>,
    /// 错误识别掩码（`AVCodecContext.err_recognition`）。`None` = FFmpeg 默认。
    err_recognition: Option<i32>,
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
            codec_opts: None,
            hw_device_config: None,
            thread_count: None,
            flags: None,
            scale_algorithm: ScaleAlgorithm::default(),
            scale_quality: ScaleQuality::default_quality().to_vec(),
            scale_pool: false,
            resize: None,
            pix_fmt: None,
            sample_fmt: None,
            skip_frame: None,
            err_recognition: None,
        }
    }

    /// Set decoding flags.
    pub fn with_flags(mut self, flags: AVCodecFlag) -> Self {
        self.flags = Some(flags);
        self
    }

    /// Set the codec name to use for decoding.
    /// If not set, the decoder will try to guess the codec based on the input.
    pub fn with_codec_name(mut self, codec_name: impl Into<Option<String>>) -> Self {
        self.codec_name = codec_name.into();
        self
    }

    /// codec options to use for decoding.
    ///
    /// 只用于 builder 未建模的**编解码器私有参数**。builder 有 typed setter 的项
    /// （`with_thread_count`、`with_flags`）若同时出现在这里，构建时报
    /// [`RsmediaError::InvalidConfig`]：同一项有两个配置源时无法判断以谁为准。
    pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
        self.codec_opts = options.into();
        self
    }

    /// set the thread count.
    ///
    /// 与 `with_options("threads")` 互斥：两者同时指定会在构建时报错（同一项
    /// 只能有一个配置源）。
    pub fn with_thread_count(mut self, thread_count: usize) -> Self {
        self.thread_count = Some(thread_count);
        self
    }

    /// set the filters to apply to decoded frames.
    pub fn with_filters(mut self, filters: impl Into<Option<Vec<Filter>>>) -> Self {
        self.filters = filters.into();
        self
    }

    /// Enable hardware acceleration with the specified device type.
    ///
    /// * `device_config` - Device to use for hardware acceleration.
    pub fn with_hardware_device(mut self, device_config: Option<HWDeviceConfig>) -> Self {
        self.hw_device_config = device_config;
        self
    }

    /// Set the scaling algorithm used when converting decoded frames to the
    /// output pixel format (and, with [`Self::with_resize`], to the output size).
    ///
    /// The algorithm picks the scaling kernel and is **mutually exclusive** —
    /// FFmpeg's header states *"Scaler selection options. Only one may be active
    /// at a time."* Defaults to [`ScaleAlgorithm::BICUBIC`]; use
    /// [`ScaleAlgorithm::BILINEAR`] for output consistent with FFmpeg's command
    /// line default, or [`ScaleAlgorithm::AREA`] when downscaling. The
    /// quality/behaviour bits are set separately with [`Self::with_scale_quality`].
    pub fn with_scale_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.scale_algorithm = algorithm;
        self
    }

    /// Set the scaling quality/behaviour bits used when converting decoded frames.
    ///
    /// Unlike the algorithm (exactly one bit), the quality flags are a set: pass the
    /// bits themselves as a list — `[ScaleQuality::BITEXACT]`,
    /// `[ScaleQuality::FULL_CHR_H_INT, ScaleQuality::ACCURATE_RND]`, … — and they are
    /// combined into the mask handed to FFmpeg. Taking a list of [`ScaleQuality`]
    /// values rather than a raw `u32` means an invalid flag cannot be passed.
    /// Defaults to [`ScaleQuality::default_mask`].
    pub fn with_scale_quality(mut self, quality: impl AsRef<[ScaleQuality]>) -> Self {
        self.scale_quality = quality.as_ref().to_vec();
        self
    }

    /// Enable (`true`) or disable (`false`) pooled allocation of the scaler's
    /// destination frames (see [`Scaler::with_buffer_pool`]).
    ///
    /// Off by default. With it on, frames this decoder scales are allocated from an
    /// internal `AVBufferPool` instead of being freshly allocated per frame, so a
    /// steady stream of same-geometry conversions stops allocating after a couple
    /// of frames; buffers are zero-filled before use, matching `alloc_buffer`.
    pub fn with_scale_pool(mut self, enabled: bool) -> Self {
        self.scale_pool = enabled;
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
    /// ```
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
    /// 默认只有 [`ErrRecognition::CRCCHECK`]，即**宽松容错**：损坏的码流会被
    /// 解码器尽力掩盖（`error_concealment`）成错帧继续输出，调用方拿到的是
    /// "看起来正常"的画面。需要"宁可失败也不出错帧"时，把
    /// [`ErrRecognition::EXPLODE`] 加进来，解码 API 就会以 `Err` 报告损坏，
    /// 由调用方决定重试/跳过/中止。
    ///
    /// 可传单个位，也可传组合出的原始掩码（见 [`ErrRecognition`]）。
    ///
    /// ```
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
    pub fn with_err_recognition(mut self, err_recognition: impl Into<u32>) -> Self {
        self.err_recognition = Some(err_recognition.into() as i32);
        self
    }

    /// 校验像素格式能否以数据平面承载（非位流/调色板/硬件格式）。
    fn ensure_pix_fmt_storable(fmt: PixelFormat) -> Result<()> {
        if !fmt.is_plane_storable() {
            return Err(RsmediaError::msg(format!(
                "Unsupported output pixel format: {fmt:?}; it cannot be stored as sample planes \
                 (bitstream, paletted and hardware formats are not supported)"
            )));
        }
        Ok(())
    }

    /// 某个仅对视频解码器生效的配置项被用于其它媒体类型时构造的错误。
    fn option_only_for(opt: &'static str, value: String, media_type: MediaType) -> RsmediaError {
        RsmediaError::msg(format!(
            "{opt}({value}) is only valid for {} decoders, got media type: {media_type:?}",
            media_type.get_media_name()
        ))
    }

    fn setup_codec_context(&self, decoder: &mut AVCodecContext, input: &AVStream) -> Result<()> {
        let media_type = self.media_type;
        if media_type as ffi::AVMediaType != decoder.codec_type {
            return Err(RsmediaError::msg(format!(
                "Decoder codec type not supported: {:?} vs. {:?}",
                media_type, decoder.codec_type
            )));
        }

        decoder.apply_codecpar(&input.codecpar())?;
        decoder.set_flags(self.flags.unwrap_or(AVCodecFlag::LOW_DELAY) as i32);
        decoder.set_time_base(input.time_base);
        decoder.set_pkt_timebase(input.time_base);
        if let Some(framerate) = input.guess_framerate() {
            decoder.set_framerate(framerate);
        }

        crate::codec::set_thread_count(decoder, self.thread_count.unwrap_or_else(num_cpus::get));

        // 稳定性策略：rsmpeg 未生成 skip_frame / err_recognition 访问器，直接写字段
        // （encode.rs 写 rc_max_rate 同例）。两项都必须在 `avcodec_open2` 之前生效，
        // 否则不会进入解码器初始化。
        unsafe {
            let raw = decoder.as_mut_ptr();
            if let Some(skip_frame) = self.skip_frame {
                (*raw).skip_frame = skip_frame.into();
            }
            if let Some(err_recognition) = self.err_recognition {
                (*raw).err_recognition = err_recognition;
            }
        }

        Ok(())
    }

    /// 解码器 AVOption 里由 builder typed setter 独占的键，`(option key, setter)`。
    ///
    /// 只列出**调用方显式设置过**的项（默认值不算配置过），规则见
    /// [`options::ensure_single_source`](crate::options)。
    fn owned_option_keys(&self) -> Vec<(&'static str, &'static str)> {
        let mut owned = Vec::new();
        if self.thread_count.is_some() {
            owned.push(("threads", "with_thread_count"));
        }
        if self.flags.is_some() {
            owned.push(("flags", "with_flags"));
        }
        if self.skip_frame.is_some() {
            owned.push(("skip_frame", "with_skip_frame"));
        }
        if self.err_recognition.is_some() {
            // AVOption 名是 `err_detect`（`err_recognition` 字段的选项拼写）。
            owned.push(("err_detect", "with_err_recognition"));
        }
        owned
    }

    /// 构建一个**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 适合需要精细控制 reader 生命周期的高级场景：这里只为**探测流信息**临时
    /// 打开一次 reader，构建结束即释放；解码时需每帧传入你自己的 reader
    /// （`decoder.decode(&mut reader)`），seek 也由你显式操作该 reader。
    pub fn build(self, source: impl Into<Location>) -> Result<Decoder> {
        let reader = StreamReader::new(source)?;
        self.build_from_reader(&reader)
    }

    /// 用给定的 reader 构建**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 高级/内部场景使用（如 mux 多流共享同一 reader）。解码时需每帧传入
    /// reader，且不支持 seek。
    pub fn build_from_reader<R: Reader>(self, reader: &R) -> Result<Decoder> {
        let media_type = self.media_type;
        // 单一配置源：typed setter 与 `with_options` 不得同时配置同一项。
        crate::options::ensure_single_source(self.codec_opts.as_ref(), &self.owned_option_keys())?;
        let (stream_index, codec_name) = reader.find_best_stream(media_type)?;
        let input_stream = reader
            .input()
            .streams()
            .get(stream_index)
            .ok_or(RsmediaError::msg(format!(
                "stream: {stream_index} not found!"
            )))?;

        // 优先用调用方指定的解码器名，否则用流自带的名字（`find_best_stream`
        // 的返回值）。这两个名字来自不同来源，不要写在同名绑定里——那样两个分支
        // 看起来一模一样，实际解析到不同的变量。
        let codec_name = self.codec_name.as_deref().unwrap_or(&codec_name);
        let codec = AVCodec::find_decoder_by_name(&strutils::str_to_cstring(codec_name)?)
            .context(format!("Failed to find decoder by name: '{codec_name}'"))?;

        let duration = Time::new(Some(input_stream.duration), input_stream.time_base);
        let nb_frames = input_stream.nb_frames;
        let frame_rate = (
            avutil::av_q2d(input_stream.r_frame_rate) as f32,
            avutil::av_q2d(input_stream.avg_frame_rate) as f32,
        );

        let mut decode_ctx = AVCodecContext::new(&codec);
        self.setup_codec_context(&mut decode_ctx, input_stream)?;

        // video
        let init_width = decode_ctx.width;
        let init_height = decode_ctx.height;

        let hw_context = self
            .hw_device_config
            .filter(|_cfg| {
                // hardware acceleration enabled for video
                media_type == MediaType::VIDEO
            })
            .map(|mut cfg| {
                // codec support or not for hardware acceleration
                let hw_pixel = cfg
                    .device_type
                    .find_hw_pixel_format_with_codec(&codec)
                    .ok_or_else(|| {
                        // 名字来自 FFmpeg，可能是非 UTF-8；它只用于文案，
                        // 拿不到合法字符串并不改变"不支持"这一结论。
                        let codec_name = strutils::cstr_to_string(codec.name())
                            .unwrap_or_else(|_| "unknown".to_owned());
                        RsmediaError::msg(format!(
                            "Decoder with HW acceleration is not supported for codec: {codec_name}"
                        ))
                    })?;

                // 以 `avcodec_get_hw_config` 的声明为准回写配置：hw frames 按这个字段
                // 分配，配置里的值若与编解码器声明不符，会得到错误的输出格式。
                // 未知格式（绑定/FFmpeg 版本不匹配）报错而不是 panic。
                cfg.hw_pixel_format = PixelFormat::from_ffi_checked(hw_pixel).ok_or_else(|| {
                    RsmediaError::msg(format!(
                        "Codec {} reports an unknown hardware pixel format: {hw_pixel}",
                        strutils::cstr_to_string_lossy(codec.name())
                    ))
                })?;

                tracing::info!(
                    "Video decoder with HW acceleration codec: {:?}, hw_pixel: {:?}, config: {:#?}",
                    codec.name(),
                    cfg.hw_pixel_format,
                    cfg
                );

                // create hardware context
                HWContext::new(cfg)
                    .and_then(|ctx| {
                        // 注意：setup_decoder_frames 可能会改变 decode_ctx.pix_fmt
                        ctx.setup_decoder_frames(&mut decode_ctx, init_width, init_height)?;
                        Ok(ctx)
                    })
                    .context("Hardware acceleration context initialization failed")
            })
            .transpose()?;

        let dict = self.codec_opts.and_then(|opts| opts.into_dict());
        decode_ctx
            .open(dict)
            .context("Failed to open decoder for stream")?;

        let stream_info = StreamInfo::from_stream(input_stream)?;
        tracing::info!("{stream_info}");

        // 输出像素格式：仅视频有效。任何能表示为数据平面的格式都接受
        // （布局由描述符推导，见 `PixelFormat::data_layout`）；位流 / 调色板 /
        // 硬件格式在构建期快速失败，而不是拖到运行时。解码输出经 swscale
        // 统一转换到目标格式。
        // 非视频类型配置了 pix_fmt 视为调用方错误，快速失败而非静默忽略。
        let output_pix_fmt = match (media_type, self.pix_fmt) {
            (MediaType::VIDEO, Some(fmt)) => {
                Self::ensure_pix_fmt_storable(fmt)?;
                fmt
            }
            (_, None) => PixelFormat::YUV420P,
            (media_type, Some(fmt)) => {
                return Err(Self::option_only_for(
                    "with_pix_fmt",
                    format!("{fmt:?}"),
                    media_type,
                ));
            }
        };

        // 输出采样格式：仅音频有效。`None` = 保留编解码器原生格式（默认），
        // 代价为零；指定后解码帧在进滤镜图之前统一转换到目标格式。
        // 非音频类型配置了 sample_fmt 视为调用方错误，快速失败而非静默忽略。
        let output_sample_fmt = match (media_type, self.sample_fmt) {
            (MediaType::AUDIO, None) => None,
            (MediaType::AUDIO, Some(fmt)) if fmt != SampleFormat::NONE => Some(fmt),
            (MediaType::AUDIO, Some(fmt)) => {
                return Err(RsmediaError::msg(format!(
                    "Unsupported output sample format: {fmt:?}"
                )));
            }
            (media_type, Some(fmt)) => {
                return Err(Self::option_only_for(
                    "with_sample_fmt",
                    format!("{fmt:?}"),
                    media_type,
                ));
            }
            (_, None) => None,
        };

        // 滤镜链声明的图输入格式（见 [`Filter::with_input_format`]）：帧在进图
        // 之前会被转成它，图内 buffer 源按同一格式声明，两者必须一致（声明与
        // 实喂不符会被 av_buffersrc 拒绝）。未声明时图输入 = 解码输出格式，
        // 零额外转换，与从前完全一致。
        let filter_input_format = self
            .filters
            .as_ref()
            .and_then(|filters| filters.iter().find_map(|f| f.input_format()));

        let filter_graph = if let Some(filters) = self.filters {
            let filter_params = match media_type {
                MediaType::VIDEO => FilterParams::Video(VideoParams {
                    width: init_width,
                    height: init_height,
                    src_format: filter_input_format
                        .and_then(FrameFormat::into_pixel)
                        .unwrap_or(output_pix_fmt),
                    // sink 格式 = 解码输出格式：图内负责把链的输出转回来，
                    // `MediaFrame` 因此拿到的仍是 `with_pix_fmt` 承诺的格式。
                    format: output_pix_fmt,
                    time_base: decode_ctx.time_base,
                    frame_rate: decode_ctx.framerate,
                    pixel_aspect: decode_ctx.sample_aspect_ratio,
                }),
                MediaType::AUDIO => FilterParams::Audio(AudioParams {
                    nb_channels: decode_ctx.ch_layout.nb_channels,
                    sample_rate: decode_ctx.sample_rate,
                    // sink 格式 = 解码输出采样格式（未指定则保留原生）。
                    format: output_sample_fmt.unwrap_or(SampleFormat::from(decode_ctx.sample_fmt)),
                    src_format: filter_input_format
                        .and_then(FrameFormat::into_sample)
                        .unwrap_or(
                            output_sample_fmt.unwrap_or(SampleFormat::from(decode_ctx.sample_fmt)),
                        ),
                    time_base: decode_ctx.time_base,
                }),
                _ => {
                    return Err(RsmediaError::msg(format!(
                        "Unsupported filter for media type: {media_type:?}"
                    )));
                }
            };

            // 滤镜链的媒体类型与可用性校验都在 `init` 内（缺失滤镜 →
            // `FilterNotFound`），这里不再重复一遍。
            let graph = FilterGraph::build(&filter_params, filters.as_slice())?;

            // 参数随图一起留下：重启流水线时必须重建一张新图。
            Some(DecodeFilterChain {
                graph,
                params: filter_params,
                filters,
            })
        } else {
            None
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
            audio_converter: resample::StreamingConverter::new(),
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
    scaler: Scaler,
    resize: Option<Resize>,
    /// 解码输出目标像素格式（仅视频）
    output_pix_fmt: PixelFormat,
    /// 解码输出目标采样格式（仅音频）；`None` = 保留编解码器原生格式
    output_sample_fmt: Option<SampleFormat>,
    /// 滤镜链声明的**进图**格式（见 [`Filter::with_input_format`]）；`None` =
    /// 图输入就是解码输出格式（零额外转换）。有值时进图前的帧会被转成它。
    filter_input_format: Option<FrameFormat>,
    /// 音频输出格式转换器；跨帧复用同一个 `SwrContext`，避免逐帧重建丢掉重采样延迟。
    audio_converter: resample::StreamingConverter,
}

impl Decoder {
    /// Create a decoder to decode the specified source.
    ///
    /// # Arguments
    ///
    /// * `reader` - A [`Reader`] to read the source from.
    #[inline]
    pub fn new_video(source: impl Into<Location>) -> Result<Decoder> {
        DecoderBuilder::new(MediaType::VIDEO).build(source)
    }

    /// Create a decoder to decode the audio stream of the specified source.
    ///
    /// # Arguments
    ///
    /// * `reader` - A [`Reader`] to read the source from.
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
    /// * `reader` - A [`Reader`] to read the source from.
    #[inline]
    pub fn new_subtitle(source: impl Into<Location>) -> Result<Decoder> {
        DecoderBuilder::new(MediaType::SUBTITLE).build(source)
    }

    /// Get the decoders input size width
    #[inline(always)]
    pub fn width(&self) -> i32 {
        self.context.width
    }

    /// Get the decoders input size height
    #[inline(always)]
    pub fn height(&self) -> i32 {
        self.context.height
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
    pub fn time_base(&self) -> ffi::AVRational {
        self.duration.time_base
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
    /// A tuple of the frame timestamp (relative to the stream) and the frame itself.
    ///
    /// # Example
    ///
    /// ```ignore
    /// loop {
    ///     let (ts, frame) = decoder.decode::<u8>().unwrap();
    ///     // Do something with frame...
    /// }
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
    pub fn decode_frame(&mut self, reader: &mut impl Reader) -> Result<Option<MediaFrame<u8>>> {
        self.decode::<u8>(reader)
    }

    /// Decode a single frame and return the raw ffmpeg `AvFrame`.
    ///
    /// # Arguments
    ///
    /// * `reader` - A [`Reader`] to read the source from.
    ///
    /// # Return value
    ///
    /// The decoded raw frame that after decoding, HW download, and filtering as [`AVFrame`].
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
            return Err(RsmediaError::msg(format!(
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
        for _ in 0..crate::MAX_DRAIN_ITERATIONS {
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
            crate::MAX_DRAIN_ITERATIONS
        )))
    }

    /// Decode one [`AVPacket`].
    ///
    /// Feeds the packet to the decoder and returns a frame if there is one available. The caller
    /// should keep feeding packets until the decoder returns a frame.
    ///
    /// # Return value
    ///
    /// A tuple of the [`AVFrame`] and timestamp (relative to the stream) and the frame itself if the
    /// decoder has a frame available, [`None`] if not.
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
    /// # Errors
    ///
    /// Returns an error once the decoder has been flushed (`reset` is required before
    /// decoding again) or when the decoder itself fails.
    ///
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    pub fn decode_raw_packet(&mut self, packet: &AVPacket) -> Result<Option<AVFrame>> {
        // 与 `decode`/`decode_raw` 同一阶段守卫：`drain_raw` 之后解码器已收到 EOS，
        // 再送包会被 FFmpeg 拒绝（EINVAL），必须在 `reset()` 之后才能复用。
        self.ensure_normal()?;
        self.send_packet_to_decoder(Some(packet))?;
        self.receive_normalized_frame()
    }

    /// Drain one frame from the decoder.
    ///
    /// The first call sends end-of-stream and puts the decoder in draining mode;
    /// afterwards the normal decode path returns an error until
    /// [`reset`](Self::reset) is called.
    ///
    /// # Return value
    ///
    /// A tuple of the [`AVFrame`] and timestamp (relative to the stream) and the frame itself if the
    /// decoder has a frame available, [`None`] if not.
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
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    ///
    /// 作为低层手动解码 API 的一部分公开：配合 [`decode_raw_packet`](Self::decode_raw_packet)
    /// 使用，可逐 packet 送入解码器并排空缓冲帧。需要 [`MediaFrame`] 的高级调用请使用
    /// [`drain`](Self::drain)。
    pub fn drain_raw(&mut self) -> Result<Option<AVFrame>> {
        if self.state.is_normal() {
            self.send_packet_to_decoder(None)?;
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
    /// This is `avcodec_flush_buffers`: after a seek the decoder still holds
    /// frames from the old position, and this drops them so decoding restarts at
    /// the new one. It is deliberately **not** named `flush` — on an
    /// [`Encoder`](crate::encode::Encoder), `flush` *drains* everything that is
    /// still buffered towards the writer, the opposite direction.
    ///
    /// The filter graph is part of the pipeline and holds frames of its own
    /// (a delay filter such as `fps` keeps one, and anything queued behind it
    /// stays inside the graph). `avcodec_flush_buffers` knows nothing about it, so
    /// this rebuilds the graph as well; without that, frames from **before** the
    /// seek would be emitted after it.
    pub fn flush_buffers(&mut self) -> Result<()> {
        unsafe {
            ffi::avcodec_flush_buffers(self.context.as_mut_ptr());
        }
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

    /// 把一个包（`None` = EOS）送进解码器。
    ///
    /// 时间戳换算不在这里做：进入解码器的包带参流时间基，解码器按 `pkt_timebase`
    /// 解释（见 `DecoderBuilder::setup_codec_context`）。
    fn send_packet_to_decoder(&mut self, packet: Option<&AVPacket>) -> Result<()> {
        self.context
            .send_packet(packet)
            .context("Failed to send packet to decoder")
    }

    /// Pulls one frame out of the decoder and returns it in a uniform shape.
    ///
    /// This is the single place where a decoded frame is normalised, in this
    /// order: download hardware frames to system memory, convert the video pixel
    /// format / resize through swscale, convert the audio sample format through
    /// swresample, and finally run the result through the filter graph (when one
    /// is configured). Callers above see only the resulting software frame.
    fn receive_normalized_frame(&mut self) -> Result<Option<AVFrame>> {
        // 1. 从解码器获取原始帧
        let decoded_frame = match self.decoder_receive_frame() {
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
                                Some(frame) => return Ok(Some(frame)),
                                None => {
                                    // 已无更多缓冲帧（graph 此时已 Flushed）
                                    debug_assert!(chain.graph.is_flushed());
                                }
                            }
                        }
                        Ok(None)
                    }
                };
            }
            Err(e) => return Err(e),
        };

        // 2. 处理硬件加速帧下载,
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

        // 3. 统一视频输出格式（如 YUV420P / RGB24，由 `with_pix_fmt` 配置）
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
                            RsmediaError::msg(format!(
                                "Cannot resize frame {w}x{h} into {resize:?}"
                            ))
                        })?,
                    None => (sw_frame.width as u32, sw_frame.height as u32),
                };
                self.scaler.scale_if_needed(
                    sw_frame,
                    out_w as i32,
                    out_h as i32,
                    target_sw_pix_fmt,
                )?
            }
            MediaType::AUDIO => match self
                .filter_input_format
                .and_then(FrameFormat::into_sample)
                .or(self.output_sample_fmt)
            {
                // 统一音频输出格式（由 `with_sample_fmt` 配置，或滤镜声明的输入
                // 格式）。与视频侧一样在进滤镜图之前完成，图内因此按目标格式声明
                // 输入（见 build_from_reader）。只在格式真的不同、且帧确实带样本时
                // 转换：默认（未指定目标）与「目标 == 原生」两种情况都零开销，空帧
                // 也无从转换。
                Some(target)
                    if target != SampleFormat::from(sw_frame.format) && sw_frame.nb_samples > 0 =>
                {
                    self.audio_converter
                        .convert(
                            &sw_frame,
                            sw_frame.ch_layout,
                            target.into(),
                            sw_frame.sample_rate,
                        )
                        .context("Failed to convert decoded audio to the output sample format")?
                }
                _ => sw_frame,
            },
            _ => {
                // do nothing
                sw_frame
            }
        };

        // 4. 应用 Filter Graph
        if let Some(chain) = self.filter_graph.as_mut() {
            // filter process
            match chain.graph.process_frame(Some(raw_frame))? {
                Some(filtered_frame) => Ok(Some(filtered_frame)),
                // `process_frame` 只在把图置为 `Drained`（还要更多输入）或
                // `Flushed`（EOF）之后才返回 `None`，因此这里没有第三种情况：
                // 返回 `None` 让外层循环继续驱动解码器，或就此收尾。
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
            // 如果没有 Filter Graph，直接返回 CPU 帧
            Ok(Some(raw_frame))
        }
    }

    /// Pull a decoded frame from the decoder. This function also implements retry mechanism in case
    /// the decoder signals `EAGAIN` and `EOF`
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
                        if drained_iterations >= crate::MAX_DRAIN_ITERATIONS {
                            return Err(RsmediaError::msg(format!(
                                "Decoder keeps returning EAGAIN after EOF, aborting after {} \
                                 iterations",
                                crate::MAX_DRAIN_ITERATIONS
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
                match self.send_packet_to_decoder(None) {
                    Ok(()) => true,
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
                // 兜底上限见 `MAX_DRAIN_ITERATIONS`。
                let mut iterations = 0usize;
                loop {
                    if iterations >= crate::MAX_DRAIN_ITERATIONS {
                        tracing::warn!(
                            "Decoder drain exceeded {} iterations, forcing EOF.",
                            crate::MAX_DRAIN_ITERATIONS
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

    #[test]
    fn test_decode_video() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");

        // drawtext 依赖 libfreetype 编译进 FFmpeg，部分构建未启用：前置探测
        // 滤镜是否存在，缺失时降级为仅 scale，而非匹配错误字符串。
        let scale = filter::video::scale(1280, 720, None);
        let mut reader = StreamReader::new(video_path)?;
        let filters = if filter::is_available("drawtext") {
            let drawtext = filter::video::DrawText::new("Hello", 10, 10, 24, "white").build();
            vec![scale, drawtext]
        } else {
            println!("SKIP drawtext (libfreetype not available)");
            vec![scale]
        };
        let build_decoder = |filters: Vec<Filter>| -> Result<Decoder> {
            DecoderBuilder::new(MediaType::VIDEO)
                .with_filters(filters)
                .build_from_reader(&reader)
        };
        let mut decoder = build_decoder(filters)?;

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

    #[test]
    fn test_decode_audio() -> Result<()> {
        let audio_path = std::path::Path::new("assets/wav.wav");

        let filters = vec![
            filter::audio::resample(2, 48000, SampleFormat::FLTP),
            filter::audio::volume(1.5),
        ];

        let mut reader = StreamReader::new(audio_path)?;
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
        let audio_path = std::path::Path::new("assets/wav.wav");

        // 统一到 FLTP：decode::<f32> 全程可用，且每帧的格式都是目标格式。
        let mut reader = StreamReader::new(audio_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::AUDIO)
            .with_sample_fmt(SampleFormat::FLTP)
            .build_from_reader(&reader)?;
        let mut unified_samples = 0u64;
        while let Some(frame) = decoder.decode::<f32>(&mut reader)? {
            assert_eq!(
                frame.format().and_then(|f| f.into_sample()),
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

        let video_path = std::path::Path::new("assets/mp4.mp4");

        let mut reader = StreamReader::new(video_path)?;
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
        let video_path = std::path::Path::new("assets/mp4.mp4");

        let mut reader = StreamReader::new(video_path)?;
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
    /// 且在构建时返回错误而非 panic。
    #[test]
    fn test_decode_video_with_pix_fmt_unsupported() {
        let video_path = std::path::Path::new("assets/mp4.mp4");
        for fmt in [
            PixelFormat::MONOWHITE, // 位流：分量不足一字节
            PixelFormat::PAL8,      // 调色板格式：样本指向独立调色板
            PixelFormat::VAAPI,     // 硬件格式：没有主机端样本
        ] {
            let reader = StreamReader::new(video_path).unwrap();
            let result = DecoderBuilder::new(MediaType::VIDEO)
                .with_pix_fmt(fmt)
                .build_from_reader(&reader);
            assert!(result.is_err(), "{fmt:?} should be rejected");
        }
    }

    /// 平面 / 半平面格式现在同样可以作为解码输出（每平面一个数组），
    /// 输出格式不再是「YUV420P + packed 8bit」白名单。
    #[test]
    fn test_decode_video_with_planar_pix_fmt_accepted() {
        let video_path = std::path::Path::new("assets/mp4.mp4");
        for fmt in [PixelFormat::NV12, PixelFormat::YUV422P, PixelFormat::GBRP] {
            let reader = StreamReader::new(video_path).unwrap();
            let result = DecoderBuilder::new(MediaType::VIDEO)
                .with_pix_fmt(fmt)
                .build_from_reader(&reader);
            assert!(result.is_ok(), "{fmt:?} should be accepted");
        }
    }

    /// `with_pix_fmt` 对音频解码器应快速失败，而非静默忽略。
    #[test]
    fn test_decode_audio_with_pix_fmt_fails() {
        let video_path = std::path::Path::new("assets/mp4.mp4");
        let reader = StreamReader::new(video_path).unwrap();
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

        let video_path = std::path::Path::new("assets/mp4.mp4");

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
            .with_filters(vec![filter::video::scale(320, 240, None)])
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
            .with_filters(vec![filter::video::scale(640, 480, None)])
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

    /// builder 的 `with_scale_pool` 必须进入解码器持有的 [`Scaler`]：
    /// 默认关闭，显式开启后为真（真实 build 路径）。
    #[test]
    fn test_builder_scale_pool_reaches_decoder_scaler() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");

        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .build_from_reader(&StreamReader::new(video_path)?)?;
        assert!(!decoder.scaler.pool_enabled(), "池化默认关闭");

        let decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_scale_pool(true)
            .build_from_reader(&StreamReader::new(video_path)?)?;
        assert!(
            decoder.scaler.pool_enabled(),
            "with_scale_pool 应进入 Scaler"
        );
        Ok(())
    }

    /// 解码到尾：`is_flushed` 只说解码器自己到 EOF，`is_finished` 才是整条流水线
    /// （含滤镜图）结束。此后继续解码必须报错，而不是静默返回 `None`。
    #[test]
    fn test_decode_reaches_finished_state() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");
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

        assert!(decoded > 0, "decoded nothing from {}", video_path.display());
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
        let path = std::path::Path::new("assets/mp4.mp4");

        let mut reader = StreamReader::new(path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(vec![crate::filter::video::fps(10.0)])
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
        let mut reader = StreamReader::new(path)?;
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
            .with_filters(vec![crate::filter::video::fps(FILTER_FPS as f32)])
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
        let path = std::path::Path::new("assets/mp4.mp4");
        let mut reader = StreamReader::new(path)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        while decoder.decode_raw(&mut reader)?.is_some() {}
        assert!(decoder.is_finished());

        let mut source = StreamReader::new(path)?;
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

    /// 造一段可预测的视频：`content` 为 `true` 时逐像素填噪声（损坏实验用，
    /// 噪声让码流对字节翻转更敏感），否则留空（静态画面）。
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
    fn decode_all_pts(path: &std::path::Path, strict: bool) -> Result<Vec<i64>> {
        let mut reader = StreamReader::new(path)?;
        let builder = DecoderBuilder::new(MediaType::VIDEO);
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
