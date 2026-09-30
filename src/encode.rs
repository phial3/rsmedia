use crate::codec::{AVCodecFlag, AVCodecFlag2, CodecConfig, ThreadType};
use crate::error::{Context, Result, RsmediaError};
use crate::filter::{AudioParams, Filter, FilterGraph, FilterParams, VideoParams};
use crate::flags::FlagSet;
use crate::fmt::FrameFormat;
use crate::frame::{ElementType, MediaFrame};
use crate::hwaccel::{HWContext, HWDeviceConfig};
use crate::io::Writer;
use crate::options::{CRF_CAPABLE_CODECS, Options, Quality, VideoProfile};
use crate::pixel::PixelFormat;
use crate::resample;
use crate::scale::{ScaleAlgorithm, ScaleQuality, Scaler, VideoSpec};
use crate::state::ProcessState;
use crate::strutils;
use crate::subtitle::SubtitleSegment;
use crate::time::{self, Rational, Rescale};
use crate::{MediaType, SampleFormat};

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParameters, AVPacket, AVSubtitle};
use rsmpeg::avutil::{self, AVAudioFifo, AVChannelLayout, AVChannelLayoutRef, AVFrame};
use rsmpeg::ffi;

use std::collections::VecDeque;
use std::sync::Arc;

/// Builds an [`Encoder`].
#[derive(Debug)]
pub struct EncoderBuilder {
    /// Video
    /// 最近一次传给 [`EncoderBuilder::with_fps`] 的浮点帧率，仅供 `build()` 校验。
    ///
    /// 非正或非有限的 `fps` 不会写进 [`Self::frame_rate`]（那里仍是上一个有效值），
    /// 所以必须把调用方原本写的浮点数留到 `build()`，才能 fail fast 而不是静默
    /// 沿用默认帧率。
    fps: Option<f32>,
    width: i32,
    height: i32,
    /// `None` = 未显式指定，`build()` 时按编码器支持列表自动协商。
    pixel_format: Option<PixelFormat>,
    /// Audio
    /// Channel count. `i32` — FFmpeg's own width (`AVChannelLayout.nb_channels`
    /// is a `c_int`), so it reaches `AVChannelLayout::from_nb_channels` and the
    /// supported-channel-count list without a cast.
    nb_channels: i32,
    /// Sample rate in Hz. `i32`, matching `AVCodecContext.sample_rate` (an FFmpeg
    /// `int`) and the codec's own supported-sample-rate list.
    sample_rate: i32,
    /// `None` = 未显式指定，`build()` 时按编码器支持列表自动协商。
    sample_format: Option<SampleFormat>,
    /// Common
    /// 目标码率；`None` = 按媒体类型取默认值（视频 [`Self::VIDEO_BIT_RATE`]、
    /// 音频 [`Self::AUDIO_BIT_RATE`]，与 ffmpeg CLI 一致）。
    bit_rate: Option<i64>,
    /// 瞬时码率上限（`AVCodecContext.rc_max_rate`，ffmpeg CLI 的 `-maxrate`）。
    max_bit_rate: Option<i64>,
    /// VBV 缓冲大小（`AVCodecContext.rc_buffer_size`，ffmpeg CLI 的 `-bufsize`）。
    buffer_size: Option<i32>,
    /// 关键帧间隔；`None` = 不设置，沿用编码器自身的默认值（通用 12，libx264 为
    /// `-1` = 交给 x264，见 [`Self::with_gop_size`]）。
    gop_size: Option<i32>,
    /// B 帧上限；`None` = 不设置，沿用编码器自身的默认值（通用 0，libx264 为 `-1`
    /// = 交给 x264，`medium` preset 下是 3，见 [`Self::with_max_b_frames`]）。
    max_b_frames: Option<i32>,
    frame_rate: Rational,
    /// config
    global_header: bool,
    /// `AVCodecContext.flags`（`AV_CODEC_FLAG_*`）中由调用方显式设置的部分。
    /// `None` = 不额外设置；`GLOBAL_HEADER` 由 [`Self::with_global_header`] 单独管理，
    /// 两者在 `build()` 里按位合并（不同来源的位，不会互相覆盖）。
    flags: Option<FlagSet<AVCodecFlag>>,
    /// `AVCodecContext.flags2`（`AV_CODEC_FLAG2_*`）。`None` = FFmpeg 默认。
    flags2: Option<FlagSet<AVCodecFlag2>>,
    /// `AVCodecContext.thread_type`（`FF_THREAD_*`）。`None` = FFmpeg 默认。
    thread_type: Option<FlagSet<ThreadType>>,
    /// `None` = 未显式设置，构建时取 [`num_cpus::get`]。
    thread_count: Option<i32>,
    media_type: MediaType,
    codec_name: Option<String>,
    codec_opts: Option<Options>,
    level: Option<String>,
    quality: Option<Quality>,
    profile: Option<VideoProfile>,
    /// Subtitle ASS script header (`[Script Info]` + `[V4+ Styles]` + `[Events]`
    /// format line). Required for subtitle encoders; in a transcode pipeline
    /// forward it from the decoded subtitle stream instead of hand-crafting one.
    subtitle_header: Option<String>,
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
}

impl EncoderBuilder {
    /// This is the assumed FPS for the encoder to use.
    /// Note that this does not need to be correct exactly.
    const FRAME_RATE: i32 = 30;

    /// Max numerator/denominator when converting a float fps via `av_d2q`.
    const FPS_MAX: i32 = 100_000;

    /// Default bit rate.
    /// 分辨率(width, height) + 推荐比特率（单位：bps）
    /// * 标清 Sd_480p:          (640, 480)   => 1_000_000,   // 1 Mbps
    /// * 高清 Hd_720p:          (1280, 720)  => 2_500_000,   // 2.5 Mbps
    /// * 全高清 FullHd(1080p):  (1920, 1080) => 5_000_000,   // 5 Mbps
    /// * 超高清 FullHd_2k:      (2560, 1440) => 8_000_000,   // 8 Mbps
    /// * 超高清 UltraHd_4K:     (3840, 2160) => 20_000_000,  // 20 Mbps
    /// * 超高清 FullUltraHd_8K: (7680, 4320) => 60_000_000,  // 60 Mbps
    const VIDEO_BIT_RATE: i64 = 1_000_000;

    /// Default audio bit rate，与 ffmpeg CLI 的 `-b:a` 默认一致（128 kbps）。
    ///
    /// 音频编码器此前沿用 `VIDEO_BIT_RATE`（1 Mbps），是音频合理码率的 8 倍。
    const AUDIO_BIT_RATE: i64 = 128_000;

    /// default video codec
    const VIDEO_CODEC_NAME: &'static str = "libx264";
    /// default audio codec
    const AUDIO_CODEC_NAME: &'static str = "aac";
    /// 字幕默认编码器：subrip（通用文本格式）。MP4 容器请用 指定 [`mov_text`]
    const SUBTITLE_CODEC_NAME: &'static str = "subrip";

    /// 字幕编码器时间基分母：1/1000 秒（毫秒精度），与 ffmpeg CLI 行为一致。
    const SUBTITLE_TIME_BASE_DEN: i32 = 1000;

    /// 单条字幕编码缓冲的**下限**：mov_text 载荷 = 2 字节大端长度 + 文本，
    /// subrip = 纯文本；实际缓冲按文本长度的 2 倍 + 余量分配（见
    /// [`Encoder::encode_subtitle_segment`]），长段落不会因固定缓冲不足而失败。
    const MIN_SUBTITLE_BUFFER_SIZE: usize = 256;

    /// Create a video encoder with the specified destination
    ///
    /// The default codec is `libx264`, with default frame rate and bit rate.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the video stream.
    /// * `height` - The height of the video stream.
    pub fn new_video(width: i32, height: i32) -> Self {
        Self::default().with_width(width).with_height(height)
    }

    /// Create an audio encoder with the specified parameters.
    ///
    /// The default codec is `aac`, with default bit rate of 128k.
    ///
    /// # Arguments
    ///
    /// * `bit_rate` - The bit rate of the audio stream.
    /// * `nb_channels` - The number of channels in the audio stream.
    /// * `sample_rate` - The sample rate of the audio stream.
    /// * `sample_format` - The sample format of the audio stream.
    ///
    /// `nb_channels` and `sample_rate` are `i32`, FFmpeg's own width: the channel
    /// count is an `AVChannelLayout.nb_channels` (`c_int`) and the sample rate is
    /// an `AVCodecContext.sample_rate` (`int`), and the codec's supported-value
    /// lists are signed too. Keeping that width means no cast can silently turn a
    /// nonsensical negative into a huge positive on the way to FFmpeg.
    pub fn new_audio(
        bit_rate: i64,
        nb_channels: i32,
        sample_rate: i32,
        sample_format: SampleFormat,
    ) -> Self {
        Self::default()
            .with_bit_rate(bit_rate)
            .with_nb_channels(nb_channels)
            .with_sample_rate(sample_rate)
            .with_sample_fmt(sample_format)
            .with_media_type(MediaType::AUDIO)
    }

    /// Create a subtitle encoder.
    ///
    /// The default codec is `subrip`; for MP4 output select `mov_text` via
    /// [`Self::with_codec_name`]. The encoder time base defaults to 1/1000
    /// (millisecond precision), matching the ffmpeg CLI.
    ///
    /// A subtitle encoder requires an ASS script header before opening:
    /// provide it via [`Self::with_subtitle_header`] — in a transcode pipeline
    /// forward the header from the decoded subtitle stream, otherwise pass a
    /// complete `[Script Info]` / `[V4+ Styles]` / `[Events]` header (see the
    /// `subtitle` example).
    pub fn new_subtitle() -> Self {
        Self::default().with_media_type(MediaType::SUBTITLE)
    }

    /// Set the ASS script header for subtitle encoders.
    ///
    /// Subtitle encoders (subrip, mov_text, ...) fail to open with
    /// `AVERROR_INVALIDDATA` unless an ASS header is set. An empty or partial
    /// header opens but silently mis-parses dialogues, so always provide a
    /// complete header. In a transcode pipeline copy the header from the
    /// decoder side (it is populated by subtitle decoders on open).
    pub fn with_subtitle_header(mut self, header: impl Into<String>) -> Self {
        self.subtitle_header = Some(header.into());
        self
    }

    /// Set the width of the video stream.
    pub fn with_width(mut self, width: i32) -> Self {
        self.width = width;
        self
    }

    /// Set the height of the video stream.
    pub fn with_height(mut self, height: i32) -> Self {
        self.height = height;
        self
    }

    // 与 DecoderBuilder 共有的那批 setter：定义与文档在 `macros.rs` 的宏里，
    // 改一次两端同时生效（见 `impl_codec_builder_setters` 的说明）。
    impl_codec_builder_setters!();

    /// Set the bit rate.
    ///
    /// 未设置时按媒体类型取默认值：视频 1 Mbps、音频 128 kbps（与 ffmpeg CLI
    /// 的 `-b:a` 默认一致）。
    pub fn with_bit_rate(mut self, bit_rate: i64) -> Self {
        self.bit_rate = Some(bit_rate);
        self
    }

    /// 限制瞬时码率（`rc_max_rate`，等价 ffmpeg CLI 的 `-maxrate`）。
    ///
    /// 目标码率 [`Self::with_bit_rate`] 是**整段平均**，编码器可以在复杂段落大幅超
    /// 出它；`max_bit_rate` 把瞬时码率也压在上限内（VBV）。与 [`Self::with_buffer_size`]
    /// 一起设置、并令 `max_bit_rate == bit_rate` 时即为 CBR（恒定码率）——推流场景
    /// 常用它避免突发码率把上行打满。
    ///
    /// 必须为正；非正值视为未设置，不写入 `rc_max_rate`（`0` 即"不限瞬时码率"）。
    pub fn with_max_bit_rate(mut self, max_bit_rate: i64) -> Self {
        self.max_bit_rate = Some(max_bit_rate);
        self
    }

    /// 设置 VBV 缓冲大小（`rc_buffer_size`，等价 ffmpeg CLI 的 `-bufsize`）。
    ///
    /// 缓冲越大，码率在 `max_bit_rate` 附近的短期波动越自由（质量更稳、上限更松）；
    /// 越小则越贴近严格 CBR，但画面质量波动更明显。必须与
    /// [`Self::with_max_bit_rate`] 配套使用，否则编码器只看到一个巨大的缓冲，
    /// 起不到限流作用。
    ///
    /// 必须为正；非正值视为未设置，不写入 `rc_buffer_size`。
    pub fn with_buffer_size(mut self, buffer_size: i32) -> Self {
        self.buffer_size = Some(buffer_size);
        self
    }

    /// Set the rate control strategy.
    ///
    /// * [`Quality::Bitrate`] — explicit target bit rate in bits/s, overriding
    ///   [`Self::with_bit_rate`]. **Not** video-only: it is written to
    ///   `AVCodecContext::bit_rate` for audio encoders as well. A non-positive
    ///   value counts as "unset" and the media-type default is used instead.
    /// * [`Quality::Crf`] — quality-targeted encoding, **video only**. Applied via
    ///   the codec's `crf` private option for the encoders in [`CRF_CAPABLE_CODECS`];
    ///   any other codec — and every audio codec — falls back to bit-rate control
    ///   (whatever [`Self::with_bit_rate`] or the media-type default yields) with
    ///   a warning, leaving the stream bit rate untouched.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Set the H.264-style profile (`baseline`, `main`, `high`, `high10`,
    /// `high422`, `high444`) via the codec's `profile` private option.
    ///
    /// Works out of the box for `libx264`; other encoders map what they
    /// support and silently ignore unknown profile names (check the encoder
    /// documentation, or pass a codec-specific option via
    /// [`Self::with_options`]). Video only.
    pub fn with_profile(mut self, profile: VideoProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the codec level, e.g. `"4.1"` for H.264 ( Annex A level) — passed
    /// through to the codec's `level` private option when available
    /// (`libx264` accepts Annex A strings like `"4.1"`). Video only.
    pub fn with_level(mut self, level: impl ToString) -> Self {
        self.level = Some(level.to_string());
        self
    }

    /// Set the video frame rate from a floating-point number of frames per second.
    ///
    /// The value is converted to a **reduced rational** via FFmpeg's
    /// `av_d2q(fps, 100_000)` and used as the encoder frame rate.
    ///
    /// ⚠️ That conversion approximates *the `f32` you passed*, not the fraction
    /// you had in mind, and the two are not always the same: `24_000.0 / 1_001.0`
    /// is not representable in `f32`, so the nearest rational to its `f32`
    /// neighbours is `86_002/3_587` — 5.6e-7 fps off, and a **non-standard time
    /// base in the container**. Nothing reports this; the file simply plays
    /// fractionally fast.
    ///
    /// Use [`Self::with_frame_rate`] whenever the rate is a known rational
    /// (anything with a `1_001` denominator, or any rate you need to match
    /// exactly) — it takes the fraction verbatim. Reserve this method for
    /// rates that genuinely come from a float measurement.
    ///
    /// 非正或非有限的 `fps` 会在 [`Self::build`] 时报错（fail fast），而不是静默
    /// 退回默认帧率——那会产出"帧率与预期不符"这类最难排查的结果。
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = Some(fps);
        if fps > 0.0 && fps.is_finite() {
            self.frame_rate = Rational::from(avutil::av_d2q(fps as f64, Self::FPS_MAX));
        }
        self
    }

    /// Set the video frame rate **exactly**, as a rational number of frames per
    /// second.
    ///
    /// This is the entry point for rates that have to be right: the value is used
    /// verbatim, with no float conversion and no `av_d2q` approximation step. It
    /// is what [`Self::with_fps`] cannot express — see that method for the
    /// `86_002/3_587` failure mode this avoids.
    ///
    /// ```
    /// use rsmedia::{EncoderBuilder, Rational};
    ///
    /// // 23.976 fps, exactly 24000/1001 — the film-on-NTSC rate.
    /// let rate = Rational::new(24_000, 1_001)?;
    /// let builder = EncoderBuilder::new_video(1920, 1080).with_frame_rate(rate);
    /// # let _ = builder;
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    ///
    /// The numerator must be positive; a zero or negative one is rejected by
    /// [`Self::build`], the same fail-fast treatment [`Self::with_fps`] gives an
    /// unusable float. A zero denominator is not representable — [`Rational`]
    /// guarantees that by construction.
    pub fn with_frame_rate(mut self, frame_rate: impl Into<Rational>) -> Self {
        self.frame_rate = frame_rate.into();
        self
    }

    /// Set the GOP size (keyframe interval, in frames).
    ///
    /// 未设置时不覆盖 `AVCodecContext::gop_size`，落到哪个值由**编码器**决定：
    /// `avcodec_alloc_context3` 先套 FFmpeg 的通用默认值 12（`g` 选项），随后编码器
    /// 自带的 `FFCodec.defaults` 会覆盖它 —— libx264 把 `g` 置为 `-1`，而它只在
    /// `gop_size >= 0` 时才去覆盖 x264 的设定，于是 x264 沿用自己的 `keyint`
    /// （默认 250）；native 编码器（mpeg4 等）则沿用 12。
    ///
    /// 注意 `0` 会被 libx264 解释为**全 I 帧**（`i_keyint_max` 取 1），除非
    /// 明确想要全帧内编码，否则不要传 0。
    pub fn with_gop_size(mut self, gop_size: i32) -> Self {
        self.gop_size = Some(gop_size);
        self
    }

    /// Set the maximum number of B-frames.
    ///
    /// 未设置时不覆盖 `AVCodecContext::max_b_frames`，默认值同样是**按编码器**的：
    /// FFmpeg 通用默认值是 `0`（`bf` 选项），而 libx264 通过 `FFCodec.defaults`
    /// 把它改成 `-1`；libx264 只在 `max_b_frames >= 0` 时才覆盖 x264 的设定，于是
    /// x264 沿用 preset 自带的值（`medium` 下 `bframes=3`）。
    ///
    /// 注意 `0` 会**显式禁用** B 帧，而不是"交给编码器"；负值在 native 编码器
    /// （mpegvideo 系）上直接是 `AVERROR(EINVAL)`（"max b frames must be 0 or
    /// positive"），只有 libx264 这类把它当"未设置"的编码器才接受。
    pub fn with_max_b_frames(mut self, max_b_frames: i32) -> Self {
        self.max_b_frames = Some(max_b_frames);
        self
    }

    /// Set the pixel format.
    ///
    /// When not set, [`Self::build`] negotiates one from the encoder's
    /// supported list (preferring [`PixelFormat::YUV420P`]).
    pub fn with_pix_fmt(mut self, pixel_format: PixelFormat) -> Self {
        self.pixel_format = Some(pixel_format);
        self
    }

    /// explicit media type, default is `MediaType::VIDEO`
    pub fn with_media_type(mut self, media_type: MediaType) -> Self {
        self.media_type = media_type;
        self
    }

    /// Set the number of audio channels.
    ///
    /// The count is expanded into a channel layout by
    /// `AVChannelLayout::from_nb_channels` at build time, so mono/stereo names
    /// are derived from the count rather than passed explicitly.
    pub fn with_nb_channels(mut self, nb_channels: i32) -> Self {
        self.nb_channels = nb_channels;
        self
    }

    /// Set the audio sample rate in Hz.
    ///
    /// Must appear in the encoder's supported-rate list, otherwise [`Self::build`]
    /// fails with [`RsmediaError::invalid_config`](crate::RsmediaError::invalid_config).
    pub fn with_sample_rate(mut self, sample_rate: i32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the sample format.
    ///
    /// When not set, [`Self::build`] negotiates one from the encoder's
    /// supported list (preferring [`SampleFormat::FLTP`]).
    pub fn with_sample_fmt(mut self, sample_format: SampleFormat) -> Self {
        self.sample_format = Some(sample_format);
        self
    }

    /// Whether the encoder writes its parameter sets into the container's
    /// extradata instead of in-band with each keyframe.
    ///
    /// `true` (the default) sets `AV_CODEC_FLAG_GLOBAL_HEADER`, which is what
    /// container formats expect: MP4/MKV/... store the parameter sets once, in
    /// `AVCodecParameters.extradata`, and every keyframe refers to them.
    ///
    /// `false` is required for a **raw elementary stream** (`-f h264` / `.h264`,
    /// `.h265`). Those muxers write no extradata at all, so with the flag on the
    /// SPS/PPS are simply dropped and the resulting file cannot be decoded. With
    /// it off the encoder repeats them in-band, exactly like the `ffmpeg` CLI.
    pub fn with_global_header(mut self, enabled: bool) -> Self {
        self.global_header = enabled;
        self
    }

    /// 编码器使用的 time_base。
    ///
    /// 视频由用户 fps 取倒数得到 `1/fps`（libx264 等编码器在这种 time_base 下才
    /// 能正确输出 packet duration，避免 MP4 muxer 丢弃末帧）；音频按
    /// `1/sample_rate` 推导，对所有音频编码器一致。
    ///
    /// # Errors
    ///
    /// 视频帧率为 0 时取不到倒数（`build()` 早已拒绝非正分子，故只在手工构造
    /// `EncoderBuilder` 时才会遇到）；音频采样率为 0 时 `1/0` 不是有理数。
    fn effective_time_base(&self) -> Result<Rational> {
        match self.media_type {
            MediaType::VIDEO => self.frame_rate.inverse(),
            MediaType::AUDIO => Rational::new(1, self.sample_rate),
            // 字幕：1/1000（毫秒精度），与 ffmpeg CLI 一致
            MediaType::SUBTITLE => Rational::new(1, Self::SUBTITLE_TIME_BASE_DEN),
            // 其它媒体类型（DATA 等）没有可推导的时间基，用 FFmpeg 的微秒基准。
            _ => Ok(time::TIME_BASE),
        }
    }

    /// The bit rate actually applied to the codec context: an explicit
    /// [`Quality::Bitrate`] overrides [`Self::bit_rate`].
    fn effective_bit_rate(&self) -> i64 {
        match self.quality {
            Some(Quality::Bitrate(bit_rate)) if bit_rate > 0 => bit_rate,
            _ => match self.media_type {
                MediaType::AUDIO => self.bit_rate.unwrap_or(Self::AUDIO_BIT_RATE),
                _ => self.bit_rate.unwrap_or(Self::VIDEO_BIT_RATE),
            },
        }
    }

    /// Apply the settings to an encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Encoder to apply settings to.
    /// * `use_crf` - Whether CRF rate control is active (skip bit rate).
    ///
    /// # Return value
    ///
    /// New encoder with settings applied.
    fn setup_codec_context(
        &self,
        encoder: &mut AVCodecContext,
        use_crf: bool,
        pixel_format: PixelFormat,
        sample_format: SampleFormat,
        config: &CodecConfig,
    ) -> Result<()> {
        let media_type = self.media_type;
        if media_type as ffi::AVMediaType != encoder.codec_type {
            return Err(RsmediaError::invalid_config(format!(
                "encoder was built for media type {media_type:?}, but the codec of the given \
                 stream is {:?}",
                encoder.codec_type
            )));
        }

        if media_type == MediaType::VIDEO {
            encoder.set_width(self.width);
            encoder.set_height(self.height);
            // CRF 模式下不设置 bit_rate（CRF 以质量为目标，码率由编码器自行
            // 决定；写默认 1Mbps 会让 muxer 元数据与实际输出不符）。
            if !use_crf {
                encoder.set_bit_rate(self.effective_bit_rate());
            }
            // gop_size 未设置时不覆盖：值由 `avcodec_alloc_context3` + 编码器自己的
            // `FFCodec.defaults` 共同决定（通用 12，libx264 是 -1 = 交给 x264）。
            // 显式设 0 反而会被 libx264 解释为全 I 帧。
            if let Some(gop_size) = self.gop_size {
                encoder.set_gop_size(gop_size);
            }
            // B 帧上限未设置时不覆盖：通用默认 0，libx264 由 `FFCodec.defaults`
            // 改成 -1 = 交给 x264（medium preset 下 3）；显式设 0 会禁用 B 帧。
            if let Some(max_b_frames) = self.max_b_frames {
                encoder.set_max_b_frames(max_b_frames);
            }
            encoder.set_framerate(self.frame_rate.into());
            encoder.set_time_base(self.effective_time_base()?.into());
            // packet 时间戳在编码器自己的时间基里产出（写包时按
            // `time_base() -> 输出流时间基` 换算），故 pkt_timebase 与它一致。
            // 三种媒体类型一律如此设置——早先只有视频设置了它，音频/字幕留 0/1。
            encoder.set_pkt_timebase(self.effective_time_base()?.into());
            encoder.set_pix_fmt(pixel_format.into());
            encoder.set_sample_aspect_ratio(Rational::ONE.into());
        } else if media_type == MediaType::AUDIO {
            if !config.supports_channel_count(self.nb_channels) {
                return Err(RsmediaError::InvalidConfig(format!(
                    "encoder '{}' does not support nb_channels {}",
                    config.name().to_string_lossy(),
                    self.nb_channels
                )));
            }
            if !config.supports_sample_rate(self.sample_rate) {
                return Err(RsmediaError::InvalidConfig(format!(
                    "encoder '{}' does not support sample rate {}",
                    config.name().to_string_lossy(),
                    self.sample_rate
                )));
            }
            encoder.set_ch_layout(AVChannelLayout::from_nb_channels(self.nb_channels).into_inner());
            encoder.set_bit_rate(self.effective_bit_rate());
            encoder.set_sample_rate(self.sample_rate);
            encoder.set_sample_fmt(sample_format as _);
            encoder.set_time_base(self.effective_time_base()?.into());
            encoder.set_pkt_timebase(self.effective_time_base()?.into());
        } else if media_type == MediaType::SUBTITLE {
            // 字幕编码器只需 time_base（毫秒精度），无像素/采样格式、码率等概念
            encoder.set_time_base(self.effective_time_base()?.into());
            encoder.set_pkt_timebase(self.effective_time_base()?.into());
        } else {
            return Err(RsmediaError::unsupported(format!(
                "media type {media_type:?}"
            )));
        }

        // 速率控制的可选约束（VBV）：rsmpeg 未生成 rc_* 访问器，直接写字段
        // （普通整型，无所有权/无缓冲）——与 `crate::codec::set_thread_count` 同法。
        //
        // 非正值一律不写字段（`rc_max_rate`/`rc_buffer_size` 只接受正值），但不同
        // 来源给不同的诊断级别：负值只可能是调用方写错，用 `warn!`；显式 `0` 在文档
        // 里是"视为未设置"的合法写法（`max_bit_rate = 0` 即"不限瞬时码率"），只在
        // `debug!` 里留痕，免得把正常用法刷成警告。
        // SAFETY: `encoder` 在此处独占（`&mut`），`deref_mut` 只在块内存活。
        unsafe {
            let ctx_raw = encoder.deref_mut();
            match self.max_bit_rate {
                Some(rate) if rate.is_positive() => ctx_raw.rc_max_rate = rate,
                Some(rate) if rate < 0 => tracing::warn!(
                    "max_bit_rate {rate} is negative and was not applied; \
                     rc_max_rate stays unset (no instantaneous rate cap)"
                ),
                Some(rate) => {
                    tracing::debug!("max_bit_rate {rate} means \"no cap\"; rc_max_rate left unset")
                }
                None => {}
            }
            match self.buffer_size {
                Some(size) if size.is_positive() => ctx_raw.rc_buffer_size = size,
                Some(size) if size < 0 => tracing::warn!(
                    "buffer_size {size} is negative and was not applied; rc_buffer_size stays unset"
                ),
                Some(size) => {
                    tracing::debug!("buffer_size {size} was treated as unset; rc_buffer_size unset")
                }
                None => {}
            }
        }

        // 参数集进 extradata（容器格式）还是随每个关键帧 in-band（裸流），
        // 由 `with_global_header` 决定，见该方法。
        //
        // 起手值必须是 `encoder.flags` 而不是 0：上下文创建时已带有 FFmpeg 的
        // 默认位（实测 `AVCodecContext::new` 返回 `AV_CODEC_FLAG_CLOSED_GOP`，
        // 与 FFmpeg CLI 默认一致——CLI 只有显式 `-flags 0` 才编出 open GOP 的
        // 流）。从 0 重建会把这类默认位清掉，静默改变输出码流（closed GOP →
        // open GOP），因此这里一律在既有位上合并。调用方 `with_flags` 与
        // builder 自管的 `GLOBAL_HEADER` 各占不同位，同理按位合并。
        let mut flags = encoder.flags;
        if let Some(extra) = self.flags {
            flags |= extra.bits() as i32;
        }
        if self.global_header {
            flags |= AVCodecFlag::GLOBAL_HEADER.as_raw() as i32;
        }
        encoder.set_flags(flags);
        if let Some(flags2) = self.flags2 {
            crate::codec::set_flags2(encoder, flags2.bits() as i32);
        }
        if let Some(thread_type) = self.thread_type {
            crate::codec::set_thread_type(encoder, thread_type.bits() as i32);
        }
        // 未显式设置时取本机 CPU 数；`0`（自行推导）与负数跳过
        crate::codec::set_thread_count(
            encoder,
            self.thread_count.unwrap_or_else(|| num_cpus::get() as i32),
        );

        Ok(())
    }

    /// 解析编码目标像素格式（P0-2 自动格式协商）。
    ///
    /// * 显式指定（[`Self::with_pix_fmt`]）：软件路径立即校验编码器
    ///   是否支持，不支持时 `build()` 报错（fail fast）；硬件路径跳过校验
    ///   （`setup_encoder_frames` 会按 HW 要求重设 pix_fmt，HW 私有格式不在
    ///   软件支持列表内）。
    /// * 未指定：取编码器支持列表
    ///   （[`CodecConfig::supported_pixel_formats`](crate::CodecConfig::supported_pixel_formats)）
    ///   的**第一个**。FFmpeg 的 `AVCodec::pix_fmts` 就是按编码器偏好顺序书写的，
    ///   首个即编码器自己最想要的格式——常见编码器（libx264、libvpx、libx265…）
    ///   首个正好是 [`PixelFormat::YUV420P`]，而 mjpeg 是
    ///   [`PixelFormat::YUVJ420P`]、png 是 [`PixelFormat::RGB24`]。比在这里硬编码
    ///   一个"通用首选"更贴近编码器意图。
    ///   列表为 `None`（FFmpeg 未声明限制）或查询失败时回退
    ///   [`PixelFormat::YUV420P`]。
    fn resolve_pixel_format(&self, config: &CodecConfig, codec_name: &str) -> Result<PixelFormat> {
        match self.pixel_format {
            Some(fmt) => {
                if self.hw_device_config.is_none() && !config.supports_pixel_format(fmt as i32) {
                    return Err(RsmediaError::InvalidConfig(format!(
                        "encoder '{codec_name}' does not support pixel format {fmt:?}"
                    )));
                }
                Ok(fmt)
            }
            None => {
                let negotiated = match config.supported_pixel_formats() {
                    Ok(Some(list)) if !list.is_empty() => PixelFormat::from(list[0]),
                    _ => PixelFormat::YUV420P,
                };
                tracing::debug!(
                    "negotiated pixel format {negotiated:?} for encoder '{codec_name}'"
                );
                Ok(negotiated)
            }
        }
    }

    /// 解析编码目标采样格式（P0-2 自动格式协商），策略同
    /// [`Self::resolve_pixel_format`]：显式指定则校验（fail fast），
    /// 未指定优先 [`SampleFormat::FLTP`]（aac/mp3 等原生平面浮点格式），
    /// 不支持时取支持列表首个（如 `pcm_s16le` 为 S16）。
    fn resolve_sample_format(
        &self,
        config: &CodecConfig,
        codec_name: &str,
    ) -> Result<SampleFormat> {
        match self.sample_format {
            Some(fmt) => {
                if !config.supports_sample_format(fmt as i32) {
                    return Err(RsmediaError::InvalidConfig(format!(
                        "encoder '{codec_name}' does not support sample format {fmt:?}"
                    )));
                }
                Ok(fmt)
            }
            None => {
                let negotiated = match config.supported_sample_formats() {
                    Ok(Some(list)) if !list.is_empty() => {
                        if list.contains(&(SampleFormat::FLTP as i32)) {
                            SampleFormat::FLTP
                        } else {
                            SampleFormat::from(list[0])
                        }
                    }
                    _ => SampleFormat::FLTP,
                };
                tracing::debug!(
                    "negotiated sample format {negotiated:?} for encoder '{codec_name}'"
                );
                Ok(negotiated)
            }
        }
    }

    /// Build an [`Encoder`].
    ///
    /// Create an encoder from a [`StreamWriter`](crate::io::StreamWriter).
    ///
    /// # Arguments
    ///
    /// * `writer` - [`StreamWriter`](crate::io::StreamWriter) to create encoder from.
    /// * `interleaved` - Whether to use interleaved write.
    /// * `settings` - Encoder settings to use.
    pub fn build(self) -> Result<Encoder> {
        let media_type = self.media_type;
        // 帧率的合法性分两条判据，对应两个入口：浮点入口只可能"非正/非有限"，
        // 精确入口只可能"分子非正"（分母非零由 [`Rational`] 的类型保证）。
        if let Some(fps) = self.fps
            && !(fps > 0.0 && fps.is_finite())
        {
            return Err(RsmediaError::invalid_config(format!(
                "fps must be a positive, finite number, got {fps}"
            )));
        }
        if self.frame_rate.num() <= 0 {
            return Err(RsmediaError::invalid_config(format!(
                "a frame rate must have a positive numerator, got {}",
                self.frame_rate
            )));
        }

        if media_type == MediaType::VIDEO {
            // `width`/`height` 是 `i32`，会原样写进 `AVCodecContext`（`AVFrame` 的宽高
            // 也是 `int`）；0 或负数都不是合法画面尺寸（`buffer` 源要求正数）
            for (name, value) in [("width", self.width), ("height", self.height)] {
                if value <= 0 {
                    return Err(RsmediaError::invalid_config(format!(
                        "{name} must be positive, got {value}"
                    )));
                }
            }
        }

        let codec_name: String = match &self.codec_name {
            Some(codec_name) => codec_name.clone(),
            None => match media_type {
                MediaType::VIDEO => Self::VIDEO_CODEC_NAME.to_string(),
                MediaType::AUDIO => Self::AUDIO_CODEC_NAME.to_string(),
                MediaType::SUBTITLE => Self::SUBTITLE_CODEC_NAME.to_string(),
                _ => {
                    return Err(RsmediaError::unsupported(format!(
                        "media type {media_type:?}",
                    )));
                }
            },
        };

        let codec = AVCodec::find_encoder_by_name(&strutils::str_to_cstring(&codec_name)?)
            .ok_or_else(|| {
                RsmediaError::unsupported(format!(
                    "encoder '{codec_name}' is not available in this FFmpeg build"
                ))
            })?;

        // CRF 速率控制：仅对支持 crf 私有选项的视频编码器生效，其余编码器
        // 回退到 bit_rate 控制（与 ffmpeg CLI 行为一致，只是多一个警告）。
        let use_crf = media_type == MediaType::VIDEO
            && match self.quality {
                Some(Quality::Crf(_)) => {
                    let capable = CRF_CAPABLE_CODECS.contains(&codec_name.as_str());
                    if !capable {
                        tracing::warn!(
                            "codec '{codec_name}' has no CRF support, falling back to bit rate control"
                        );
                    }
                    capable
                }
                _ => false,
            };

        let mut encode_ctx = AVCodecContext::new(&codec);
        let config = CodecConfig::from_codec(codec);
        // P0-2 自动格式协商：显式指定的格式立即校验（fail fast），未指定的
        // 从编码器支持列表中挑选，避免把帧转进一个编码器不支持的格式后才
        // 在写入阶段报错。字幕编码器无像素/采样格式概念，跳过协商。
        let (pixel_format, sample_format) = if media_type == MediaType::SUBTITLE {
            (PixelFormat::YUV420P, SampleFormat::FLTP)
        } else {
            (
                self.resolve_pixel_format(&config, &codec_name)?,
                self.resolve_sample_format(&config, &codec_name)?,
            )
        };

        self.setup_codec_context(
            &mut encode_ctx,
            use_crf,
            pixel_format,
            sample_format,
            &config,
        )?;

        // 编码器输入时间基：与滤镜图 buffer 源（下方 FilterParams）和"滤镜未改写
        // 帧率时的编码器 time_base"同源。必须在 self 被部分 move 之前求值。
        let input_time_base = self.effective_time_base()?;

        // 在 hw_device_config / codec_opts 被 move 之前构造 filter graph：
        // 此位置 self 尚未被部分 move，可直接借用 self 计算 time_base。
        // 滤镜链可声明要求的输入格式（如 GIF 调色板链要求 RGB 输入、输出
        // pal8；音频链可声明输入采样格式）；未声明时输入格式=编码器协商格式
        // （src/sink 同格式，零行为变化）。
        let filter_input_format = self
            .filters
            .as_ref()
            .and_then(|filters| filters.iter().find_map(|f| f.input_format()));
        let mut filter_graph = if let Some(filters) = self.filters.as_ref() {
            let filter_params = match media_type {
                MediaType::VIDEO => {
                    FilterParams::Video(VideoParams {
                        width: self.width,
                        height: self.height,
                        src_format: filter_input_format
                            .and_then(FrameFormat::into_pixel)
                            .unwrap_or(pixel_format),
                        format: pixel_format,
                        time_base: input_time_base,
                        frame_rate: self.frame_rate,
                        // sample aspect ratio (0/1 if unknown)
                        pixel_aspect: encode_ctx.sample_aspect_ratio.into(),
                    })
                }
                MediaType::AUDIO => {
                    FilterParams::Audio(AudioParams {
                        nb_channels: self.nb_channels,
                        sample_rate: self.sample_rate,
                        format: sample_format,
                        src_format: filter_input_format
                            .and_then(FrameFormat::into_sample)
                            .unwrap_or(sample_format),
                        time_base: input_time_base, // time_base = 1 / sample_rate
                    })
                }
                _ => {
                    return Err(RsmediaError::invalid_config(format!(
                        "a {media_type:?} filter cannot be used on this stream"
                    )));
                }
            };
            // 滤镜链的媒体类型与可用性校验都在 `init` 内（缺失滤镜 →
            // `InvalidConfig`），这里不再重复一遍。
            let graph = FilterGraph::build(&filter_params, filters.as_slice())?;
            Some(graph)
        } else {
            None
        };

        // 滤镜可能改变输出帧率/时间基（如 `framerate`、`fps`、`setpts`）以及输出尺寸
        // （如 `scale`、`crop`、`pad`、`rotate`、`transpose`）。此时编码器必须采用滤镜
        // 输出帧率/时间基/尺寸，否则按输入参数推导的 time_base 会与滤镜输出 pts 不匹配
        // （B 帧重排 dts 乱序、mux 报错），或 codec context 尺寸与滤镜输出帧尺寸不符
        // 导致 send_frame 报错。
        // 注：滤镜输出时间基无需在此缓存——`send_frame_post_filter` 会在发送前按需
        // 从滤镜图实时查询，用于把 pts 换算到编码器时间基。
        let (filter_frame_rate, filter_size) = match filter_graph.as_mut() {
            Some(graph) => (Some(graph.output_frame_rate()?), Some(graph.output_size()?)),
            None => (None, None),
        };
        if media_type == MediaType::VIDEO {
            if let Some(out_fr) = filter_frame_rate {
                let changed = out_fr != self.frame_rate;
                if out_fr.num() > 0 && changed {
                    // 分子为正 ⇒ 倒数必然存在（`Rational` 的分母恒不为零）。
                    let time_base = out_fr.inverse()?;
                    tracing::info!(
                        "Filter changes frame rate: {} -> {}",
                        self.frame_rate,
                        out_fr
                    );
                    encode_ctx.set_framerate(out_fr.into());
                    encode_ctx.set_time_base(time_base.into());
                }
            }
            if let Some((fw, fh)) = filter_size
                && fw > 0
                && fh > 0
                && (fw != encode_ctx.width || fh != encode_ctx.height)
            {
                tracing::info!(
                    "Filter changes size: {}x{} -> {}x{}",
                    encode_ctx.width,
                    encode_ctx.height,
                    fw,
                    fh
                );
                encode_ctx.set_width(fw);
                encode_ctx.set_height(fh);
            }
        }

        let hw_context = self
            .hw_device_config
            .filter(|_cfg| {
                // hardware acceleration enabled for video
                media_type == MediaType::VIDEO
            })
            .map(|cfg| {
                // codec support or not for hardware acceleration
                tracing::info!(
                    "Video Encoder with HW acceleration codec: {:?}, config: {:#?}",
                    self.codec_name,
                    cfg
                );

                // create hardware context
                let (width, height) = (encode_ctx.width, encode_ctx.height);
                HWContext::new(cfg)
                    .and_then(|ctx| {
                        // *注意*: setup_encoder_frames 会根据 HW 能力修改 encode_ctx.pix_fmt
                        ctx.setup_encoder_frames(
                            &mut encode_ctx,
                            width,
                            height,
                            self.hw_pool_size
                                .unwrap_or(crate::hwaccel::DEFAULT_HW_POOL_SIZE),
                        )?;
                        Ok(ctx)
                    })
                    .context("Hardware acceleration context initialization failed")
            })
            .transpose()?;

        // 打开编码器前的私有选项：quality/profile/level 写成 AVOption；用户
        // codec_opts 只用于补充 builder 未建模的键 —— 同一个键两边都给时以用户透传为准
        // （typed setter 写的是 `AVCodecContext` 字段，`avcodec_open2` 在字段写入之后
        // 才应用这个字典，见 `EncoderBuilder::with_options` 文档）。
        let mut opts = Options::new();
        if use_crf && let Some(Quality::Crf(crf)) = self.quality {
            opts.set("crf", crf.to_string());
        }
        if media_type == MediaType::VIDEO {
            if let Some(profile) = self.profile {
                opts.set("profile", profile.as_option_str());
            }
            if let Some(level) = &self.level {
                opts.set("level", level);
            }
        }
        if let Some(user_opts) = self.codec_opts {
            // 这里能直接看出重叠的只有刚写进 `opts` 的 `crf`/`profile`/`level`；其余
            // typed setter 直接写上下文字段，是否被字典覆盖只有 `avcodec_open2` 内部
            // 知道，无法在此检测，故在 `with_options` 文档里声明规则。
            for (key, _) in user_opts.iter() {
                if opts.contains_key(key) {
                    tracing::warn!(
                        "codec option '{key}' from with_options overrides the builder setting \
                         for the same AVOption"
                    );
                }
            }
            opts.merge(user_opts);
        }

        // 字幕编码器（mov_text/subrip 等）init 时会执行
        // `ff_ass_split(avctx->subtitle_header)`，未设置时返回 NULL →
        // AVERROR_INVALIDDATA。header 必须由调用方提供：转码时从解码器侧
        // 传递（解码器 open 时填充），authoring 场景用
        // [`EncoderBuilder::with_subtitle_header`] 显式给出。
        if media_type == MediaType::SUBTITLE {
            let Some(header) = &self.subtitle_header else {
                return Err(RsmediaError::invalid_config(
                    "subtitle encoder requires an ASS script header: provide it via \
                     EncoderBuilder::with_subtitle_header, or forward it from the decoded \
                     subtitle stream in a transcode pipeline",
                ));
            };
            let header_c =
                std::ffi::CString::new(header.as_str()).context("Invalid subtitle header")?;
            encode_ctx
                .set_subtitle_header(header_c.as_c_str())
                .context("Failed to set subtitle header")?;
        }

        encode_ctx
            .open(opts.into_dict())
            .context("Failed to open encode context")?;

        Ok(Encoder {
            config,
            hw_context,
            media_type,
            filter_input_format,
            filter_graph,
            context: encode_ctx,
            state: ProcessState::Normal,
            scaler: Scaler::new_with_options(self.scale_algorithm, self.scale_quality)
                .with_buffer_pool(self.scale_pool),
            pending_packets: VecDeque::new(),
            audio_fifo: None,
            next_pts: 0,
            input_time_base,
            filter_resampler: None,
            encode_resampler: None,
        })
    }
}

impl Default for EncoderBuilder {
    fn default() -> Self {
        Self {
            // video
            width: 0,
            height: 0,
            pixel_format: None,
            bit_rate: None,
            max_bit_rate: None,
            buffer_size: None,
            frame_rate: Rational::integer(Self::FRAME_RATE),
            fps: None,
            gop_size: None,
            max_b_frames: None,
            global_header: true,
            // audio
            nb_channels: 2,
            sample_rate: 44100,
            sample_format: None,
            // common
            media_type: MediaType::VIDEO,
            thread_count: None,
            flags: None,
            flags2: None,
            thread_type: None,
            codec_name: None,
            codec_opts: None,
            quality: None,
            profile: None,
            level: None,
            filters: None,
            subtitle_header: None,
            hw_device_config: None,
            hw_pool_size: None,
            scale_algorithm: ScaleAlgorithm::default(),
            scale_quality: ScaleQuality::default_mask(),
            scale_pool: false,
        }
    }
}

/// Encodes frames into a video stream.
///
/// # Example
///
/// ```ignore
/// let decoder = Decoder::new("video_out.mkv").unwrap();
/// decoder
///     .decode_iter()
///     .take_while(Result::is_ok)
///     .map(|frame| encoder
///         .encode(frame.unwrap())
///         .expect("Failed to encode frame."),
///     );
/// ```
pub struct Encoder {
    config: CodecConfig,
    context: AVCodecContext,
    filter_graph: Option<FilterGraph>,
    /// 滤镜图输入格式声明（来自 [`Filter::input_format`]）。仅当滤镜图
    /// 存在时可能为 `Some`；输入帧进图前需转换到该格式（视频=像素格式，
    /// 音频=采样格式；默认=编码器协商格式）。
    filter_input_format: Option<FrameFormat>,
    hw_context: Option<Arc<HWContext>>,
    media_type: MediaType,
    /// 编解码上下文的阶段，与解码器共用一套 [`ProcessState`]：
    /// `Normal`（在读帧）→ `Drained`（EOS 已送出、仍在出包）→ `Flushed`（EOF）。
    ///
    /// **只有真正送出 EOS 才允许推进到 `Drained`。** `receive_packet` 的 EAGAIN
    /// 只表示"此刻暂无包可出"，read 阶段的编码器（B 帧、lookahead 缓冲）同样会
    /// 返回它；把 EAGAIN 记成 `Drained` 会让 [`is_drained`](Self::is_drained) 在流
    /// 中段就永久为真，于是 `flush` 的排空循环在没有 EOS 的情况下空转。
    state: ProcessState,
    /// 视频像素缩放器。与音频的 [`Self`]`::filter_resampler`/`encode_resampler` 不同，
    /// 它**不是** `Option`：`Scaler::new` 不需要输入规格（swscale 从帧属性推导源格式），
    /// 而 `Resampler::new` 要输入输出两侧都给全、输入侧只有第一帧才知道 —— 后者因此惰性
    /// 建在 `resample_if_needed` 里。视频侧的目标规格随用途变（进图前只换格式、编码前
    /// 换格式），故每次调用传 `dst_spec` 而不存在 scaler 里。
    scaler: Scaler,
    /// 编码器缓冲满（send_frame 返回 EAGAIN）时，先行排空的已就绪包暂存于此， 由 `receive_packet` 优先取出，
    /// 避免丢包。按 FIFO 出队（`pop_front`）， 保证与编码器输出顺序一致（否则 dts 会乱序、mux 报错）。
    pending_packets: VecDeque<AVPacket>,
    /// 音频样本缓冲：固定帧长编码器（如 aac，frame_size=1024）要求每次 `send_frame`
    /// 恰好给出 `frame_size` 个样本，而待编码音频帧大小可能可变（滤镜输出、或用户
    /// 输入不足一帧），需先累积补齐到帧长再送编码器。
    audio_fifo: Option<AVAudioFifo>,
    /// 下一个自动分配的 pts（`input_time_base` 下的 tick 计数）。
    ///
    /// 用户未设置 pts（`AV_NOPTS_VALUE`）的帧由此计数器自动编号：视频每帧 +1
    /// （输入时间基 = `1/fps`，每帧恰一 tick），音频按样本数递增（输入时间基 =
    /// `1/sample_rate`，样本位置即时间轴）。用户设置了 pts 的帧照常使用其值，
    /// 但计数器仍跳到其后，保证后续未设置的帧能接续正确的时间轴。
    ///
    /// 固定帧长音频（aac 等）由 `audio_fifo` 重新切帧：输出帧的 pts 无法取自
    /// 任何单个输入帧，只能用"已输出累计样本数"。因此首次缓冲时以首帧 pts
    /// （若设置）播种本计数器，此后每切出一帧按 `frame_size` 递增、flush 末帧
    /// 按 `remaining` 递增——与视频/直发路径共用同一个计数器。
    next_pts: i64,
    /// 编码器**输入**时间基：滤镜存在时为滤镜图 buffer 源的时间基（= 建图时的
    /// `effective_time_base()`），否则等于编码器 time_base。滤镜改写输出帧率时
    /// 编码器 time_base 会被改为 `1/滤镜输出fps`，与输入时间基不再相等，因此
    /// 必须显式保存，供 pts 换算与自动编号使用。
    input_time_base: Rational,
    /// 送进**滤镜图**前的音频采样格式转换：把帧对齐到图的输入格式（`filter_input_format`
    /// 声明的格式，未声明时是编码器采样格式）。只换采样格式，布局/采样率跟着帧走。
    ///
    /// 与 [`Self::encode_resampler`] 是**两个**而非一个，不是冗余而是正确性：一个
    /// [`Resampler`] 只记**一种**输出格式，而这两处要的目标不同（进图前 = 图输入格式，
    /// 滤镜后 = 编码器完整规格）。共用一个会让它的输出格式在"图输入格式"与"编码器格式"
    /// 之间每帧翻一次、从而每帧重建两次 swr 上下文。视频侧没有这个问题是因为 [`Scaler`]
    /// 不记输出格式（每次调用传 `dst_spec`、变了就重建），音频侧则把它记在上下文里。
    ///
    /// `None` = 尚未遇到需要转换的帧：上下文的输入格式要到第一帧才知道（见
    /// [`resample::resample_if_needed`]）。没有滤镜图时本字段从不使用。
    ///
    /// **不需要排空**：它的输出布局与采样率都取自帧本身，只换采样格式，上下文因此
    /// 不做重采样、也就没有延迟线（见
    /// `resample::tests::test_a_format_only_resampler_keeps_no_delay_line`）。
    /// 需要排空的是 [`Self::encode_resampler`] —— 它的输出采样率是编码器的。
    filter_resampler: Option<resample::Resampler>,
    /// 送进**编码器**前的音频重采样：把帧对齐到编码器的完整规格（`self.audio_spec()` 的
    /// 布局 + 采样格式 + 采样率）。滤镜输出（或无滤镜时的原始帧）在这里做最终对齐。
    ///
    /// `None` = 尚未遇到需要转换的帧；见 [`Self::filter_resampler`]。
    ///
    /// 输出采样率取自**编码器**（与输入可能不同），因此这个上下文真的会重采样、真的
    /// 有延迟线 —— EOF 时由 [`Self::drain_resampler_tail`] 排空。
    encode_resampler: Option<resample::Resampler>,
}

impl Encoder {
    /// Create a video encoder with the specified destination
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the video stream.
    /// * `height` - The height of the video stream.
    ///
    /// note: default video codec is `libx264`
    #[inline]
    pub fn new_video(width: i32, height: i32) -> Result<Encoder> {
        EncoderBuilder::new_video(width, height).build()
    }

    /// Create a audio encoder with the specified parameters.
    ///
    /// * `bit_rate` - Bit rate in bits per second. default is 128k.
    /// * `nb_channels` - Number of channels.
    /// * `sample_rate` - Sample rate in Hz.
    /// * `sample_format` - Sample format.
    ///
    /// note: default audio codec is `aac`
    #[inline]
    pub fn new_audio(
        nb_channels: i32,
        sample_rate: i32,
        sample_format: SampleFormat,
    ) -> Result<Encoder> {
        EncoderBuilder::new_audio(128_000, nb_channels, sample_rate, sample_format).build()
    }

    /// Returns `true` if end-of-stream has been sent and the encoder is still
    /// producing its remaining packets — i.e. the draining phase, before
    /// [`is_flushed`](Self::is_flushed).
    ///
    /// This is **not** "the last receive returned EAGAIN": that also happens
    /// mid-stream, when the encoder simply wants more input, and must not move
    /// the phase. The phase is read straight off the encoder's state — the same
    /// single flag [`Decoder::is_drained`](crate::Decoder::is_drained) uses.
    pub fn is_drained(&self) -> bool {
        self.state.is_drained()
    }

    /// Returns `true` if the encoder is fully flushed and finished.
    pub fn is_flushed(&self) -> bool {
        self.state.is_flushed()
    }

    /// Encode a high-level frame (a single frame ndarray-based)
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to encode in `HWC` format and standard layout.
    pub fn encode<T>(&mut self, frame: MediaFrame<T>) -> Result<Vec<AVPacket>>
    where
        T: ElementType,
    {
        let raw_frame = frame.to_avframe()?;
        self.encode_raw(raw_frame)
    }

    /// Encode a single raw frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to encode.
    ///
    /// # Returns
    ///
    /// 所有已就绪的编码包。一次输入帧可能（在编码器缓冲满、滤镜升帧率等场景下）
    /// 产出 0 或多包，因此返回集合而非单个包。
    pub fn encode_raw(&mut self, frame: AVFrame) -> Result<Vec<AVPacket>> {
        if !self.state.is_normal() {
            return Err(RsmediaError::invalid_config(format!(
                "Encoder cannot encode after being flushed (state {:?}); \
                 an FFmpeg encoder cannot be un-flushed, build a new one",
                self.state
            )));
        }

        // send frame
        self.send_frame_to_encoder(Some(frame))?;

        // receive packet: 排空所有已就绪的包（含 EAGAIN 时暂存的 pending_packets）
        let mut packets = Vec::new();
        while let Some(pkt) = self.receive_packet()? {
            packets.push(pkt);
        }
        Ok(packets)
    }

    /// 编码一条字幕段落（仅字幕编码器）。
    ///
    /// 字幕编码走 rsmpeg 的 [`AVCodecContext::encode_subtitle`]（同步 API，无
    /// send_frame/receive_packet 队列），与音视频的 [`Self::encode_raw`] 不同：
    /// 每条段落恰好产出 0 或 1 包，无需 flush。pts/duration 已按编码器
    /// time_base（1/1000 毫秒精度）设置。
    pub fn encode_subtitle_segment(&mut self, segment: &SubtitleSegment) -> Result<Vec<AVPacket>> {
        if self.media_type != MediaType::SUBTITLE {
            return Err(RsmediaError::unsupported(format!(
                "encode_subtitle_segment requires a subtitle encoder, got media type: {:?}",
                self.media_type
            )));
        }
        if segment.text.is_empty() {
            return Ok(Vec::new());
        }

        // Build a single ASS text rect. `AVCodecContext::encode_subtitle`
        // hands `rect.ass` to the codec's `ff_ass_split_dialog`, whose
        // hardcoded field list is `ReadOrder, Layer, Style, Name, MarginL,
        // MarginR, MarginV, Effect, Text` — **no start/end timestamps**: the
        // timings are carried by the packet pts/duration set below. A full
        // `Dialogue:` line would shift every field by two (the timestamps get
        // eaten by Layer/Style) and surface as a stray comma prepended to the
        // decoded text plus a spurious zero-style record.
        let mut subtitle = AVSubtitle::new();
        let dialogue = format!("0,0,Default,,0,0,0,,{}", segment.text);
        let dialogue_c =
            std::ffi::CString::new(dialogue).context("Subtitle text contains NUL byte")?;
        subtitle
            .push_ass_rect(dialogue_c.as_c_str())
            .context("Failed to build subtitle rect")?;

        // 输出大小与文本长度成正比（mov_text: 2 字节长度前缀 + 文本；subrip: 纯文本），
        // 按需分配而不是固定 8KB —— 否则超长段落会被截断或直接编码失败。
        let capacity = segment
            .text
            .len()
            .saturating_mul(2)
            .saturating_add(EncoderBuilder::MIN_SUBTITLE_BUFFER_SIZE);
        let mut buf = vec![0u8; capacity];
        let len = self
            .context
            .encode_subtitle(&subtitle, &mut buf)
            .context("Subtitle encoding failed")?;
        if len == 0 {
            return Ok(Vec::new());
        }

        // 手工构造 packet：rsmpeg 没有「从字节构造 AVPacket」的接口。
        // `av_new_packet` 分配一块自有缓冲（含 `AV_INPUT_BUFFER_PADDING_SIZE`
        // 的尾部填充），因此下面的拷贝严格落在已分配范围内。
        let len_i32 = i32::try_from(len)
            .map_err(|_| RsmediaError::invalid_config("encoded subtitle packet too large"))?;
        let mut packet = AVPacket::new();
        // SAFETY: `packet` 由 `AVPacket::new` 创建、析构前一直有效；
        // `av_new_packet` 成功（ret >= 0）后 `packet->data` 指向至少
        // `len_i32` 字节的可写缓冲，且 `buf.len() >= len`（len 由 FFmpeg 写入 buf）。
        let ret = unsafe { ffi::av_new_packet(packet.as_mut_ptr(), len_i32) };
        if ret < 0 {
            return Err(RsmediaError::FFmpeg(rsmpeg::error::RsmpegError::from(ret)));
        }
        // SAFETY: 见上；两个缓冲不重叠（一个来自 Vec，一个由 FFmpeg 分配）。
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), packet.deref_mut().data, len);
        }

        // 编码器 time_base 为 1/1000，pts/duration 直接使用毫秒值；
        // dts = pts（字幕无 B 帧重排）。
        let pts = segment.start_ms;
        packet.set_pts(pts);
        packet.set_dts(pts);
        packet.set_duration(segment.duration_ms().max(1));
        packet.set_pos(-1);

        Ok(vec![packet])
    }

    fn send_frame_to_encoder(&mut self, frame_opt: Option<AVFrame>) -> Result<()> {
        if let Some(mut frame) = frame_opt {
            // sample_rate 补齐与 pts 处理都必须在这里完成、早于滤镜与格式转换：滤镜
            // （`fps`/`framerate` 等）需要有效 pts 才能正确工作，时间基换算要以
            // 滤镜图输入时间基为基准，而帧采样率会被滤镜输入转换、`rescale`、
            // `check_frame` 三处读取（见 `assign_pts_sample_rate`）。
            self.assign_pts_sample_rate(&mut frame);
            // 硬件帧只能直接进编码器：软件滤镜图里没有 `hwdownload`，硬塞进去
            // 会以 FFmpeg 的格式错误收场。要过滤镜就先自行下载到内存，或去掉滤镜。
            if !frame.hw_frames_ctx.is_null() && self.filter_graph.is_some() {
                return Err(RsmediaError::invalid_config(format!(
                    "input frame is a hardware frame ({:?}) but this encoder has a software filter graph; \
                     download the frame to system memory before encoding, or drop the filters",
                    PixelFormat::from(frame.format)
                )));
            }
            // 正常编码帧：经过 filter（如有）
            // 滤镜 buffer/abuffer 源按"滤镜图输入格式"配置（声明优先，见
            // `Filter::with_input_format`；默认=编码器协商格式）。输入帧格式
            // 与图输入格式不一致时需先转换再进图：
            // - 视频：像素格式转换（swscale）。直接送入不匹配的 buffer 源会
            //   触发 FFmpeg 自动格式转换路径的越界读写（SIGSEGV）。此转换与
            //   无滤镜时 `send_frame_post_filter` 里的 `rescale` 行为一致。
            // - 音频：采样格式转换（swresample，速率/声道不变）。abuffer 拒绝
            //   属性不符的帧（"Changing frame properties on the fly"），图内
            //   格式变化由滤镜链（如 aformat）完成，滤镜后 `rescale` 兜底。
            let graph_input_format = self.filter_input_format.unwrap_or_else(|| {
                // 无声明时图输入=编码器协商格式
                match self.media_type {
                    MediaType::VIDEO => FrameFormat::Pixel(self.input_sw_pix_fmt()),
                    _ => FrameFormat::Sample(self.sample_fmt()),
                }
            });
            // 先把帧转换到滤镜图输入格式（此转换需要 `&mut self` 以复用可变的
            // `scaler`/重采样上下文），转换完成后再借用 `filter_graph` 处理。
            //
            // 硬件帧不进软件格式转换：swscale 处理不了 hw 帧，而 hw 帧也不能喂给
            // 软件滤镜图（图里没有 `hwdownload`），故上面已拒绝"hw 帧 + 滤镜图"的
            // 组合，这里只把它原样交给 `send_frame_post_filter` 做 frames context 映射。
            let converted = if !frame.hw_frames_ctx.is_null() {
                frame
            } else {
                match graph_input_format {
                    // 只换像素格式：目标尺寸取自帧本身。已经匹配时 `scale_if_needed`
                    // 原样返回（与下面的音频分支同一形状）。
                    FrameFormat::Pixel(dst) => {
                        let dst_spec = VideoSpec::from_frame(&frame)?.with_pix_fmt(dst);
                        self.scaler.scale_if_needed(frame, dst_spec)?
                    }
                    FrameFormat::Sample(dst) => {
                        // 只换采样格式：目标布局/采样率取自帧本身。已经匹配时
                        // `resample_if_needed` 原样返回。
                        let out_spec =
                            resample::AudioSpec::from_frame(&frame).with_sample_fmt(dst as _);
                        resample::resample_if_needed(&mut self.filter_resampler, frame, out_spec)?
                    }
                }
            };
            if let Some(graph) = self.filter_graph.as_mut() {
                match graph.process_frame(Some(converted))? {
                    Some(filtered) => self.send_frame_post_filter(filtered)?,
                    None => {
                        // filter 暂未输出（内部缓冲中），等待后续帧驱动
                        tracing::debug!("Filter graph drained, waiting for more input.");
                    }
                }
            } else {
                self.send_frame_post_filter(converted)?;
            }
            Ok(())
        } else {
            // EOF：向编码器发送 EOS。filter 的缓冲帧已由 `flush()` 单独冲刷送走，
            // 这里不应再调用 `process_frame(None)`，否则对已 flushed 的 graph 会报错。
            // 编码器缓冲可能仍满（EAGAIN），由 `send_frame_with_retry` 先排空再重试。
            self.send_frame_with_retry(None)
        }
    }

    /// 帧进滤镜/编码器前的归一化：补齐音频采样率、换算时间基、自动编号 pts。
    ///
    /// **采样率**：音频帧的 0 表示"调用方没有声明"，而不是"0 Hz"——音频编码器的目标
    /// 采样率在 [`EncoderBuilder::new_audio`] 时就必须给定并写入 `AVCodecContext`
    /// （见 [`effective_time_base`](Self::effective_time_base)），是唯一权威值，故以它
    /// 补齐。**只有非正值（`<= 0`，含负数的脏值）会被替换**：正数一律视为调用方
    /// 声明的真实源率，与编码器不同时
    /// 照常重采样（这是"任意采样率输入"功能的依据）。这一步必须在下游三处消费者之前
    /// 完成——滤镜输入格式转换（`resample::convert_frame` 把帧率当**源率**）、
    /// [`rescale`](Self::rescale) 的重采样判断、[`check_frame`](Self::check_frame) 的
    /// 采样率校验；它们都假定该值有效，未声明时会一路走到 `check_resampler_input`
    /// 而以 `Invalid input frame.` 失败。
    ///
    /// **pts**：FFmpeg 在编码器输入侧**忽略** `AVFrame.time_base`，pts 一律按
    /// `AVCodecContext.time_base` 解释。因此两件事必须在这里完成：
    ///
    /// 1. **时间基换算**：帧携带了有效且不同的 `time_base` 时（解码侧容器流时间基，
    ///    如 mp4 的 `1/15360`），把 pts 换算到编码器输入时间基，否则时间轴被静默
    ///    误读（时长/帧率全错）。
    /// 2. **自动编号**：pts 为 `AV_NOPTS_VALUE`（用户未设置）时，用运行计数器
    ///    [`next_pts`](Self::next_pts) 编号——视频每帧 +1 tick（输入时间基 =
    ///    `1/fps`），音频按样本位置递增。固定帧长音频（aac 等）除外：其帧切分由
    ///    `audio_fifo` 完成，本方法不为输入帧编号（NOPTS 原样通过），而由
    ///    `buffer_audio_frame` 首次缓冲时用首帧 pts 播种 `next_pts`，此后每个
    ///    输出帧按已输出样本数递增。
    ///
    /// 计数器在用户设置了 pts 的帧上同样前进（跳到该 pts 之后），使后续未设置
    /// pts 的帧能接续正确的时间轴。
    ///
    /// 离开本方法时：音频帧的采样率必定有效，任何帧的时间基必定是编码器输入时间基。
    fn assign_pts_sample_rate(&mut self, frame: &mut AVFrame) {
        let is_audio = self.media_type == MediaType::AUDIO;
        let input_tb = self.input_time_base;

        // 采样率：未声明（0）时以编码器的目标率补齐。
        if is_audio && frame.sample_rate <= 0 {
            frame.set_sample_rate(self.sample_rate());
        }

        // 时间基：帧自带有效且与编码器**不同**的时间基时（解码侧容器时间基，如 mp4 的
        // 1/15360），pts 需先换算过来；无论换算与否，离开时帧都带编码器输入时间基。
        let frame_tb = Rational::from(frame.time_base);
        let needs_rescale =
            frame.pts != ffi::AV_NOPTS_VALUE && frame_tb.num() > 0 && frame_tb != input_tb;
        if needs_rescale {
            frame.set_pts(frame.pts.rescale(frame_tb, input_tb));
        }
        frame.set_time_base(input_tb.into());

        // 固定帧长音频（aac 等）的输出 pts 由 `audio_fifo` 按已输出样本数维护
        // （见 buffer_audio_frame / drain_audio_fifo），故不为输入帧编号、计数器也不前进。
        if is_audio && self.frame_size() > 0 {
            return;
        }
        if frame.pts == ffi::AV_NOPTS_VALUE {
            frame.set_pts(self.next_pts);
        }
        // 输入时间基：视频 `1/fps`（每帧恰一 tick），音频 `1/sample_rate`（样本位置即时间轴）。
        let step = if is_audio {
            frame.nb_samples.max(1) as i64
        } else {
            1
        };
        self.next_pts = frame.pts + step;
    }

    /// 编码器要求的目标音频格式（声道布局、采样格式、采样率）。
    ///
    /// 三者就是 `send_frame_post_filter` 前必须对齐的那套规格；打包成
    /// [`AudioSpec`](resample::AudioSpec) 之后，重采样与"已经对齐了吗"的判断都用它一处。
    fn audio_spec(&self) -> resample::AudioSpec {
        resample::AudioSpec::new(
            self.ch_layout().clone().into_inner(),
            self.sample_fmt().into(),
            self.sample_rate(),
        )
    }

    /// 将已通过 filter（或无 filter）的帧做 rescale/hw 上传后发送给编码器。
    ///
    /// 注意：`flush()` 阶段 filter 已进入 Flushed 状态，不能再把缓冲帧送回
    /// `process_frame`（会因 EAGAIN 被丢弃），因此缓冲帧必须直接走本方法。
    fn send_frame_post_filter(&mut self, frame: AVFrame) -> Result<()> {
        // 滤镜输出帧的 pts 位于滤镜输出时间基（如 `framerate` 输出 1/120），而编码器
        // 时间基已按滤镜输出帧率对齐（如 1/30）。发送前需把 pts 换算到编码器时间基，
        // 否则 B 帧重排得到的 dts 会乱序、mux 报 AVERROR(-22)。
        // 时间基按需从滤镜图实时查询，避免缓存冗余状态。
        let mut frame = frame;
        if let Some(filter_tb) = self
            .filter_graph
            .as_mut()
            .map(|g| g.output_time_base())
            .transpose()?
        {
            let enc_tb = Rational::from(self.context.time_base);
            if frame.pts != ffi::AV_NOPTS_VALUE {
                frame.set_pts(frame.pts.rescale(filter_tb, enc_tb));
                frame.set_time_base(enc_tb.into());
            }
        }

        // 确保帧的格式匹配编码器要求
        let scaled_frame = self.rescale(frame)?;
        self.send_frame_ready(scaled_frame)
    }

    /// 把**已经对齐编码器规格、且 pts 已在编码器时间基上**的帧送进编码器。
    ///
    /// 与 [`Self::send_frame_post_filter`] 只差前置的两步（格式转换与滤镜输出时间基
    /// 换算）：重采样延迟排出的尾帧由 [`Self::drain_resampler_tail`] 交到这里，它们
    /// 已经是编码器规格、pts 也已在编码器时间基上 —— 再走一遍滤镜输出时间基的换算
    /// 会把同一个 pts 换算两次。
    fn send_frame_ready(&mut self, frame: AVFrame) -> Result<()> {
        // 转换硬件帧
        let hw_frame = match self.hw_context.as_ref() {
            Some(hw_ctx) if hw_ctx.is_sw_frame(&frame) => {
                // sw_frame -> hw_frame
                hw_ctx
                    .hw_upload(&mut self.context, &frame)
                    .context("Failed to upload frame to HW")?
            }
            // 已是硬件帧，但属于**别的** frames context（解码器/另一台设备/调用方
            // 自建）：映射进编码器自己那份，同设备时零拷贝。已经在编码器 frames
            // context 里的帧由 `map_hw_frame` 原样返回。
            Some(hw_ctx) if hw_ctx.is_hw_frame(&frame) => hw_ctx
                .map_hw_frame(&mut self.context, frame)
                .context("Failed to map the input hardware frame into the encoder")?,
            _ => frame, // 不需要上传或已经是 HW frame
        };

        // 固定帧长音频编码器（如 aac，frame_size=1024）要求每次 `send_frame` 恰好给出
        // frame_size 个样本，而待编码帧大小可能可变（滤镜输出、或用户输入不足一帧），
        // 需先进 `audio_fifo` 累积补齐后再送编码器；无固定帧长（frame_size=0）的
        // 编码器（如部分无损格式）直接发送。
        if self.media_type == MediaType::AUDIO && self.frame_size() > 0 {
            self.buffer_audio_frame(hw_frame)
        } else {
            self.check_frame(Some(&hw_frame))?;

            tracing::debug!(
                "Send frame to encoder: {:?}, time_base: {:?}, media_type: {:?}",
                hw_frame,
                self.time_base(),
                self.media_type()
            );

            self.send_frame_with_retry(Some(&hw_frame))
        }
    }

    /// 将一帧已 rescale 的音频帧写入 `audio_fifo`，凑满 `frame_size` 后送出。
    ///
    /// 固定帧长编码器必须在每次 `send_frame` 时恰好给出 `frame_size` 个样本，
    /// 因此先把待编码帧写入 `audio_fifo` 累积，凑满 `frame_size` 再送编码器；
    /// 不足 `frame_size` 的剩余样本，由 `flush` 阶段作为末帧截取。
    fn buffer_audio_frame(&mut self, frame: AVFrame) -> Result<()> {
        let frame_size = self.frame_size();
        if self.audio_fifo.is_none() {
            let channels = self.ch_layout().nb_channels;
            let sample_fmt = self.sample_fmt() as _;
            // 首次缓冲时以首帧 pts（编码器时间基下的样本位置，`assign_pts_sample_rate` 已
            // 完成换算/自动编号的对齐）播种样本计数器；未设置则保持 0 起步。
            if frame.pts != ffi::AV_NOPTS_VALUE {
                self.next_pts = frame.pts;
            }
            self.audio_fifo = Some(AVAudioFifo::new(sample_fmt, channels, frame_size));
        }
        unsafe {
            self.audio_fifo
                .as_mut()
                .unwrap()
                .write(frame.data.as_ptr(), frame.nb_samples)?;
        }
        self.drain_audio_fifo(frame_size)
    }

    /// 从 `audio_fifo` 中取出满帧长样本，拼成帧送编码器，直至剩余不足一帧。
    fn drain_audio_fifo(&mut self, frame_size: i32) -> Result<()> {
        loop {
            // 借用范围限定在这次判断内：`fifo_pop_frame` 需要 &mut self。
            let ready = match self.audio_fifo.as_ref() {
                Some(fifo) => fifo.size() >= frame_size,
                None => false,
            };
            if !ready {
                return Ok(());
            }
            let frame = self.fifo_pop_frame(frame_size)?;
            self.check_frame(Some(&frame))?;
            self.send_frame_with_retry(Some(&frame))?;
        }
    }

    /// 从 `audio_fifo` 取出 `count` 个样本组成一帧，并按已输出样本数编 pts。
    ///
    /// `drain_audio_fifo`（凑满一帧）与 `flush_audio_fifo`（冲刷不足一帧的尾巴）
    /// 只差一个样本数，帧的组装与编号完全一致，故共用此处。
    fn fifo_pop_frame(&mut self, count: i32) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(count);
        frame.set_ch_layout(self.ch_layout().clone().into_inner());
        frame.set_format(self.sample_fmt() as _);
        frame.set_sample_rate(self.sample_rate());
        frame.set_time_base(self.time_base().into());
        // SAFETY: `frame` 已分配缓冲，`fifo.read` 最多写入 `count` 个样本/声道。
        unsafe {
            frame
                .alloc_buffer()
                .context("Failed to allocate audio frame buffer")?;
            let fifo = self
                .audio_fifo
                .as_mut()
                .ok_or_else(|| RsmediaError::msg("Audio FIFO is not initialised"))?;
            fifo.read(frame.data.as_ptr(), count)?;
        }
        frame.set_pts(self.next_pts);
        self.next_pts += count as i64;
        Ok(frame)
    }

    /// 冲刷音频缓冲中不足一帧的剩余样本，作为末帧送编码器。
    fn flush_audio_fifo(&mut self) -> Result<()> {
        let remaining = self.audio_fifo.as_ref().map_or(0, |fifo| fifo.size());
        if remaining <= 0 {
            return Ok(());
        }
        let frame = self.fifo_pop_frame(remaining)?;
        self.check_frame(Some(&frame))?;
        self.send_frame_with_retry(Some(&frame))
    }

    /// 冲刷 [`Self::encode_resampler`] 延迟线上的尾样，并按正常帧的路径送进编码器。
    ///
    /// 采样率变化时 `swr` 会把滤波延迟里的几十毫秒样本留在上下文里，只有 EOF 之后才
    /// 吐得出来（见 [`Resampler::flush_frames`](resample::Resampler::flush_frames)）；
    /// 不排空就丢掉音频末尾的一小段 —— 而且丢得毫无征兆：帧数、pts、包数都对，
    /// 只是末尾少了一点声音。
    ///
    /// 排出的帧走 [`Self::send_frame_ready`]（而不是
    /// [`Self::send_frame_post_filter`]）：它们已经出过重采样，pts 也在编码器时间基
    /// 上，再走一遍滤镜输出的换算会换算两次。固定帧长音频随后仍由
    /// [`Self::flush_audio_fifo`] 把不足一帧的尾巴送出，因此这里必须**先于**它调用。
    ///
    /// 只有 [`Self::encode_resampler`] 需要这一步：[`Self::filter_resampler`] 的目标
    /// 布局与采样率都取自帧本身（只换采样格式），上下文不做重采样，因而没有延迟线。
    fn drain_resampler_tail(&mut self) -> Result<()> {
        // take() 取出以避免同时可变借用 `self`；尾样排完就不再需要它。
        let Some(mut resampler) = self.encode_resampler.take() else {
            return Ok(());
        };
        let time_base = self.time_base();
        // 固定帧长音频的输出 pts 由 `audio_fifo` 按已输出样本数维护（见
        // `assign_pts_sample_rate`），这里再按尾样推进计数器会与 `fifo_pop_frame`
        // 重复累加一次。
        let numbered_by_fifo = self.media_type == MediaType::AUDIO && self.frame_size() > 0;
        for mut frame in resampler.flush_frames()? {
            frame.set_time_base(time_base.into());
            if !numbered_by_fifo {
                frame.set_pts(self.next_pts);
                self.next_pts += i64::from(frame.nb_samples);
            }
            self.send_frame_ready(frame)?;
        }
        Ok(())
    }

    /// 向编码器发送一帧（或 EOF）已就绪的输入；若编码器缓冲已满（EAGAIN），
    /// 先排空已就绪包再重试。
    ///
    /// 重试次数有上限（[`crate::MAX_DRAIN_ITERATIONS`]）：个别编码器在 EOS 之后
    /// 会持续返回 EAGAIN 而不再产出包，无上限循环会挂死；达到上限即报错，
    /// 而不是无限等待。
    fn send_frame_with_retry(&mut self, frame: Option<&AVFrame>) -> Result<()> {
        let mut retries = 0usize;
        loop {
            match self.context.send_frame(frame) {
                Ok(()) => return Ok(()),
                Err(rsmpeg::error::RsmpegError::SendFrameAgainError) => {
                    retries += 1;
                    if retries > crate::MAX_DRAIN_ITERATIONS {
                        return Err(RsmediaError::msg(format!(
                            "Encoder keeps returning EAGAIN after {} retries (eof: {}); aborting",
                            crate::MAX_DRAIN_ITERATIONS,
                            frame.is_none()
                        )));
                    }
                    tracing::debug!("Encoder buffer full (EAGAIN), draining ready packets first.");
                    self.drain_encoder_packets()?;
                }
                Err(e) => return Err(RsmediaError::FFmpeg(e)),
            }
        }
    }

    /// 编码器缓冲已满（send_frame 返回 EAGAIN）时，先排空已就绪包到 `pending_packets`，
    /// 供 `receive_packet` 优先返回，避免丢包，随后由调用方重试发送。
    fn drain_encoder_packets(&mut self) -> Result<()> {
        loop {
            match self.context.receive_packet() {
                Ok(pkt) => self.pending_packets.push_back(pkt),
                // 要更多输入，或已排空
                Err(rsmpeg::error::RsmpegError::EncoderDrainError)
                | Err(rsmpeg::error::RsmpegError::EncoderFlushedError) => break,
                Err(e) => return Err(RsmediaError::FFmpeg(e)),
            }
        }
        Ok(())
    }

    fn rescale(&mut self, frame: AVFrame) -> Result<AVFrame> {
        let scaled_frame = match self.media_type {
            MediaType::VIDEO => {
                // 已是硬件帧：swscale 处理不了 hw 帧，且软件编码器也吃不下它。
                // 跨 frames context 的搬运（本编码器自己的那份）由
                // `send_frame_post_filter` 用 `av_hwframe_map` 完成，这里原样透传。
                if !frame.hw_frames_ctx.is_null() {
                    let hw_ctx = self.hw_context.as_ref().ok_or_else(|| {
                        RsmediaError::invalid_config(format!(
                            "input frame is a hardware frame ({:?}) but this encoder has no hardware \
                             device configured; enable hardware acceleration with \
                             `with_hardware_device`, or download the frame to system memory first",
                            PixelFormat::from(frame.format)
                        ))
                    })?;
                    if !hw_ctx.is_hw_frame(&frame) {
                        return Err(RsmediaError::invalid_config(format!(
                            "input hardware frame format {:?} does not match this encoder's hardware \
                             format {:?}; download the frame to system memory, or encode with the \
                             same hardware backend",
                            PixelFormat::from(frame.format),
                            PixelFormat::from(hw_ctx.get_format(true))
                        )));
                    }
                    return Ok(frame);
                }
                let target_sw_pix_fmt = if let Some(hw_ctx) = self.hw_context.as_ref() {
                    hw_ctx.get_format(false).into()
                } else {
                    self.pix_fmt()
                };
                let dst_spec = VideoSpec::from_frame(&frame)?.with_pix_fmt(target_sw_pix_fmt);
                self.scaler.scale_if_needed(frame, dst_spec)?
            }
            MediaType::AUDIO => {
                // 目标格式先取成值：`self.audio_spec()` 借用 `self`，而下一行要可变借用
                // `self.encode_resampler`。
                let out_spec = self.audio_spec();
                resample::resample_if_needed(&mut self.encode_resampler, frame, out_spec)?
            }
            _ => {
                // do nothing
                return Err(RsmediaError::unsupported(format!(
                    "no frame conversion path for media type: {:?}",
                    self.media_type
                )));
            }
        };
        Ok(scaled_frame)
    }

    /// Check if the frame is valid for encoding.
    fn check_frame(&self, frame: Option<&AVFrame>) -> Result<()> {
        let Some(frame) = frame else {
            return Ok(());
        };
        match self.media_type {
            MediaType::VIDEO => {
                // 硬件帧的像素格式（如 NV12/HW 私有格式）不在软件编码器的
                // `supported_pixel_formats()` 列表中，跳过该检查以免误报。
                if !frame.hw_frames_ctx.is_null() {
                    return Ok(());
                }
                if !self.config.supports_pixel_format(frame.format) {
                    return Err(RsmediaError::unsupported(format!(
                        "this encoder cannot encode frames in pixel format {:?}",
                        frame.format
                    )));
                }
                // 编码器上下文的尺寸是权威值（`build` 时若滤镜改了尺寸已同步到
                // 上下文）。尺寸不符时 `avcodec_send_frame` 只在深处报一句
                // "Frame parameters mismatch context ..." 的 EINVAL，这里提前失败
                // 并说清该改哪一边。上下文尺寸为 0 表示未设置，交给编码器自行决定。
                let (ctx_width, ctx_height) = (self.context.width, self.context.height);
                if ctx_width > 0
                    && ctx_height > 0
                    && (frame.width != ctx_width || frame.height != ctx_height)
                {
                    return Err(RsmediaError::invalid_config(format!(
                        "frame size {}x{} does not match the encoder's {}x{}; \
                         resize the frame before encoding, or configure the encoder \
                         with the frame's size",
                        frame.width, frame.height, ctx_width, ctx_height
                    )));
                }
            }

            MediaType::AUDIO => {
                if !self.config.supports_sample_format(frame.format) {
                    return Err(RsmediaError::unsupported(format!(
                        "this encoder cannot encode frames in sample format {:?}",
                        frame.format
                    )));
                }

                if !self.config.supports_sample_rate(frame.sample_rate) {
                    return Err(RsmediaError::unsupported(format!(
                        "this encoder cannot encode audio at sample rate {:?}",
                        frame.sample_rate
                    )));
                }

                // 注意：不在此校验 `nb_samples == frame_size`。固定帧长音频编码器
                // 已由 `audio_fifo` 缓冲切帧（切出帧恒为 frame_size，flushed 末帧
                // 允许不足一帧），此处切出的帧长短不由待编码帧决定；且末帧不足一帧
                // 是编码器合法接受的，故帧长正确性由缓冲路径保证，不在此拦截。
            }
            _ => {}
        }
        Ok(())
    }

    /// Get encoder time base.
    #[inline]
    pub fn time_base(&self) -> Rational {
        self.context.time_base.into()
    }

    /// The frame rate the encoder is actually using (`AVCodecContext.framerate`).
    ///
    /// This is the value that reaches the container, after any negotiation: the
    /// exact rational from [`EncoderBuilder::with_frame_rate`], the `av_d2q`
    /// approximation of [`EncoderBuilder::with_fps`], or a filter graph's output
    /// rate when a filter like `fps` rewrote it. Read it to verify what a
    /// pipeline really produces instead of assuming the requested rate survived.
    #[inline]
    pub fn frame_rate(&self) -> Rational {
        self.context.framerate.into()
    }

    /// Width of the encoder's negotiated video frame, in pixels.
    ///
    /// A width is non-negative by nature, so this is returned as `u32` — the same
    /// width [`EncoderBuilder::with_width`] takes, which keeps a set/get
    /// round-trip cast-free. The underlying `AVCodecContext.width` field is an
    /// FFmpeg `int`; the conversion here can never see a negative value.
    #[inline]
    pub fn width(&self) -> i32 {
        self.context.width
    }

    /// Height of the encoder's negotiated video frame, in pixels.
    ///
    /// Returned as `u32` for the same reason as [`Self::width`].
    #[inline]
    pub fn height(&self) -> i32 {
        self.context.height
    }

    #[inline]
    pub fn pix_fmt(&self) -> PixelFormat {
        self.context.pix_fmt.into()
    }

    /// 编码器期望的**软件输入**像素格式。
    ///
    /// 软件编码器即编码器协商格式（[`Self::pix_fmt`]）；硬件编码器时
    /// `codec_ctx.pix_fmt` 被 [`crate::hwaccel::HWContext::setup_encoder_frames`]
    /// 覆写为硬件私有格式（如 `AV_PIX_FMT_VIDEOTOOLBOX`），而输入帧仍需以
    /// **软件格式**（如 NV12）做 swscale/上传，故此处返回 `hw_ctx` 的软件格式。
    #[inline]
    fn input_sw_pix_fmt(&self) -> PixelFormat {
        match self.hw_context.as_ref() {
            Some(hw_ctx) => hw_ctx.get_format(false).into(),
            None => self.pix_fmt(),
        }
    }

    /// Each submitted frame except the last must contain exactly frame_size samples per channel.
    /// May be 0 when the codec has AV_CODEC_CAP_VARIABLE_FRAME_SIZE set, then the frame size is not restricted.
    ///
    /// Unlike [`Self::width`]/[`Self::height`] this stays `i32`: `0` is a
    /// meaningful value here ("no fixed frame size"), so the sentinel and the
    /// quantity share one range, exactly as in FFmpeg's `int` field.
    #[inline]
    pub fn frame_size(&self) -> i32 {
        self.context.frame_size
    }

    /// Audio samples per second.
    ///
    /// Returned as `i32`, the width of the underlying `AVCodecContext.sample_rate`
    /// field, matching [`EncoderBuilder::with_sample_rate`] and
    /// [`MediaFrame`]'s audio metadata.
    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.context.sample_rate
    }

    /// audio sample format
    #[inline]
    pub fn sample_fmt(&self) -> SampleFormat {
        SampleFormat::from(self.context.sample_fmt)
    }

    #[inline]
    pub fn ch_layout(&self) -> AVChannelLayoutRef<'_> {
        self.context.ch_layout()
    }

    #[inline]
    pub fn media_type(&self) -> MediaType {
        self.media_type
    }

    #[inline]
    pub fn codecpar(&self) -> AVCodecParameters {
        self.context.extract_codecpar()
    }

    /// `AVCodecContext.flags` 位集（`AV_CODEC_FLAG_*`，取值见 [`AVCodecFlag`]）。
    ///
    /// 读的是 `avcodec_open2` **之后**的实际值，因此 `with_global_header(true)`
    /// 并入的 `GLOBAL_HEADER` 位也在内；编解码器自行调整过的位同样会反映出来。
    /// 查询用 [`FlagSet::contains`]，需要原始整数时用 [`FlagSet::bits`]。
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
    /// 读的是 `avcodec_open2` **之后**的实际值：帧级线程的编解码器会在
    /// `thread_count == 0` 时把它改写成自动推导出的线程数（见 FFmpeg
    /// `ff_frame_thread_init`），非 0 的调用方设置则原样保留。
    ///
    /// 类型与 FFmpeg 的字段一致（`int` 而非 `u32`）：与
    /// [`Decoder::thread_count`](crate::Decoder::thread_count) 保持同型。
    #[inline]
    pub fn thread_count(&self) -> i32 {
        self.context.thread_count
    }

    /// 单帧时长（编码器 time_base 单位），用于补全缺失的 packet duration。
    ///
    /// 先求单帧时长（秒），再换算到编码器 time_base 的整数 tick：
    /// `ticks = av_rescale_q(1, frame_dur_sec, time_base)`。
    pub(crate) fn packet_duration(&self) -> i64 {
        let tb = self.time_base();
        let frame_dur_sec = match self.media_type {
            // 视频：1 / frame_rate（帧率为 0 —— 未协商出帧率 —— 时无从得知单帧时长）
            MediaType::VIDEO => match self.frame_rate().inverse() {
                Ok(frame_duration) => frame_duration,
                Err(_) => return 0,
            },
            // 音频：frame_size / sample_rate
            MediaType::AUDIO => {
                let fs = self.frame_size();
                if fs <= 0 {
                    return 0;
                }
                Rational::new(fs, self.sample_rate()).unwrap_or(Rational::ZERO)
            }
            _ => return 0,
        };
        avutil::av_rescale_q(1, frame_dur_sec.into(), tb.into()).max(1)
    }

    /// Internal: Pull an encoded packet from the decoder.
    ///
    /// Handles `EAGAIN`, drained, and flushed states.
    ///
    /// # Returns
    ///
    /// `Some(packet)` if a packet is returned, `None` if waiting or end.
    fn receive_packet(&mut self) -> Result<Option<AVPacket>> {
        // 优先返回暂存区（send_frame EAGAIN 排空时存入）的包。
        // 按 FIFO 顺序出队（`pop_front`），保证与编码器输出顺序一致，
        // 避免 dts 乱序、mux 报错。
        if let Some(pkt) = self.pending_packets.pop_front() {
            return Ok(Some(pkt));
        }
        match self.context.receive_packet() {
            Ok(pkt) => Ok(Some(pkt)),
            Err(rsmpeg::error::RsmpegError::EncoderDrainError) => {
                // EAGAIN：此刻无包可出，需要继续喂帧（read 阶段）或继续排空
                // （已送出 EOS）。这里**不能**改状态——read 阶段同样会走到这里，
                // 置成 `Drained` 会让 `is_drained()` 在流中段就永久为真
                // （见 `Encoder::state` 的说明）。
                tracing::debug!("Encoder drained, try send new frame again.");
                Ok(None)
            }
            Err(rsmpeg::error::RsmpegError::EncoderFlushedError) => {
                tracing::debug!("Encoder flushed, EOF reached.");
                self.state = ProcessState::Flushed;
                Ok(None)
            }
            Err(err) => Err(RsmediaError::FFmpeg(err)),
        }
    }

    /// 输入侧收尾：把还压在滤镜、重采样延迟与音频 FIFO 里的样本全部送进编码器，
    /// 然后发 EOS。
    ///
    /// 只在 `Normal`（EOS 尚未送出）时调用一次 —— 之后编码器进入排空阶段，不允许
    /// 再送帧。顺序不是随意的：滤镜的缓冲帧要先全部推出（它们还要过重采样），
    /// 重采样延迟线排在滤镜之后，音频 FIFO 的末帧排在重采样之后 —— 每一步的产物
    /// 都是下一步的输入。
    fn finish_input(&mut self) -> Result<()> {
        if let Some(filter) = self.filter_graph.as_mut() {
            let frames = filter.flush()?;
            for frame in frames {
                // filter 已 Flushed，缓冲帧直接走 post-filter 路径，不可再进 process_frame
                self.send_frame_post_filter(frame)?;
            }
        }

        // 采样率变化时重采样器还压着几十毫秒的尾样，只有 EOF 之后才吐得出来。
        self.drain_resampler_tail()?;

        // 冲刷音频缓冲中不足一帧的剩余样本（作为末帧送编码器）
        self.flush_audio_fifo()?;

        // EOF: Notify the encoder that the last frame has been sent.
        self.send_frame_to_encoder(None)?;

        // 只有 EOS 真正送出、才进入排空阶段（此后不允许再送帧）。置位点必须在这里，
        // 而不是在 `receive_packet` 的 EAGAIN 分支——那里 read 阶段也会走到。
        // 与 `Decoder::drain_raw` 同一写法：阶段只由 `state` 表示。
        self.state = ProcessState::Drained;
        Ok(())
    }

    /// Flush the encoder and write any remaining packets.
    ///
    /// This function sends an end-of-stream signal to the encoder, and continues
    /// to pull packets until the encoder is fully flushed.
    ///
    /// # Arguments
    ///
    /// * `writer` - Writer to write encoded packets.
    /// * `interleaved` - Whether to write packets in interleaved mode (typical for most formats).
    /// * `index` - Stream index for the output stream.
    /// * `out_stream_time_base` - Time base of the output stream.
    ///
    /// 产出的字节留在 `writer` 里（见 [`Writer`] 的输出模型），由调用方按需取。
    ///
    /// May return an error if writing fails or encoder returns an error.
    ///
    /// Idempotent **only once the drain really finished** ([`is_flushed`](Self::is_flushed)):
    /// a call after a failed drain (the encoder is `Drained` — EOS sent, packets
    /// still buffered) retries the drain instead of reporting success, so a caller
    /// that retries — or [`Muxer::finish`](crate::mux::Muxer::finish) running a
    /// second time — cannot turn a truncated stream into a silent success.
    pub fn flush<W: Writer>(
        &mut self,
        writer: &mut W,
        interleaved: bool,
        index: usize,
        out_stream_time_base: Rational,
    ) -> Result<()> {
        // 幂等只在**真正排空完成**时成立。`Drained` 表示 EOS 已送出、但上一次没排完
        // （排空循环报错，或撞上迭代上限）：那种情况必须重试排空。若按"不是 Normal"
        // 就返回，第二次调用会直接成功，`Muxer::finish` 随后照样写 trailer，
        // 把截断的输出当成成功。
        if self.is_flushed() {
            tracing::debug!("Encoder already flushed ({:?}), nothing to do.", self.state);
            return Ok(());
        }

        // 字幕编码器走同步 API（avcodec_encode_subtitle），无内部缓冲，
        // 不支持 send/receive flush（send_frame(None) 会崩溃），直接返回。
        // 仍然标记 Flushed：对字幕而言"排空"没有下一步可做，且 Drop 的
        // "未 flush" 告警只应针对真的丢了缓冲的编码器。
        if self.media_type == MediaType::SUBTITLE {
            self.state = ProcessState::Flushed;
            return Ok(());
        }

        // `Normal` = EOS 还没送出：先排滤镜与音频 FIFO，再送 EOS；
        // `Drained` = EOS 已送出，只是上次没排完，直接从下面的排空循环继续。
        if self.state.is_normal() {
            self.finish_input()?;
        }

        // drain the items still on the queue before giving up.
        // EOF 已发送，理论上编码器最终会返回 EOF；但为防御个别编码器在 EOS 后
        // 持续返回 EAGAIN（Drained）而不返回 EOF，增加迭代上限，避免死循环。
        let mut drained_iterations = 0usize;
        let mut written_packets = 0usize;
        loop {
            match self.receive_packet() {
                Ok(Some(mut packet)) => {
                    drained_iterations = 0;
                    packet.set_pos(-1);
                    packet.set_stream_index(index as i32);
                    // 编码器输出的 packet 常不带 duration（libx264 等），若缺失则按
                    // 帧率/采样率补上，否则 MP4 等容器无法推导**最后一帧**的时长，
                    // 导致末帧被 muxer 丢弃。
                    if packet.duration <= 0 {
                        packet.set_duration(self.packet_duration());
                    }
                    // 将编码器输出的数据包时间戳，从编码器时间基转换到输出流时间基
                    // encode_ctx_timebase => out_stream_time_base
                    packet.rescale_ts(self.time_base().into(), out_stream_time_base.into());
                    if interleaved {
                        writer.write_interleaved(&mut packet)?
                    } else {
                        writer.write_frame(&mut packet)?
                    };
                    written_packets += 1;
                }
                Ok(None) => {
                    if self.is_drained() {
                        tracing::debug!("Encoder drained, try send new frame again.");
                        drained_iterations += 1;
                        if drained_iterations >= crate::MAX_DRAIN_ITERATIONS {
                            return Err(RsmediaError::msg(format!(
                                "Encoder keeps returning EAGAIN after EOF for {} iterations; \
                                 flush aborted after {written_packets} packet(s), output is truncated",
                                crate::MAX_DRAIN_ITERATIONS
                            )));
                        }
                        continue;
                    } else {
                        tracing::debug!("Encoder flushed, EOF reached.");
                        break;
                    }
                }
                Err(e) => {
                    // 排空阶段的错误不能降级成日志：那会把"被截断的输出"当成成功返回。
                    // 已经写进 writer 的包无法回收，错误信息里带上数量便于定位。
                    return Err(e.with_context(format!(
                        "Failed to drain encoder during flush after {written_packets} packet(s); \
                         output is truncated"
                    )));
                }
            }
        }

        Ok(())
    }
}

impl Drop for Encoder {
    /// Automatically called when the `Encoder` is dropped.
    ///
    /// **Warning**: This does NOT automatically flush the encoder.
    /// The user is responsible for calling [`Encoder::flush`] manually
    /// before dropping the encoder to ensure all frames are written.
    fn drop(&mut self) {
        if !self.is_flushed() {
            tracing::error!("Encoder dropped without flushing, data may be lost.");
        }
    }
}

/// SAFETY:
/// - Encoder contains `AVCodecContext`, which is not inherently thread-safe.
///   Sharing `&Encoder` across threads (`Sync`) cannot be guaranteed, so only
///   `Send` is implemented: moving an Encoder to another thread for exclusive
///   use is safe, as all resources move with the object.
unsafe impl Send for Encoder {}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // 核心方法单元测试
    //
    // 只覆盖编码器自身的决策逻辑 —— 格式协商、时间基/码率推导、pts 自动编号、
    // 帧校验、builder 选项落点 —— 不落盘、不做编解码往返，因此不需要 `MediaFrame`，
    // 也不依赖任何测试媒体文件。
    //
    // 需要真实文件的端到端功能测试（容器矩阵 / 编解码往返 / 滤镜 / 转码 /
    // 音频切帧）见 `tests/encode_pipeline.rs`。
    // ====================================================================

    /// 本构建是否提供默认视频编码器（`EncoderBuilder::VIDEO_CODEC_NAME`）。
    ///
    /// 环境差异（构建未编入 libx264）应当跳过而不是失败，所以要**先探测可用性**
    fn default_video_encoder_available() -> bool {
        let name = std::ffi::CString::new(EncoderBuilder::VIDEO_CODEC_NAME)
            .expect("codec name is a NUL-free literal");
        AVCodec::find_encoder_by_name(&name).is_some()
    }

    /// `with_flags`/`with_flags2`/`with_thread_type` 的位集参数落到 `AVCodecContext`
    /// 的对应字段（单个标志与 `|` 组合都原样保留）；
    /// `with_flags` 与 builder 自管的 `GLOBAL_HEADER` 是不同位，按位合并而非互相覆盖。
    #[test]
    fn test_builder_codec_flags_reach_context() -> Result<()> {
        use crate::codec::{AVCodecFlag, AVCodecFlag2, ThreadType};

        let encoder = EncoderBuilder::new_video(64, 64)
            .with_flags(AVCodecFlag::CLOSED_GOP | AVCodecFlag::LOW_DELAY)
            .with_flags2(AVCodecFlag2::FAST | AVCodecFlag2::CHUNKS)
            .with_thread_type(ThreadType::SLICE)
            .build()?;

        let want = AVCodecFlag::CLOSED_GOP.as_raw() | AVCodecFlag::LOW_DELAY.as_raw();
        assert_eq!(
            encoder.flags().bits() & want,
            want,
            "caller flags must survive the GLOBAL_HEADER merge"
        );
        assert_eq!(encoder.flags2(), AVCodecFlag2::FAST | AVCodecFlag2::CHUNKS);
        assert_eq!(encoder.thread_type(), ThreadType::SLICE.into());

        // 单个标志（不组合）同样可传，且不会带进别的位。
        let single = EncoderBuilder::new_video(64, 64)
            .with_flags2(AVCodecFlag2::FAST)
            .build()?;
        assert_eq!(single.flags2(), AVCodecFlag2::FAST.into());
        assert!(single.flags2().contains(AVCodecFlag2::FAST));

        // 组合位（`FF_THREAD_FRAME | FF_THREAD_SLICE`）原样落到位集。
        let encoder = EncoderBuilder::new_video(64, 64)
            .with_thread_type(ThreadType::FRAME | ThreadType::SLICE)
            .build()?;
        assert_eq!(encoder.thread_type(), ThreadType::FRAME | ThreadType::SLICE);

        // GLOBAL_HEADER 与调用方的 flags 是两个来源的不同位，必须同时存在。
        let encoder = EncoderBuilder::new_video(64, 64)
            .with_global_header(true)
            .with_flags(AVCodecFlag::LOW_DELAY)
            .build()?;
        assert!(encoder.flags().contains(AVCodecFlag::GLOBAL_HEADER));
        assert!(encoder.flags().contains(AVCodecFlag::LOW_DELAY));

        // 调用方一个 flags 都不设时，上下文自带的默认位（FFmpeg 在
        // `avcodec_alloc_context3` 里写入，实测含 `CLOSED_GOP`）必须原样保留：
        // 把这些位从 0 重建会静默改变输出码流（closed GOP → open GOP），
        // 而调用方并没有要求这个变化。对照值直接取自全新上下文的读数，不写死
        // 具体位（默认位随 FFmpeg 版本变化）。
        let codec = AVCodec::find_encoder_by_name(c"libx264")
            .expect("libx264 is the default video encoder used by this test");
        let defaults = AVCodecContext::new(&codec).flags as u32;
        // 自检：下面那条断言只有在默认位非空时才有意义（6.1~9.0 实测都是
        // `AV_CODEC_FLAG_CLOSED_GOP`）。若某个版本真把默认位清空了，这条会先
        // 失败提醒，而不是让上面的断言悄悄变成恒真。
        assert_ne!(defaults, 0, "no default flags to preserve on this FFmpeg");
        let plain = EncoderBuilder::new_video(64, 64).build()?;
        assert_eq!(
            plain.flags().bits() & defaults,
            defaults,
            "builder dropped FFmpeg's own default flags: got {:#x}, want at least {defaults:#x}",
            plain.flags().bits()
        );
        Ok(())
    }

    /// 未设置时落到本机 CPU 数；显式正数原样落入上下文；`0`（FFmpeg 的"自行推导"）
    /// 与负数（没有合法语义）都由 [`crate::codec::set_thread_count`] 跳过不写，上下文
    /// 保持 FFmpeg 默认的 `0` = 由编码器自行推导。负数另有一条 `warn!`，让"线程数
    /// 没生效"对调用方可见（日志断言需要 subscriber，故此处只锁行为）。
    #[test]
    fn test_builder_thread_count_non_positive_is_ignored() -> Result<()> {
        let explicit = EncoderBuilder::new_video(64, 64)
            .with_thread_count(3)
            .build()?;
        assert_eq!(explicit.thread_count(), 3);

        let cpu_count = num_cpus::get() as i32;
        let default = EncoderBuilder::new_video(64, 64).build()?;
        assert_eq!(default.thread_count(), cpu_count);

        for ignored in [0, -1, i32::MIN] {
            let non_positive = EncoderBuilder::new_video(64, 64)
                .with_thread_count(ignored)
                .build()?;
            assert_eq!(
                non_positive.thread_count(),
                0,
                "a non-positive thread_count ({ignored}) must be ignored, \
                 leaving FFmpeg's default"
            );
        }
        Ok(())
    }

    #[test]
    fn test_video_size_out_of_range_is_invalid_config() {
        // `Encoder` 没有 `Debug`，只能这样取错误。
        fn build_err(builder: EncoderBuilder) -> RsmediaError {
            let Err(err) = builder.build() else {
                panic!("an out-of-range video size must fail to build");
            };
            err
        }

        // `width`/`height` 是 `i32`（FFmpeg 的 `AVFrame.width`/`height` 也是 `int`），
        // 非正数不是合法画面尺寸。
        for (width, height, want) in [
            (0i32, 480, "width"),
            (640, 0, "height"),
            (-1, 480, "width"),
            (640, -1, "height"),
        ] {
            let err = build_err(EncoderBuilder::new_video(width, height));
            assert!(err.is_invalid_config(), "{err}");
            assert!(err.to_string().contains(want), "{err}");
        }

        // 正常尺寸不受影响（防止校验过严）。
        assert!(EncoderBuilder::new_video(8, 8).build().is_ok());
    }

    /// `with_max_bit_rate`/`with_buffer_size` 的非正值不写进 `rc_*` 字段：
    /// `0` 在文档里是"视为未设置"（`max_bit_rate = 0` 即不限瞬时码率），负值只可能
    /// 是写错。两者都不生效，负值额外打 `warn!`（日志不便断言，这里锁住"负值不会
    /// 污染字段"这一 fail-safe 行为）。
    #[test]
    fn test_non_positive_rate_control_is_not_applied() -> Result<()> {
        let applied = EncoderBuilder::new_video(64, 64)
            .with_bit_rate(500_000)
            .with_max_bit_rate(600_000)
            .with_buffer_size(1_200_000)
            .build()?;
        assert_eq!(applied.context.rc_max_rate, 600_000);
        assert_eq!(applied.context.rc_buffer_size, 1_200_000);

        for (max_rate, buffer_size) in [(0i64, 0i32), (-1, -1), (i64::MIN, i32::MIN)] {
            let ignored = EncoderBuilder::new_video(64, 64)
                .with_bit_rate(500_000)
                .with_max_bit_rate(max_rate)
                .with_buffer_size(buffer_size)
                .build()?;
            assert_eq!(
                ignored.context.rc_max_rate, 0,
                "max_bit_rate {max_rate} must not reach rc_max_rate"
            );
            assert_eq!(
                ignored.context.rc_buffer_size, 0,
                "buffer_size {buffer_size} must not reach rc_buffer_size"
            );
        }
        Ok(())
    }

    /// 未显式指定像素格式时按编码器能力协商；显式指定且编码器不支持时
    /// 立即报错（fail fast），而不是把帧转进去后在写入阶段才失败。
    #[test]
    fn test_resolve_pixel_format_negotiation() -> Result<()> {
        let config = CodecConfig::new_with_name(c"libx264")?;

        // 未指定 → 协商为 YUV420P（兼容性最好的默认值）。
        let builder = EncoderBuilder::new_video(64, 64);
        assert_eq!(
            builder.resolve_pixel_format(&config, "libx264")?,
            PixelFormat::YUV420P
        );

        // 显式且受支持 → 原样采用。
        let builder = EncoderBuilder::new_video(64, 64).with_pix_fmt(PixelFormat::YUV420P);
        assert_eq!(
            builder.resolve_pixel_format(&config, "libx264")?,
            PixelFormat::YUV420P
        );

        // 显式但不受支持 → InvalidConfig。
        let builder = EncoderBuilder::new_video(64, 64).with_pix_fmt(PixelFormat::RGB24);
        let err = builder
            .resolve_pixel_format(&config, "libx264")
            .expect_err("libx264 does not accept RGB24");
        assert!(
            matches!(err, RsmediaError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn test_missing_encoder_reports_unsupported() {
        let Err(err) = EncoderBuilder::new_video(64, 64)
            .with_codec_name("no_such_encoder")
            .build()
        else {
            panic!("unknown codec name must fail");
        };
        assert!(
            err.is_unsupported(),
            "expected an unsupported build capability, got {err:?}"
        );
        assert!(!err.is_invalid_config(), "{err}");
        assert!(err.to_string().contains("no_such_encoder"), "{err}");
    }

    /// 采样格式协商：未指定时优先 FLTP，编码器不支持 FLTP 时取支持列表首个；
    /// 显式指定且不受支持时立即报错。
    #[test]
    fn test_resolve_sample_format_negotiation() -> Result<()> {
        // pcm_s16le 只接受 S16，而默认优先级是 FLTP → 落到列表首个 S16。
        let config = CodecConfig::new_with_name(c"pcm_s16le")?;
        let builder = EncoderBuilder::default()
            .with_media_type(MediaType::AUDIO)
            .with_codec_name("pcm_s16le")
            .with_nb_channels(2)
            .with_sample_rate(44_100);
        assert_eq!(
            builder.resolve_sample_format(&config, "pcm_s16le")?,
            SampleFormat::S16
        );

        // 显式指定不支持的格式 → InvalidConfig。
        let builder = builder.with_sample_fmt(SampleFormat::FLTP);
        let err = builder
            .resolve_sample_format(&config, "pcm_s16le")
            .expect_err("pcm_s16le does not accept FLTP");
        assert!(
            matches!(err, RsmediaError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );

        // aac 原生平面浮点 → 未指定时协商为 FLTP。
        let config = CodecConfig::new_with_name(c"aac")?;
        let builder = EncoderBuilder::default()
            .with_media_type(MediaType::AUDIO)
            .with_codec_name("aac")
            .with_nb_channels(2)
            .with_sample_rate(44_100);
        assert_eq!(
            builder.resolve_sample_format(&config, "aac")?,
            SampleFormat::FLTP
        );
        Ok(())
    }

    /// 编码器输入时间基的推导：视频 `1/fps`、音频 `1/sample_rate`、
    /// 字幕 `1/1000`（毫秒精度，与 ffmpeg CLI 一致）。
    #[test]
    fn test_effective_time_base_per_media_type() {
        let video = EncoderBuilder::new_video(64, 64).with_fps(25.0);
        assert_eq!(
            video.effective_time_base().unwrap(),
            Rational::new(1, 25).unwrap(),
            "video input time base = 1/fps"
        );

        let audio = EncoderBuilder::new_audio(128_000, 2, 44_100, SampleFormat::FLTP);
        assert_eq!(
            audio.effective_time_base().unwrap(),
            Rational::new(1, 44_100).unwrap(),
            "audio input time base = 1/sample_rate"
        );

        let subtitle = EncoderBuilder::new_subtitle();
        assert_eq!(
            subtitle.effective_time_base().unwrap(),
            Rational::new(1, EncoderBuilder::SUBTITLE_TIME_BASE_DEN).unwrap(),
            "subtitle input time base = 1/1000"
        );

        // 帧率为 0 ⇒ 取不到倒数：音频/视频都报 `InvalidConfig`，而不是给出 1/0。
        let zero_rate = EncoderBuilder::new_video(64, 64).with_frame_rate(Rational::ZERO);
        assert!(
            zero_rate
                .effective_time_base()
                .unwrap_err()
                .is_invalid_config()
        );
    }

    /// 码率默认值按媒体类型区分：视频 1 Mbps、音频 128k（ffmpeg CLI 的 `-b:a`
    /// 默认）；显式 `with_bit_rate` 与 `Quality::Bitrate` 均可覆盖。
    #[test]
    fn test_default_bit_rate_per_media_type() {
        assert_eq!(
            EncoderBuilder::new_video(64, 64).effective_bit_rate(),
            EncoderBuilder::VIDEO_BIT_RATE,
            "video default bit rate"
        );
        assert_eq!(
            EncoderBuilder::default()
                .with_media_type(MediaType::AUDIO)
                .effective_bit_rate(),
            EncoderBuilder::AUDIO_BIT_RATE,
            "audio default bit rate is 128k, not the video default"
        );
    }

    /// `Quality::Bitrate` 覆盖 `with_bit_rate`，其他 `Quality` 不参与码率推导。
    #[test]
    fn test_effective_bit_rate_precedence() {
        fn bitrate_builder() -> EncoderBuilder {
            EncoderBuilder::new_video(64, 64).with_bit_rate(500_000)
        }

        assert_eq!(bitrate_builder().effective_bit_rate(), 500_000);
        assert_eq!(
            bitrate_builder()
                .with_quality(Quality::Bitrate(2_000_000))
                .effective_bit_rate(),
            2_000_000,
            "Quality::Bitrate overrides with_bit_rate"
        );
        assert_eq!(
            bitrate_builder()
                .with_quality(Quality::Crf(20))
                .effective_bit_rate(),
            500_000,
            "Crf is a video quality knob and must not touch the bit rate"
        );
        assert_eq!(
            bitrate_builder()
                .with_quality(Quality::Bitrate(0))
                .effective_bit_rate(),
            500_000,
            "a non-positive Quality::Bitrate counts as unset"
        );
        assert_eq!(
            EncoderBuilder::new_video(64, 64).effective_bit_rate(),
            EncoderBuilder::VIDEO_BIT_RATE,
            "default bit rate"
        );
    }

    /// `check_frame` 按媒体类型校验帧格式：`None`（flush 空帧）直接放行，
    /// 编码器不支持的像素/采样格式被拒。
    #[test]
    fn test_check_frame_validates_frame_formats() -> Result<()> {
        let video = EncoderBuilder::new_video(64, 64).build()?;
        video.check_frame(None)?;

        let mut frame = AVFrame::new();
        frame.set_width(64);
        frame.set_height(64);
        frame.set_format(PixelFormat::YUV420P as i32);
        video.check_frame(Some(&frame))?;

        frame.set_format(PixelFormat::RGB24 as i32);
        assert!(
            video.check_frame(Some(&frame)).is_err(),
            "libx264 does not accept RGB24 frames"
        );

        // 尺寸不符要在本地就被拦下：否则只有 `avcodec_send_frame` 深处一句
        // "Frame parameters mismatch context" 的 EINVAL，看不出该改哪边。
        frame.set_format(PixelFormat::YUV420P as i32);
        frame.set_width(32);
        let err = video
            .check_frame(Some(&frame))
            .expect_err("32x64 frame must not reach a 64x64 encoder");
        assert!(err.is_invalid_config(), "{err}");
        assert!(err.to_string().contains("32x64"), "{err}");

        let audio = EncoderBuilder::new_audio(128_000, 2, 44_100, SampleFormat::FLTP).build()?;
        audio.check_frame(None)?;

        let mut frame = AVFrame::new();
        frame.set_format(SampleFormat::FLTP as i32);
        frame.set_sample_rate(44_100);
        audio.check_frame(Some(&frame))?;

        frame.set_format(SampleFormat::U8 as i32);
        assert!(
            audio.check_frame(Some(&frame)).is_err(),
            "aac does not accept U8 frames"
        );
        Ok(())
    }

    /// 视频自动 pts：未设置 pts 的帧按每帧 1 tick 编号，用户设置的 pts 原样
    /// 使用并把计数器跳到其后，后续未设置的帧接续编号。
    #[test]
    fn test_assign_pts_sample_rate_auto_numbering_video() -> Result<()> {
        let mut encoder = EncoderBuilder::new_video(64, 64).with_fps(25.0).build()?;
        assert_eq!(encoder.next_pts, 0);

        let mut first = AVFrame::new();
        encoder.assign_pts_sample_rate(&mut first);
        assert_eq!(first.pts, 0, "first auto pts");
        assert_eq!(encoder.next_pts, 1);

        let mut second = AVFrame::new();
        encoder.assign_pts_sample_rate(&mut second);
        assert_eq!(second.pts, 1, "second auto pts");
        assert_eq!(encoder.next_pts, 2);

        // 输入时间基被统一为编码器的 1/fps。
        assert_eq!(
            Rational::from(second.time_base),
            Rational::new(1, 25).unwrap()
        );

        // 用户显式 pts：原样使用，计数器跳到该 pts 之后。
        let mut explicit = AVFrame::new();
        explicit.set_pts(100);
        encoder.assign_pts_sample_rate(&mut explicit);
        assert_eq!(explicit.pts, 100);
        assert_eq!(encoder.next_pts, 101);

        let mut resumed = AVFrame::new();
        encoder.assign_pts_sample_rate(&mut resumed);
        assert_eq!(
            resumed.pts, 101,
            "auto numbering resumes after the explicit pts"
        );
        Ok(())
    }

    /// 音频帧未声明采样率（0）时回退到编码器自己的率；已声明的不被覆盖。
    #[test]
    fn test_assign_pts_sample_rate_fills_missing_rate() -> Result<()> {
        let mut encoder =
            EncoderBuilder::new_audio(128_000, 2, 44_100, SampleFormat::FLTP).build()?;

        // 未声明（0）→ 用编码器的率补齐。用户直接构造的 AVFrame 就是这个状态。
        let mut undeclared = AVFrame::new();
        undeclared.set_sample_rate(0);
        encoder.assign_pts_sample_rate(&mut undeclared);
        assert_eq!(undeclared.sample_rate, 44_100);

        // 已声明 → 原样保留，否则"任意采样率输入、自动重采样"会失效。
        let mut declared = AVFrame::new();
        declared.set_sample_rate(16_000);
        encoder.assign_pts_sample_rate(&mut declared);
        assert_eq!(
            declared.sample_rate, 16_000,
            "an explicitly declared source rate must not be overwritten"
        );

        // 视频帧不涉及采样率，不应被改写。
        let mut video = EncoderBuilder::new_video(64, 64).build()?;
        let mut vframe = AVFrame::new();
        vframe.set_sample_rate(0);
        video.assign_pts_sample_rate(&mut vframe);
        assert_eq!(vframe.sample_rate, 0);

        Ok(())
    }

    /// 固定帧长音频（aac，frame_size = 1024）的输出 pts 由 `audio_fifo` 切帧时
    /// 按已输出样本数维护，`assign_pts_sample_rate` 不参与编号。
    #[test]
    fn test_assign_pts_sample_rate_defers_to_audio_fifo_for_fixed_frame_size() -> Result<()> {
        let mut encoder =
            EncoderBuilder::new_audio(128_000, 2, 44_100, SampleFormat::FLTP).build()?;
        assert!(encoder.frame_size() > 0, "aac has a fixed frame size");

        let mut frame = AVFrame::new();
        encoder.assign_pts_sample_rate(&mut frame);
        assert_eq!(
            frame.pts,
            ffi::AV_NOPTS_VALUE,
            "fixed-frame-size audio is numbered by the fifo, not here"
        );
        Ok(())
    }

    /// 采样率变化时 `swr` 的滤波延迟里压着尾样（见
    /// `resample::tests::test_streaming_resampler_carries_delay_and_flushes`），只有
    /// EOF 之后才吐得出来。收尾必须先把它们排进编码器，否则音频末尾静默短掉一小段。
    #[test]
    fn test_finish_input_drains_the_resampler_delay_line() -> Result<()> {
        const IN_RATE: i32 = 48_000;
        const OUT_RATE: i32 = 44_100;
        const FRAMES: i32 = 8;
        const NB_SAMPLES: i32 = 1024;

        let mut encoder =
            EncoderBuilder::new_audio(128_000, 2, OUT_RATE, SampleFormat::FLTP).build()?;
        assert!(encoder.frame_size() > 0, "aac has a fixed frame size");

        for _ in 0..FRAMES {
            let mut frame = AVFrame::new();
            frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
            frame.set_ch_layout(AVChannelLayout::from_nb_channels(2).into_inner());
            frame.set_sample_rate(IN_RATE);
            frame.set_nb_samples(NB_SAMPLES);
            frame.set_pts(ffi::AV_NOPTS_VALUE);
            frame
                .alloc_buffer()
                .context("Failed to allocate the input frame")?;
            // 必须填真实样本：`alloc_buffer` 给的是未初始化内存，把里面的随机浮点
            // 喂给 aac 会让它以 EINVAL 失败 —— 而且是间歇性的（取决于那块内存的内容）。
            for plane in 0..2 {
                // SAFETY: 平面已按 `nb_samples` 个 f32 分配（FLTP，每平面一条声道）。
                unsafe {
                    let samples = std::slice::from_raw_parts_mut(
                        frame.deref_mut().data[plane] as *mut f32,
                        NB_SAMPLES as usize,
                    );
                    // 静音即可：本测试只关心样本**数量**，不关心内容。
                    samples.fill(0.0);
                }
            }
            // 固定帧长音频不为输入帧编号（见 `assign_pts_sample_rate`），所以 `next_pts`
            // 从 0 起步、只由 `fifo_pop_frame` 按已输出样本数推进。
            encoder.send_frame_to_encoder(Some(frame))?;
        }

        encoder.finish_input()?;

        // 送进编码器的样本总数 = 已从 fifo 切出的（`next_pts`）+ 还压在 fifo 里的。
        let buffered = encoder.audio_fifo.as_ref().map_or(0, |fifo| fifo.size());
        let total = encoder.next_pts + i64::from(buffered);
        let expected = i64::from(FRAMES * NB_SAMPLES) * i64::from(OUT_RATE) / i64::from(IN_RATE);
        assert!(
            (total - expected).abs() <= 2,
            "expected ~{expected} samples at the encoder input, got {total} \
             (the resampler's delay line was not drained)"
        );
        Ok(())
    }

    /// 只读访问器反映 builder 的配置。
    #[test]
    fn test_encoder_accessors_reflect_builder() -> Result<()> {
        let encoder = EncoderBuilder::new_video(320, 240)
            .with_fps(30.0)
            .with_pix_fmt(PixelFormat::YUV420P)
            .build()?;

        assert_eq!(encoder.width(), 320);
        assert_eq!(encoder.height(), 240);
        assert_eq!(encoder.pix_fmt(), PixelFormat::YUV420P);
        assert_eq!(encoder.media_type(), MediaType::VIDEO);
        assert_eq!(encoder.frame_size(), 0, "video codecs have no frame size");
        assert_eq!(
            encoder.time_base(),
            Rational::new(1, 30).unwrap(),
            "time base = 1/fps"
        );
        assert_eq!(
            encoder.frame_rate(),
            Rational::integer(30),
            "frame rate = fps"
        );
        assert!(!encoder.is_drained(), "a fresh encoder is not drained");
        assert!(!encoder.is_flushed(), "a fresh encoder is not flushed");
        Ok(())
    }

    /// 阶段只由 `state` 表示：流中段的 EAGAIN（"此刻暂无包"）**不是**排空阶段。
    ///
    /// 这条不变式正是当年 `Encoder::draining` 那个额外标志位要兜住的东西。它同时
    /// 也说明了为什么不该用标志位兜：`is_drained()` 现在直接读 `state`，所以任何
    /// 把它写回 EAGAIN 分支的实现都会在这里失败（旧写法下这个测试反而抓不到 bug）。
    ///
    /// 前几帧必然在编码器内部触发 EAGAIN：libx264 默认 B 帧 + lookahead 会先缓冲
    /// 输入，`receive_packet` 因此返回"暂无包"。
    #[test]
    fn test_eagain_mid_stream_is_not_the_draining_phase() -> Result<()> {
        // 默认编码器缺失的环境（构建不含 libx264）跳过：先按名字探测可用性，
        // 而不是把构建错误当成"环境差异"吞掉——分类合并后后者也是 InvalidConfig。
        if !default_video_encoder_available() {
            println!(
                "SKIP: {} is not in this FFmpeg build",
                EncoderBuilder::VIDEO_CODEC_NAME
            );
            return Ok(());
        }
        let mut encoder = EncoderBuilder::new_video(64, 64)
            .with_fps(25.0)
            .with_pix_fmt(PixelFormat::YUV420P)
            .build()?;

        for index in 0..4i64 {
            let mut frame = AVFrame::new();
            frame.set_width(64);
            frame.set_height(64);
            frame.set_format(i32::from(PixelFormat::YUV420P));
            frame
                .alloc_buffer()
                .context("Failed to allocate test frame buffer")?;
            frame.set_pts(index);
            encoder.encode_raw(frame)?;

            assert!(
                !encoder.is_drained(),
                "frame {index}: still reading, so EAGAIN must not look like draining"
            );
            assert!(!encoder.is_flushed(), "frame {index}: not at EOF yet");
        }
        Ok(())
    }

    /// 自动探测硬件设备；探不到就返回 `None`（"本机此刻没有可用 GPU"不是测试失败）。
    fn try_auto_hw_config() -> Option<HWDeviceConfig> {
        match HWDeviceConfig::auto_platform() {
            Ok(config) => Some(config),
            Err(e) => {
                println!("SKIP: no hardware acceleration device probed: {e}");
                None
            }
        }
    }

    /// 某设备类型对应的 H.264 硬件编码器名（FFmpeg 的命名约定）。
    ///
    /// 只列出与**编码**设备一一对应的那些；纯解码后端（dxva2/d3d11va/vulkan/opencl）
    /// 返回 `None`，测试据此跳过。
    fn hw_encoder_for(device_type: crate::hwaccel::HWDeviceType) -> Option<&'static str> {
        use crate::hwaccel::HWDeviceType as T;
        match device_type {
            T::VIDEOTOOLBOX => Some("h264_videotoolbox"),
            T::CUDA => Some("h264_nvenc"),
            T::VAAPI => Some("h264_vaapi"),
            T::QSV => Some("h264_qsv"),
            T::MEDIACODEC => Some("h264_mediacodec"),
            _ => None,
        }
    }

    /// 硬件帧输入：帧属于**别的** frames context（解码器 / 另一台设备 / 调用方自建）
    /// 时，`rescale` 不许让 swscale 去碰它，`send_frame_post_filter` 必须把它搬进
    /// 编码器自己那份 frames context（优先 `av_hwframe_map` 零拷贝，后端不支持
    /// map 时退回 download + upload）。
    ///
    /// 断言分两层：搬运结果的 `hw_frames_ctx` 必须**就是**编码器上下文持有的那个
    /// frames context；随后送 EOS 排空编码器并逐包计数——FFmpeg 会拒绝 frames
    /// context 与编码器不匹配的硬件帧，"能编码出包"因此反过来证明搬运确实发生了。
    /// 出包断言必须在 draining 后做：硬件编码器带 `AV_CODEC_CAP_DELAY`，EOS 前
    /// 可以合法地零输出（包在内部异步管线里），flush 前计数不稳定且与版本/负载有关。
    ///
    /// 无 GPU / 无对应硬件编码器的环境跳过（先探测再跳过，不把环境差异当失败）。
    #[test]
    fn test_encode_raw_accepts_foreign_hw_frame() -> Result<()> {
        // 本测试会创建并持有 `HWContext`（进程级缓存），必须与其它硬件测试串行，
        // 否则会破坏按引用计数断言的缓存释放测试。
        let _guard = crate::test_support::hw_cache_test_lock();

        let Some(config) = try_auto_hw_config() else {
            return Ok(());
        };
        let Some(codec_name) = hw_encoder_for(config.device_type) else {
            println!(
                "SKIP: {:?} has no matching hardware encoder",
                config.device_type
            );
            return Ok(());
        };
        let codec_c = std::ffi::CString::new(codec_name).expect("codec name is NUL-free");
        if AVCodec::find_encoder_by_name(&codec_c).is_none() {
            println!("SKIP: {codec_name} is not in this FFmpeg build");
            return Ok(());
        }

        let (width, height) = (64i32, 64i32);
        let hw_ctx = HWContext::new(config.clone()).context("hardware device must open")?;

        // 源帧来自**另一个** frames context（同一台设备）——正是解码器输出帧的样子。
        let mut src_frames = hw_ctx.create_hw_frames_ctx(width, height, 2)?;

        let mut encoder = EncoderBuilder::new_video(width, height)
            .with_codec_name(codec_name)
            .with_hardware_device(Some(config))
            .with_hw_pool_size(2)
            .build()?;

        let frame_count: i64 = 4;
        let mut packets = 0usize;
        for index in 0..frame_count {
            let mut src = AVFrame::new();
            src.set_width(width);
            src.set_height(height);
            src.set_format(hw_ctx.get_format(true));
            src_frames.get_buffer(&mut src)?;
            src.set_pts(index);

            // 映射进编码器自己的 frames context。
            let mapped = hw_ctx.map_hw_frame(&mut encoder.context, src)?;
            // 比对 AVHWFramesContext 对象本身（`AVBufferRef::data`）：map/upload 都会
            // 新建一个 AVBufferRef 指向同一个 frames context，结构体地址并不可比。
            let expected_data = {
                let frames = encoder
                    .context
                    .hw_frames_ctx()
                    .expect("encoder owns a frames context");
                unsafe { (*frames.as_ptr()).data as usize }
            };
            assert!(
                unsafe { (*mapped.hw_frames_ctx).data as usize } == expected_data,
                "frame {index}: mapped frame must live in the encoder's frames context"
            );

            packets += encoder.encode_raw(mapped)?.len();
        }

        // VideoToolbox 等硬件编码器带 `AV_CODEC_CAP_DELAY`：send/receive API 允许
        // 编码器在 EOS 前合法地零输出（包压在内部异步管线里），所以不能在 flush 前
        // 断言出包。送 NULL 进入 draining，把编码器排空后再断言"每帧一包"。
        encoder.send_frame_to_encoder(None)?;
        let mut eagain = 0;
        while !encoder.is_flushed() {
            match encoder.receive_packet()? {
                // 只统计包数：包本身用 `_` 匹配，就地释放。
                Some(_) => {
                    packets += 1;
                    eagain = 0;
                }
                None => {
                    // 与 `flush()` 同一防御：正常 draining 不会持续 EAGAIN，
                    // 持续不推进说明编码器异常，避免死循环。
                    eagain += 1;
                    assert!(
                        eagain < crate::MAX_DRAIN_ITERATIONS,
                        "hardware encoder did not reach EOF after draining"
                    );
                }
            }
        }
        assert_eq!(
            packets, frame_count as usize,
            "hardware frames must produce one packet per frame after flush"
        );
        Ok(())
    }

    /// `with_fps` is fail-fast: a non-positive or non-finite rate is rejected by
    /// `build()` rather than silently falling back to the default 30 fps (which
    /// would produce a stream at the wrong speed, the hardest kind of bug to
    /// notice).
    #[test]
    fn test_with_fps_rejects_invalid_rates() {
        for fps in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let err = match EncoderBuilder::new_video(64, 64).with_fps(fps).build() {
                Ok(_) => panic!("an invalid fps must not build"),
                Err(e) => e,
            };
            assert!(err.is_invalid_config(), "invalid fps {fps} gave: {err}");
        }

        // 合法帧率仍可构建（默认编码器缺席的环境下跳过：先探测可用性，
        // 免得把真正的配置错误当成环境差异）。
        if !default_video_encoder_available() {
            println!(
                "SKIP: {} is not in this FFmpeg build",
                EncoderBuilder::VIDEO_CODEC_NAME
            );
            return;
        }
        match EncoderBuilder::new_video(64, 64).with_fps(24.0).build() {
            Ok(_) => {}
            Err(e) => panic!("a valid fps must build: {e}"),
        }
    }

    /// `with_fps` 无法表达分母带 `1001` 的标准帧率：`24_000.0 / 1_001.0` 先被舍
    /// 入成 `f32`，`av_d2q` 再逼近**那个 f32**，得到的不是调用方写下的分数（实测
    /// `86002/3587`）。偏差只有 5.6e-7 fps，短片段看不出来，长片会累积；而
    /// `with_frame_rate` 走的是精确入口，原样使用调用方给的分数。
    ///
    /// 这个断言记录的是"浮点入口会逼近、精确入口不会"这一事实本身，所以它既
    /// 不依赖某个具体的错误有理数，也不会因为 FFmpeg 换了逼近算法而失效。
    #[test]
    fn test_with_fps_cannot_express_a_1001_denominator_rate() {
        const FILM_ON_NTSC: (i32, i32) = (24_000, 1_001);
        let exact = Rational::new(FILM_ON_NTSC.0, FILM_ON_NTSC.1).unwrap();

        let approx = EncoderBuilder::new_video(64, 64).with_fps(24_000.0 / 1_001.0);
        let approx = approx.frame_rate;
        assert_ne!(approx, exact, "`f32` 表示不出 24000/1001 这个分数");
        let error = (approx.as_f64() - exact.as_f64()).abs();
        assert!(
            error > 0.0 && error < 1e-5,
            "浮点入口给出的应当是「差之毫厘」的近似，实际 {approx}（偏差 {error}）"
        );

        let verbatim = EncoderBuilder::new_video(64, 64).with_frame_rate(exact);
        assert_eq!(verbatim.frame_rate, exact);
    }

    /// 精确入口的合法性判据只有一条：分子为正（分母非零由 [`Rational`] 保证）。
    /// 非法值同样在 `build()` 里 fail fast，而不是静默退回默认帧率。
    #[test]
    fn test_with_frame_rate_rejects_non_positive_numerator() {
        for numerator in [0, -1] {
            let rate = Rational::new(numerator, 1).unwrap();
            let err = match EncoderBuilder::new_video(64, 64)
                .with_frame_rate(rate)
                .build()
            {
                Ok(_) => panic!("a non-positive frame rate must not build"),
                Err(e) => e,
            };
            assert!(err.is_invalid_config(), "rate {rate} gave: {err}");
        }
    }

    /// 精确设置帧率后，编码器实际采用的就是那个有理数（不经过任何浮点往返）。
    #[test]
    fn test_with_frame_rate_reaches_the_encoder() -> Result<()> {
        if !default_video_encoder_available() {
            println!(
                "SKIP: {} is not in this FFmpeg build",
                EncoderBuilder::VIDEO_CODEC_NAME
            );
            return Ok(());
        }

        let rate = Rational::new(30_000, 1_001).unwrap();
        let encoder = EncoderBuilder::new_video(64, 64)
            .with_frame_rate(rate)
            .build()?;
        assert_eq!(encoder.frame_rate(), rate);
        assert_eq!(
            encoder.time_base(),
            Rational::new(1_001, 30_000).unwrap(),
            "编码器时间基应当是帧率的倒数"
        );
        Ok(())
    }

    /// 字幕编码器在打开时必须已有 ASS 脚本 header，否则 `build()` 报错
    /// （`ff_ass_split(NULL)` 会让 init 返回 `AVERROR_INVALIDDATA`）。
    #[test]
    fn test_subtitle_builder_requires_ass_header() {
        const ASS_HEADER: &str = "[Script Info]\n\
             ScriptType: v4.00+\n\
             \n\
             [V4+ Styles]\n\
             Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, \
             OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, \
             ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
             Alignment, MarginL, MarginR, MarginV, Encoding\n\
             Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n\
             \n\
             [Events]\n\
             Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

        // 缺 header：必须是 header 校验失败（编码器缺失时跳过，环境差异不算失败）。
        match EncoderBuilder::new_subtitle()
            .with_codec_name("mov_text")
            .build()
        {
            Err(e) if e.is_invalid_config() => {}
            Err(e) => {
                println!("SKIP: mov_text encoder unavailable in this build ({e})");
            }
            Ok(_) => panic!("subtitle encoder must not open without an ASS header"),
        }

        // 提供 header 后可正常构建。
        if let Err(e) = EncoderBuilder::new_subtitle()
            .with_codec_name("mov_text")
            .with_subtitle_header(ASS_HEADER)
            .build()
        {
            panic!("subtitle encoder must open once an ASS header is provided: {e}");
        }
    }

    // ====================================================================
    // 既有单元测试（原 `video_tests` 模块中不落盘的几条，其余已移到
    // `tests/encode_pipeline.rs`）
    // ====================================================================

    /// Quality::Bitrate 覆盖 with_bit_rate，并反映到 codecpar。
    #[test]
    fn test_quality_bitrate_applied() -> Result<()> {
        let encoder = EncoderBuilder::new_video(64, 64)
            .with_bit_rate(500_000)
            .with_quality(Quality::Bitrate(2_000_000))
            .build()?;
        assert_eq!(encoder.codecpar().bit_rate, 2_000_000);
        Ok(())
    }

    /// Crf 请求在不支持的编码器上回退为 bit_rate 控制（默认 1Mbps）。
    #[test]
    fn test_crf_fallback_bitrate() -> Result<()> {
        let encoder = EncoderBuilder::new_video(64, 64)
            .with_fps(25.0)
            .with_codec_name("mpeg4")
            .with_quality(Quality::Crf(20))
            .build()?;
        assert_eq!(encoder.codecpar().bit_rate, EncoderBuilder::VIDEO_BIT_RATE);
        Ok(())
    }

    /// 音频编码器忽略 Crf（视频概念），正常按 bit_rate 打开。
    #[test]
    fn test_audio_ignores_crf() -> Result<()> {
        let encoder = EncoderBuilder::new_audio(128_000, 2, 44100, SampleFormat::FLTP)
            .with_quality(Quality::Crf(20))
            .build()?;
        assert_eq!(encoder.codecpar().bit_rate, 128_000);
        Ok(())
    }

    /// builder 的缩放选项进入编码器持有的 [`Scaler`]：算法位一个、质量位可多个。
    #[test]
    fn test_builder_scale_options_reach_the_scaler() -> Result<()> {
        use crate::scale::{ScaleAlgorithm, ScaleQuality};

        // 多质量位（`|` 组合）+ 非默认算法。
        let encoder = EncoderBuilder::new_video(320, 240)
            .with_scale_algorithm(ScaleAlgorithm::LANCZOS)
            .with_scale_quality(
                ScaleQuality::FULL_CHR_H_INT | ScaleQuality::ACCURATE_RND | ScaleQuality::BITEXACT,
            )
            .build()?;
        assert_eq!(encoder.scaler.algorithm(), ScaleAlgorithm::LANCZOS);
        assert_eq!(encoder.scaler.quality(), ScaleQuality::default_mask());
        assert_eq!(
            encoder.scaler.flags(),
            ScaleAlgorithm::LANCZOS.as_raw() | ScaleQuality::default_mask().bits()
        );

        // 默认策略：BICUBIC + 默认质量掩码。
        let encoder = EncoderBuilder::new_video(320, 240).build()?;
        assert_eq!(encoder.scaler.algorithm(), ScaleAlgorithm::default());
        assert_eq!(encoder.scaler.quality(), ScaleQuality::default_mask());

        // 单个质量位、以及空集（= 无质量位）。
        let encoder = EncoderBuilder::new_video(320, 240)
            .with_scale_quality(ScaleQuality::BITEXACT)
            .build()?;
        assert_eq!(encoder.scaler.quality(), ScaleQuality::BITEXACT.into());
        let encoder = EncoderBuilder::new_video(320, 240)
            .with_scale_quality(FlagSet::EMPTY)
            .build()?;
        assert!(encoder.scaler.quality().is_empty());

        // 池化开关进入 Scaler：默认关闭，with_buffer_pool(true) 打开。
        assert!(!encoder.scaler.pool_enabled());
        let encoder = EncoderBuilder::new_video(320, 240)
            .with_buffer_pool(true)
            .build()?;
        assert!(encoder.scaler.pool_enabled());
        Ok(())
    }

    /// 排空失败后**必须能重试**：`Drained`（EOS 已送出、包还没排完）不等于"已经
    /// flush 过"。回归点：守卫曾经只看"状态不是 `Normal`"，于是第二次 `flush` 直接
    /// 返回空累积器，而 `Muxer::finish` 随后照样写 trailer ⇒ **截断的流以成功返回**。
    ///
    /// 用可切换失败的 writer 制造"排空循环中途报错"：第一次必然 `Err`；第二次仍必须
    /// `Err`（说明它真的又去排空了，而不是报告成功）；切回正常 writer 后必须把剩下的
    /// 包全部写出、进入 `Flushed`，此后再调用才允许幂等返回空。
    #[test]
    fn test_flush_retries_drain_after_a_failed_attempt() -> Result<()> {
        if !default_video_encoder_available() {
            println!(
                "SKIP: {} is not in this FFmpeg build",
                EncoderBuilder::VIDEO_CODEC_NAME
            );
            return Ok(());
        }

        let mut encoder = EncoderBuilder::new_video(64, 64)
            .with_fps(25.0)
            .with_pix_fmt(PixelFormat::YUV420P)
            .build()?;
        // 多喂几帧：EOS 时编码器内部还压着多个包（lookahead / B 帧重排序），
        // 这样"某次写失败"之后仍有包可排，重试才有可观测的效果。
        for index in 0..12i64 {
            let mut frame = AVFrame::new();
            frame.set_width(64);
            frame.set_height(64);
            frame.set_format(i32::from(PixelFormat::YUV420P));
            frame
                .alloc_buffer()
                .context("Failed to allocate test frame buffer")?;
            frame.set_pts(index);
            encoder.encode_raw(frame)?;
        }

        let out_time_base = Rational::new(1, 25).unwrap();
        let mut writer = FlakyWriter::new("matroska")?;
        writer.add_stream(encoder.codecpar(), out_time_base)?;
        writer.write_header()?;

        writer.fail = true;
        let first = encoder
            .flush(&mut writer, false, 0, out_time_base)
            .expect_err("writing must fail while the writer is switched to fail");
        assert!(
            !encoder.is_flushed(),
            "a failed drain must not look flushed: {first}"
        );

        encoder
            .flush(&mut writer, false, 0, out_time_base)
            .expect_err("the retry must drain again instead of reporting success");

        writer.fail = false;
        encoder.flush(&mut writer, false, 0, out_time_base)?;
        assert!(encoder.is_flushed(), "a complete drain must reach Flushed");
        assert!(
            writer.packets > 0,
            "the retry must write the packets that were still buffered"
        );
        writer.write_trailer()?;
        assert!(writer.bytes > 0, "the container must really have bytes");

        // 只有真正排空完成之后才是幂等的 no-op（返回值是 `()`，可用写入的包数没变来验收）
        let packets_before = writer.packets;
        encoder.flush(&mut writer, false, 0, out_time_base)?;
        assert_eq!(
            writer.packets, packets_before,
            "the idempotent retry must not write anything"
        );
        Ok(())
    }

    /// 包一层 [`BufferWriter`](crate::io::BufferWriter)，可切换"写包即失败"，
    /// 用来在排空循环中途制造一次 I/O 错误。
    ///
    /// 单独记成功写出的包数：muxer 会把数据攒在内部（header/簇/trailer 才吐出来），
    /// 所以"排空确实写了包"必须用包数而不是字节数验收。
    struct FlakyWriter {
        inner: crate::io::BufferWriter,
        fail: bool,
        bytes: usize,
        packets: usize,
    }

    impl FlakyWriter {
        fn new(format: &str) -> Result<Self> {
            Ok(Self {
                inner: crate::io::BufferWriter::new(format)?,
                fail: false,
                bytes: 0,
                packets: 0,
            })
        }

        /// 记账（拉模型：字节留在内层 writer，这里只累计本次新增的量）。
        fn count(&mut self) {
            self.bytes += self.inner.take_written().len();
            self.packets += 1;
        }
    }

    impl Writer for FlakyWriter {
        fn write_header(&mut self) -> Result<()> {
            self.inner.write_header()?;
            self.count();
            Ok(())
        }

        fn write_frame(&mut self, packet: &mut AVPacket) -> Result<()> {
            if self.fail {
                return Err(RsmediaError::msg("simulated writer failure"));
            }
            self.inner.write_frame(packet)?;
            self.count();
            Ok(())
        }

        fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<()> {
            if self.fail {
                return Err(RsmediaError::msg("simulated writer failure"));
            }
            self.inner.write_interleaved(packet)?;
            self.count();
            Ok(())
        }

        fn write_trailer(&mut self) -> Result<()> {
            self.inner.write_trailer()?;
            self.count();
            Ok(())
        }

        fn output(&self) -> &rsmpeg::avformat::AVFormatContextOutput {
            self.inner.output()
        }

        fn output_mut(&mut self) -> &mut rsmpeg::avformat::AVFormatContextOutput {
            self.inner.output_mut()
        }

        fn is_header_written(&self) -> bool {
            self.inner.is_header_written()
        }
    }
}
