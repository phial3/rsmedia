use crate::error::{Result, RsmediaError};
use crate::stream::MediaType;
use crate::strutils;
use crate::time::Rational;

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecRef};
use rsmpeg::ffi;

use std::ffi::CStr;
use std::fmt;

/// 设置 `AVCodecContext::thread_count`。
///
/// rsmpeg 的 `settable!` 字段表不含 `thread_count`，只能直接写字段；把这个
/// unsafe 收敛在这里，`Encoder`/`Decoder` 都不再自己碰裸指针。
///
/// **不做任何取值判断**：值原样写入，包括 `0`（FFmpeg 的"自行推导"语义）与负数。
///
/// 调用方一侧的约定才是"用谁的"：只有用户**显式**调过 `with_thread_count` 才
/// 写这个字段；没设置就完全不碰它，让 FFmpeg 用自己的默认值。
pub(crate) fn set_thread_count(context: &mut AVCodecContext, thread_count: i32) {
    // SAFETY: `context` 由 `AVCodecContext::new` 分配、在借用期内一直有效
    unsafe {
        context.deref_mut().thread_count = thread_count;
    }
}

/// 设置 `AVCodecContext::thread_type`（多线程的并行方式，取值见 [`ThreadType`]）。
///
/// 与 [`set_thread_count`] 同因：`settable!` 字段表不含 `thread_type`。必须在
/// `avcodec_open2` 之前写入才会生效（`avcodec_open2` 会按它选择线程实现）。
/// `thread_type` 是 `FF_THREAD_*` 的位掩码（`FRAME | SLICE` 合法）。
pub(crate) fn set_thread_type(context: &mut AVCodecContext, thread_type: i32) {
    // SAFETY: 同 `set_thread_count`；`FF_THREAD_*` 各占一位，远在 `i32` 范围内。
    unsafe {
        context.deref_mut().thread_type = thread_type;
    }
}

/// 设置 `AVCodecContext::flags2`（取值见 [`AVCodecFlag2`]）。
///
/// 与 [`set_thread_count`] 同因：`settable!` 只暴露了 `flags`，没有 `flags2`。
/// 同样必须在 `avcodec_open2` 之前写入。`flags` 走 rsmpeg 的
/// [`set_flags`](AVCodecContext::set_flags)，本函数只补它缺的那半边。
pub(crate) fn set_flags2(context: &mut AVCodecContext, flags2: i32) {
    // SAFETY: 同 `set_thread_count`；`AV_CODEC_FLAG2_*` 是 `int` 位集。
    unsafe {
        context.deref_mut().flags2 = flags2;
    }
}

ffi_enum!(
    /// 对应 FFmpeg `FF_THREAD_*`，即 `AVCodecContext.thread_type` 的位集。
    ///
    /// 选择多线程的并行粒度：`FRAME` 帧级并行（每个线程缓冲一帧，解码延迟增加
    /// 一帧/线程）、`SLICE` 片级并行（延迟低，但需要编解码器支持切片）。两者
    /// 可组合：`ThreadType::FRAME | ThreadType::SLICE`。
    ///
    /// 未显式设置时保持 FFmpeg 默认（`avcodec_open2` 自行选择），rsmedia 不干预。
    #[allow(non_camel_case_types)]
    ThreadType, u32 {
        FRAME => ffi::FF_THREAD_FRAME;
        SLICE => ffi::FF_THREAD_SLICE;
    }
);

ffi_enum!(
    /// 对应 FFmpeg `AV_CODEC_FLAG_*`
    #[allow(non_camel_case_types)]
    AVCodecFlag, u32 {
    UNALIGNED => ffi::AV_CODEC_FLAG_UNALIGNED;
    QSCALE => ffi::AV_CODEC_FLAG_QSCALE;
    X4MV => ffi::AV_CODEC_FLAG_4MV;
    OUTPUT_CORRUPT => ffi::AV_CODEC_FLAG_OUTPUT_CORRUPT;
    QPEL => ffi::AV_CODEC_FLAG_QPEL;
    RECON_FRAME => ffi::AV_CODEC_FLAG_RECON_FRAME;
    COPY_OPAQUE => ffi::AV_CODEC_FLAG_COPY_OPAQUE;
    FRAME_DURATION => ffi::AV_CODEC_FLAG_FRAME_DURATION;
    PASS1 => ffi::AV_CODEC_FLAG_PASS1;
    PASS2 => ffi::AV_CODEC_FLAG_PASS2;
    LOOP_FILTER => ffi::AV_CODEC_FLAG_LOOP_FILTER;
    GRAY => ffi::AV_CODEC_FLAG_GRAY;
    PSNR => ffi::AV_CODEC_FLAG_PSNR;
    INTERLACED_DCT => ffi::AV_CODEC_FLAG_INTERLACED_DCT;
    LOW_DELAY => ffi::AV_CODEC_FLAG_LOW_DELAY;
    GLOBAL_HEADER => ffi::AV_CODEC_FLAG_GLOBAL_HEADER;
    BITEXACT => ffi::AV_CODEC_FLAG_BITEXACT;
    AC_PRED => ffi::AV_CODEC_FLAG_AC_PRED;
    INTERLACED_ME => ffi::AV_CODEC_FLAG_INTERLACED_ME;
    CLOSED_GOP => ffi::AV_CODEC_FLAG_CLOSED_GOP;
});

ffi_enum!(
    /// 对应 FFmpeg `AV_CODEC_FLAG2_*`
    #[allow(non_camel_case_types)]
    AVCodecFlag2, u32 {
    FAST => ffi::AV_CODEC_FLAG2_FAST;
    NO_OUTPUT => ffi::AV_CODEC_FLAG2_NO_OUTPUT;
    LOCAL_HEADER => ffi::AV_CODEC_FLAG2_LOCAL_HEADER;
    CHUNKS => ffi::AV_CODEC_FLAG2_CHUNKS;
    IGNORE_CROP => ffi::AV_CODEC_FLAG2_IGNORE_CROP;
    #[cfg(feature = "ffmpeg9")]
    FIXED_FRAME_SIZE => ffi::AV_CODEC_FLAG2_FIXED_FRAME_SIZE;
    SHOW_ALL => ffi::AV_CODEC_FLAG2_SHOW_ALL;
    EXPORT_MVS => ffi::AV_CODEC_FLAG2_EXPORT_MVS;
    SKIP_MANUAL => ffi::AV_CODEC_FLAG2_SKIP_MANUAL;
    RO_FLUSH_NOOP => ffi::AV_CODEC_FLAG2_RO_FLUSH_NOOP;
    ICC_PROFILES => ffi::AV_CODEC_FLAG2_ICC_PROFILES;
});

pub struct CodecConfig {
    codec: AVCodecRef<'static>,
    #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
    context: AVCodecContext,
}

impl CodecConfig {
    pub fn new(id: ffi::AVCodecID) -> Result<Self> {
        let codec = AVCodec::find_encoder(id)
            .or_else(|| AVCodec::find_decoder(id))
            .ok_or_else(|| {
                RsmediaError::unsupported(format!(
                    "codec '{id}' is not available in this FFmpeg build"
                ))
            })?;
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(Self { codec })
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let context = AVCodecContext::new(&codec);
            Ok(Self { codec, context })
        }
    }

    pub fn new_with_name(codec_name: &CStr) -> Result<Self> {
        let codec = AVCodec::find_encoder_by_name(codec_name)
            .or_else(|| AVCodec::find_decoder_by_name(codec_name))
            .ok_or_else(|| {
                RsmediaError::unsupported(format!(
                    "codec '{}' is not available in this FFmpeg build",
                    codec_name.to_string_lossy()
                ))
            })?;
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(Self { codec })
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let context = AVCodecContext::new(&codec);
            Ok(Self { codec, context })
        }
    }

    pub fn from_codec(codec: AVCodecRef<'static>) -> Self {
        #[cfg(feature = "ffmpeg6")]
        {
            Self { codec }
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let context = AVCodecContext::new(&codec);
            Self { codec, context }
        }
    }

    pub fn id(&self) -> ffi::AVCodecID {
        self.codec.id
    }

    pub fn name(&self) -> &CStr {
        self.codec.name()
    }

    pub fn long_name(&self) -> &CStr {
        self.codec.long_name()
    }

    pub fn is_encoder(&self) -> bool {
        // SAFETY: `self.codec` holds a live `AVCodecRef` for as long as `self`
        // exists, so the pointer stays valid; `av_codec_is_encoder` only reads
        // the codec's descriptor.
        unsafe { ffi::av_codec_is_encoder(self.codec.as_ptr()) != 0 }
    }

    pub fn is_decoder(&self) -> bool {
        // SAFETY: same as `is_encoder` — a pure read of a live descriptor.
        unsafe { ffi::av_codec_is_decoder(self.codec.as_ptr()) != 0 }
    }

    /// `AV_CODEC_CAP_VARIABLE_FRAME_SIZE`: an audio encoder may be handed any
    /// frame size, rather than multiples of one fixed value.
    pub fn supports_variable_frame_size(&self) -> bool {
        self.codec.capabilities & ffi::AV_CODEC_CAP_VARIABLE_FRAME_SIZE as i32 != 0
    }

    /// `AV_CODEC_CAP_DELAY`: the codec buffers input and emits packets only
    /// later, so the encoder must be flushed (and the decoder drained) to get the
    /// tail out.
    pub fn supports_delay(&self) -> bool {
        self.codec.capabilities & ffi::AV_CODEC_CAP_DELAY as i32 != 0
    }
}

impl CodecConfig {
    pub fn supported_pixel_formats(&self) -> Result<Option<&[ffi::AVPixelFormat]>> {
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(self.codec.pix_fmts())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let fmts = self.context.get_supported_pix_fmts(Some(&self.codec))?;
            Ok(if fmts.is_empty() { None } else { Some(fmts) })
        }
    }

    pub fn supported_sample_formats(&self) -> Result<Option<&[ffi::AVSampleFormat]>> {
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(self.codec.sample_fmts())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let fmts = self.context.get_supported_sample_fmts(Some(&self.codec))?;
            Ok(if fmts.is_empty() { None } else { Some(fmts) })
        }
    }

    /// Lists the frame rates the codec declares support for.
    ///
    /// `None` means "no restriction" — that is how FFmpeg spells an empty
    /// list. Unlike the other `supported_*` accessors this returns an owned
    /// [`Vec`]: FFmpeg hands out an `AVRational` array, and [`Rational`] has a
    /// different layout, so the values cannot be re-borrowed as a slice.
    ///
    /// # Errors
    ///
    /// Returns an error if FFmpeg fails to query the codec's configuration.
    pub fn supported_frame_rates(&self) -> Result<Option<Vec<Rational>>> {
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(self
                .codec
                .supported_framerates()
                .map(|rates| rates.iter().copied().map(Rational::from).collect()))
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let rates = self.context.get_supported_frame_rates(Some(&self.codec))?;
            Ok(if rates.is_empty() {
                None
            } else {
                Some(rates.iter().copied().map(Rational::from).collect())
            })
        }
    }

    pub fn supported_sample_rates(&self) -> Result<Option<&[i32]>> {
        #[cfg(feature = "ffmpeg6")]
        {
            Ok(self.codec.supported_samplerates())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let rates = self.context.get_supported_sample_rates(Some(&self.codec))?;
            Ok(if rates.is_empty() { None } else { Some(rates) })
        }
    }

    fn supported_channel_counts(&self) -> Option<Vec<i32>> {
        let layouts: &[ffi::AVChannelLayout] = {
            #[cfg(feature = "ffmpeg6")]
            {
                // ffmpeg6 无 `AV_CODEC_CONFIG_CHANNEL_LAYOUT` 能力接口，改用旧式
                // `AVCodec.ch_layouts` 字段（`*const AVChannelLayout`，以 zeroed layout 结尾）。
                // `build_array` 依赖字节相等性判断终止，zeroed layout 即终止哨兵。
                // SAFETY: `self.codec.ch_layouts` is FFmpeg's own zero-terminated static array;
                // `build_array` reads until the zeroed sentinel and copies the entries out.
                unsafe {
                    rsmpeg::build_array::<ffi::AVChannelLayout>(
                        self.codec.ch_layouts,
                        std::mem::zeroed(),
                    )?
                }
            }
            #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
            {
                // SAFETY: rsmpeg marks `get_supported_config` unsafe because the
                // caller must match the type parameter to the config id — the
                // request below is `AV_CODEC_CONFIG_CHANNEL_LAYOUT`, and
                // AVChannelLayout is exactly the type FFmpeg fills for it. The
                // returned buffer is FFmpeg-allocated and owned by the caller;
                // `self.codec` is borrowed for the duration of the call.
                unsafe {
                    self.context.get_supported_config::<ffi::AVChannelLayout>(
                        Some(&self.codec),
                        ffi::AV_CODEC_CONFIG_CHANNEL_LAYOUT,
                    )
                }
                .ok()?
            }
        };
        let counts: Vec<i32> = layouts
            .iter()
            .map(|l| l.nb_channels)
            .filter(|n| *n > 0)
            .collect();
        if counts.is_empty() {
            None
        } else {
            Some(counts)
        }
    }

    ///////////////
    ///////////////

    /// 查询结果为 `None`（FFmpeg 未限制，支持所有值）或查询失败（如媒体类型
    /// 不匹配的配置项）时按"支持"处理，避免误拦合法帧。
    pub(crate) fn supports_pixel_format(&self, pix_fmt: i32) -> bool {
        match self.supported_pixel_formats() {
            Ok(None) | Err(_) => true,
            Ok(Some(formats)) => formats.contains(&pix_fmt),
        }
    }

    pub(crate) fn supports_sample_format(&self, sample_fmt: i32) -> bool {
        match self.supported_sample_formats() {
            Ok(None) | Err(_) => true,
            Ok(Some(formats)) => formats.contains(&sample_fmt),
        }
    }

    pub(crate) fn supports_sample_rate(&self, sample_rate: i32) -> bool {
        match self.supported_sample_rates() {
            Ok(None) | Err(_) => true,
            Ok(Some(rates)) => rates.contains(&sample_rate),
        }
    }

    /// 编码器是否支持指定声道数；未声明限制或查询失败时按"支持"处理，
    /// 与 [`CodecConfig::supports_sample_rate`] 语义一致。
    pub(crate) fn supports_channel_count(&self, nb_channels: i32) -> bool {
        match self.supported_channel_counts() {
            None => true,
            Some(counts) => counts.contains(&nb_channels),
        }
    }
}

impl CodecConfig {
    /// Media type (video/audio/subtitle/...) this codec handles.
    pub fn media_type(&self) -> MediaType {
        MediaType::from(self.codec.type_)
    }

    /// Whether this is a hardware-accelerated implementation
    /// (e.g. `h264_nvenc`, `h264_vaapi`, `h264_videotoolbox`).
    pub fn is_hardware(&self) -> bool {
        self.codec.capabilities & ffi::AV_CODEC_CAP_HARDWARE as i32 != 0
    }

    /// Profiles the codec supports (e.g. High/Main/Baseline for H.264).
    /// Empty when the codec does not declare a profile list.
    ///
    /// Note: FFmpeg 9 leaves `AVCodec.profiles` NULL for FFCodec-based
    /// codecs (most encoders), so this may be empty there — use
    /// [`CodecConfig::profile_name`] to resolve a known profile id instead.
    pub fn profiles(&self) -> Vec<Profile> {
        let mut out = Vec::new();
        // SAFETY: `self.codec.profiles` is FFmpeg's static, `AV_PROFILE_UNKNOWN`-terminated
        // array (or NULL, checked below); the loop stops at the sentinel and only reads each
        // entry, copying the name out via `c_char_to_str`.
        unsafe {
            let mut p = self.codec.profiles;
            if p.is_null() {
                return out;
            }
            while (*p).profile != ffi::AV_PROFILE_UNKNOWN {
                let name = strutils::c_char_to_str((*p).name);
                out.push(Profile {
                    id: (*p).profile,
                    name,
                });
                p = p.add(1);
            }
        }
        out
    }

    /// Resolve the human readable name of a profile id via
    /// `avcodec_profile_name`, e.g. `FF_PROFILE_H264_HIGH` -> "High".
    /// Works on all supported FFmpeg versions, even when [`CodecConfig::profiles`]
    /// returns an empty list.
    pub fn profile_name(&self, profile_id: i32) -> Option<String> {
        // SAFETY: `avcodec_profile_name` returns FFmpeg's static string (or NULL);
        // `c_char_to_str` handles NULL and copies the contents out immediately.
        let pname = unsafe { ffi::avcodec_profile_name(self.codec.id, profile_id) };
        unsafe { Some(strutils::c_char_to_str(pname)) }
    }

    /// All codecs registered in this FFmpeg build (encoders and decoders).
    pub fn all() -> Vec<Self> {
        AVCodec::iterate().map(Self::from_codec).collect()
    }

    /// All registered encoder implementations.
    pub fn encoders() -> Vec<Self> {
        Self::all().into_iter().filter(|c| c.is_encoder()).collect()
    }

    /// All registered decoder implementations.
    pub fn decoders() -> Vec<Self> {
        Self::all().into_iter().filter(|c| c.is_decoder()).collect()
    }

    /// All encoder implementations for one codec id — e.g. for
    /// `AV_CODEC_ID_H264` this typically lists `libx264`, `h264_nvenc`,
    /// `h264_videotoolbox`, ... depending on the FFmpeg build and platform.
    pub fn encoders_for(id: ffi::AVCodecID) -> Vec<Self> {
        Self::encoders()
            .into_iter()
            .filter(|c| c.id() == id)
            .collect()
    }

    /// All decoder implementations for one codec id.
    pub fn decoders_for(id: ffi::AVCodecID) -> Vec<Self> {
        Self::decoders()
            .into_iter()
            .filter(|c| c.id() == id)
            .collect()
    }
}

/// One entry of a codec's profile list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    /// FFmpeg profile constant (e.g. `FF_PROFILE_H264_HIGH`).
    pub id: i32,
    /// Human readable profile name (e.g. "High").
    pub name: String,
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name.is_empty() {
            write!(f, "{}", self.id)
        } else {
            write!(f, "{}", self.name)
        }
    }
}

/// Owned summary of a muxer/demuxer container format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatInfo {
    /// Short name list (comma-separated aliases), e.g. "matroska,webm".
    pub name: String,
    /// Human readable description, e.g. "Matroska".
    pub long_name: String,
    /// Typical file extensions, e.g. ["mkv"].
    pub extensions: Vec<String>,
}

impl FormatInfo {
    /// Primary short name (the first alias of [`FormatInfo::name`]),
    /// e.g. "matroska" for the Matroska demuxer whose name list is
    /// "matroska,webm".
    pub fn short_name(&self) -> &str {
        self.name.split(',').next().unwrap_or(&self.name)
    }

    fn new_info(
        name: *const std::os::raw::c_char,
        long_name: *const std::os::raw::c_char,
        extensions: *const std::os::raw::c_char,
    ) -> Self {
        // SAFETY: the three pointers come from FFmpeg's static codec tables (NULL or
        // NUL-terminated); `c_char_to_str`/`c_char_to_str_list` accept NULL and copy out at once.
        Self {
            name: unsafe { strutils::c_char_to_str(name) },
            long_name: unsafe { strutils::c_char_to_str(long_name) },
            extensions: unsafe { strutils::c_char_to_str_list(extensions) },
        }
    }

    /// All muxers (output container formats) in this FFmpeg build.
    pub fn muxers() -> Vec<Self> {
        rsmpeg::avformat::AVOutputFormat::iterate()
            .map(|outfmt| Self::new_info(outfmt.name, outfmt.long_name, outfmt.extensions))
            .collect()
    }

    /// All demuxers (input container formats) in this FFmpeg build.
    pub fn demuxers() -> Vec<Self> {
        rsmpeg::avformat::AVInputFormat::iterate()
            .map(|infmt| Self::new_info(infmt.name, infmt.long_name, infmt.extensions))
            .collect()
    }

    /// Whether a short name (or one of its aliases) matches `short_name`.
    ///
    /// A format's `name` field is a comma-separated alias list (e.g.
    /// `"matroska,webm"`), so a lookup must also check each alias.
    fn name_matches(name: &str, short_name: &str) -> bool {
        name == short_name || name.split(',').any(|alias| alias.trim() == short_name)
    }

    /// Look up a muxer by its short name (or one of its aliases),
    /// e.g. "mp4", "mkv", "matroska".
    pub fn find_muxer(short_name: &str) -> Option<Self> {
        Self::muxers()
            .into_iter()
            .find(|f| Self::name_matches(&f.name, short_name))
    }

    /// Look up a demuxer by its short name (or one of its aliases).
    pub fn find_demuxer(short_name: &str) -> Option<Self> {
        Self::demuxers()
            .into_iter()
            .find(|f| Self::name_matches(&f.name, short_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 断言视频编码/解码器支持的非空像素格式列表非空。
    /// 若 FFmpeg 返回 `None`（表示"所有值均支持"）则视为通过。
    fn assert_video_config(config: &CodecConfig, name: &str) -> Result<()> {
        let pix = config.supported_pixel_formats()?;
        assert!(
            pix.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: expected non-empty supported pixel formats"
        );
        Ok(())
    }

    /// 断言音频编码/解码器支持的采样率、采样格式列表非空。
    fn assert_audio_config(config: &CodecConfig, name: &str) -> Result<()> {
        let rates = config.supported_sample_rates()?;
        assert!(
            rates.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: sample rates should be non-empty if specified"
        );

        let fmts = config.supported_sample_formats()?;
        assert!(
            fmts.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: expected non-empty supported sample formats"
        );
        Ok(())
    }

    #[test]
    fn test_supported_video_codec() -> Result<()> {
        for id in [
            ffi::AV_CODEC_ID_H264,
            ffi::AV_CODEC_ID_MPEG4,
            ffi::AV_CODEC_ID_VP8,
            ffi::AV_CODEC_ID_VP9,
            ffi::AV_CODEC_ID_HEVC,
            ffi::AV_CODEC_ID_AV1,
        ] {
            let config = CodecConfig::new(id)?;
            assert_video_config(&config, &format!("video codec {id}"))?;
            assert!(
                config.is_encoder() || config.is_decoder(),
                "video codec {id} should be encoder or decoder"
            );
        }
        Ok(())
    }

    #[test]
    fn test_supported_video_codec_name() -> Result<()> {
        for name in [
            c"libx264",
            c"libx265",
            c"mpeg4",
            c"mpeg1video",
            c"mpeg2video",
        ] {
            let config = CodecConfig::new_with_name(name)?;
            assert_video_config(&config, &format!("video codec {name:?}"))?;
        }
        Ok(())
    }

    #[test]
    fn test_supported_audio_codec() -> Result<()> {
        for id in [
            ffi::AV_CODEC_ID_AAC,
            ffi::AV_CODEC_ID_FLAC,
            ffi::AV_CODEC_ID_MP3,
            ffi::AV_CODEC_ID_OPUS,
            ffi::AV_CODEC_ID_VORBIS,
        ] {
            let config = CodecConfig::new(id)?;
            assert_audio_config(&config, &format!("audio codec {id}"))?;
            assert!(
                config.is_encoder() || config.is_decoder(),
                "audio codec {id} should be encoder or decoder"
            );
        }
        Ok(())
    }

    #[test]
    fn test_supported_audio_codec_name() -> Result<()> {
        // 外部编码器是否存在取决于 FFmpeg 编译配置（如 libvorbis 并非总是启用），
        // 缺失时跳过该编码器；但至少要有一个可用，否则视为构建异常。
        let mut available = 0;
        for name in [c"libmp3lame", c"libopus", c"libvorbis"] {
            if AVCodec::find_encoder_by_name(name).is_none() {
                println!("skip codec {name:?}: not available in this FFmpeg build");
                continue;
            }
            let config = CodecConfig::new_with_name(name)?;
            assert_audio_config(&config, &format!("audio codec {name:?}"))?;
            available += 1;
        }
        assert!(available > 0, "no external audio codecs available");
        Ok(())
    }

    #[test]
    fn test_codec_discovery() {
        let all = CodecConfig::all();
        assert!(!all.is_empty(), "no codecs registered");

        let encoders = CodecConfig::encoders();
        let decoders = CodecConfig::decoders();
        assert!(!encoders.is_empty() && !decoders.is_empty());
        assert!(encoders.iter().all(|c| c.is_encoder()));
        assert!(decoders.iter().all(|c| c.is_decoder()));

        // h264 decoder is available in every FFmpeg build.
        let h264 = CodecConfig::decoders_for(ffi::AV_CODEC_ID_H264);
        assert!(!h264.is_empty(), "h264 decoder missing");
        assert_eq!(h264[0].media_type(), MediaType::VIDEO);
    }

    #[test]
    fn test_audio_supported_capabilities() {
        // AAC 是最常见的软件音频编码器：断言其声道数/采样率能力可被
        // supports_channel_count / supports_sample_rate 识别，且对
        // 非法值返回 false（证明校验并非恒真 no-op）。
        let Some(config) = AVCodec::find_encoder_by_name(c"aac") else {
            eprintln!("aac encoder not available, skipping");
            return;
        };
        let config = CodecConfig::from_codec(config);
        // 常见合法组合必须被认定为支持。
        assert!(
            config.supports_channel_count(2),
            "aac should support stereo (2 channels)"
        );
        assert!(
            config.supports_sample_rate(44100),
            "aac should support 44100 Hz"
        );
        assert!(
            !config.supports_sample_rate(-1),
            "aac must not report support for negative sample rate"
        );
    }

    #[test]
    fn test_encoders_for_h264() {
        let encoders = CodecConfig::encoders_for(ffi::AV_CODEC_ID_H264);
        assert!(!encoders.is_empty(), "no h264 encoder in this FFmpeg build");
        // Software implementations are not flagged as hardware.
        if let Some(sw) = encoders.iter().find(|c| c.name() == c"libx264") {
            assert!(!sw.is_hardware(), "libx264 must not be hardware");
        }
        let names: Vec<_> = encoders
            .iter()
            .map(|c| c.name().to_string_lossy().into_owned())
            .collect();
        println!("h264 encoders: {names:?}");
    }

    #[test]
    fn test_format_discovery() {
        // 显式注册 libavdevice，让设备格式进入 `muxers()`/`demuxers()` 的迭代
        // 范围：其中有 `long_name == NULL` 的项（`NULL_IF_CONFIG_SMALL` 在
        // `CONFIG_SMALL` 构建下就是 NULL，实测本机构建的 `libcdio` 解复用器
        // 如此）。不显式注册的话，这条判空路径要靠其它用例先调 `init()` 才会
        // 被覆盖，且并行执行时表现为 SEGV 偶发——本用例把它变成确定性覆盖。
        crate::init::init().expect("rsmedia init failed");
        let muxers = FormatInfo::muxers();
        let demuxers = FormatInfo::demuxers();
        assert!(!muxers.is_empty() && !demuxers.is_empty());

        assert!(
            muxers.iter().any(|f| f.name == "matroska"),
            "matroska muxer missing"
        );
        assert!(muxers.iter().any(|f| f.name == "mp4"), "mp4 muxer missing");
        // The mov demuxer's short-name field is the full alias list
        // ("mov,mp4,m4a,3gp,3g2,mj2"), so look it up by alias.
        assert!(
            FormatInfo::find_demuxer("mov").is_some(),
            "mov demuxer missing"
        );
        assert!(
            FormatInfo::find_demuxer("matroska").is_some(),
            "matroska demuxer missing"
        );

        let mp4 = FormatInfo::find_muxer("mp4").expect("mp4 muxer lookup failed");
        assert_eq!(mp4.name, "mp4");
        assert!(mp4.extensions.contains(&"mp4".to_string()));

        // "mkv" is only an output alias; the demuxer is registered as
        // "matroska" (with the alias list "matroska,webm").
        let mkv = FormatInfo::find_demuxer("matroska").expect("matroska demuxer lookup failed");
        assert_eq!(mkv.short_name(), "matroska");
        assert!(mkv.extensions.contains(&"mkv".to_string()));
    }

    #[test]
    fn test_libx264_profiles() {
        let config = CodecConfig::new_with_name(c"libx264").expect("libx264 missing");
        let profiles = config.profiles();
        let names: Vec<_> = profiles.iter().map(|p| p.name.to_lowercase()).collect();
        if profiles.is_empty() {
            // FFmpeg 9: AVCodec.profiles is NULL for FFCodec-based encoders;
            // fall back to the profile-id lookup which works everywhere.
            let high = config
                .profile_name(ffi::AV_PROFILE_H264_HIGH as i32)
                .expect("h264 High profile name lookup failed");
            assert_eq!(high, "High");
        } else {
            assert!(names.contains(&"high".to_string()), "profiles: {names:?}");
            assert!(
                names.contains(&"baseline".to_string()),
                "profiles: {names:?}"
            );
        }
        assert_eq!(config.media_type(), MediaType::VIDEO);
        assert!(!config.is_hardware());
    }
}
