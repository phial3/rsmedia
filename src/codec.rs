use anyhow::{Error, Result};
#[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::avcodec::{AVCodec, AVCodecRef};
use rsmpeg::ffi;
use std::ffi::CStr;

pub struct CodecConfig {
    codec: AVCodecRef<'static>,
    #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
    context: AVCodecContext,
}

impl CodecConfig {
    pub fn new(id: ffi::AVCodecID) -> Result<Self> {
        let codec = AVCodec::find_encoder(id)
            .or_else(|| AVCodec::find_decoder(id))
            .ok_or_else(|| Error::msg(format!("Codec id:{id} not found.")))?;
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
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
            .ok_or_else(|| Error::msg(format!("Codec not found by name: '{codec_name:?}'")))?;
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
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
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
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
        unsafe { ffi::av_codec_is_encoder(self.codec.as_ptr()) != 0 }
    }

    pub fn is_decoder(&self) -> bool {
        unsafe { ffi::av_codec_is_decoder(self.codec.as_ptr()) != 0 }
    }

    /// for audio codec, check if it supports variable frame size
    pub fn is_support_variable_frame_size(&self) -> bool {
        self.codec.capabilities & ffi::AV_CODEC_CAP_VARIABLE_FRAME_SIZE as i32 != 0
    }

    /// for codec, check if it supports delay
    pub fn is_support_delayed_frame(&self) -> bool {
        self.codec.capabilities & ffi::AV_CODEC_CAP_DELAY as i32 != 0
    }
}

impl CodecConfig {
    pub fn supported_pixel_formats(&self) -> Result<Option<&[ffi::AVPixelFormat]>> {
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
        {
            Ok(self.codec.pix_fmts())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let fmts = self.context.get_supported_pix_fmts(Some(&self.codec))?;
            // FFmpeg 约定：查询结果为 NULL 表示"支持所有值"，rsmpeg 将其映射为
            // 空切片；归一化为 None，与 FFmpeg 6 静态字段为 NULL 的语义一致。
            Ok(if fmts.is_empty() { None } else { Some(fmts) })
        }
    }

    pub fn supported_sample_formats(&self) -> Result<Option<&[ffi::AVSampleFormat]>> {
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
        {
            Ok(self.codec.sample_fmts())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let fmts = self.context.get_supported_sample_fmts(Some(&self.codec))?;
            // 同上：空列表（FFmpeg NULL）表示"支持所有值"，归一化为 None。
            Ok(if fmts.is_empty() { None } else { Some(fmts) })
        }
    }

    pub fn supported_frame_rates(&self) -> Result<Option<&[ffi::AVRational]>> {
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
        {
            Ok(self.codec.supported_framerates())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        unsafe {
            let rates: &[ffi::AVRational] = self
                .context
                .get_supported_config(Some(&self.codec), ffi::AV_CODEC_CONFIG_FRAME_RATE)?;
            // 空列表（FFmpeg NULL）表示"支持所有值"，归一化为 None。
            Ok(if rates.is_empty() { None } else { Some(rates) })
        }
    }

    pub fn supported_sample_rates(&self) -> Result<Option<&[i32]>> {
        #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
        {
            Ok(self.codec.supported_samplerates())
        }
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        unsafe {
            let rates: &[i32] = self
                .context
                .get_supported_config(Some(&self.codec), ffi::AV_CODEC_CONFIG_SAMPLE_RATE)?;
            // 空列表（FFmpeg NULL）表示"支持所有值"，归一化为 None。
            Ok(if rates.is_empty() { None } else { Some(rates) })
        }
    }

    ///////////////
    ///////////////

    /// 查询结果为 `None`（FFmpeg 未限制，支持所有值）或查询失败（如媒体类型
    /// 不匹配的配置项）时按"支持"处理，避免误拦合法帧。
    pub(crate) fn is_support_pixel_format(&self, pix_fmt: i32) -> bool {
        match self.supported_pixel_formats() {
            Ok(None) | Err(_) => true,
            Ok(Some(formats)) => formats.contains(&pix_fmt),
        }
    }

    pub(crate) fn is_support_sample_format(&self, sample_fmt: i32) -> bool {
        match self.supported_sample_formats() {
            Ok(None) | Err(_) => true,
            Ok(Some(formats)) => formats.contains(&sample_fmt),
        }
    }

    /// 注意：对音频编码器查询帧率会得到 EINVAL（音频无帧率概念），
    /// 此时按"支持"处理。
    pub(crate) fn is_support_frame_rates(&self, frame_rate: ffi::AVRational) -> bool {
        match self.supported_frame_rates() {
            Ok(None) | Err(_) => true,
            Ok(Some(rates)) => rates
                .iter()
                .any(|r| r.num == frame_rate.num && r.den == frame_rate.den),
        }
    }

    pub(crate) fn is_support_sample_rate(&self, sample_rate: i32) -> bool {
        match self.supported_sample_rates() {
            Ok(None) | Err(_) => true,
            Ok(Some(rates)) => rates.contains(&sample_rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 断言视频编码/解码器支持的非空像素格式列表非空。
    /// 若 FFmpeg 返回 `None`（表示"所有值均支持"）则视为通过。
    fn assert_video_config(config: &CodecConfig, name: &str) {
        let pix = config
            .supported_pixel_formats()
            .unwrap_or_else(|e| panic!("{name}: query pixel formats failed: {e}"));
        assert!(
            pix.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: expected non-empty supported pixel formats"
        );
    }

    /// 断言音频编码/解码器支持的采样率、采样格式列表非空。
    fn assert_audio_config(config: &CodecConfig, name: &str) {
        let rates = config
            .supported_sample_rates()
            .unwrap_or_else(|e| panic!("{name}: query sample rates failed: {e}"));
        assert!(
            rates.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: sample rates should be non-empty if specified"
        );

        let fmts = config
            .supported_sample_formats()
            .unwrap_or_else(|e| panic!("{name}: query sample formats failed: {e}"));
        assert!(
            fmts.map(|v| !v.is_empty()).unwrap_or(true),
            "{name}: expected non-empty supported sample formats"
        );
    }

    #[test]
    fn test_supported_video_codec() {
        for id in [
            ffi::AV_CODEC_ID_H264,
            ffi::AV_CODEC_ID_MPEG4,
            ffi::AV_CODEC_ID_VP8,
            ffi::AV_CODEC_ID_VP9,
            ffi::AV_CODEC_ID_HEVC,
            ffi::AV_CODEC_ID_AV1,
        ] {
            let config = CodecConfig::new(id).unwrap();
            assert_video_config(&config, &format!("video codec {id}"));
            assert!(
                config.is_encoder() || config.is_decoder(),
                "video codec {id} should be encoder or decoder"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_supported_video_codec_name() {
        for name in [
            c"libx264",
            c"libx265",
            c"mpeg4",
            c"mpeg1video",
            c"mpeg2video",
        ] {
            let config = CodecConfig::new_with_name(name)
                .unwrap_or_else(|e| panic!("could not find codec {name:?}: {e}"));
            assert_video_config(&config, &format!("video codec {name:?}"));
        }
    }

    #[test]
    fn test_supported_audio_codec() {
        for id in [
            ffi::AV_CODEC_ID_AAC,
            ffi::AV_CODEC_ID_FLAC,
            ffi::AV_CODEC_ID_MP3,
            ffi::AV_CODEC_ID_OPUS,
            ffi::AV_CODEC_ID_VORBIS,
        ] {
            let config = CodecConfig::new(id).unwrap();
            assert_audio_config(&config, &format!("audio codec {id}"));
            assert!(
                config.is_encoder() || config.is_decoder(),
                "audio codec {id} should be encoder or decoder"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    #[ignore = "skip test_supported_audio_codec_name"]
    fn test_supported_audio_codec_name() {
        for name in [c"libmp3lame", c"libopus", c"libvorbis"] {
            let config = CodecConfig::new_with_name(name)
                .unwrap_or_else(|e| panic!("could not find codec {name:?}: {e}"));
            assert_audio_config(&config, &format!("audio codec {name:?}"));
        }
    }
}
