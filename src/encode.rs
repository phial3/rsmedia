use crate::codec::CodecConfig;
use crate::filter::{AudioParams, Filter, FilterGraph, FilterParams, VideoParams};
use crate::flags::AvFormatFlags;
#[cfg(feature = "ndarray")]
use crate::frame::{MediaFrame, MediaFrameType};
use crate::hwaccel::{HWContext, HWDeviceConfig};
use crate::io::Writer;
use crate::options::Options;
use crate::pixel::PixelFormat;
use crate::stream::StreamInfo;
use crate::swctx::ScaleAlgorithm;
use crate::{swctx, time, utils, Location, MediaType, SampleFormat, StreamWriter};

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParameters, AVPacket};
use rsmpeg::avutil::{self, AVChannelLayout, AVChannelLayoutRef, AVFrame};
use rsmpeg::ffi;

use anyhow::{Context, Error, Result};
use std::sync::Arc;

/// Builds an [`Encoder`].
#[derive(Debug)]
pub struct EncoderBuilder {
    /// Video
    fps: f32,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    /// Audio
    nb_channels: i32,
    sample_rate: i32,
    sample_format: SampleFormat,
    /// Common
    bit_rate: i64,
    gop_size: i32,
    max_b_frames: i32,
    time_base: ffi::AVRational,
    pkt_time_base: ffi::AVRational,
    frame_rate: ffi::AVRational,
    /// config
    oformat_flags: i32,
    thread_count: usize,
    media_type: MediaType,
    codec_name: Option<String>,
    codec_opts: Option<Options>,
    filters: Option<Vec<Filter>>,
    hw_device_config: Option<HWDeviceConfig>,
    scale_algorithm: ScaleAlgorithm,
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

    /// default codec
    const VIDEO_CODEC_NAME: &'static str = "libx264";
    const AUDIO_CODEC_NAME: &'static str = "aac";

    /// Create a video encoder with the specified destination
    ///
    /// The default codec is `libx264`, with default frame rate and bit rate.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the video stream.
    /// * `height` - The height of the video stream.
    pub fn new_video(width: usize, height: usize) -> Self {
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
            .with_sample_format(sample_format)
            .with_media_type(MediaType::AUDIO)
    }

    /// Set the width of the video stream.
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the height of the video stream.
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Set the codec name.
    /// video codec default is `libx264`
    /// audio codec default is `aac`
    pub fn with_codec_name(mut self, codec_name: impl Into<Option<String>>) -> Self {
        self.codec_name = codec_name.into();
        self
    }

    /// Set the thread count.
    pub fn with_thread_count(mut self, thread_count: usize) -> Self {
        self.thread_count = thread_count;
        self
    }

    /// Set the bit rate.
    pub fn with_bit_rate(mut self, bit_rate: i64) -> Self {
        self.bit_rate = bit_rate;
        self
    }

    // /// Set the frame rate.
    // pub fn with_frame_rate_ra(mut self, frame_rare: ffi::AVRational) -> Self {
    //     self.frame_rate = frame_rare;
    //     self
    // }
    //
    // pub fn with_frame_rate(mut self, num: i32, den: i32) -> Self {
    //     self.frame_rate = time::new_rational(num, den);
    //     self
    // }

    /// Set the video frame rate from a floating-point number of frames per second.
    ///
    /// Convenience for [`with_frame_rate`](Self::with_frame_rate) that accepts a
    /// plain `fps` value (e.g. `30.0`, `29.97`). The value is converted to a
    /// reduced rational via FFmpeg's `av_d2q` and used as the encoder frame rate.
    pub fn with_fps(mut self, fps: f32) -> Self {
        if fps > 0.0 && fps.is_finite() {
            self.fps = fps;
            self.frame_rate = avutil::av_d2q(fps as f64, Self::FPS_MAX);
        }
        self
    }

    // /// Set the time base.
    // pub fn with_time_base_ra(mut self, time_base: ffi::AVRational) -> Self {
    //     self.time_base = time_base;
    //     self
    // }
    //
    // pub fn with_time_base(mut self, num: i32, den: i32) -> Self {
    //     self.time_base = time::new_rational(num, den);
    //     self
    // }
    //
    // /// Set the packet time base.
    // pub fn with_pkt_time_base_ra(mut self, pkt_time_base: ffi::AVRational) -> Self {
    //     self.pkt_time_base = pkt_time_base;
    //     self
    // }
    //
    // pub fn with_pkt_time_base(mut self, num: i32, den: i32) -> Self {
    //     self.pkt_time_base = time::new_rational(num, den);
    //     self
    // }

    /// Set the GOP size.
    pub fn with_gop_size(mut self, gop_size: i32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Set the maximum number of B-frames.
    pub fn with_max_b_frames(mut self, max_b_frames: i32) -> Self {
        self.max_b_frames = max_b_frames;
        self
    }

    /// Set the pixel format.
    pub fn with_pixel_format(mut self, pixel_format: PixelFormat) -> Self {
        self.pixel_format = pixel_format;
        self
    }

    /// codec options used for encoder
    pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
        self.codec_opts = options.into();
        self
    }

    /// filters used for encoder
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

    /// Set the scaling algorithm used when converting input frames to the
    /// encoder's target pixel format (e.g. RGB24 -> YUV420P).
    ///
    /// Defaults to [`ScaleAlgorithm::Bicubic`].
    pub fn with_scale_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.scale_algorithm = algorithm;
        self
    }

    /// explicit media type, default is `MediaType::VIDEO`
    pub fn with_media_type(mut self, media_type: MediaType) -> Self {
        self.media_type = media_type;
        self
    }

    pub fn with_nb_channels(mut self, nb_channels: i32) -> Self {
        assert!(
            nb_channels > 0 && nb_channels < 9,
            "nb_channels should be in range [1, 8]"
        );
        self.nb_channels = nb_channels;
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: i32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_sample_format(mut self, sample_format: SampleFormat) -> Self {
        self.sample_format = sample_format;
        self
    }

    /// Some formats want stream headers to be separate.
    pub fn with_oformat_flags(mut self, flags: AvFormatFlags) -> Self {
        self.oformat_flags = flags as i32;
        self
    }

    /// 编码器使用的 time_base。
    ///
    /// 视频由用户 fps 推导为 `1/fps`（libx264 等编码器在这种 time_base 下才
    /// 能正确输出 packet duration，避免 MP4 muxer 丢弃末帧）；音频按
    /// `1/sample_rate` 推导，对所有音频编码器一致。
    fn effective_time_base(&self) -> ffi::AVRational {
        match self.media_type {
            MediaType::VIDEO => avutil::av_inv_q(self.frame_rate),
            MediaType::AUDIO => time::new_rational(1, self.sample_rate),
            _ => self.time_base,
        }
    }

    /// Apply the settings to an encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Encoder to apply settings to.
    ///
    /// # Return value
    ///
    /// New encoder with settings applied.
    fn setup_codec_context(&self, encoder: &mut AVCodecContext) -> Result<()> {
        let media_type = self.media_type;
        if media_type as ffi::AVMediaType != encoder.codec_type {
            return Err(Error::msg(format!(
                "Encoder codec type not supported: {:?} vs. {:?}",
                media_type, encoder.codec_type
            )));
        }

        if media_type == MediaType::VIDEO {
            encoder.set_width(self.width as i32);
            encoder.set_height(self.height as i32);
            encoder.set_bit_rate(self.bit_rate);
            encoder.set_gop_size(self.gop_size);
            encoder.set_max_b_frames(self.max_b_frames);
            encoder.set_framerate(self.frame_rate);
            encoder.set_time_base(self.effective_time_base());
            encoder.set_pkt_timebase(self.pkt_time_base);
            encoder.set_pix_fmt(self.pixel_format.into());
            encoder.set_sample_aspect_ratio(time::new_rational(1, 1));
        } else if media_type == MediaType::AUDIO {
            encoder.set_ch_layout(AVChannelLayout::from_nb_channels(self.nb_channels).into_inner());
            encoder.set_bit_rate(self.bit_rate);
            encoder.set_sample_rate(self.sample_rate);
            encoder.set_sample_fmt(self.sample_format as _);
            encoder.set_time_base(self.effective_time_base());
        } else {
            return Err(Error::msg(format!(
                "Unsupported media type: {media_type:?}"
            )));
        }

        // Some formats want stream headers to be separate.
        if self.oformat_flags & ffi::AVFMT_GLOBALHEADER as i32 != 0 {
            encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
        }
        unsafe {
            (*encoder.as_mut_ptr()).thread_count = self.thread_count as i32;
        }

        Ok(())
    }

    /// Build an [`EncoderWrapper`] with a [`StreamWriter`].
    pub fn build_wrapped(
        self,
        destination: impl Into<Location>,
    ) -> Result<EncoderWrapper<StreamWriter>> {
        let writer = StreamWriter::new(destination)?;
        self.build_wrapped_with_writer(writer, true)
    }

    /// Build an [`EncoderWrapper`] with a custom writer.
    pub fn build_wrapped_with_writer<W: Writer>(
        self,
        mut writer: W,
        interleaved: bool,
    ) -> Result<EncoderWrapper<W>> {
        let encoder = self.build()?;
        let index = writer.add_stream(encoder.codecpar(), encoder.time_base());
        Ok(EncoderWrapper::new(encoder, writer, index, interleaved))
    }

    /// Build an [`Encoder`].
    ///
    /// Create an encoder from a [`StreamWriter`].
    ///
    /// # Arguments
    ///
    /// * `writer` - [`StreamWriter`] to create encoder from.
    /// * `interleaved` - Whether to use interleaved write.
    /// * `settings` - Encoder settings to use.
    pub fn build(self) -> Result<Encoder> {
        let media_type = self.media_type;
        let codec = {
            let codec_name = if let Some(codec_name) = &self.codec_name {
                codec_name.as_ref()
            } else {
                match media_type {
                    MediaType::VIDEO => Self::VIDEO_CODEC_NAME,
                    MediaType::AUDIO => Self::AUDIO_CODEC_NAME,
                    _ => {
                        return Err(Error::msg(
                            format!("Unsupported media type:{media_type:?}",),
                        ))
                    }
                }
            };
            AVCodec::find_encoder_by_name(&utils::from_str(codec_name))
                .context(format!("Failed to find encoder by name: '{codec_name}'"))?
        };

        let mut encode_ctx = AVCodecContext::new(&codec);
        self.setup_codec_context(&mut encode_ctx)?;
        let config = CodecConfig::new_with_ctx(&encode_ctx);

        // 在 hw_device_config / codec_opts 被 move 之前构造 filter graph：
        // 此位置 self 尚未被部分 move，可直接借用 self 计算 time_base。
        let filter_graph = if let Some(filters) = self.filters.as_ref() {
            let filter_params = match media_type {
                MediaType::VIDEO => {
                    FilterParams::Video(VideoParams {
                        width: self.width as i32,
                        height: self.height as i32,
                        format: self.pixel_format,
                        time_base: self.effective_time_base(),
                        frame_rate: self.frame_rate,
                        pixel_aspect: encode_ctx.sample_aspect_ratio, // sample aspect ratio (0 if unknown)
                    })
                }
                MediaType::AUDIO => {
                    FilterParams::Audio(AudioParams {
                        nb_channels: self.nb_channels,
                        sample_rate: self.sample_rate,
                        format: self.sample_format,
                        time_base: self.effective_time_base(), // time_base = 1 / sample_rate
                    })
                }
                _ => {
                    panic!("Unsupported filter for media type: {media_type:?}");
                }
            };
            let mut graph = FilterGraph::new();
            // check Filter media type
            if !filters.iter().all(|f| f.media_type() == media_type) {
                return Err(Error::msg(format!(
                    "Filter media type mismatch for encoder type {media_type:?}"
                )));
            }
            graph
                .init(&filter_params, filters.as_slice())
                .context("Failed to initialize filter graph")?;

            Some(graph)
        } else {
            None
        };

        let hw_context = self
            .hw_device_config
            .filter(|_cfg| {
                // hardware acceleration enabled for video
                media_type == MediaType::VIDEO
            })
            .map(|cfg| {
                // codec support or not for hardware acceleration
                log::info!(
                    "Video Encoder with HW acceleration codec: {:?}, config: {:#?}",
                    codec.name(),
                    cfg
                );

                // create hardware context
                let (width, height) = (encode_ctx.width, encode_ctx.height);
                HWContext::new(cfg)
                    .and_then(|ctx| {
                        // *注意*: setup_hw_frames 会根据 HW 能力修改 encode_ctx.pix_fmt
                        ctx.setup_hw_frames(false, &mut encode_ctx, width, height)?;
                        // 更新 Builder 中记录的目标格式，以反映 HW 的要求
                        // self.pixel_format = PixelFormat::from(encode_ctx.pix_fmt);
                        Ok(ctx)
                    })
                    .context("Hardware acceleration context initialization failed")
            })
            .transpose()?;

        let dict = self.codec_opts.map(|opts| opts.into_dict());
        encode_ctx
            .open(dict)
            .context("Failed to open encode context")?;

        Ok(Encoder {
            config,
            hw_context,
            media_type,
            filter_graph,
            context: encode_ctx,
            state: EncoderState::Normal,
            scale_algorithm: self.scale_algorithm,
        })
    }
}

impl Default for EncoderBuilder {
    fn default() -> Self {
        Self {
            // video
            width: 0,
            height: 0,
            pixel_format: PixelFormat::YUV420P,
            time_base: time::TIME_BASE,
            pkt_time_base: time::TIME_BASE,
            bit_rate: Self::VIDEO_BIT_RATE,
            frame_rate: time::new_rational(Self::FRAME_RATE, 1),
            fps: Self::FRAME_RATE as f32,
            gop_size: 0,
            max_b_frames: 0,
            oformat_flags: AvFormatFlags::GLOBAL_HEADER as i32,
            // audio
            nb_channels: 2,
            sample_rate: 44100,
            sample_format: SampleFormat::FLTP,
            // common
            media_type: MediaType::VIDEO,
            thread_count: num_cpus::get(),
            codec_name: None,
            codec_opts: None,
            filters: None,
            hw_device_config: None,
            scale_algorithm: ScaleAlgorithm::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum EncoderState {
    Normal,
    Drained,
    Flushed,
}

/// Encodes frames into a video stream.
///
/// # Example
///
/// ```ignore
/// let decoder = Decoder::new(Path::new("video_out.mkv")).unwrap();
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
    hw_context: Option<Arc<HWContext>>,
    media_type: MediaType,
    state: EncoderState,
    scale_algorithm: ScaleAlgorithm,
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
    pub fn new_video(width: usize, height: usize) -> Result<Encoder> {
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

    /// Returns `true` if the encoder is in the "drained" state.
    ///
    /// This means all input has been processed, but not fully flushed.
    pub fn is_drained(&self) -> bool {
        self.state == EncoderState::Drained
    }

    /// Returns `true` if the encoder is fully flushed and finished.
    pub fn is_flushed(&self) -> bool {
        self.state == EncoderState::Flushed
    }

    /// Encode a high-level frame (a single frame ndarray-based)
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to encode in `HWC` format and standard layout.
    #[cfg(feature = "ndarray")]
    pub fn encode<T>(&mut self, frame: MediaFrame<T>) -> Result<Option<AVPacket>>
    where
        T: MediaFrameType,
    {
        let raw_frame = frame.to_avframe()?;
        self.encode_raw(raw_frame)
    }

    /// Encode a single raw frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to encode.
    pub fn encode_raw(&mut self, frame: AVFrame) -> Result<Option<AVPacket>> {
        log::info!("{:?}, time_base: {:?}", frame, frame.time_base);

        // send frame
        self.send_frame_to_encoder(Some(frame))?;

        // receive packet
        self.receive_packet()
    }

    fn send_frame_to_encoder(&mut self, frame_opt: Option<AVFrame>) -> Result<()> {
        if let Some(frame) = frame_opt {
            // 正常编码帧：经过 filter（如有）
            if let Some(graph) = self.filter_graph.as_mut() {
                match graph.process_frame(Some(frame))? {
                    Some(filtered) => self.send_frame_post_filter(filtered)?,
                    None => {
                        // filter 暂未输出（内部缓冲中），等待后续帧驱动
                        log::debug!("Filter graph drained, waiting for more input.");
                    }
                }
            } else {
                self.send_frame_post_filter(frame)?;
            }
            Ok(())
        } else {
            // EOF：向编码器发送 EOS。filter 的缓冲帧已由 `flush()` 单独冲刷送走，
            // 这里不应再调用 `process_frame(None)`，否则对已 flushed 的 graph 会报错。
            self.context.send_frame(None)?;
            Ok(())
        }
    }

    /// 将已通过 filter（或无 filter）的帧做 rescale/hw 上传后发送给编码器。
    ///
    /// 注意：`flush()` 阶段 filter 已进入 Flushed 状态，不能再把缓冲帧送回
    /// `process_frame`（会因 EAGAIN 被丢弃），因此缓冲帧必须直接走本方法。
    fn send_frame_post_filter(&mut self, frame: AVFrame) -> Result<()> {
        // 确保帧的格式匹配编码器要求
        let scaled_frame = self.rescale(frame)?;

        // 转换硬件帧
        let hw_frame = match self.hw_context.as_ref() {
            Some(hw_ctx) if hw_ctx.is_sw_frame(&scaled_frame) => {
                // sw_frame -> hw_frame
                hw_ctx
                    .hw_upload(&mut self.context, &scaled_frame)
                    .context("Failed to upload frame to HW")?
            }
            _ => scaled_frame, // 不需要上传或已经是 HW frame
        };

        // check frame valid
        self.check_frame(Some(&hw_frame))?;

        log::debug!(
            "Send frame to encoder: {:?}, time_base: {:?}, media_type: {:?}",
            hw_frame,
            self.time_base(),
            self.media_type()
        );

        self.context.send_frame(Some(&hw_frame))?;
        Ok(())
    }

    fn rescale(&self, frame: AVFrame) -> Result<AVFrame> {
        let scaled_frame = match self.media_type {
            MediaType::VIDEO => {
                let target_sw_pix_fmt = if let Some(hw_ctx) = self.hw_context.as_ref() {
                    hw_ctx.get_format(false).into()
                } else {
                    self.pix_fmt()
                };
                if frame.format != target_sw_pix_fmt.into() {
                    swctx::scale_with_flags(
                        &frame,
                        frame.width,
                        frame.height,
                        target_sw_pix_fmt,
                        self.scale_algorithm,
                    )?
                } else {
                    frame
                }
            }
            MediaType::AUDIO => {
                let ch_layout = self.ch_layout();
                if frame.sample_rate != self.sample_rate()
                    || frame.format != self.sample_fmt() as i32
                    || frame.ch_layout.nb_channels != ch_layout.nb_channels
                {
                    swctx::convert_frame(
                        &frame,
                        ch_layout.clone().into_inner(),
                        self.sample_fmt() as _,
                        self.sample_rate(),
                    )?
                } else {
                    frame
                }
            }
            _ => {
                // do nothing
                return Err(Error::msg(format!(
                    "Unsupported encode frame media type: {:?}",
                    self.media_type
                )));
            }
        };
        Ok(scaled_frame)
    }

    /// Check if the frame is valid for encoding.
    fn check_frame(&self, frame: Option<&AVFrame>) -> Result<()> {
        if frame.is_none() {
            return Ok(());
        }
        let frame = frame.unwrap();
        match self.media_type {
            MediaType::VIDEO => {
                // 硬件帧的像素格式（如 NV12/HW 私有格式）不在软件编码器的
                // `supported_pixel_formats()` 列表中，跳过该检查以免误报。
                if !frame.hw_frames_ctx.is_null() {
                    return Ok(());
                }
                let pix_fmts_opt = self.config.supported_pixel_formats()?;
                if let Some(pix_fmts) = pix_fmts_opt {
                    if !pix_fmts.contains(&frame.format) {
                        return Err(Error::msg(format!(
                            "Unsupported video encoder frame pixel format: {:?}",
                            frame.format
                        )));
                    }
                }
            }

            MediaType::AUDIO => {
                let ch_layouts_opt = self.config.supported_channel_layouts()?;
                if let Some(ch_layouts) = ch_layouts_opt {
                    ch_layouts
                        .iter()
                        .find(|ch_layout| ch_layout.nb_channels == frame.ch_layout.nb_channels)
                        .ok_or_else(|| {
                            Error::msg(format!(
                                "Unsupported audio encoder frame channel layout: {:?}",
                                frame.ch_layout.nb_channels
                            ))
                        })?;
                }

                let sample_fmts_opt = self.config.supported_sample_formats()?;
                if let Some(sample_fmts) = sample_fmts_opt {
                    if !sample_fmts.contains(&frame.format) {
                        return Err(Error::msg(format!(
                            "Unsupported encode frame sample format: {:?}",
                            frame.format
                        )));
                    }
                }

                let sample_rates_opt = self.config.supported_sample_rates()?;
                if let Some(sample_rates) = sample_rates_opt {
                    if !sample_rates.contains(&frame.sample_rate) {
                        return Err(Error::msg(format!(
                            "Unsupported encode frame sample rate: {:?}",
                            frame.sample_rate
                        )));
                    }
                }

                // variable frame size, do nothing
                // if fixed frame size, require frame size
                if !self.config.support_variable_frame_size()
                    && frame.nb_samples != self.frame_size()
                {
                    return Err(Error::msg(format!(
                        "Unsupported encode frame sample size: {:?}, expect {:?}",
                        frame.nb_samples,
                        self.frame_size()
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get encoder time base.
    #[inline]
    pub fn time_base(&self) -> ffi::AVRational {
        self.context.time_base
    }

    #[inline]
    pub fn frame_rate(&self) -> ffi::AVRational {
        self.context.framerate
    }

    #[inline]
    pub fn width(&self) -> i32 {
        self.context.width
    }

    #[inline]
    pub fn height(&self) -> i32 {
        self.context.height
    }

    #[inline]
    pub fn pix_fmt(&self) -> PixelFormat {
        self.context.pix_fmt.into()
    }

    /// Each submitted frame except the last must contain exactly frame_size samples per channel.
    /// May be 0 when the codec has AV_CODEC_CAP_VARIABLE_FRAME_SIZE set, then the frame size is not restricted.
    #[inline]
    pub fn frame_size(&self) -> i32 {
        self.context.frame_size
    }

    /// audio samples per second
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

    /// 单帧时长（编码器 time_base 单位），用于补全缺失的 packet duration。
    ///
    /// 先求单帧时长（秒），再换算到编码器 time_base 的整数 tick：
    /// `ticks = av_rescale_q(1, frame_dur_sec, time_base)`。
    fn packet_duration(&self) -> i64 {
        let tb = self.time_base();
        let frame_dur_sec = match self.media_type {
            // 视频：1 / frame_rate
            MediaType::VIDEO => avutil::av_inv_q(self.frame_rate()),
            // 音频：frame_size / sample_rate
            MediaType::AUDIO => {
                let fs = self.frame_size();
                if fs <= 0 {
                    return 0;
                }
                time::new_rational(fs, self.sample_rate())
            }
            _ => return 0,
        };
        avutil::av_rescale_q(1, frame_dur_sec, tb).max(1)
    }

    /// Internal: Pull an encoded packet from the decoder.
    ///
    /// Handles `EAGAIN`, drained, and flushed states.
    ///
    /// # Returns
    ///
    /// `Some(packet)` if a packet is returned, `None` if waiting or end.
    fn receive_packet(&mut self) -> Result<Option<AVPacket>> {
        match self.context.receive_packet() {
            Ok(pkt) => Ok(Some(pkt)),
            Err(rsmpeg::error::RsmpegError::EncoderDrainError) => {
                log::debug!("Encoder drained, try send new frame again.");
                self.state = EncoderState::Drained;
                Ok(None)
            }
            Err(rsmpeg::error::RsmpegError::EncoderFlushedError) => {
                log::debug!("Encoder flushed, EOF reached.");
                self.state = EncoderState::Flushed;
                Ok(None)
            }
            Err(err) => Err(Error::new(err)),
        }
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
    /// # Returns
    ///
    /// Returns `Ok(())` if flushing completes successfully.
    /// May return an error if writing fails or encoder returns an error.
    pub fn flush<W: Writer>(
        &mut self,
        writer: &mut W,
        interleaved: bool,
        index: usize,
        out_stream_time_base: ffi::AVRational,
    ) -> Result<()> {
        // 确定编码器是否支持延迟（delay）
        // 如果编码器不支持延迟，那么就没有必要进行 flush 操作，因为在这种情况下，编码器不会保留任何未处理的数据。
        // 如果编码器支持延迟（delay），则在结束编码之前发送 EOS 包是有必要的，
        // 因为编码器可能还在缓冲一些数据，直到接收到 EOS 信号才会处理完这些数据并输出剩余的包。
        if !self.config.support_delayed_frame() {
            return Ok(());
        }

        if let Some(filter) = self.filter_graph.as_mut() {
            let frames = filter.flush()?;
            for frame in frames {
                // filter 已 Flushed，缓冲帧直接走 post-filter 路径，不可再进 process_frame
                self.send_frame_post_filter(frame)?;
            }
        }

        // EOF: Notify the encoder that the last frame has been sent.
        self.send_frame_to_encoder(None)?;

        // drain the items still on the queue before giving up.
        // EOF 已发送，理论上编码器最终会返回 EOF；但为防御个别编码器在 EOS 后
        // 持续返回 EAGAIN（Drained）而不返回 EOF，增加迭代上限，避免死循环。
        let mut drained_iterations = 0u32;
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
                    packet.rescale_ts(self.time_base(), out_stream_time_base);
                    if interleaved {
                        writer.write_interleaved(&mut packet)?;
                    } else {
                        writer.write_frame(&mut packet)?;
                    }
                }
                Ok(None) => {
                    if self.is_drained() {
                        log::debug!("Encoder drained, try send new frame again.");
                        drained_iterations += 1;
                        if drained_iterations > 1_000 {
                            log::error!(
                                "Encoder keeps returning EAGAIN after EOF, aborting flush."
                            );
                            break;
                        }
                        continue;
                    } else {
                        log::debug!("Encoder flushed, EOF reached.");
                        break;
                    }
                }
                Err(e) => {
                    log::debug!("Encode packet error: {e}");
                    break;
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
        //! let _ = self.flush();
        if !self.is_flushed() {
            log::error!("Encoder dropped without flushing, data may be lost.");
        }
    }
}

/// SAFETY:
/// - Encoder contains `AVCodecContext`, which is not inherently thread-safe.
/// - We implement `Send`/`Sync` only because `Encoder` is guaranteed to be used
///   in a single-threaded context or externally synchronized by the caller.
/// - If used across threads, caller must ensure no concurrent access.
unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

/// 编码器包装器，持有编码器和写入器
pub struct EncoderWrapper<W: Writer> {
    writer: W,
    encoder: Encoder,
    interleaved: bool,
    stream_index: usize,
    stream_info: StreamInfo,
    have_written_header: bool,
    have_written_trailer: bool,
    /// 自动时间戳的当前位置，由 [`write_frame`](Self::write_frame) 维护。
    position: time::Time,
    frame_duration: time::Time,
}

impl<W: Writer> EncoderWrapper<W> {
    /// 创建一个新的编码器包装器
    pub fn new(encoder: Encoder, writer: W, stream_index: usize, interleaved: bool) -> Self {
        let stream_info = StreamInfo::from_writer(&writer, stream_index).unwrap();
        // 当前帧时长：视频按帧率，音频按采样数/采样率
        let duration = match encoder.media_type {
            // 帧时长 = 1 / frame_rate，即取帧率(fr.num / fr.den)的倒数 (fr.den / fr.num)
            MediaType::VIDEO => {
                let fr = encoder.frame_rate();
                time::Time::new(Some(1), time::new_rational(fr.den, fr.num.max(1)))
            }
            // 帧时长 = nb_samples / sample_rate
            MediaType::AUDIO => time::Time::new(
                Some(encoder.frame_size() as i64),
                time::new_rational(1, encoder.sample_rate().max(1)),
            ),
            _ => panic!("No supported encoder for media_type."),
        };
        Self {
            writer,
            encoder,
            interleaved,
            stream_index,
            stream_info,
            have_written_header: false,
            have_written_trailer: false,
            position: time::Time::zero(),
            frame_duration: duration,
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn encode<T: MediaFrameType>(&mut self, frame: MediaFrame<T>) -> Result<()> {
        let raw_frame = frame.to_avframe()?;
        self.encode_raw(raw_frame)
    }

    pub fn encode_raw(&mut self, frame: AVFrame) -> Result<()> {
        // Write file header if we hadn't done that yet.
        if !self.have_written_header {
            self.writer.write_header()?;
            self.have_written_header = true;
        }

        if let Some(mut packet) = self.encoder.encode_raw(frame)? {
            packet.set_pos(-1);
            packet.set_stream_index(self.stream_index as i32);
            if packet.duration <= 0 {
                packet.set_duration(self.encoder.packet_duration());
            }
            // 实时获取输出流时间基（write_header 后 muxer 可能调整 timescale）。
            packet.rescale_ts(
                self.time_base(),
                self.writer.stream_time_base(self.stream_index),
            );

            if self.interleaved {
                self.writer.write_interleaved(&mut packet)?;
            } else {
                self.writer.write_frame(&mut packet)?;
            }
        }

        Ok(())
    }

    /// 写入一帧，并自动维护时间戳（pts）。
    ///
    /// 与 [`encode`](Self::encode) 不同，`write_frame` 会按帧率（视频）或
    /// 采样率/采样数（音频）自动递增 pts，用户无需手动
    /// [`set_pts`](MediaFrame::set_pts)。适合需要"开箱即用"地逐帧写出时使用。
    ///
    /// 若需要完全控制时间戳，请使用 [`encode`](Self::encode)。
    #[cfg(feature = "ndarray")]
    pub fn write_frame<T: MediaFrameType>(&mut self, mut frame: MediaFrame<T>) -> Result<()> {
        let pts = self
            .position
            .aligned_with_rational(self.encoder.time_base())
            .into_value()
            .unwrap_or(0);
        frame.set_pts(pts);

        self.position = self.position.aligned_with(self.frame_duration).add();

        self.encode(frame)
    }

    /// Signal to the encoder that writing has finished. This will cause any packets in the encoder
    /// to be flushed and a trailer to be written if the container format has one.
    ///
    /// Note: If you don't call this function before dropping the encoder, it will be called
    /// automatically. This will block the caller thread. Any errors cannot be propagated in this
    /// case.
    pub fn finish(&mut self) -> Result<()> {
        if self.have_written_header && !self.have_written_trailer {
            self.have_written_trailer = true;
            self.flush()?;
            self.writer.write_trailer()?;
        }

        Ok(())
    }

    /// 刷新编码器并写入剩余数据
    fn flush(&mut self) -> Result<()> {
        // 实时获取（write_header 后 muxer 可能已调整 timescale）
        let out_stream_time_base = self.writer.stream_time_base(self.stream_index);
        self.encoder.flush(
            &mut self.writer,
            self.interleaved,
            self.stream_index,
            out_stream_time_base,
        )
    }

    pub fn time_base(&self) -> ffi::AVRational {
        self.encoder.time_base()
    }

    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    /// 获取内部编码器的可变引用
    pub fn encoder_mut(&mut self) -> &mut Encoder {
        &mut self.encoder
    }

    /// 获取内部写入器的可变引用
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// 解构并返回内部组件
    pub fn into_parts(self) -> (Encoder, W) {
        (self.encoder, self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    /// 生成一帧纯色（RGB24）测试帧，颜色随相位 `p` 在彩虹色相上变化。
    #[cfg(feature = "ndarray")]
    fn rainbow_frame(w: usize, h: usize, p: f32) -> MediaFrame<u8> {
        use crate::colors;
        let rgb = colors::hsv_to_rgb(p * 360.0, 100.0, 100.0);
        let mut frame =
            MediaFrame::<u8>::new_video_frame(w, h, PixelFormat::RGB24, time::new_rational(1, 24))
                .unwrap();
        for y in 0..h {
            for x in 0..w {
                frame.data[[y, x, 0]] = rgb[0];
                frame.data[[y, x, 1]] = rgb[1];
                frame.data[[y, x, 2]] = rgb[2];
            }
        }
        frame
    }

    /// 视频/容器格式参数（内部测试辅助）
    #[allow(dead_code)]
    struct VideoFormatParams {
        time_base: (i32, i32),
        codec_name: String,
        /// 支持的帧率列表（VIDEO）
        supported_frame_rates: Option<Vec<ffi::AVRational>>,
        /// 支持的像素格式列表（VIDEO）
        supported_pix_fmts: Vec<ffi::AVPixelFormat>,
        /// 特定编码器选项
        codec_options: Option<HashMap<String, String>>,
        /// 特定格式选项
        format_options: Option<HashMap<String, String>>,
    }

    /// 动态码率/帧率调整、关键帧间隔控制
    /// 完善主流编码格式支持（H.264/265, VP9/AV1, AAC/Opus）
    ///
    /// | 容器格式 | 标准时间基         | 说明                      |
    /// | :------- | :-------------- | :------------------------|
    /// | MP4      | 1/90_000        | 90kHz，源自MPEG-2标准      |
    /// | MOV      | 1/10_000_000    | 10MHz，苹果QuickTime格式   |
    /// | MKV      | 1/1_000_000_000 | 纳秒级精度                 |
    /// | FLV      | 1/1_000         | 毫秒级，Flash视频标准       |
    /// | TS       | 1/90_000        | 90kHz，MPEG传输流          |
    /// | AVI      | 1/{帧率}         | 基于帧计数，或1/1000        |
    /// | WebM     | 1/1_000_000_000 | 纳秒级，基于MKV            |
    /// | 3GP      | 1/90_000        | 移动设备视频标准            |
    /// | ASF/WMV  | 1/10_000_000    | 100纳秒单位，Windows Media |
    /// | OGG/OGV  | 1/1_000_000     | 微秒级，开源标准            |
    /// | MPEG     | 1/90_000        | 90kHz，MPEG标准           |
    #[cfg(feature = "ndarray")]
    fn test_encode_video_for_container(container_type: &str, fps: f64) -> Result<()> {
        use crate::filter;
        use crate::time::Time;
        use std::path::Path;

        // 使用单一match获取基本参数
        let (codec_name, time_base, codec_options, format_options) = match container_type {
            // 常见流媒体/通用格式
            "mp4" => (
                None,
                (1, 90_000), // 90kHz
                None,
                None,
            ),
            "mov" => {
                let mut format_opts = HashMap::new();
                format_opts.insert(
                    "movflags".to_string(),
                    "frag_keyframe+empty_moov".to_string(),
                );

                (
                    None,
                    (1, 90_000), // 90kHz
                    None,
                    Some(format_opts),
                )
            }
            "mkv" => {
                let mut format_opts = HashMap::new();
                format_opts.insert("strict".to_string(), "experimental".to_string());

                (
                    None,
                    (1, 1_000_000_000), // 纳秒级
                    None,
                    Some(format_opts),
                )
            }
            "webm" => (
                Some("libvpx-vp9".to_string()),
                (1, 1_000_000_000), // 纳秒级
                None,
                None,
            ),
            "flv" => {
                let mut format_opts = HashMap::new();
                format_opts.insert("flvflags".to_string(), "no_duration_filesize".to_string());

                (
                    None,
                    (1, 1_000), // 毫秒级
                    None,
                    Some(format_opts),
                )
            }
            "ts" | "mts" | "m2ts" => (
                None,
                (1, 90_000), // 90kHz
                None,
                None,
            ),
            "avi" => {
                // AVI通常使用帧率作为时间基
                let fps_rounded = fps.round() as i32;

                let mut codec_opts = HashMap::new();
                codec_opts.insert("profile".to_string(), "baseline".to_string());
                codec_opts.insert("level".to_string(), "3.0".to_string());

                let mut format_opts = HashMap::new();
                format_opts.insert("strict".to_string(), "normal".to_string());

                (None, (1, fps_rounded), Some(codec_opts), Some(format_opts))
            }
            "3gp" => (
                None,
                (1, 90_000), // 通常为90kHz
                None,
                None,
            ),
            "wmv" | "asf" => {
                let mut format_opts = HashMap::new();
                format_opts.insert("strict".to_string(), "normal".to_string());

                (
                    None,
                    (1, 10_000_000), // 100纳秒单位
                    None,
                    Some(format_opts),
                )
            }
            "ogg" | "ogv" => (
                Some("libtheora".to_string()),
                (1, 1_000_000), // 微秒级
                None,
                None,
            ),
            "mpg" | "mpeg" => (
                None,
                (1, 90_000), // 90kHz
                None,
                None,
            ),

            // 专业广电格式
            "mxf" => {
                let mut format_opts = HashMap::new();
                format_opts.insert("strict".to_string(), "experimental".to_string());
                format_opts.insert("mxf_operational_pattern".to_string(), "1a".to_string());

                let mut codec_opts = HashMap::new();
                codec_opts.insert("profile".to_string(), "main".to_string());
                codec_opts.insert("r".to_string(), "25".to_string());
                codec_opts.insert("g".to_string(), "15".to_string());
                codec_opts.insert("b".to_string(), "5M".to_string());

                (
                    Some("mpeg2video".to_string()),
                    (1, 25),
                    Some(codec_opts),
                    Some(format_opts),
                )
            }
            "gxf" | "ps" => (Some("mpeg2video".to_string()), (1, 90_000), None, None),
            "xavc" => (None, (1, 90_000), None, None),

            // 硬件设备格式
            "vob" => (Some("mpeg2video".to_string()), (1, 90_000), None, None),
            "rmvb" | "divx" => (None, (1, 90_000), None, None),

            // 特殊格式
            "heif" => {
                let mut codec_opts = HashMap::new();
                codec_opts.insert("x265-params".to_string(), "lossless=1".to_string());

                let mut format_opts = HashMap::new();
                format_opts.insert("brand".to_string(), "heic".to_string());
                format_opts.insert("hvc1_flag".to_string(), "1".to_string());

                (
                    Some("libx265".to_string()),
                    (1, 90_000),
                    Some(codec_opts),
                    Some(format_opts),
                )
            }
            "f4v" | "dav" | "evo" | "h264" => (None, (1, 90_000), None, None),
            "h265" => (Some("libx265".to_string()), (1, 90_000), None, None),
            "cmaf" => {
                let mut format_opts = HashMap::new();
                format_opts.insert(
                    "movflags".to_string(),
                    "cmaf+dash+frag_keyframe+negative_cts_offsets".to_string(),
                );
                format_opts.insert("use_template".to_string(), "1".to_string());
                format_opts.insert("use_timeline".to_string(), "1".to_string());

                (None, (1, 90_000), None, Some(format_opts))
            }

            // 默认值（用于未明确定义的格式）
            _ => (
                None,
                (1, 90_000), // 90kHz为最安全的默认值
                None,
                None,
            ),
        };

        let codec_name = utils::from_str(&codec_name.unwrap_or_else(|| "libx264".to_string()));
        let codec_config = CodecConfig::new_with_name(&codec_name)?;
        assert!(
            codec_config.is_encoder(),
            "Codec:'{:?}' is not an encoder.",
            codec_name
        );

        let config = VideoFormatParams {
            time_base,
            codec_name: codec_name.to_str()?.to_string(),
            supported_frame_rates: codec_config
                .supported_frame_rates()?
                .map(|fps| fps.to_vec()),
            supported_pix_fmts: codec_config.supported_pixel_formats()?.unwrap().to_vec(),
            codec_options,
            format_options,
        };

        let filters = vec![
            filter::video::scale(1920, 1080, None),
            filter::video::DrawText::new("Watermark", 50, 50, 24, "white@0.5").build(),
            filter::video::crop(0, 0, 640, 360),
        ];

        // 视频编码参数
        let width = 1280;
        let height = 720;
        // 确定输出路径和扩展名
        let output_file = format!("/tmp/test_encode_video.{}", container_type);
        let output_path = Path::new(output_file.as_str());

        // 创建编码器
        let mut encoder = EncoderBuilder::new_video(width as usize, height as usize)
            .with_codec_name(config.codec_name)
            .with_options(config.codec_options.map(|opts| opts.into()))
            .with_filters(filters)
            .build_wrapped(output_path)?;

        let actual_timebase = encoder.time_base();
        let frame_duration_seconds = 1.0 / fps;

        // 将秒转换为对应时间基单位
        let duration_units = (frame_duration_seconds * actual_timebase.den as f64
            / actual_timebase.num as f64)
            .round() as i64;

        let duration = Time::new(Some(duration_units), actual_timebase);

        // 初始化position时使用正确的时间基
        let mut position = Time::new(Some(0), time::new_rational(time_base.0, time_base.1));

        println!(
            "Encoding {} with actual timebase: {}/{}, duration units: {}, fps: {}",
            container_type, actual_timebase.num, actual_timebase.den, duration_units, fps
        );

        // 帧编码并写入文件
        for i in 0..10 {
            let mut frame = rainbow_frame(width as usize, height as usize, i as f32 / 10.0);
            frame.set_pts(
                position
                    .aligned_with_rational(encoder.time_base())
                    .into_value()
                    .unwrap(),
            );

            encoder.encode(frame)?;

            // 使用aligned_with确保时间基一致进行加法操作
            position = position.aligned_with(duration).add();
        }

        // flush encoder
        encoder.finish().unwrap();

        Ok(())
    }

    #[cfg(feature = "ndarray")]
    #[test]
    #[rustfmt::skip]
    #[ignore = "ignore video output file"]
    fn test_encode_video() -> Result<()> {

        let video_formats = [
            // 通用/主流视频容器
            "mp4",   // MPEG-4 Part 14，最通用的视频格式
            "mkv",   // Matroska，开源高灵活性容器
            "webm",  // Web优化的Matroska子集
            "avi",   // Audio Video Interleave，传统通用格式
            "mov",   // QuickTime格式，苹果生态常用
            "wmv",   // Windows Media Video
            "flv",   // Flash Video，流媒体
            "mpg",   // MPEG-1/2 Program Stream
            "mpeg",  // 同上
            "asf",   // Advanced Systems Format

            // 广播/专业视频容器
            "mxf",   // Material eXchange Format，广电行业标准
            "gxf",   // General eXchange Format，磁带元数据
            "ts",    // MPEG Transport Stream，广播流
            "m2ts",  // Blu-ray MPEG-2 Transport Stream
            "mts",   // 同上
            "xavc",  // Sony's 4K/8K专业格式
            "ps",    // MPEG-2 Program Stream (专业用途)

            // 流媒体/网络视频容器
            "f4v",   // Adobe Flash MP4衍生格式
            "cmaf",  // Common Media Application Format (DASH/HLS)
            "ismv",  // Microsoft Smooth Streaming

            // 移动设备/特殊视频容器
            "3gp",   // 3GPP移动多媒体格式
            "3g2",   // 3GPP2多媒体格式
            "ogv",   // Ogg Video
            "rmvb",  // RealMedia Variable Bitrate
            "rm",    // RealMedia
            "vob",   // DVD Video Object
            "divx",  // DivX Media Format
            "amv",   // Anime Music Video

            // 图像序列/原始视频格式
            "heif",  // High Efficiency Image File Format
            "heic",  // HEIF的iOS实现
            "dav",   // 大华监控专用
            "h264",  // 裸H.264比特流
            "h265",  // 裸H.265/HEVC比特流
            "y4m",   // YUV4MPEG2原始格式

            // 旧版/特殊格式
            "swf",   // Shockwave Flash
            "evo",   // HD-DVD格式
            "ifo",   // DVD信息文件
            "m4v",   // MPEG-4视频文件
        ];

        let mut err_encoder = Vec::new();
        for format in video_formats {
            println!("Testing format: {}...", format);
            match test_encode_video_for_container(format, 24.0) {
                Ok(_) => println!("Testing format: {} passed.", format),
                Err(e) => {
                    println!("Testing format: {} failed: {}", format, e);
                    err_encoder.push(format.to_string());
                }
            }
        }

        if !err_encoder.is_empty() {
            eprintln!("Failed encoders: {:#?}", err_encoder)
        }

        Ok(())
    }

    /// 自包含的编解码往返测试：编码若干帧到临时文件，再解码回，验证帧数与尺寸一致。
    /// 不依赖任何外部媒体文件，可自动运行。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_encode_decode_roundtrip() -> Result<()> {
        use crate::{DecoderBuilder, MediaType};

        let width = 64usize;
        let height = 64usize;
        let n_frames = 10;
        let fps = 25.0;

        let path = std::env::temp_dir().join("rsmedia_roundtrip.mp4");
        let _ = std::fs::remove_file(&path);

        // 1) 编码：用 write_frame 自动维护 pts
        let mut encoder = EncoderBuilder::new_video(width, height)
            .with_fps(fps)
            .build_wrapped(path.as_path())?;
        for i in 0..n_frames {
            let frame = rainbow_frame(width, height, i as f32 / n_frames as f32);
            encoder.write_frame(frame)?;
        }
        encoder.finish()?;

        // 2) 解码回：验证帧数与解码尺寸
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_wrapped(path.as_path())?;
        let mut decoded = 0usize;
        while let Some(frame) = decoder.decode_frame()? {
            assert_eq!(frame.width, width);
            assert_eq!(frame.height, height);
            decoded += 1;
        }
        assert_eq!(
            decoded, n_frames,
            "decoded frame count mismatch: got {decoded}, expected {n_frames}"
        );

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    /// 回归测试：带延迟滤镜（framerate，内部缓冲运动插值帧、flush 时才输出剩余帧）
    /// 编码时，flush() 阶段取出的缓冲帧必须全部写盘，不能因再次送入已 flushed 的
    /// filter 而被丢弃。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_encode_delayed_filter_roundtrip() -> Result<()> {
        use crate::{filter::Filter, DecoderBuilder, MediaType};

        let width = 64usize;
        let height = 64usize;
        let n_frames = 30;
        let fps = 30.0;

        let path = std::env::temp_dir().join("rsmedia_delayed_filter.mp4");
        let _ = std::fs::remove_file(&path);

        // framerate 滤镜内部缓冲运动插值帧，输入 30 帧@30fps=1s，输出仍约 30 帧，
        // 其中尾部的插值帧要等 flush(EOF) 才输出。若 flush 缓冲帧被丢弃会偏少。
        let mut encoder = EncoderBuilder::new_video(width, height)
            .with_fps(fps as f32)
            .with_filters(vec![Filter::new(
                "framerate",
                MediaType::VIDEO,
                "framerate=fps=30".to_string(),
            )])
            .build_wrapped(path.as_path())?;
        for i in 0..n_frames {
            let frame = rainbow_frame(width, height, i as f32 / n_frames as f32);
            encoder.write_frame(frame)?;
        }
        encoder.finish()?;

        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_wrapped(path.as_path())?;
        let mut decoded = 0usize;
        while let Some(_frame) = decoder.decode_frame()? {
            decoded += 1;
        }
        assert!(
            decoded >= n_frames,
            "delayed filter roundtrip lost frames: got {decoded}, expected >= {n_frames}"
        );

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    /// 验证 `write_frame` 自动维护的 pts 单调递增且与帧率一致（不含 B 帧时每帧
    /// 时长为 1/fps，按编码器时间基换算）。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_write_frame_auto_pts() -> Result<()> {
        use crate::{DecoderBuilder, MediaType};

        let width = 64usize;
        let height = 64usize;
        let n_frames = 8;
        let fps: f64 = 30.0;

        let path = std::env::temp_dir().join("rsmedia_auto_pts.mp4");
        let _ = std::fs::remove_file(&path);

        let mut encoder = EncoderBuilder::new_video(width, height)
            .with_fps(fps as f32)
            .build_wrapped(path.as_path())?;

        // 帧时长 = 1/fps（秒）。解码输出的 pts 位于输出流 time_base（movenc
        // 可能调整，如 MP4 用 1/15360），故在解码后按实际帧 time_base 计算期望增量。
        for i in 0..n_frames {
            let frame = rainbow_frame(width, height, i as f32 / n_frames as f32);
            encoder.write_frame(frame)?;
        }
        encoder.finish()?;

        // 解码回，收集真实 pts，验证相邻帧 pts 差恒定
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_wrapped(path.as_path())?;
        let mut pts_list: Vec<i64> = Vec::new();
        while let Some(frame) = decoder.decode_frame()? {
            pts_list.push(frame.pts);
        }
        assert_eq!(pts_list.len(), n_frames);

        // 解码 pts 位于输出流 time_base（movenc 可能调整，如 MP4 用 1/15360）
        let tb = decoder.decoder_mut().time_base();
        let expected_delta = (tb.den as f64 / tb.num as f64 / fps).round() as i64;

        // 排除 B 帧重排的影响：仅断言存在一致的正增量（B 帧可能为 0/负，取出现最多的增量）
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for d in pts_list.windows(2).map(|w| w[1] - w[0]) {
            if d > 0 {
                *counts.entry(d).or_default() += 1;
            }
        }
        let delta = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(d, _)| d)
            .unwrap_or(expected_delta);
        assert_eq!(delta, expected_delta, "pts delta mismatch vs 1/fps");

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    /// 验证不同帧率下：编码器 time_base 恒为 `1/fps`，且编码→解码往返帧数一致。
    ///
    /// 这是对 fps → time_base 推导（`av_inv_q`）与末帧 duration 修复的回归测试。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_encode_video_multiple_fps() -> Result<()> {
        use crate::{DecoderBuilder, MediaType};

        for fps in [24.0f32, 25.0, 30.0, 60.0, 29.97] {
            let path = std::env::temp_dir().join(format!("rsmedia_fps_{fps}.mp4"));
            let _ = std::fs::remove_file(&path);

            let n_frames = 12;
            let mut encoder = EncoderBuilder::new_video(64, 64)
                .with_fps(fps)
                .build_wrapped(path.as_path())?;

            // 1) 编码器 time_base 必须等于 1/fps
            let tb = encoder.time_base();
            let expected_tb = avutil::av_inv_q(avutil::av_d2q(fps as f64, EncoderBuilder::FPS_MAX));
            assert_eq!(
                (tb.num, tb.den),
                (expected_tb.num, expected_tb.den),
                "fps={fps}: time_base {}/{} != 1/fps",
                tb.num,
                tb.den
            );

            for i in 0..n_frames {
                let frame = rainbow_frame(64, 64, i as f32 / n_frames as f32);
                encoder.write_frame(frame)?;
            }
            encoder.finish()?;

            // 2) 解码回，帧数必须与编码一致（验证末帧未被 muxer 丢弃）
            let mut decoder =
                DecoderBuilder::new(MediaType::VIDEO).build_wrapped(path.as_path())?;
            let mut decoded = 0usize;
            while let Some(frame) = decoder.decode_frame()? {
                assert_eq!(frame.width, 64);
                assert_eq!(frame.height, 64);
                decoded += 1;
            }
            assert_eq!(
                decoded, n_frames,
                "fps={fps}: decoded {decoded} frames, expected {n_frames}"
            );

            let _ = std::fs::remove_file(&path);
        }
        Ok(())
    }

    /// 音频编解码往返测试：编码若干 AAC 音频帧，解码回验证采样率/通道数/总采样数。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_encode_decode_audio_roundtrip() -> Result<()> {
        use crate::frame::MediaFrame;
        use crate::{DecoderBuilder, MediaType};

        let sample_rate = 44_100u32;
        let channels = 2u32;
        let format = SampleFormat::FLTP;
        // AAC 默认 frame_size = 1024 采样/帧
        let samples_per_frame = 1024u32;
        let frames_to_write = 10u32;

        let path = std::env::temp_dir().join("rsmedia_audio_roundtrip.m4a");
        let _ = std::fs::remove_file(&path);

        let mut encoder =
            EncoderBuilder::new_audio(128_000, channels as i32, sample_rate as i32, format)
                .build_wrapped(path.as_path())?;

        // 1) 音频 time_base 应为 1/sample_rate
        let tb = encoder.time_base();
        let expected_tb = time::new_rational(1, sample_rate as i32);
        assert_eq!(
            (tb.num, tb.den),
            (expected_tb.num, expected_tb.den),
            "audio time_base {}/{} != 1/sample_rate",
            tb.num,
            tb.den
        );

        for _ in 0..frames_to_write {
            let frame = MediaFrame::<f32>::new_audio_frame(
                format,
                channels,
                samples_per_frame,
                sample_rate,
                time::new_rational(1, sample_rate as i32),
            )?;
            encoder.write_frame(frame)?;
        }
        encoder.finish()?;

        // 2) 解码验证：采样率、通道数、采样量（AAC 有编码延迟/padding，总采样数应覆盖输入）
        // 音频 FLTP 用 f32 解码（decode_frame 固定返回 u8，仅适用于视频）。
        let mut decoder = DecoderBuilder::new(MediaType::AUDIO).build_wrapped(path.as_path())?;
        let mut total_samples = 0u64;
        let mut decoded_frames = 0usize;
        while let Some(frame) = decoder.decode::<f32>()? {
            assert_eq!(frame.sample_rate, sample_rate, "sample rate mismatch");
            assert_eq!(frame.nb_channels, channels, "channel count mismatch");
            total_samples += frame.nb_samples as u64;
            decoded_frames += 1;
        }
        let expected = frames_to_write as u64 * samples_per_frame as u64;
        assert!(
            total_samples >= expected,
            "decoded {total_samples} samples, expected >= {expected}"
        );
        assert!(decoded_frames > 0, "no audio frames decoded");

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
