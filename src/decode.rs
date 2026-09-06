use crate::filter::{AudioParams, Filter, FilterGraph, FilterParams, VideoParams};
use crate::flags::AvCodecFlags;
#[cfg(feature = "ndarray")]
use crate::frame::{MediaFrame, MediaFrameType};
use crate::hwaccel::{HWContext, HWDeviceConfig};
use crate::io::Reader;
use crate::options::Options;
use crate::resize::Resize;
use crate::stream::StreamInfo;
use crate::swctx::ScaleAlgorithm;
use crate::{Location, MediaType, PixelFormat, SampleFormat, StreamReader, Time, swctx, utils};

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket};
use rsmpeg::avformat::AVStream;
use rsmpeg::avutil::{self, AVChannelLayoutRef, AVFrame};
use rsmpeg::ffi;

use anyhow::{Context, Error, Result};
use std::sync::Arc;

/// Builds a [`Decoder`].
#[derive(Debug)]
pub struct DecoderBuilder {
    flags: AvCodecFlags,
    thread_count: usize,
    media_type: MediaType,
    codec_name: Option<String>,
    codec_opts: Option<Options>,
    filters: Option<Vec<Filter>>,
    hw_device_config: Option<HWDeviceConfig>,
    scale_algorithm: ScaleAlgorithm,
    resize: Option<Resize>,
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
            thread_count: num_cpus::get(),
            flags: AvCodecFlags::LOW_DELAY,
            scale_algorithm: ScaleAlgorithm::default(),
            resize: None,
        }
    }

    /// Set decoding flags.
    pub fn with_flags(mut self, flags: AvCodecFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set the codec name to use for decoding.
    /// If not set, the decoder will try to guess the codec based on the input.
    pub fn with_codec_name(mut self, codec_name: impl Into<Option<String>>) -> Self {
        self.codec_name = codec_name.into();
        self
    }

    /// codec options to use for decoding.
    pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
        self.codec_opts = options.into();
        self
    }

    /// set the thread count.
    pub fn with_thread_count(mut self, thread_count: usize) -> Self {
        self.thread_count = thread_count;
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

    /// Set the scaling algorithm used when converting decoded frames to a
    /// canonical pixel format (e.g. YUV420P).
    ///
    /// Defaults to [`ScaleAlgorithm::Bicubic`]. Use
    /// [`ScaleAlgorithm::Bilinear`] for output consistent with FFmpeg's command
    /// line default, or [`ScaleAlgorithm::Area`] when downscaling.
    pub fn with_scale_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.scale_algorithm = algorithm;
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

    fn setup_codec_context(&self, decoder: &mut AVCodecContext, input: &AVStream) -> Result<()> {
        let media_type = self.media_type;
        if media_type as ffi::AVMediaType != decoder.codec_type {
            return Err(Error::msg(format!(
                "Decoder codec type not supported: {:?} vs. {:?}",
                media_type, decoder.codec_type
            )));
        }

        decoder.apply_codecpar(&input.codecpar())?;
        decoder.set_flags(self.flags as i32);
        decoder.set_time_base(input.time_base);
        decoder.set_pkt_timebase(input.time_base);
        if let Some(framerate) = input.guess_framerate() {
            decoder.set_framerate(framerate);
        }

        unsafe {
            (*decoder.as_mut_ptr()).thread_count = self.thread_count as i32;
        }

        Ok(())
    }

    /// 构建一个**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 适合需要精细控制 reader 生命周期的高级场景：解码时需每帧传入 reader
    /// （`decoder.decode(&mut reader)`），且**不支持 seek**（seek 需要 reader，
    /// 请改用 [`build_wrapped`](DecoderBuilder::build_wrapped)）。
    pub fn build(self, source: impl Into<Location>) -> Result<Decoder> {
        let reader = StreamReader::new(source)?;
        self.build_from_reader(&reader)
    }

    /// 构建一个持有 reader 的 [`DecoderWrapper`]（**推荐入口**）。
    ///
    /// 返回的 [`DecoderWrapper`] 封装了 reader，解码无需每帧传参
    /// （`decoder.decode()` / `decoder.decode_frame()`），并支持直接
    /// [`seek_to_frame`](DecoderWrapper::seek_to_frame) /
    /// [`seek_to_timestamp`](DecoderWrapper::seek_to_timestamp)。
    pub fn build_wrapped(
        self,
        source: impl Into<Location>,
    ) -> Result<DecoderWrapper<StreamReader>> {
        let reader = StreamReader::new(source)?;
        self.build_wrapped_with_reader(reader)
    }

    /// 用自定义 reader 构建持有 reader 的 [`DecoderWrapper`]。
    ///
    /// 当需要从自定义 [`Reader`]（如网络流、内存缓冲）解码时使用，行为与
    /// [`build_wrapped`](DecoderBuilder::build_wrapped) 一致。
    pub fn build_wrapped_with_reader<R: Reader>(self, reader: R) -> Result<DecoderWrapper<R>> {
        let decoder = self.build_from_reader(&reader)?;
        Ok(DecoderWrapper::new(decoder, reader))
    }

    /// 用给定的 reader 构建**裸** [`Decoder`]（不持有 reader）。
    ///
    /// 高级/内部场景使用（如 mux 多流共享同一 reader）。解码时需每帧传入
    /// reader，且不支持 seek。日常使用请优先 [`build_wrapped`](DecoderBuilder::build_wrapped)。
    pub fn build_from_reader<R: Reader>(self, reader: &R) -> Result<Decoder> {
        let media_type = self.media_type;
        let (stream_index, codec_name) = reader.find_best_stream(media_type)?;
        let input_stream = reader
            .input()
            .streams()
            .get(stream_index)
            .ok_or(Error::msg(format!("stream: {stream_index} not found!")))?;

        let codec = {
            let codec_name = if let Some(ref codec_name) = self.codec_name {
                codec_name.as_str()
            } else {
                codec_name.as_str()
            };
            AVCodec::find_decoder_by_name(&utils::from_str(codec_name))
                .context(format!("Failed to find decoder by name: '{codec_name}'"))?
        };

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
            .map(|cfg| {
                // codec support or not for hardware acceleration
                let hw_pixel = cfg
                    .device_type
                    .find_hw_pixel_format_with_codec(&codec)
                    .ok_or_else(|| {
                        let codec_name = utils::to_string(codec.name()).unwrap();
                        Error::msg(format!(
                            "Decoder with HW acceleration is not supported for codec: {codec_name}"
                        ))
                    })?;

                log::info!(
                    "Video decoder with HW acceleration codec: {:?}, hw_pixel: {:?}, config: {:#?}",
                    codec.name(),
                    PixelFormat::from(hw_pixel),
                    cfg
                );

                // create hardware context
                HWContext::new(cfg)
                    .and_then(|ctx| {
                        // 注意：setup_hw_frames 可能会改变 decode_ctx.pix_fmt
                        ctx.setup_hw_frames(true, &mut decode_ctx, init_width, init_height)?;
                        Ok(ctx)
                    })
                    .context("Hardware acceleration context initialization failed")
            })
            .transpose()?;

        let dict = self.codec_opts.map(|opts| opts.into_dict());
        decode_ctx
            .open(dict)
            .context("Failed to open decoder for stream")?;

        let stream_info = StreamInfo::from_stream(input_stream)?;
        log::info!("{stream_info}");

        let filter_graph = if let Some(filters) = self.filters {
            let filter_params = match media_type {
                MediaType::VIDEO => {
                    FilterParams::Video(VideoParams {
                        width: init_width,
                        height: init_height,
                        format: PixelFormat::YUV420P, // 确保视频帧 filter 的输入格式是 YUV420P
                        time_base: decode_ctx.time_base,
                        frame_rate: decode_ctx.framerate,
                        pixel_aspect: decode_ctx.sample_aspect_ratio,
                    })
                }
                MediaType::AUDIO => FilterParams::Audio(AudioParams {
                    nb_channels: decode_ctx.ch_layout.nb_channels,
                    sample_rate: decode_ctx.sample_rate,
                    format: SampleFormat::from(decode_ctx.sample_fmt),
                    time_base: decode_ctx.time_base,
                }),
                _ => panic!("Unsupported filter for media type: {media_type:?}"),
            };

            let mut graph = FilterGraph::new();
            // 验证 Filter 链的媒体类型是否与当前流匹配
            if !filters.iter().all(|f| f.media_type() == media_type) {
                return Err(Error::msg(format!(
                    "Filter media type mismatch for stream type {media_type:?}"
                )));
            }
            graph
                .init(&filter_params, filters.as_slice())
                .context("Failed to initialize filter graph")?;

            Some(graph)
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
            state: DecoderState::Normal,
            scale_algorithm: self.scale_algorithm,
            resize: self.resize,
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum DecoderState {
    Normal,
    Drained,
    Flushed,
}

/// Decode video files and streams.
///
/// # Example
///
/// ```ignore
/// let decoder = Decoder::new(Path::new("video.mp4")).unwrap();
/// decoder
///     .decode_iter()
///     .take_while(Result::is_ok)
///     .for_each(|frame| println!("Got frame!"),
/// );
/// ```
pub struct Decoder {
    context: AVCodecContext,
    filter_graph: Option<FilterGraph>,
    hw_context: Option<Arc<HWContext>>,
    /// (r_frame_rate, avg_frame_rate)
    frame_rate: (f32, f32),
    nb_frames: i64,
    duration: Time,
    stream_index: usize,
    media_type: MediaType,
    state: DecoderState,
    scale_algorithm: ScaleAlgorithm,
    resize: Option<Resize>,
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

    #[inline]
    pub fn pix_fmt(&self) -> PixelFormat {
        self.context.pix_fmt.into()
    }

    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.context.sample_rate
    }

    #[inline]
    pub fn sample_fmt(&self) -> SampleFormat {
        SampleFormat::from(self.context.sample_fmt)
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

    /// Get the decoders input stream number of frames
    #[inline(always)]
    pub fn frames(&self) -> i64 {
        self.nb_frames
    }

    /// Get the decoders input frame rate
    ///
    /// # Return
    /// A tuple of the frame rate of float values
    ///
    /// `0`: r_frame_rate
    /// `1`: avg_frame_rate
    ///
    #[inline(always)]
    pub fn frame_rate(&self) -> (f32, f32) {
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
        self.state == DecoderState::Drained
    }

    pub fn is_flushed(&self) -> bool {
        self.state == DecoderState::Flushed
    }

    /// 解码器是否已完全结束：解码器到达 EOF，且 filter（如有）内部缓冲帧也已全部
    /// 冲刷完毕。仅当二者都满足时，才禁止继续调用 `decode`/`decode_raw`。否则
    /// （解码器已 Flushed 但 filter 仍有多余缓冲帧待冲刷，如延迟滤镜 `framerate`），
    /// 仍需允许继续调用以取回剩余帧，否则会丢帧或报"cannot decode after flushed"。
    fn is_complete(&self) -> bool {
        self.is_flushed()
            && match &self.filter_graph {
                Some(graph) => graph.is_flushed(),
                None => true,
            }
    }

    /// Decode a single frame.
    ///
    /// # Return value
    ///
    /// A tuple of the frame timestamp (relative to the stream) and the frame itself.
    ///
    /// # Example
    ///
    /// ```ignore
    /// loop {
    ///     let (ts, frame) = decoder.decode::<u8>()?;
    ///     // Do something with frame...
    /// }
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn decode<T>(&mut self, reader: &mut impl Reader) -> Result<Option<MediaFrame<T>>>
    where
        T: MediaFrameType,
    {
        if self.is_complete() {
            return Err(Error::msg(
                "Decoder cannot decode after flushed. Call reset().",
            ));
        }

        let mut read_exhausted = false;
        Ok(loop {
            if !read_exhausted {
                match reader.read_packet() {
                    Ok(Some((stream, packet))) => {
                        if stream.index() != self.stream_index() {
                            // skip other streams
                            log::trace!("skip stream index: {}, {:?}", stream.index(), packet);
                            continue;
                        }
                        if let Some(frame) = self.decode_packet(&packet)? {
                            break Some(frame);
                        }
                    }
                    Ok(None) => {
                        log::debug!("No more packets, Reader exhausted.");
                        read_exhausted = true;
                        continue;
                    }
                    Err(e) => {
                        log::error!("Error reading packet: {e}");
                        return Err(e);
                    }
                }
            } else {
                match self.drain() {
                    Ok(Some(frame)) => {
                        break Some(frame);
                    }
                    Ok(None) => {
                        // None 可能来自 Drained（EAGAIN，解码器仍有缓冲帧待产出）或
                        // Flushed（EOF）。若是 Drained 需继续 drain，否则会丢失尾部帧
                        // （多见于含 B 帧的码流）。
                        if self.is_drained() {
                            log::debug!("Decoder drained, keep draining.");
                            continue;
                        }
                        log::debug!("Decoder flushed. EOF reached.");
                        // self.reset();
                        // read_exhausted = false;
                        break None;
                    }
                    Err(e) => {
                        log::error!("Error to drain decoder: {e}");
                        return Err(e);
                    }
                }
            }
        })
    }

    /// Decode a single frame as a `MediaFrame<u8>` (video) or `MediaFrame<f32>` (audio).
    ///
    /// Convenience for `decode::<u8>()` which is the common video path.
    ///
    /// # Return value
    ///
    /// The decoded frame, or [`None`] at end of stream.
    #[cfg(feature = "ndarray")]
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
        if self.is_complete() {
            return Err(Error::msg(
                "Decoder cannot decode after flushed. Call reset().",
            ));
        }

        let mut read_exhausted = false;
        Ok(loop {
            if !read_exhausted {
                match reader.read_packet() {
                    Ok(Some((stream, packet))) => {
                        if stream.index() != self.stream_index() {
                            // skip other streams
                            log::trace!("skip stream index: {}, {:?}", stream.index(), packet);
                            continue;
                        }
                        if let Some(frame) = self.decode_raw_packet(&packet)? {
                            break Some(frame);
                        }
                    }
                    Ok(None) => {
                        log::debug!("No more packets, Reader exhausted.");
                        read_exhausted = true;
                        continue;
                    }
                    Err(e) => {
                        log::error!("Error reading packet: {e}");
                        return Err(e);
                    }
                }
            } else {
                match self.drain_raw() {
                    Ok(Some(frame)) => {
                        break Some(frame);
                    }
                    Ok(None) => {
                        if self.is_drained() {
                            log::debug!("Decoder drained, keep draining.");
                            continue;
                        }
                        log::debug!("Decoder flushed. EOF reached.");
                        // self.reset();
                        // read_exhausted = false;
                        break None;
                    }
                    Err(e) => {
                        log::error!("Error to drain decoder: {e}");
                        return Err(e);
                    }
                }
            }
        })
    }

    /// Decode a [`Packet`].
    ///
    /// Feeds the packet to the decoder and returns a frame if there is one available. The caller
    /// should keep feeding packets until the decoder returns a frame.
    ///
    /// # Panics
    ///
    /// Panics if in draining mode.
    ///
    /// # Return value
    ///
    /// A tuple of the [`Frame`] and timestamp (relative to the stream) and the frame itself if the
    /// decoder has a frame available, [`None`] if not.
    #[cfg(feature = "ndarray")]
    pub fn decode_packet<T>(&mut self, packet: &AVPacket) -> Result<Option<MediaFrame<T>>>
    where
        T: MediaFrameType,
    {
        match self.decode_raw_packet(packet) {
            Ok(Some(raw_frame)) => Ok(Some(self.raw_frame_to_media_frame(raw_frame)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Decode a [`Packet`].
    ///
    /// Feeds the packet to the decoder and returns a frame if there is one available. The caller
    /// should keep feeding packets until the decoder returns a frame.
    ///
    /// # Panics
    ///
    /// Panics if in draining mode.
    ///
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    pub fn decode_raw_packet(&mut self, packet: &AVPacket) -> Result<Option<AVFrame>> {
        self.send_packet_to_decoder(Some(packet))?;
        self.receive_frame_from_decoder()
    }

    /// Drain one frame from the decoder.
    ///
    /// After calling drain once the decoder is in draining mode and the caller may not use normal
    /// decode anymore, or it will panic.
    ///
    /// # Return value
    ///
    /// A tuple of the [`Frame`] and timestamp (relative to the stream) and the frame itself if the
    /// decoder has a frame available, [`None`] if not.
    #[cfg(feature = "ndarray")]
    pub fn drain<T>(&mut self) -> Result<Option<MediaFrame<T>>>
    where
        T: MediaFrameType,
    {
        match self.drain_raw() {
            Ok(Some(raw_frame)) => Ok(Some(self.raw_frame_to_media_frame(raw_frame)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    #[cfg(feature = "ndarray")]
    fn raw_frame_to_media_frame<T>(&self, frame: AVFrame) -> Result<MediaFrame<T>>
    where
        T: MediaFrameType,
    {
        // Video Frame pixel YUV420P, RGB24 is supported
        MediaFrame::<T>::from_avframe(&frame)
    }

    /// Drain one frame from the decoder.
    ///
    /// After calling drain once the decoder is in draining mode and the caller may not use normal
    /// decode anymore, or it will panic.
    ///
    /// # Return value
    ///
    /// The decoded raw frame as [`AVFrame`] if the decoder has a frame available, [`None`] if not.
    ///
    /// 作为低层手动解码 API 的一部分公开：配合 [`into_parts`](Self::into_parts) 与
    /// [`decode_raw_packet`](Self::decode_raw_packet) 使用，可逐 packet 送入解码器并排空
    /// 缓冲帧。需要 [`MediaFrame`] 的高级调用请使用 [`drain`](Self::drain)。
    pub fn drain_raw(&mut self) -> Result<Option<AVFrame>> {
        if self.state == DecoderState::Normal {
            self.send_packet_to_decoder(None)?;
            // 已发送 EOS，进入 draining 模式。此后 EAGAIN 表示"仍在 drain"，
            // 而非 read 阶段缺包，因此在此处显式置位。
            self.state = DecoderState::Drained;
        }
        self.receive_frame_from_decoder()
    }

    /// Reset the decoder to be used again after draining.
    pub fn reset(&mut self) {
        self.flush();
        self.state = DecoderState::Normal;
    }

    fn flush(&mut self) {
        unsafe {
            ffi::avcodec_flush_buffers(self.context.as_mut_ptr());
        }
    }

    /// Send packet to decoder.
    /// Ensure rescaling timestamps accordingly before sending to decoder.
    fn send_packet_to_decoder(&mut self, packet: Option<&AVPacket>) -> Result<()> {
        self.context
            .send_packet(packet)
            .context("Failed to send packet to decoder")?;
        Ok(())
    }

    /// Receive packet from decoder. Will handle hwaccel conversions and scaling as well.
    fn receive_frame_from_decoder(&mut self) -> Result<Option<AVFrame>> {
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
                match self.state {
                    DecoderState::Normal | DecoderState::Drained => return Ok(None),
                    DecoderState::Flushed => {
                        if let Some(graph) = self.filter_graph.as_mut()
                            && !graph.is_flushed()
                        {
                            match graph.process_frame(None)? {
                                Some(frame) => return Ok(Some(frame)),
                                None => {
                                    // 已无更多缓冲帧（graph 此时已 Flushed）
                                    debug_assert!(graph.is_flushed());
                                }
                            }
                        }
                        return Ok(None);
                    }
                }
            }
            Err(e) => return Err(e),
        };

        // 2. 处理硬件加速帧下载,
        let sw_frame = match &self.hw_context {
            Some(hw_ctx) if hw_ctx.is_hw_frame(&decoded_frame) => {
                // hw_frame -> sw_frame
                hw_ctx
                    .hw_download(&mut self.context, &decoded_frame)
                    .context("Failed HW frame download")?
            }
            _ => {
                // 已经是 CPU 帧或无 HWaccel
                decoded_frame
            }
        };

        // 3. 确保输入Filter的视频帧的格式为 YUV420P
        // 例如：
        // 无硬件加速，默认解码格式 YUV420P
        // 存在硬件加速帧，则转换 NV12 -> YUV420P
        // 注意：这里无论是否有 Filter，都会统一转成 YUV420P（因为
        // `MediaFrame` 视频目前主要支持 YUV420P/RGB24）。因此即使源是
        // yuv444p / nv12 / 10-bit，解码输出也会被转成 YUV420P，不会颜色错乱。
        let raw_frame = match self.media_type {
            MediaType::VIDEO => {
                let target_sw_pix_fmt = PixelFormat::YUV420P;
                // 计算目标尺寸：无 resize 时保持源尺寸，有 resize 时按策略计算
                let (out_w, out_h) = match self.resize {
                    Some(resize) => resize
                        .compute_for((sw_frame.width as u32, sw_frame.height as u32))
                        .ok_or_else(|| {
                            let (w, h) = (sw_frame.width, sw_frame.height);
                            Error::msg(format!("Cannot resize frame {w}x{h} into {resize:?}"))
                        })?,
                    None => (sw_frame.width as u32, sw_frame.height as u32),
                };
                if sw_frame.format != target_sw_pix_fmt.into()
                    || sw_frame.width != out_w as i32
                    || sw_frame.height != out_h as i32
                {
                    swctx::scale_with_flags(
                        &sw_frame,
                        out_w as i32,
                        out_h as i32,
                        target_sw_pix_fmt,
                        self.scale_algorithm,
                    )?
                } else {
                    sw_frame
                }
            }
            _ => {
                // do nothing
                sw_frame
            }
        };

        // 4. 应用 Filter Graph
        if let Some(graph) = self.filter_graph.as_mut() {
            // filter process
            match graph.process_frame(Some(raw_frame))? {
                Some(filtered_frame) => Ok(Some(filtered_frame)),
                None => {
                    if graph.is_drained() {
                        // Filter graph 当前输入帧未能产生输出帧，需要继续尝试拉取
                        log::debug!("Filter graph drained, trying again.");
                        // 在这种情况下，我们应该返回 Ok(None)，让外层循环继续驱动解码器 或 filter graph
                    } else if graph.is_flushed() {
                        // Filter graph 当前输入帧未能产生输出帧，已经到达 EOF
                        log::error!("Filter graph flushed. EOF reached, should not happened.");
                    } else {
                        log::warn!("Filter graph did not output a frame.");
                    }
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
                log::debug!("Decoder drained. try send new packet again.");
                // self.state = DecoderState::Drained;
                Ok(None)
            }
            Err(rsmpeg::error::RsmpegError::DecoderFlushedError) => {
                log::debug!("Decoder flushed. EOF reached.");
                self.state = DecoderState::Flushed;
                Ok(None)
            }
            Err(e) => {
                log::warn!("Failed to receive frame from decoder: {e}");
                Err(Error::new(e))
            }
        }
    }
}

/// Important note: Do not forget to drain the decoder after the reader is exhausted. It may still
/// contain frames. Run `drain_raw()` or `drain()` in a loop until no more frames are produced.
impl Drop for Decoder {
    fn drop(&mut self) {
        // 1. Flush Filter Graph if exists.
        if let Some(graph) = self.filter_graph.as_mut() {
            match graph.flush() {
                Ok(frames) => {
                    if !frames.is_empty() {
                        log::warn!(
                            "{} frames dropped during Decoder drop filter flush.",
                            frames.len()
                        );
                    }
                    log::debug!("Filter graph flushed during Decoder drop.");
                }
                Err(e) => log::error!("Failed to flush filter graph during Decoder drop: {e}"),
            }
        }

        // We need to drain the items still in the decoders queue.
        match self.send_packet_to_decoder(None) {
            Ok(_) => {
                // 兜底上限：个别解码器可能持续返回 EAGAIN 而迟迟不结束，
                // 与 encode.rs 的 1_000 保护一致，防止 Drop 排空无限循环。
                const MAX_DRAIN_ITERATIONS: usize = 1_000;
                let mut iterations = 0usize;
                loop {
                    if iterations >= MAX_DRAIN_ITERATIONS {
                        log::warn!(
                            "Decoder drain exceeded {MAX_DRAIN_ITERATIONS} iterations, forcing EOF."
                        );
                        break;
                    }
                    iterations += 1;
                    match self.decoder_receive_frame() {
                        Ok(Some(_frame)) => {
                            // If receive a frame, we continue to drain the queue.
                            log::debug!("continue draining decoder queue.");
                        }
                        Ok(None) => {
                            if self.is_drained() {
                                // If we need more, we continue to drain the queue.
                                log::debug!("Decoder draining. continue...");
                                continue;
                            } else {
                                log::debug!("Decoder flushed. EOF reached.");
                                break;
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to drain decoder: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                log::warn!("Failed to send flush packet to decoder: {e}")
            }
        }

        unsafe {
            // explicitly drop the hw_context to release the hardware resources
            // 1. malloc(): unsorted double linked list corrupted
            // 2. malloc(): mismatching next->prev_size (unsorted)
            // 3. free(): invalid pointer
            // 4. double free or corruption (!prev)
            // 5. corrupted double-linked list Aborted (core dumped)
            let codec_ctx_ptr = self.context.as_mut_ptr();
            if !codec_ctx_ptr.is_null() {
                if !(*codec_ctx_ptr).hw_frames_ctx.is_null() {
                    let _hw_frames = (*codec_ctx_ptr).hw_frames_ctx;
                    (*codec_ctx_ptr).hw_frames_ctx = std::ptr::null_mut();
                }

                if !(*codec_ctx_ptr).hw_device_ctx.is_null() {
                    let _hw_device = (*codec_ctx_ptr).hw_device_ctx;
                    (*codec_ctx_ptr).hw_device_ctx = std::ptr::null_mut();
                }
            }
        }
    }
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

/// 解码器包装器，持有 Decoder 和 Reader
pub struct DecoderWrapper<R: Reader> {
    reader: R,
    decoder: Decoder,
    stream_info: StreamInfo,
}

impl<R: Reader> DecoderWrapper<R> {
    /// 创建一个新的解码器包装器
    pub fn new(decoder: Decoder, reader: R) -> Self {
        let stream_info = StreamInfo::from_reader(&reader, decoder.stream_index()).unwrap();
        Self {
            reader,
            decoder,
            stream_info,
        }
    }

    /// 解码下一帧（媒体帧）
    #[cfg(feature = "ndarray")]
    pub fn decode<T: MediaFrameType>(&mut self) -> Result<Option<MediaFrame<T>>> {
        self.decoder.decode(&mut self.reader)
    }

    /// 解码下一帧（`MediaFrame<u8>` 便捷方法，等价于 `decode::<u8>()`）
    #[cfg(feature = "ndarray")]
    pub fn decode_frame(&mut self) -> Result<Option<MediaFrame<u8>>> {
        self.decoder.decode(&mut self.reader)
    }

    /// 解码下一帧（原始帧）
    pub fn decode_raw(&mut self) -> Result<Option<AVFrame>> {
        self.decoder.decode_raw(&mut self.reader)
    }

    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    /// 获取内部解码器的可变引用
    pub fn decoder_mut(&mut self) -> &mut Decoder {
        &mut self.decoder
    }

    /// 获取内部读取器的可变引用
    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// 解构并返回内部组件
    pub fn into_parts(self) -> (Decoder, R) {
        (self.decoder, self.reader)
    }

    /// Seek in reader.
    ///
    /// See [`StreamReader::seek_to_time`](crate::io::StreamReader::seek_to_timestamp) for more information.
    #[inline]
    pub fn seek_to_timestamp(&mut self, timestamp_milliseconds: i64) -> Result<()> {
        if let Some(stream_reader) = self.reader.as_any_mut().downcast_mut::<StreamReader>() {
            stream_reader
                .seek_to_timestamp(timestamp_milliseconds)
                .inspect(|_| self.decoder.flush())
        } else {
            Err(Error::msg("Seek is only supported for StreamReader"))
        }
    }

    /// Seek to specific frame in reader.
    ///
    /// See [`StreamReader::seek_to_frame`](crate::io::StreamReader::seek_to_frame) for more information.
    #[inline]
    pub fn seek_to_frame(&mut self, frame_number: i64) -> Result<()> {
        if let Some(stream_reader) = self.reader.as_any_mut().downcast_mut::<StreamReader>() {
            stream_reader
                .seek_to_frame(
                    self.decoder.stream_index(),
                    frame_number,
                    ffi::AVSEEK_FLAG_ANY as i32,
                )
                .inspect(|_| self.decoder.flush())
        } else {
            Err(Error::msg(
                "Seek to frame is only supported for StreamReader",
            ))
        }
    }

    /// Seek to start of reader.
    ///
    /// See [`StreamReader::seek_to_start`](crate::io::StreamReader::seek_to_start) for more information.
    #[inline]
    pub fn seek_to_start(&mut self) -> Result<()> {
        if let Some(stream_reader) = self.reader.as_any_mut().downcast_mut::<StreamReader>() {
            stream_reader
                .seek_to_start()
                .inspect(|_| self.decoder.flush())
        } else {
            Err(Error::msg(
                "Seek to start is only supported for StreamReader",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter;
    use std::collections::HashSet;

    #[test]
    fn test_decode_video() -> Result<()> {
        let video_path = std::path::Path::new("assets/mp4.mp4");

        // drawtext 依赖 libfreetype 编译进 FFmpeg，部分构建未启用，失败时降级为仅 scale。
        let scale = filter::video::scale(1280, 720, None);
        let drawtext = filter::video::DrawText::new("Hello", 10, 10, 24, "white").build();
        let build_decoder = |filters| {
            DecoderBuilder::new(MediaType::VIDEO)
                .with_filters(filters)
                .build_wrapped(video_path)
        };
        let mut decoder = match build_decoder(vec![scale, drawtext]) {
            Ok(d) => d,
            Err(e)
                if format!("{e:#}").to_lowercase().contains("no such filter")
                    || format!("{e:#}").to_lowercase().contains("not found") =>
            {
                println!("SKIP drawtext (libfreetype not available): {e:#}");
                build_decoder(vec![filter::video::scale(1280, 720, None)])?
            }
            Err(e) => return Err(e),
        };

        loop {
            match decoder.decode_raw() {
                Ok(Some(frame)) => {
                    println!("video frame: {:?}, timebase:{:?}", frame, frame.time_base);
                }
                Ok(None) => {
                    println!("No more frames, decoder flushed");
                    break;
                }
                Err(e) => {
                    log::error!("Error decoding frame: {}", e);
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

        let mut decoder = DecoderBuilder::new(MediaType::AUDIO)
            .with_filters(filters)
            .build_wrapped(audio_path)?;

        loop {
            match decoder.decode_raw() {
                Ok(Some(frame)) => {
                    println!("audio frame: {:?}, timebase:{:?}", frame, frame.time_base);
                }
                Ok(None) => {
                    println!("No more frames, decoder flushed");
                    break;
                }
                Err(e) => {
                    log::error!("Error decoding frame: {}", e);
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_video_with_resize() -> Result<()> {
        use crate::Resize;

        let video_path = std::path::Path::new("assets/mp4.mp4");

        let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .build_wrapped(video_path)?;

        let mut frames = 0usize;
        while let Some(frame) = decoder.decode_raw()? {
            assert_eq!(frame.width, 320);
            assert_eq!(frame.height, 240);
            frames += 1;
        }
        assert!(frames > 0, "expected at least one decoded frame");

        Ok(())
    }

    /// 验证 `with_resize` 与 `scale` filter 两种缩放方式结果一致，且同时使用时
    /// 按「先 resize 后 filter」的顺序叠加，不冲突。
    #[test]
    fn test_resize_vs_filter_scale() -> Result<()> {
        use crate::Resize;

        let video_path = std::path::Path::new("assets/mp4.mp4");

        // A) 仅 with_resize
        eprintln!("[A] with_resize only");
        let mut dec_a = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .build_wrapped(video_path)?;
        let mut a_dims = HashSet::new();
        while let Some(f) = dec_a.decode_raw()? {
            a_dims.insert((f.width, f.height));
        }

        // B) 仅 scale filter
        eprintln!("[B] filter only");
        let mut dec_b = DecoderBuilder::new(MediaType::VIDEO)
            .with_filters(vec![filter::video::scale(320, 240, None)])
            .build_wrapped(video_path)?;
        let mut b_dims = HashSet::new();
        while let Some(f) = dec_b.decode_raw()? {
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
        let mut dec_c = DecoderBuilder::new(MediaType::VIDEO)
            .with_resize(Resize::Exact(320, 240))
            .with_filters(vec![filter::video::scale(640, 480, None)])
            .build_wrapped(video_path)?;
        let mut c_dims = HashSet::new();
        while let Some(f) = dec_c.decode_raw()? {
            c_dims.insert((f.width, f.height));
        }
        assert_eq!(
            c_dims,
            HashSet::from([(640i32, 480i32)]),
            "resize+filter should compose to the filter size"
        );

        Ok(())
    }

    /// 生成 `n_frames` 帧、`fps` 帧率的纯色小视频（不含 B 帧），供延迟滤镜 EOF 回归测试使用。
    #[cfg(feature = "ndarray")]
    fn make_test_video(
        path: &std::path::Path,
        width: usize,
        height: usize,
        n_frames: usize,
        fps: f32,
    ) -> Result<()> {
        use crate::{colors, encode::EncoderBuilder};

        let mut encoder = EncoderBuilder::new_video(width, height)
            .with_fps(fps)
            .build_wrapped(path)?;
        for i in 0..n_frames {
            let rgb = colors::hsv_to_rgb(i as f32 / n_frames as f32 * 360.0, 100.0, 100.0);
            let mut frame = MediaFrame::<u8>::new_video_frame(
                width,
                height,
                PixelFormat::RGB24,
                crate::time::new_rational(1, 24),
            )?;
            for y in 0..height {
                for x in 0..width {
                    frame.data[[y, x, 0]] = rgb[0];
                    frame.data[[y, x, 1]] = rgb[1];
                    frame.data[[y, x, 2]] = rgb[2];
                }
            }
            encoder.write_frame(frame)?;
        }
        encoder.finish()?;
        Ok(())
    }

    /// 自包含回归测试：验证解码器带「延迟滤镜」时，EOF 阶段的 filter flush 不会报错或丢帧。
    /// 延迟滤镜（如 `framerate` 缓冲插值帧、`setpts` 重排帧）需在解码器 EOF 后逐帧冲刷；
    /// 若 flush 重复向 buffersrc 发送 EOF，会得到 `AVERROR_EOF` 并中断解码（即已修复的
    /// filter-EOF 类 BUG）。
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_decode_delayed_filter_eof() -> Result<()> {
        let width = 64usize;
        let height = 64usize;
        // (滤镜名, 参数, 输入帧数, fps, 期望最小输出帧数)
        let cases: &[(&str, &str, usize, f32, usize)] = &[
            ("framerate", "framerate=fps=30", 30, 30.0, 30),
            ("setpts", "setpts=PTS*2", 24, 24.0, 23),
        ];

        for (i, (name, spec, n_frames, fps, min_frames)) in cases.iter().enumerate() {
            let path = crate::test_utils::test_output_path(
                "decode",
                &format!("rsmedia_decode_delayed_{i}.mp4"),
            );
            crate::test_utils::remove_test_output(&path);
            make_test_video(&path, width, height, *n_frames, *fps)?;

            let filters = vec![Filter::new(name, MediaType::VIDEO, spec.to_string())];
            let mut decoder = DecoderBuilder::new(MediaType::VIDEO)
                .with_filters(filters)
                .build_wrapped(path.as_path())?;
            let mut count = 0usize;
            while let Some(_f) = decoder.decode_raw()? {
                count += 1;
            }
            assert!(
                count >= *min_frames,
                "{name} delayed-filter decode dropped frames: got {count}, expected >= {min_frames}"
            );

            let _ = std::fs::remove_file(&path);
        }
        Ok(())
    }
}
