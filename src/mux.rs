use crate::error::{Context, Result, RsmediaError};
use crate::filter::Filter;
use crate::hwaccel::HWDeviceConfig;
use crate::io::{Reader, Writer};
use crate::options::{Metadata, Options};
use crate::stream::{MediaType, StreamInfo};
use crate::subtitle::SubtitleSegment;
use crate::{
    Decoder, DecoderBuilder, Encoder, EncoderBuilder, Location, StreamReader, StreamWriter,
};

use rsmpeg::avcodec::AVPacket;
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use std::collections::HashMap;

/// A container chapter mark (MP4/MKV chapters), with times in seconds.
///
/// Chapter times are stored internally on a millisecond time base (1/1000),
/// matching what `ffmetadata` and most container tooling use.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Chapter id requested from the container; `None` (the [`Chapter::new`]
    /// default) means "auto-assign".
    ///
    /// The muxer decides which ids actually reach the file: MP4 (`movenc`) numbers
    /// chapters by position from 0, Matroska from 1 — an explicit `Some(..)` is not
    /// round-tripped. Reading chapters back therefore always yields `Some(..)`,
    /// holding the value the container chose.
    pub id: Option<i64>,
    /// Human-readable chapter title, stored as the chapter's `title` metadata.
    pub title: String,
    /// Chapter start time in seconds.
    pub start: f64,
    /// Chapter end time in seconds (exclusive).
    pub end: f64,
}

impl Chapter {
    /// Creates a chapter with an auto-assigned id and the given title.
    pub fn new(title: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            id: None,
            title: title.into(),
            start,
            end,
        }
    }

    /// Returns the chapter duration in seconds.
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Represents a muxer. A muxer allows muxing media packets into a new container format. Muxing does
/// not require encoding and/or decoding.
///
/// # Examples
///
/// Mux to an MKV file:
///
/// ```no_run
/// use std::path::Path;
/// use rsmedia::mux::Muxer;
/// let mut muxer = Muxer::new("to_file.mkv").unwrap();
/// // Add streams and mux packets...
/// muxer.finish().unwrap();
/// ```
///
/// Mux from file to MP4 and print length of first 100 buffer segments:
///
/// ```no_run
/// use std::path::Path;
/// use rsmedia::mux::Muxer;
/// use rsmedia::error::Result;
/// fn main() -> Result<()> {
///     let mut muxer = Muxer::new("output.mp4")?;
///     // Add streams and mux packets...
///     muxer.finish()?;
///     Ok(())
/// }
/// ```
pub struct Muxer<W: Writer> {
    pub writer: W,
    streams: Vec<MuxerStream>,
    interleaved: bool,
    /// `true` 时把提交进来的时间戳整体平移到 0 起点，见
    /// [`Muxer::set_normalize_timestamps`]。
    normalize_timestamps: bool,
    /// 归一化基准（微秒，`AV_TIME_BASE`），首个带有效时间戳的提交单元建立。
    /// 存微秒而**不是**某个流的时间基单位：视频（1/fps）与音频（1/sample_rate）
    /// 减的必须是同一物理时刻，否则会破坏音视频同步。
    pts_base_us: Option<i64>,
    have_written_header: bool,
    have_written_trailer: bool,
    /// Container-level metadata (e.g. "title", "artist"), applied to the
    /// format context right before the header is written.
    metadata: Metadata,
    /// Per-stream metadata (e.g. "language"), keyed by the output stream
    /// index returned from [`Muxer::add_encoder`], applied right before the
    /// header is written.
    stream_metadata: HashMap<usize, Metadata>,
    /// Container chapters, applied right before the header is written.
    chapters: Vec<Chapter>,
    /// `true` once [`Self::apply_chapters`] has transferred the chapter nodes to
    /// the format context, so a retried header write does not allocate a second
    /// set of nodes and leak the first one.
    chapters_applied: bool,
}

/// 单个输出流。既可以是编码流（持有 [`Encoder`]，由 [`Muxer::add_encoder`]
/// 创建，输入 `AVFrame`），也可以是**透传流**（`encoder` 为 `None`，由
/// [`Muxer::add_copy_stream`] 创建，直接写入原始 `AVPacket`）。
pub struct MuxerStream {
    pub encoder: Option<Encoder>,
    pub stream_info: StreamInfo,
    pub media_type: MediaType,
    pub stream_index: usize,
    /// 透传/remux 模式下源流的时间基，用于把 `mux_packet` 的 pts/dts
    /// 从源流时间基换算到输出流时间基。
    pub src_time_base: Option<ffi::AVRational>,
}

impl MuxerStream {
    pub fn new_encoded(encoder: Encoder, stream_info: StreamInfo) -> Self {
        let media_type = encoder.media_type();
        let stream_index = stream_info.index;
        Self {
            encoder: Some(encoder),
            media_type,
            stream_info,
            stream_index,
            src_time_base: None,
        }
    }

    /// 透传流：直接拷贝源的编解码参数，`src_time_base` 用于时间戳换算。
    pub fn new_copy(stream_info: StreamInfo, src_time_base: ffi::AVRational) -> Self {
        let media_type = stream_info.media_type;
        let stream_index = stream_info.index;
        Self {
            encoder: None,
            media_type,
            stream_info,
            stream_index,
            src_time_base: Some(src_time_base),
        }
    }
}

impl Muxer<StreamWriter> {
    pub fn new(destination: impl Into<Location>) -> Result<Self> {
        let writer = StreamWriter::new(destination)?;
        Ok(Self::new_from_writer(writer))
    }

    /// 打开**分段录制**输出：按时间/大小把输出切成多个文件（`segment` muxer）。
    ///
    /// `pattern` 是文件名模板，`%d`/`%03d` 由 `segment` muxer 按段序号展开
    /// （如 `"out_%03d.mp4"`）；切分条件与段内参数经 `options` 给出，常用的有：
    ///
    /// | 选项 | 含义 |
    /// |------|------|
    /// | `segment_time` | 每段时长（秒，支持 `"2.5"`；默认由 muxer 决定） |
    /// | `segment_time_delta` | 切点容差（秒），吸收时间戳抖动，避免段长忽长忽短 |
    /// | `segment_size` | 每段字节上限（与 `segment_time` 取先到者） |
    /// | `segment_format` / `segment_format_options` | 段容器与段容器参数（默认取模板扩展名） |
    /// | `reset_timestamps` | `1` = 每段时间戳从 0 重新开始（播放器/上传友好） |
    ///
    /// 切点落在**关键帧**上：默认只在参考流的关键帧处开新段（`break_non_keyframes=1`
    /// 可放宽）。要精确控制切点，用
    /// [`MediaFrame::force_key_frame`](crate::MediaFrame::force_key_frame) 在
    /// 目标位置强制插关键帧，并让
    /// [`with_gop_size`](crate::EncoderBuilder::with_gop_size) 与段长相称。
    ///
    /// 与 [`Muxer::new`] 的差别只在打开方式：`segment` 是 `AVFMT_NOFILE` 容器，
    /// 由它自己按模板开/关每个段文件，因此**必须**显式指定格式，模板也不会被
    /// 当成一个真实文件名创建。
    ///
    /// ```no_run
    /// use rsmedia::mux::Muxer;
    /// use rsmedia::options::Options;
    /// use rsmedia::{EncoderBuilder, MediaFrame, PixelFormat};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// let mut opts = Options::new();
    /// opts.insert("segment_time", "10");
    /// opts.insert("reset_timestamps", "1");
    /// let mut muxer = Muxer::new_segmented("/tmp/out_%03d.mp4", opts)?;
    ///
    /// let encoder = EncoderBuilder::new_video(640, 480).with_fps(25.0).build()?;
    /// let tb = encoder.time_base();
    /// let idx = muxer.add_encoder(encoder)?;
    /// let mut frame = MediaFrame::<u8>::new_video_frame(640, 480, PixelFormat::RGB24)?;
    /// frame.set_pts(0);
    /// let mut av = frame.to_avframe()?;
    /// av.set_time_base(tb);
    /// muxer.mux(av, idx)?;
    /// muxer.finish()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_segmented(
        pattern: impl Into<Location>,
        options: impl Into<Option<Options>>,
    ) -> Result<Self> {
        let writer = crate::io::StreamWriterBuilder::new(pattern)
            .with_format("segment")
            .with_options(options)
            .build()
            .context("Failed to open segmented output")?;
        Ok(Self::new_from_writer(writer))
    }
}

impl<W: Writer> Muxer<W> {
    pub fn new_from_writer(writer: W) -> Self {
        Self {
            writer,
            streams: Vec::new(),
            // 默认交错写入，与 ffmpeg CLI（`av_interleaved_write_frame`）一致。
            // 编码器含 B 帧时 packet 按解码序输出（dts 先于 pts、首包 dts 为负），
            // 非交错直写会让 FLV 等容器报 "Packets poorly interleaved / not in
            // the proper order with respect to DTS"（AVERROR(EINVAL)）。
            interleaved: true,
            // 默认不平移：库不悄悄改用户给的时间戳，需要的人显式开启
            // （live 采集的 wallclock/epoch 时间戳等，见 set_normalize_timestamps）。
            normalize_timestamps: false,
            pts_base_us: None,
            have_written_header: false,
            have_written_trailer: false,
            metadata: Metadata::new(),
            stream_metadata: HashMap::new(),
            chapters: Vec::new(),
            chapters_applied: false,
        }
    }

    /// A human-readable description of one output stream's [`StreamInfo`].
    ///
    /// Named for what it returns rather than for `av_dump_format`: it reports a
    /// single stream (not the whole container), it does not print, and it cannot
    /// fail silently — the caller decides where the text goes.
    pub fn dump_stream_info(&self, index: usize) -> Result<String> {
        Ok(format!("{:?}", self.get_stream(index)?.stream_info))
    }

    /// 开关交错写入（interleaved）。
    ///
    /// **默认开启**，等价于 ffmpeg CLI 的 `av_interleaved_write_frame`：跨流按
    /// dts 交错排序、缓冲 B 帧导致的乱序，保证首包 dts 非负。编码器输出含
    /// B 帧时（默认即含，见 [`EncoderBuilder::with_max_b_frames`]）必须使用
    /// 交错写入，否则 FLV 等容器直接报错。
    ///
    /// 透传（remux）单流等场景可关闭，回到逐包直写
    /// （`av_write_frame`）的旧行为。
    pub fn set_interleaved(&mut self, interleaved: bool) -> &mut Self {
        // header 写出后 `AVFormatContext` 的写队列已按当时的设置建立（交错写入
        // 的内部 buffer 属于 muxer），此时再改只影响后续包，语义静默改变——
        // 与 metadata/chapter 一致地告警。
        if self.have_written_header {
            tracing::warn!(
                "set_interleaved({interleaved}) after header write only affects subsequent \
                 packets; set it before the first mux()/mux_packet()"
            );
        }
        self.interleaved = interleaved;
        self
    }

    /// 开关时间戳归一化：以**首个带有效时间戳的提交单元**为基准，把全部流的时间戳
    /// 平移到 0 起点，等价于 ffmpeg CLI 默认的 `ts_offset`（把输入起点移到 0）行为。
    ///
    /// 用于时间戳本身很大的输入——live 采集的 wallclock/epoch 时间戳（`avformat`
    /// 的 `use_wallclock_as_timestamps=1`）、SDI 时间码、带绝对起点的 RTP 流等。
    /// **FLV/RTMP 的时间戳字段只有 32 位毫秒**，直接透传 epoch 值会溢出回绕：产物
    /// 起始 pts 落在几十万秒处，播放器的时长/缓冲计算随之失真。
    ///
    /// 只做整体平移，帧间间隔（真实到达节奏、卡顿造成的 PTS 跳变）完全保留；
    /// 基准跨流共享同一物理时刻（内部按微秒存储、逐流换算回各自时间基），
    /// 因此音视频同步不受影响。`AV_NOPTS_VALUE`（未设 pts，由编码器自动编号）
    /// 原样透传，也不参与基准建立。
    ///
    /// 应在写入任何数据**之前**调用；header 写出后才开启时基准只能从下一个提交
    /// 单元建立，先前写出的包保持原值，时间戳会出现回跳（此处会告警）。
    pub fn set_normalize_timestamps(&mut self, normalize: bool) -> &mut Self {
        if normalize && self.have_written_header {
            tracing::warn!(
                "set_normalize_timestamps(true) after header write: the base is taken from the \
                 next submitted frame/packet; already-written packets keep their original values"
            );
        }
        self.normalize_timestamps = normalize;
        self.pts_base_us = None;
        self
    }

    /// 把一个时间戳平移到归一化基准（`tb` 是该值当前所在的时间基）。
    ///
    /// 首个有效时间戳建立基准并把自身恰好平移到 0（避免换算取整后首帧落在 ±1 tick）；
    /// 后续值减去同一基准。未开启归一化或时间戳为 `AV_NOPTS_VALUE` 时原样返回。
    fn normalize_ts(&mut self, ts: i64, tb: ffi::AVRational) -> i64 {
        if !self.normalize_timestamps || ts == ffi::AV_NOPTS_VALUE {
            return ts;
        }
        match self.pts_base_us {
            Some(base_us) => ts - rsmpeg::avutil::av_rescale_q(base_us, ffi::AV_TIME_BASE_Q, tb),
            None => {
                self.pts_base_us = Some(rsmpeg::avutil::av_rescale_q(ts, tb, ffi::AV_TIME_BASE_Q));
                0
            }
        }
    }

    pub fn add_encoder(&mut self, encoder: Encoder) -> Result<usize> {
        // header 一旦写出（首个包 mux 时懒触发），AVFormatContext 的流数组就固定了；
        // 此时再加流会让 av_interleaved_write_frame 访问越界的流索引 → SIGSEGV
        // （边界测试实测）。必须报错而不是放行。
        if self.have_written_header {
            return Err(RsmediaError::invalid_config(
                "Cannot add a stream after the container header has been written; \
                 register all streams before the first mux()/mux_packet()",
            ));
        }
        let stream_idx = self
            .writer
            .add_stream(encoder.codecpar(), encoder.time_base())?;
        let stream_info = StreamInfo::from_writer(&self.writer, stream_idx)?;
        self.streams
            .push(MuxerStream::new_encoded(encoder, stream_info));
        Ok(stream_idx)
    }

    /// 添加一个**透传（copy）流**用于 remux（转封装，不解码不重编码）。
    ///
    /// 从源 demuxer 的某个流拷贝编解码参数到输出容器，并记录源流时间基
    /// 供 [`Self::mux_packet`] 做时间戳换算。返回输出流的 index（传给
    /// [`Self::mux_packet`]）。
    ///
    /// 典型用法：`Demuxer::new_passthrough` 迭代的 packet（保留原流 index）
    /// 与 `Muxer::add_copy_stream` 输入的源流一一对应。
    pub fn add_copy_stream(&mut self, src_info: &StreamInfo) -> Result<usize> {
        // 与 `add_encoder` 同一守卫：header 写出后流数组已固定，再加流是未定义行为。
        if self.have_written_header {
            return Err(RsmediaError::invalid_config(
                "Cannot add a stream after the container header has been written; \
                 register all streams before the first mux()/mux_packet()",
            ));
        }
        let src_time_base = src_info.time_base;
        let stream_idx = self
            .writer
            .add_stream(src_info.codec_parameters.clone(), src_info.time_base)?;
        // 拷贝的 codec_parameters 携带源容器专属的 codec_tag（如 `mp4a`
        // /`avc1`）。跨容器 remux（mp4→mkv 等）时这些 tag 与目标 muxer 不
        // 兼容，清空后由目标 muxer 在 write_header 时自行指派正确 tag。
        // 与 ffmpeg remux 的 `codec_tag = 0` 语义一致。
        let stream = self.stream_ptr(stream_idx)?;
        let codecpar = unsafe { (*stream).codecpar };
        if codecpar.is_null() {
            return Err(RsmediaError::msg(format!(
                "output stream {stream_idx} has no codec parameters"
            )));
        }
        unsafe {
            (*codecpar).codec_tag = 0;
        }
        let stream_info = StreamInfo::from_writer(&self.writer, stream_idx)?;
        self.streams
            .push(MuxerStream::new_copy(stream_info, src_time_base));
        Ok(stream_idx)
    }

    /// 取得输出上下文中第 `idx` 条流的裸指针（由 `avformat_new_stream` 分配，
    /// 生命周期同 context）。
    ///
    /// [`add_stream`](crate::io::Writer::add_stream) 是公开扩展点，自定义
    /// `Writer` 实现可以返回任意索引；这里统一做边界检查，避免用越界索引写
    /// 裸指针（UB）。rsmpeg 只暴露不可变的 `AVStreamRef`，而 `codec_tag` /
    /// `disposition` 必须在 `write_header` 前就地修改，故取裸指针。
    ///
    /// 调用方只应在 `write_header` 之前使用返回的指针（此时 context 由
    /// `self.writer` 独占，无并发访问），并在同一语句内完成修改。
    fn stream_ptr(&mut self, idx: usize) -> Result<*mut ffi::AVStream> {
        let ctx = unsafe { &mut *self.writer.output_mut().as_mut_ptr() };
        let nb_streams = ctx.nb_streams as usize;
        if idx >= nb_streams {
            return Err(RsmediaError::invalid_config(format!(
                "output stream index {idx} out of range (nb_streams={nb_streams})"
            )));
        }
        // SAFETY: `ctx.streams` holds `nb_streams` non-null pointers owned by the
        // output context; `idx` was bounds-checked above.
        let stream = unsafe { *ctx.streams.add(idx) };
        if stream.is_null() {
            return Err(RsmediaError::msg(format!(
                "output stream {idx} is a null pointer (nb_streams={nb_streams})"
            )));
        }
        Ok(stream)
    }

    pub fn get_stream(&self, index: usize) -> Result<&MuxerStream> {
        self.streams
            .iter()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::invalid_config(format!("Stream index: {index} not found")))
    }

    pub fn get_stream_mut(&mut self, index: usize) -> Result<&mut MuxerStream> {
        self.streams
            .iter_mut()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::invalid_config(format!("Stream index: {index} not found")))
    }

    /// Sets a container-level metadata entry, e.g. `title`, `artist`,
    /// `comment`. Applied when the container header is written, i.e. before
    /// the first [`Self::mux`] call; entries set after the header is written
    /// are ignored (with a warning).
    pub fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<&mut Self> {
        if self.have_written_header {
            tracing::warn!("set_metadata after header write has no effect");
        }
        self.metadata.insert(key.into(), value.into());
        Ok(self)
    }

    /// Sets a per-stream metadata entry for the stream with index returned
    /// from [`Self::add_encoder`]. The common case is `language` with an
    /// ISO 639-2 code ("chi", "eng", "und", ...), which players use to pick
    /// audio/subtitle tracks.
    ///
    /// Applied when the container header is written; entries containing
    /// interior NUL bytes are skipped with a warning at that point.
    pub fn set_stream_metadata(
        &mut self,
        stream_index: usize,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<&mut Self> {
        let (key, value) = (key.into(), value.into());
        let nb_streams = self.writer.output().nb_streams as usize;
        if stream_index >= nb_streams {
            return Err(RsmediaError::invalid_config(format!(
                "stream index {stream_index} out of range (nb_streams={nb_streams})"
            )));
        }
        if self.have_written_header {
            tracing::warn!(
                "set_stream_metadata({stream_index}, {key:?}) after header write has no effect"
            );
        }
        self.stream_metadata
            .entry(stream_index)
            .or_default()
            .insert(key, value);
        Ok(self)
    }

    /// Applies container-level and per-stream metadata to the raw format
    /// context.
    ///
    /// Must be called exactly once, immediately before `write_header`, as
    /// FFmpeg only reads metadata during `avformat_write_header`. rsmpeg
    /// exposes output streams only immutably (`AVStreamRef`), so the raw
    /// stream array is accessed here instead; this is sound because the
    /// writer exclusively owns the context and no streams are added after
    /// this point.
    fn apply_metadata(&mut self) {
        if self.metadata.is_empty() && self.stream_metadata.is_empty() {
            return;
        }
        let ctx = unsafe { &mut *self.writer.output_mut().as_mut_ptr() };

        // SAFETY: `ctx.metadata` is a live dictionary slot owned by the format
        // context; `write_into_raw_dict` replaces it with our entries.
        unsafe { self.metadata.write_into_raw_dict(&mut ctx.metadata) };

        let streams =
            unsafe { std::slice::from_raw_parts_mut(ctx.streams, ctx.nb_streams as usize) };
        for (idx, entries) in &self.stream_metadata {
            let Some(stream) = streams.get_mut(*idx) else {
                tracing::warn!(
                    "stream metadata: index {idx} out of range (nb_streams={})",
                    ctx.nb_streams
                );
                continue;
            };
            // SAFETY: `(**stream).metadata` is a live dictionary slot owned by the
            // stream of the exclusively owned output context.
            unsafe { entries.write_into_raw_dict(&mut (**stream).metadata) };
        }
    }

    /// Adds a container chapter. Chapters are written when the container
    /// header is written, i.e. before the first [`Self::mux`] call; chapters
    /// added after the header is written are ignored (with a warning).
    ///
    /// Times are in seconds; internally stored on a millisecond time base.
    /// Requires a container with chapter support (MP4, MKV, ...).
    pub fn add_chapter(&mut self, chapter: Chapter) -> Result<&mut Self> {
        if self.have_written_header {
            tracing::warn!(
                "add_chapter({:?}) after header write has no effect",
                chapter.title
            );
        }
        if chapter.start < 0.0 {
            return Err(RsmediaError::invalid_config(format!(
                "chapter start must be >= 0, got {}",
                chapter.start
            )));
        }
        if chapter.end <= chapter.start {
            return Err(RsmediaError::invalid_config(format!(
                "chapter end ({}) must be greater than start ({})",
                chapter.end, chapter.start
            )));
        }
        self.chapters.push(Chapter {
            id: chapter.id,
            title: chapter.title,
            start: chapter.start,
            end: chapter.end,
        });
        Ok(self)
    }

    /// Adds a cover-art (attached picture) stream using the MP4/MKV default
    /// `mjpeg` codec. See [`Self::add_cover_art_with`] for details; use that
    /// method instead when a custom cover codec (e.g. `png`) is needed.
    ///
    /// Returns the output stream index of the cover stream.
    pub fn add_cover_art(&mut self, cover_frame: AVFrame) -> Result<usize> {
        let width = cover_frame.width as usize;
        let height = cover_frame.height as usize;
        let encoder = EncoderBuilder::new_video(width, height)
            .with_codec_name("mjpeg".to_string())
            .build()?;
        self.add_cover_art_with(encoder, cover_frame)
    }

    /// Adds a cover-art (attached picture) stream: `cover_frame` is encoded
    /// with `encoder` (typically `mjpeg` or `png`) into a dedicated video
    /// stream marked with `AV_DISPOSITION_ATTACHED_PIC` and muxed as a single
    /// frame.
    ///
    /// What that mark actually buys differs per muxer, and only MP4/MOV ends up
    /// with a real attached picture:
    ///
    /// - **MP4/MOV**: the frame is stored in the `covr` atom instead of as a track
    ///   sample (`mov_write_covr`), which requires the stream's disposition to be
    ///   **exactly** `AV_DISPOSITION_ATTACHED_PIC` — `is_cover_image()` compares for
    ///   equality, so OR-ing another disposition bit in silently drops the cover.
    ///   The stream itself stays in the file and reads back with `ATTACHED_PIC` set.
    /// - **MKV**: matroskaenc only treats a stream as an attachment when its
    ///   `codecpar->codec_type` is `AVMEDIA_TYPE_ATTACHMENT`, which a video stream
    ///   never is — so the cover is written as an ordinary single-frame **video
    ///   track**, and reading the file back shows `disposition = 0` with no
    ///   attachment tag. ffmpeg's CLI does the same for `-disposition:v attached_pic`.
    /// - Other muxers ignore the disposition.
    ///
    /// The stream is marked and the frame muxed immediately, so this must be
    /// called after all primary streams are added (the header is written on
    /// the first mux) and before any regular [`Self::mux`] call. Returns the
    /// output stream index of the cover stream.
    pub fn add_cover_art_with(&mut self, encoder: Encoder, cover_frame: AVFrame) -> Result<usize> {
        if self.have_written_header {
            // 调用顺序问题（封面必须在首个 mux() 之前加入），不是"本构建不支持"：
            // 措辞上不要用 unsupported，否则与 `RsmediaError::Unsupported` 撞概念。
            return Err(RsmediaError::invalid_config(
                "add_cover_art cannot be called after the header has been written: \
                 add the cover stream before the first mux()/mux_packet()",
            ));
        }
        let stream_idx = self.add_encoder(encoder)?;

        // Mark the stream as an attached picture. rsmpeg only exposes the
        // output stream array immutably, so the raw stream array is accessed
        // here instead; sound because the writer exclusively owns the context
        // and the disposition must be set before `write_header`.
        let stream = self.stream_ptr(stream_idx)?;
        unsafe {
            (*stream).disposition |= ffi::AV_DISPOSITION_ATTACHED_PIC as i32;
        }

        // The cover must have a valid pts so the muxer timestamps it correctly.
        let mut cover_frame = cover_frame;
        if cover_frame.pts == ffi::AV_NOPTS_VALUE {
            cover_frame.set_pts(0);
        }
        self.mux(cover_frame, stream_idx)?;
        Ok(stream_idx)
    }

    /// Writes the pending [`Chapter`] list into the raw format context.
    ///
    /// Must be called exactly once, immediately before `write_header`, as
    /// FFmpeg requires `nb_chapters` to be initialized before the header is
    /// written. Chapters are allocated with `av_mallocz` and ownership is
    /// transferred to the format context, which frees them in
    /// `avformat_free_context`.
    ///
    /// Failure handling：任一 chapter 或指针数组分配失败时，释放所有已分配的
    /// chapter 节点并保持 `ctx.chapters` 未被设置（FFmpeg 视为无章节），既避免
    /// 内存泄漏，也避免 `nb_chapters > 0` 时数组中出现空指针导致 FFmpeg 解引用崩溃。
    ///
    /// 幂等：节点所有权一旦转移给 format context，[`Self::chapters_applied`]
    /// 即置位，重试写 header（上一次 `write_header` 失败后再次 mux）不会重复
    /// 分配节点并覆盖 `ctx.chapters`（那会泄漏上一批节点）。
    fn apply_chapters(&mut self) {
        if self.chapters.is_empty() || self.chapters_applied {
            return;
        }
        let ctx = unsafe { &mut *self.writer.output_mut().as_mut_ptr() };

        let count = self.chapters.len();

        // 1) 逐章分配节点。用 Rust Vec 暂存所有权，以便失败时统一释放。
        let mut chapter_nodes: Vec<*mut ffi::AVChapter> = Vec::with_capacity(count);
        for (i, chapter) in self.chapters.iter().enumerate() {
            // Millisecond time base: times are stored as whole milliseconds.
            let start_ms = (chapter.start * 1000.0).round() as i64;
            let end_ms = (chapter.end * 1000.0).round() as i64;
            let chapter_ptr = unsafe {
                ffi::av_mallocz(std::mem::size_of::<ffi::AVChapter>()) as *mut ffi::AVChapter
            };
            if chapter_ptr.is_null() {
                tracing::error!("av_mallocz for chapter {i} failed; chapters are dropped");
                Self::free_chapter_nodes(&mut chapter_nodes);
                return;
            }
            // SAFETY: `chapter_ptr` comes from `av_mallocz` and was checked non-null
            // above; the node is exclusively owned here (it has not been linked into
            // the format context yet) and stays alive until FFmpeg takes it over, so
            // writing its fields through the raw pointer is exclusive and in-bounds.
            unsafe {
                // 未指定 id 时按顺序自动编号（FFmpeg 要求章节 id 唯一）。
                (*chapter_ptr).id = chapter.id.unwrap_or(i as i64);
                (*chapter_ptr).time_base = ffi::AVRational { num: 1, den: 1000 };
                (*chapter_ptr).start = start_ms;
                (*chapter_ptr).end = end_ms;
                // SAFETY: `(*chapter_ptr).metadata` starts out NULL and the chapter
                // node is exclusively owned here (already inside an `unsafe` block).
                Metadata::new()
                    .insert("title", &chapter.title)
                    .write_into_raw_dict(&mut (*chapter_ptr).metadata);
            }
            chapter_nodes.push(chapter_ptr);
        }

        // 2) 分配连续指针数组。
        let chapters_ptr = unsafe {
            ffi::av_calloc(count, std::mem::size_of::<*mut ffi::AVChapter>())
                as *mut *mut ffi::AVChapter
        };
        if chapters_ptr.is_null() {
            tracing::error!("av_calloc for {count} chapters failed; chapters are dropped");
            Self::free_chapter_nodes(&mut chapter_nodes);
            return;
        }

        // 3) 复制节点指针并转移所有权给 format context。
        unsafe {
            std::ptr::copy_nonoverlapping(chapter_nodes.as_ptr(), chapters_ptr, count);
        }
        // 节点指针是裸指针，Vec 直接 Drop 不会释放它们（无 Drop 实现）；
        // 所有权已交由 `chapters_ptr` 数组，FFmpeg 在 `avformat_free_context`
        // 时统一释放数组与其内节点。
        ctx.chapters = chapters_ptr;
        ctx.nb_chapters = count as u32;
        self.chapters_applied = true;
    }

    /// 释放一组尚未转移所有权的 chapter 节点（`av_free` 对空指针安全）。
    fn free_chapter_nodes(nodes: &mut Vec<*mut ffi::AVChapter>) {
        for node in nodes.drain(..) {
            unsafe {
                ffi::av_free(node as *mut std::os::raw::c_void);
            }
        }
    }

    /// Refreshes every stream's cached [`StreamInfo`] from the writer, right
    /// after the header has been written.
    ///
    /// Muxers may adjust stream parameters inside `avformat_write_header` — most
    /// notably the time base (the GIF muxer forces `1/100`, MP4 applies its movie
    /// timescale). Packet timestamps must be rescaled to the *post-header* value,
    /// so the cache is refreshed here and **every** rescale site reads it
    /// (`mux`, `mux_subtitle_segment`, `mux_packet`, `finish`): one source of
    /// truth, instead of some sites reading the cache while others query the
    /// writer — which is how the two silently disagree.
    fn refresh_stream_info(&mut self) -> Result<()> {
        for mux_stream in self.streams.iter_mut() {
            let stream_info = StreamInfo::from_writer(&self.writer, mux_stream.stream_index)?;
            let tb_changed = stream_info.time_base.num != mux_stream.stream_info.time_base.num
                || stream_info.time_base.den != mux_stream.stream_info.time_base.den;
            if tb_changed {
                tracing::debug!(
                    "Muxer changed stream {} time_base: {:?} -> {:?}",
                    mux_stream.stream_index,
                    mux_stream.stream_info.time_base,
                    stream_info.time_base
                );
            }
            mux_stream.stream_info = stream_info;
        }
        Ok(())
    }

    /// 写入 container header（并应用 metadata/chapters），带幂等判断。
    ///
    /// header 一旦写出，后续所有 `mux`/`mux_packet` 直接写包；本函数在首个
    /// 包之前被主动调用，避免每个包路径各自重复 header 逻辑。
    /// 返回 header 写入产生的输出（已写过则返回 `None`），由调用方并入自己的
    /// 结果一起返回。缓冲型 [`Writer`] 的 `Out` 是**增量**字节，**不能在这里
    /// 丢掉**：`write_header` 已经把它们从 writer 的内部累积里取走，丢弃就意味着
    /// header 那些字节永远不会到达调用方（`BufferWriter` 用户会拿到缺头的容器）。
    fn ensure_header_written(&mut self) -> Result<W::Accum> {
        if self.have_written_header {
            return Ok(W::Accum::default());
        }
        self.apply_metadata();
        self.apply_chapters();
        let header = self.writer.write_header()?;
        // 只有 header 真正写出后才置位：否则一次失败会被记成"已写"，后续 `mux`
        // 会往无头容器里塞包、`finish` 还会补一个 trailer，错误被彻底掩盖。
        self.have_written_header = true;
        // 刷新失败只告警：header 字节此刻已经从 writer 的内部累积里取出，若让
        // 错误上抛，这些字节就永远不会到达调用方（缓冲型 Writer 会拿到缺头的
        // 容器）。缓存仍是 header 之前的（旧）时间基，包按旧时间基换算——退化为
        // 与"未刷新"一致的行为，而不是丢掉已经写出的 header。
        if let Err(err) = self.refresh_stream_info() {
            tracing::warn!(
                "Failed to refresh stream info after write_header: {err:#}; \
                 packet timestamps keep using the pre-header time bases"
            );
        }
        let mut collected = W::Accum::default();
        W::merge_out(&mut collected, header);
        Ok(collected)
    }

    /// [`Self::mux_packet`] / [`Self::mux_subtitle_segment`] 的公共前置：必要时先写
    /// header，已写 trailer 则拒绝。
    ///
    /// `av_write_trailer` 之后 format context 已封闭（muxer 的内部队列被释放、
    /// 索引已回填），继续写包不是"追加数据"而是把容器写坏。编码流路径
    /// （[`Self::mux`]）不需要这里的判断：它的 packet 由 encoder 产出，而已 flush
    /// 的 encoder 自己就会拒绝再编码。
    fn begin_packet_write(&mut self) -> Result<W::Accum> {
        if self.have_written_trailer {
            return Err(RsmediaError::invalid_config(
                "Cannot write more packets after the trailer has been written; \
                 finish() already closed the container",
            ));
        }
        self.ensure_header_written()
    }

    /// 将已调整好流索引与时间戳的 packet 写入输出容器，返回容器的写入结果。
    ///
    /// 按 [`Self::set_interleaved`] 的当前设置走 `write_interleaved`（跨流按 dts
    /// 交错，B 帧乱序也安全）或 `write_frame`（顺序直写）。三条 mux 路径
    /// （编码流 / 字幕流 / 复制流）收尾共用这一个出口，交错与否只在这一处判断。
    fn write_packet(&mut self, packet: &mut AVPacket) -> Result<W::Out> {
        if self.interleaved {
            self.writer.write_interleaved(packet)
        } else {
            self.writer.write_frame(packet)
        }
    }

    /// 三条写包路径的公共出口：把 packet 归一到 `stream_idx` 输出流后写出。
    ///
    /// * `tb_from` - packet 时间戳当前所处的时间基（编码流为编码器时间基，
    ///   透传流为源流时间基）；目标时间基取流缓存的 **post-header** 值
    ///   （见 [`Self::refresh_stream_info`]），与 [`Self::finish`] 的 flush 路径
    ///   同源。
    fn write_out_packet(
        &mut self,
        packet: &mut AVPacket,
        stream_idx: usize,
        tb_from: ffi::AVRational,
    ) -> Result<W::Out> {
        let out_time_base = self.get_stream(stream_idx)?.stream_info.time_base;
        packet.set_pos(-1);
        packet.set_stream_index(stream_idx as i32);
        packet.rescale_ts(tb_from, out_time_base);
        self.write_packet(packet)
    }

    /// Mux a single frame through an encoder stream.
    ///
    /// 只适用于通过 [`Self::add_encoder`] 添加的编码流；若目标是透传流
    /// （[`Self::add_copy_stream`]），应改用 [`Self::mux_packet`]。
    ///
    /// # Arguments
    ///
    /// * `frame` - [`AVFrame`] to encode and mux.
    /// * `stream_idx` - Index of the target output stream.
    pub fn mux(&mut self, mut frame: AVFrame, stream_idx: usize) -> Result<W::Accum> {
        // 先校验目标流是编码流，再提交 header（见 `ensure_header_written`）：拿
        // 透传流调 `mux()` 属于参数错误，不该把 header 不可逆地写出去。
        let enc_time_base = self
            .get_stream(stream_idx)?
            .encoder
            .as_ref()
            .ok_or_else(|| {
                RsmediaError::invalid_config(format!(
                    "Stream {stream_idx} is a copy stream: use mux_packet() instead of mux()"
                ))
            })?
            .time_base();
        let mut collected = self.ensure_header_written()?;

        // 归一化在**编码前**应用：编码器内部的自动编号与 flush 出的延迟包就都在同一
        // 坐标系里，输出侧（含 `finish()` 的 flush 路径）无需再区分处理。
        // 帧自带有效时间基时以它为准；否则 pts 已按编码器时间基计数（见
        // `MediaFrame.time_base` 的约定）。
        let frame_time_base = if frame.time_base.num > 0 && frame.time_base.den > 0 {
            frame.time_base
        } else {
            enc_time_base
        };
        let normalized = self.normalize_ts(frame.pts, frame_time_base);
        if normalized != frame.pts {
            frame.set_pts(normalized);
        }

        let mux_stream = self.get_stream_mut(stream_idx)?;
        let encoder = mux_stream.encoder.as_mut().ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "Stream {stream_idx} is a copy stream: use mux_packet() instead of mux()"
            ))
        })?;
        let packets = encoder.encode_raw(frame)?;
        // 编码器输出的 packet 常不带 duration（mpeg4 等），若缺失则按
        // 帧率/采样率补上，否则 MP4 等交错 muxer 无法推导**最后一帧**的
        // 时长，导致末帧被丢弃（与 Encoder::flush 的补全逻辑保持一致）。
        let duration_fallback = encoder.packet_duration();
        // mux_stream 对 self.streams 的借用至此结束，之后可独占使用 self.writer

        // 一帧可能编出多个 packet（B 帧重排序、编码器内部缓冲），必须（连同
        // header 的输出一起）逐包累积而不能只留最后一个：缓冲型 Writer 的 `Out`
        // 是**增量**字节，覆盖它会让调用方拿到被截断的流。
        for mut packet in packets {
            if packet.duration <= 0 {
                packet.set_duration(duration_fallback);
            }
            let out = self.write_out_packet(&mut packet, stream_idx, enc_time_base)?;
            W::merge_out(&mut collected, out);
        }
        Ok(collected)
    }

    /// Encodes one subtitle segment through a subtitle encoder stream and
    /// muxes the resulting packet(s).
    ///
    /// Subtitle encoders (mov_text, subrip, ...) use the synchronous
    /// `avcodec_encode_subtitle` API instead of the frame-based
    /// send/receive loop, so they cannot go through [`Self::mux`]. Each
    /// segment yields exactly zero or one packet whose pts/duration are in
    /// the encoder's 1/1000 time base; they are rescaled to the output
    /// stream time base here.
    ///
    /// # Arguments
    ///
    /// * `segment`   - The text cue with start/end times in milliseconds.
    /// * `stream_idx` - Index of the subtitle stream created by
    ///   [`Self::add_encoder`] with a subtitle [`Encoder`].
    pub fn mux_subtitle_segment(
        &mut self,
        segment: &SubtitleSegment,
        stream_idx: usize,
    ) -> Result<W::Accum> {
        // 先校验目标流是字幕编码流，再提交 header（见 `begin_packet_write`）。
        {
            let mux_stream = self.get_stream(stream_idx)?;
            let encoder = mux_stream.encoder.as_ref().ok_or_else(|| {
                RsmediaError::invalid_config(format!(
                    "Stream {stream_idx} is a copy stream: subtitle segments require an encoder stream"
                ))
            })?;
            if encoder.media_type() != MediaType::SUBTITLE {
                return Err(RsmediaError::invalid_config(format!(
                    "mux_subtitle_segment requires a subtitle encoder, got {}",
                    encoder.media_type()
                )));
            }
        }
        // header 的输出要并入返回值（见 `ensure_header_written`），且必须在
        // 借出 `mux_stream` 之前调用。
        let mut collected = self.begin_packet_write()?;

        let mux_stream = self.get_stream_mut(stream_idx)?;
        let encoder = mux_stream.encoder.as_mut().ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "Stream {stream_idx} is a copy stream: subtitle segments require an encoder stream"
            ))
        })?;
        let enc_time_base = encoder.time_base();
        let packets = encoder.encode_subtitle_segment(segment)?;

        // 与 `mux` 相同的累积写法（字幕段通常是 0/1 个 packet，但没理由与 `mux` 用两套语义）。
        for mut packet in packets {
            // 与 `mux` 一致：归一化在写包前应用（字幕包的 pts/dts 在编码器的
            // 1/1000 时间基里），基准与音视频流共享，不破坏同步。
            let pts = self.normalize_ts(packet.pts, enc_time_base);
            let dts = self.normalize_ts(packet.dts, enc_time_base);
            if pts != packet.pts {
                packet.set_pts(pts);
            }
            if dts != packet.dts {
                packet.set_dts(dts);
            }
            let out = self.write_out_packet(&mut packet, stream_idx, enc_time_base)?;
            W::merge_out(&mut collected, out);
        }
        Ok(collected)
    }

    /// Mux a raw (decoded_source / remux) packet through a **copy stream**.
    ///
    /// 用于 remux（转封装）：直接把 demuxer 读到的原始 `AVPacket` 写入输出
    /// 容器，不解码、不重编码。pts/dts 从源流时间基（`add_copy_stream` 记录）
    /// 换算到输出流时间基。
    ///
    /// # Arguments
    ///
    /// * `packet`   - 来自 [`Demuxer::demux_packet`] 的原始包；其 `stream_index`
    ///   会在写入前被改写为输出流 index。
    /// * `stream_idx` - [`Self::add_copy_stream`] 返回的输出流 index。
    pub fn mux_packet(&mut self, packet: &mut AVPacket, stream_idx: usize) -> Result<W::Accum> {
        // 先确认这是透传流并取出源时间基（在 `begin_packet_write` 之前）：拿编码流调
        // `mux_packet()` 属于参数错误，不该把 header 不可逆地写出去。
        let src_time_base = self.get_stream(stream_idx)?.src_time_base.ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "Stream {stream_idx} is not a copy stream: use mux() instead of mux_packet()"
            ))
        })?;
        let mut collected = self.begin_packet_write()?;

        // 与 `mux` 一致：归一化在写包前应用（透传包的 pts/dts 在源流时间基里），
        // 基准与编码流共享同一物理时刻。
        let pts = self.normalize_ts(packet.pts, src_time_base);
        let dts = self.normalize_ts(packet.dts, src_time_base);
        if pts != packet.pts {
            packet.set_pts(pts);
        }
        if dts != packet.dts {
            packet.set_dts(dts);
        }

        // src_stream_time_base => out_stream_time_base（重复调用会重复换算，
        // 因此只在我们自己保存的源时间基与输出时间基之间进行一次换算）
        let out = self.write_out_packet(packet, stream_idx, src_time_base)?;
        W::merge_out(&mut collected, out);
        Ok(collected)
    }

    /// Signal to the muxer that writing has finished. This will cause a trailer to be written if
    /// the container format has one.
    pub fn finish(&mut self) -> Result<W::Accum> {
        // 从未 mux 过任何数据（header 也尚未写入）：没有实际内容需要 flush，
        // 直接空操作返回空累积器，不产生“无头”的残缺输出。
        if !self.have_written_header {
            return Ok(W::Accum::default());
        }

        // flush 与 trailer 的输出同样要累积（缓冲型 Writer 的 `Out` 是增量字节）。
        let mut collected = W::Accum::default();
        for mux_stream in self.streams.iter_mut() {
            // flush the encoder to ensure all packets are sent to the muxer.
            // 透传流没有编码器延迟缓冲，无需 flush。
            let Some(encoder) = mux_stream.encoder.as_mut() else {
                continue;
            };
            let out_stream_index = mux_stream.stream_index;
            let out_stream_time_base = mux_stream.stream_info.time_base;
            let flushed = encoder.flush(
                &mut self.writer,
                self.interleaved,
                out_stream_index,
                out_stream_time_base,
            )?;
            W::merge_accum(&mut collected, flushed);
        }

        // 已写 header 且未写 trailer 时才写 trailer；header + trailer 均已写说明
        // 是重复调用 finish()，此时幂等返回已累积的内容，避免重复写 trailer。
        if !self.have_written_trailer {
            let trailer = self.writer.write_trailer()?;
            // 写成功后才置位：否则一次失败会被永久记成"已完成"，重试的 finish()
            // 直接返回 Ok，缺 trailer 的容器被当成正常收尾。
            self.have_written_trailer = true;
            W::merge_out(&mut collected, trailer);
        }
        Ok(collected)
    }

    /// 若 header 已写而 trailer 未写，则补写 trailer 兜底收尾（幂等）。
    ///
    /// 这是 `finish()` 之外用于 `into_writer` / `Drop` 的统一兜底入口：
    /// 避免二者各自复制一遍「已写 header && 未写 trailer」的判定。
    fn flush_if_needed(&mut self) {
        if self.have_written_header
            && !self.have_written_trailer
            && let Err(err) = self.finish()
        {
            tracing::error!("Failed to auto-flush muxer: {err:#}");
        }
    }

    /// Consumes the muxer and returns the underlying writer.
    ///
    /// 应在 [`Self::finish`] 之后调用；此时 trailer 已写出，可从 writer 中
    /// 取回最终输出（如 [`crate::io::BufferWriter::into_bytes`] 或
    /// [`crate::io::IoWriter::into_inner`]）。若忘记调用 `finish()`，
    /// 此处会自动补写 trailer（与 `Drop` 的兜底行为一致）。
    pub fn into_writer(mut self) -> W {
        // 先补写 trailer，使 Drop 的自动 flush 逻辑成为空操作。
        self.flush_if_needed();
        // SAFETY: `Muxer` 实现了 `Drop`，不能直接 move 字段。此处用
        // `ManuallyDrop` 跳过 `Muxer::Drop`（此时其逻辑已是空操作），
        // 取走 writer 后手动析构其余字段，保证 encoder 等资源正常释放。
        unsafe {
            let mut this = std::mem::ManuallyDrop::new(self);
            let writer = std::ptr::read(&this.writer);
            std::ptr::drop_in_place(&mut this.streams);
            std::ptr::drop_in_place(&mut this.metadata);
            std::ptr::drop_in_place(&mut this.stream_metadata);
            std::ptr::drop_in_place(&mut this.chapters);
            writer
        }
    }
}

/// SAFETY: 仅承诺可移动到其他线程独占使用（`Send`）。`AVFormatContext` 及
/// 内部 encoder 均非线程安全，`&Self` 跨线程共享（`Sync`）不成立，故不实现。
unsafe impl<W: Writer + Send> Send for Muxer<W> {}

impl<W: Writer> Drop for Muxer<W> {
    fn drop(&mut self) {
        // 用户忘记调用 finish() 时（尤其是错误提前返回/panic），
        // 自动 flush 编码器延迟缓冲并写 trailer，避免生成损坏的容器文件。
        // 仅当已写过 header 时才处理，未 mux 过的空文件不做无意义写入。
        self.flush_if_needed();
    }
}

/// 校验 `writer` 尚未被写过 header（处女态）。
///
/// [`Writer`] 自己记录 header 状态（[`Writer::is_header_written`]），但该路径还要
/// 建自己的流，因此判据是"输出上下文里没有任何流、也没写过 header"：header 一旦
/// 写出，`AVFormatContext` 的流数组即固定，此时再 `add_stream` + `write_header`
/// 会让 FFmpeg 内部仍持有的旧指针失效（实测 SIGSEGV）。
///
/// 绕开 [`Muxer`] 直接操作 writer 的辅助函数（见
/// [`encode_subtitle_segments`](crate::subtitle::encode_subtitle_segments)）
/// 必须显式确认这一点。
pub(crate) fn ensure_writer_pristine<W: Writer>(writer: &W) -> Result<()> {
    let nb_streams = writer.output().nb_streams as usize;
    if writer.is_header_written() || nb_streams > 0 {
        return Err(RsmediaError::invalid_config(format!(
            "This writer is no longer pristine ({nb_streams} stream(s), header written: {}): \
             direct container writing needs a fresh writer (pass one through Muxer to have \
             header/trailer state tracked)",
            writer.is_header_written()
        )));
    }
    Ok(())
}

/// stream definition for demuxer
pub struct DemuxerStream {
    pub decoder: Decoder,
    pub stream_info: StreamInfo,
    pub media_type: MediaType,
    pub stream_index: usize,
}

impl DemuxerStream {
    pub fn new(decoder: Decoder, stream_info: StreamInfo) -> Self {
        let media_type = decoder.media_type();
        let stream_index = stream_info.index;
        Self {
            decoder,
            media_type,
            stream_info,
            stream_index,
        }
    }
}

/// Demuxer
///
/// Two construction modes:
/// - **Decode mode** ([`new`](Demuxer::new) / [`new_from_reader`](Demuxer::new_from_reader) /
///   [`new_single_stream`](Demuxer::new_single_stream)): iteration yields decoded
///   `AVFrame`s.
/// - **Passthrough mode** ([`new_passthrough`](Demuxer::new_passthrough)): no decoder is
///   built; [`demux_packet`](Demuxer::demux_packet) / [`packets`](Demuxer::packets) yield
///   raw `AVPacket`s for remuxing (re-wrapping without decode or re-encode).
pub struct Demuxer<R: Reader> {
    pub reader: R,
    /// One decoder per selected stream; always empty in passthrough mode.
    streams: Vec<DemuxerStream>,
    /// `true` when built via [`Demuxer::new_passthrough`], i.e. no decoders exist and
    /// only raw packets are produced.
    ///
    /// Kept as an explicit flag instead of inferring it from `streams.is_empty()`:
    /// an empty `streams` is also a valid decode-mode state (a container with no
    /// decodable stream), and callers must be able to tell the two apart.
    passthrough: bool,
}

impl Demuxer<StreamReader> {
    pub fn new(source: impl Into<Location>) -> Result<Self> {
        let reader = StreamReader::new(source)?;
        Self::new_from_reader(reader, None, None)
    }
}

/// 把一堆滤镜按媒体类型分组，供各解码器只取自己那条链。
///
/// `Demuxer::new_from_reader` 与 `Demuxer::new_single_stream` 需要的分组完全
/// 相同，抽到这里以免两处各改一遍。
fn group_filters(filters: Option<Vec<Filter>>) -> HashMap<MediaType, Vec<Filter>> {
    filters.unwrap_or_default().into_iter().fold(
        HashMap::<MediaType, Vec<Filter>>::new(),
        |mut map, filter| {
            map.entry(filter.media_type()).or_default().push(filter);
            map
        },
    )
}

impl<R: Reader> Demuxer<R> {
    /// 为单个流构建解码器（含硬件失败回退软件的逻辑）。
    fn build_decoder(
        reader: &R,
        stream_info: &StreamInfo,
        device_config: &Option<HWDeviceConfig>,
        filters: &HashMap<MediaType, Vec<Filter>>,
    ) -> Result<Decoder> {
        let media_type = stream_info.media_type;
        let device_type = device_config.as_ref().map(|c| c.device_type);
        let Some(codec_name) = stream_info.find_decoder_name(device_type) else {
            // 本构建里这个 codec_id 没有可用解码器 ⇒ `Unsupported`：调用方改不了
            // 自己的调用（换输入或换构建才行）。
            return Err(RsmediaError::unsupported(format!(
                "decoder for codec_id {:#x} (stream {}) is not available in this FFmpeg build",
                stream_info.codec_id, stream_info.index
            )));
        };
        match DecoderBuilder::new(media_type)
            .with_codec_name(codec_name.clone())
            .with_hardware_device(device_config.clone())
            .with_filters(filters.get(&media_type).cloned())
            .build_from_reader(reader)
        {
            Ok(decoder) => Ok(decoder),
            Err(e) if device_type.is_some() => {
                // 硬件解码器构建失败（如 hw 初始化失败）：回退软件解码器重试，
                // 与 find_decoder_name 的回退语义对齐；再失败才让错误上抛。
                tracing::warn!(
                    "HW decoder '{codec_name}' failed to build: {e:#}; \
                     falling back to software decoder"
                );
                let software_name = stream_info
                    .find_decoder_name(None)
                    .unwrap_or_else(|| codec_name.clone());
                DecoderBuilder::new(media_type)
                    .with_codec_name(software_name)
                    .with_filters(filters.get(&media_type).cloned())
                    .build_from_reader(reader)
                    .context("Failed to build decoder (hw and software both failed)")
            }
            // 用 `context` 而不是把 `{e:#}` 拼成一句话：后者会把带类型的**分类**
            // （`InvalidConfig` / `Unsupported` 等）降级成无类型消息，调用方再也
            // 没法按变体处理。
            Err(e) => Err(e).context("Failed to build decoder"),
        }
    }

    /// 全流解码模式：为容器中所有可解码的流构建解码器。
    pub fn new_from_reader(
        reader: R,
        filters: Option<Vec<Filter>>,
        device_config: Option<HWDeviceConfig>,
    ) -> Result<Demuxer<R>> {
        let nb_streams = reader.input().nb_streams as usize;
        let filter_map = group_filters(filters);

        let mut streams = Vec::new();
        for stream_idx in 0..nb_streams {
            let stream_info = StreamInfo::from_reader(&reader, stream_idx)?;
            // auto detect hardware acceleration decoder codec
            let device_type = device_config.as_ref().map(|c| c.device_type);
            if stream_info.find_decoder_name(device_type).is_none() {
                // Streams without a registered decoder (chapter tracks,
                // attached pictures, binary data, ...) are skipped instead of
                // failing the whole demuxer.
                tracing::debug!(
                    "Skipping stream {stream_idx}: no decoder for codec_id {:#x}",
                    stream_info.codec_id
                );
                continue;
            }
            let decoder = Self::build_decoder(&reader, &stream_info, &device_config, &filter_map)?;
            streams.push(DemuxerStream::new(decoder, stream_info));
        }

        Ok(Self {
            reader,
            streams,
            passthrough: false,
        })
    }

    /// 单流解码模式：只为 `media_type` 的**最佳流**构建解码器，其余流的 packet
    /// 在迭代时丢弃。适合"只抽视频帧/只取音频"的单流场景。
    ///
    /// "最佳"取自 FFmpeg 的 `av_find_best_stream`（与
    /// `Reader::find_best_stream` 同一判据），
    /// 找不到该类型的流时返回错误。早先这里取的是"该类型的第一个流"，而容器里的
    /// 流顺序并不保证最佳流排在最前——`Demuxer::new` 走 `find_best_stream`，
    /// 两者因此可能选中不同的流。
    pub fn new_single_stream(
        reader: R,
        media_type: MediaType,
        filters: Option<Vec<Filter>>,
        device_config: Option<HWDeviceConfig>,
    ) -> Result<Demuxer<R>> {
        let filter_map = group_filters(filters);

        let (stream_index, _codec_name) = reader.find_best_stream(media_type)?;
        let stream_info = StreamInfo::from_reader(&reader, stream_index)?;
        let decoder = Self::build_decoder(&reader, &stream_info, &device_config, &filter_map)?;

        Ok(Self {
            reader,
            streams: vec![DemuxerStream::new(decoder, stream_info)],
            passthrough: false,
        })
    }

    /// 透传模式（remux）：不构建任何解码器，仅通过
    /// [`demux_packet`](Self::demux_packet) / [`packets`](Self::packets)
    /// 迭代原始 `AVPacket`。
    ///
    /// 开销最小（无 codec 初始化、无解码），用于转封装：配合
    /// [`Muxer::add_copy_stream`](crate::mux::Muxer::add_copy_stream) 和
    /// [`Muxer::mux_packet`](crate::mux::Muxer::mux_packet) 可原样搬运码流。
    pub fn new_passthrough(reader: R) -> Result<Demuxer<R>> {
        Ok(Self {
            reader,
            streams: Vec::new(),
            passthrough: true,
        })
    }

    /// Decoders for the selected streams, in container stream order.
    ///
    /// Always empty in passthrough mode (no decoders are built); use
    /// [`is_passthrough`](Self::is_passthrough) to tell "empty because of
    /// passthrough" apart from "empty because the container has no decodable
    /// stream".
    pub fn streams(&self) -> &[DemuxerStream] {
        &self.streams
    }

    /// Reads back the container chapters (title/start/end in seconds).
    ///
    /// `id` 是容器里记录的实际 id（因此总是 `Some(..)`：读回来的就是具体值，
    /// 不存在"自动编号"这一说）。
    ///
    /// Returns an empty list for containers without chapter support.
    pub fn chapters(&self) -> Vec<Chapter> {
        let input = self.reader.input();
        let nb = input.nb_chapters as usize;
        if nb == 0 || input.chapters.is_null() {
            return Vec::new();
        }
        let mut chapters = Vec::with_capacity(nb);
        let raw = unsafe { std::slice::from_raw_parts(input.chapters, nb) };
        for &c in raw {
            if c.is_null() {
                continue;
            }
            // SAFETY: `c` is a non-null `AVChapter` pointer taken from `input.chapters`
            // (`nb` entries, so the slice above is in bounds). The chapters belong to
            // the input context that `input` borrows, so they stay alive for the whole
            // loop, and only their own fields are read.
            unsafe {
                // SAFETY: `(*c).metadata` is valid for as long as `input` is borrowed.
                let title = Metadata::from_raw_dict((*c).metadata)
                    .get("title")
                    .unwrap_or_default()
                    .to_string();
                let tb = (*c).time_base;
                let tb_secs = tb.num as f64 / tb.den as f64;
                chapters.push(Chapter {
                    id: Some((*c).id),
                    title,
                    start: (*c).start as f64 * tb_secs,
                    end: (*c).end as f64 * tb_secs,
                });
            }
        }
        chapters
    }

    pub fn get_stream(&self, index: usize) -> Result<&DemuxerStream> {
        self.streams
            .iter()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::invalid_config(format!("Stream index: {index} not found")))
    }

    pub fn get_stream_mut(&mut self, index: usize) -> Result<&mut DemuxerStream> {
        self.streams
            .iter_mut()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::invalid_config(format!("Stream index: {index} not found")))
    }

    /// 返回输入容器第 `index` 个流的 [`StreamInfo`]。
    ///
    /// 优先返回构建解码器时缓存的 [`DemuxerStream::stream_info`]（与
    /// [`Self::streams`] / [`Self::get_stream`] 看到的完全一致），避免同一份
    /// 信息存在"缓存"与"实时读取"两个真相源。透传模式没有解码器、也就没有
    /// 缓存，此时回退为实时读取——该模式正是用它来给 [`Muxer::add_copy_stream`]
    /// 建立输出透传流的。
    pub fn stream_info(&self, index: usize) -> Result<StreamInfo> {
        if let Some(demux_stream) = self.streams.iter().find(|s| s.stream_index == index) {
            return Ok(demux_stream.stream_info.clone());
        }
        StreamInfo::from_reader(&self.reader, index)
    }

    /// 输入流总数。
    pub fn nb_streams(&self) -> usize {
        self.reader.input().nb_streams as usize
    }

    /// Whether this demuxer was built in passthrough (remux) mode.
    pub fn is_passthrough(&self) -> bool {
        self.passthrough
    }

    /// 读取下一个**原始 packet**（不解码）。
    ///
    /// 透传模式（[`new_passthrough`](Self::new_passthrough)）的主迭代入口，
    /// 也可在解码模式下用于高级场景（如混合 remux）。注意：解码模式下调用
    /// 本方法会"消耗" packet，对应的帧将不会出现在 [`demux`](Self::demux)
    /// 的迭代中——两种迭代方式不要对同一流混用。
    ///
    /// 返回 `Ok(None)` 表示输入结束。
    pub fn demux_packet(&mut self) -> Result<Option<(usize, AVPacket)>> {
        self.reader.read_packet()
    }

    /// 原始 packet 迭代器（透传模式专用入口，等价于循环调用
    /// [`demux_packet`](Self::demux_packet)）。
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use rsmedia::mux::Demuxer;
    /// use rsmedia::io::StreamReader;
    /// use rsmedia::error::Result;
    /// fn main() -> Result<()> {
    ///     let reader = StreamReader::new("my_file.mp4")?;
    ///     let mut demuxer = Demuxer::new_passthrough(reader)?;
    ///     for result in demuxer.packets() {
    ///         let (stream_index, packet) = result?;
    ///         println!("packet from stream {stream_index}");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn packets(&mut self) -> impl Iterator<Item = Result<(usize, AVPacket)>> + '_ {
        std::iter::from_fn(|| match self.demux_packet() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
    }

    pub fn demux(&mut self) -> Result<Option<(usize, AVFrame)>> {
        if self.is_passthrough() {
            return Err(RsmediaError::invalid_config(
                "Demuxer is in passthrough mode: use demux_packet()/packets() instead of demux()"
                    .to_string(),
            ));
        }
        let mut read_exhausted = false;
        loop {
            if !read_exhausted {
                match self.reader.read_packet() {
                    Ok(Some((stream_idx, packet))) => {
                        let Some(demux_stream) = self
                            .streams
                            .iter_mut()
                            .find(|s| s.stream_index == stream_idx)
                        else {
                            // Packets of skipped streams (chapter tracks,
                            // unselected streams in single-stream mode, ...)
                            // are dropped.
                            tracing::debug!("Dropping packet of undecodable stream {stream_idx}");
                            continue;
                        };
                        if let Some(frame) = demux_stream.decoder.decode_raw_packet(&packet)? {
                            return Ok(Some((stream_idx, frame)));
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
                // 排空阶段：反复 drain 每个尚未结束的解码器，而不是"每轮一次"。
                //
                // 两个要点（旧实现两处都做错了）：
                // 1. `drain_raw` 返回 `Ok(None)` 在 Drained 态只表示"这一拍没帧"
                //    （EAGAIN —— 多线程解码、B 帧 lookahead 都会出现），**不是**
                //    结束信号。据此结束本次 `demux()`，`Iterator::next` 会把
                //    `Ok(None)` 当成迭代终止，解码器里还压着的帧就永远取不出来了。
                // 2. 停止谓词是 [`Decoder::is_finished`](crate::Decoder::is_finished)
                //    而不是 `is_flushed`：带滤镜图时解码器到 EOF 后，滤镜（如
                //    fps/framerate 这类有延迟的）仍可能压着帧，按 `is_flushed`
                //    跳过会丢掉它们。
                //
                // 循环以 `MAX_DRAIN_ITERATIONS` 为上限，与 `Decoder::drain` 一致：
                // 个别编解码器在 EOS 后可能一直回"暂无帧"，没有上限就是挂死。
                let mut drain_iterations = 0usize;
                loop {
                    let mut pending = false;
                    for demux_stream in self.streams.iter_mut() {
                        if demux_stream.decoder.is_finished() {
                            continue;
                        }
                        pending = true;
                        let stream_idx = demux_stream.stream_index;
                        match demux_stream.decoder.drain_raw() {
                            Ok(Some(frame)) => return Ok(Some((stream_idx, frame))),
                            Ok(None) => {
                                tracing::trace!(
                                    "Stream: [{stream_idx}] has no frame ready this pass; retrying drain"
                                );
                            }
                            Err(e) => {
                                tracing::error!("Stream: [{stream_idx}] Decoder Drain Error: {e}");
                                return Err(e);
                            }
                        }
                    }
                    // 所有解码器都真正结束（含滤镜图排空）→ 迭代到此为止。
                    if !pending {
                        return Ok(None);
                    }
                    drain_iterations += 1;
                    if drain_iterations >= crate::MAX_DRAIN_ITERATIONS {
                        tracing::warn!(
                            "Demuxer: decoders produced no frame after {} drain iterations; ending demux()",
                            crate::MAX_DRAIN_ITERATIONS
                        );
                        return Ok(None);
                    }
                }
            }
        }
    }
}

/// Demuxer iterator
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use rsmedia::mux::Demuxer;
/// use rsmedia::error::Result;
/// fn main() -> Result<()> {
///     let mut demuxer = Demuxer::new("my_file.mp4")?;
///     for result in demuxer {
///         let (stream_index, frame) = result?;
///         println!("stream_index: {}, frame: {}", stream_index, frame.width);
///     }
///     Ok(())
/// }
/// ```
impl<R: Reader> Iterator for Demuxer<R> {
    type Item = Result<(usize, AVFrame)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.demux() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// 仅承诺可移动到其他线程独占使用：内部的 AVFormatContextInput /
/// AVCodecContext 均为 FFmpeg 非线程安全句柄，`&Demuxer` 不可跨线程共享，
/// 故只实现 `Send`、不实现 `Sync`。
unsafe impl<R: Reader> Send for Demuxer<R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncoderBuilder, PixelFormat, SampleFormat, StreamReader, StreamWriter, strutils};

    use crate::error::{Context, Result};
    use rsmpeg::avformat::AVFormatContextOutput;
    use rsmpeg::avutil::{AVChannelLayout, AVFrame};
    use std::path::Path;

    /// 生成YUV420P格式的视频帧,彩色渐变测试图
    fn generate_video_frame(width: usize, height: usize, frame_index: i64) -> AVFrame {
        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(PixelFormat::YUV420P.into());
        frame
            .alloc_buffer()
            .context("Failed to allocate buffer for frame")
            .unwrap();

        // 获取各平面参数 (YUV420P布局)
        let y_plane = frame.data_mut()[0];
        let u_plane = frame.data_mut()[1];
        let v_plane = frame.data_mut()[2];

        let y_linesize = frame.linesize[0];
        let u_linesize = frame.linesize[1];
        let v_linesize = frame.linesize[2];

        // 基于帧索引创建动态效果
        let time_factor = (frame_index as f32 * 0.05).sin() * 0.5 + 0.5;

        // 填充Y平面 (亮度)
        for y in 0..height {
            for x in 0..width {
                let index = y * y_linesize as usize + x;
                let gradient = (x as f32 / width as f32 * 255.0) as u8;
                unsafe {
                    *y_plane.add(index) = gradient;
                }
            }
        }

        // 填充U平面 (蓝色分量)
        for y in 0..(height / 2) {
            for x in 0..(width / 2) {
                let index = y * u_linesize as usize + x;
                let u_value = ((time_factor * 128.0) as u8).wrapping_add(128);
                unsafe {
                    *u_plane.add(index) = u_value;
                }
            }
        }

        // 填充V平面 (红色分量)
        for y in 0..(height / 2) {
            for x in 0..(width / 2) {
                let index = y * v_linesize as usize + x;
                let v_value = (((1.0 - time_factor) * 128.0) as u8).wrapping_add(128);
                unsafe {
                    *v_plane.add(index) = v_value;
                }
            }
        }

        frame
    }

    /// 生成FLTP格式的正弦波音频帧
    fn generate_audio_sine_wave_frame(
        freq: f32,
        channels: usize,
        nb_samples: usize,
        sample_rate: i32,
    ) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_format(SampleFormat::FLTP as _);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(channels as i32).into_inner());
        frame.set_sample_rate(sample_rate);
        frame.set_nb_samples(nb_samples as i32);
        frame
            .alloc_buffer()
            .context("Failed to allocate buffer for frame")?;

        let sample_interval = 1.0 / sample_rate as f32;
        let two_pi_f = 2.0 * std::f32::consts::PI * freq;

        for ch in 0..channels {
            let data_ptr = unsafe {
                let ptr = (*frame.as_mut_ptr()).data[ch] as *mut f32;
                if ptr.is_null() {
                    return Err(RsmediaError::msg("Audio data pointer is null"));
                }
                std::slice::from_raw_parts_mut(ptr, nb_samples)
            };

            // 生成正弦波
            data_ptr.iter_mut().enumerate().for_each(|(i, sample)| {
                let t = i as f32 * sample_interval;
                *sample = (two_pi_f * t).sin() * 0.8;
            });
        }

        Ok(frame)
    }

    #[test]
    fn test_mux_demux_video() -> Result<()> {
        let output_path = crate::test_support::test_output_path("mux", "test_mux_demux_video.mp4");

        let (width, height) = (640, 360);
        let video_encoder = Encoder::new_video(width, height)?;

        let mut muxer = Muxer::new(&output_path)?;

        let encoder_frame_rate = video_encoder.frame_rate();
        let encoder_time_base = video_encoder.time_base();
        let video_index = muxer.add_encoder(video_encoder)?;

        // 生成测试视频帧 // 3秒视频 30fps
        for index in 0..3 * encoder_frame_rate.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);

            println!(
                "encode video frame:{:?}, time_base:{:?}, encoder_time_base:{:?}",
                frame, frame.time_base, encoder_time_base
            );
            muxer.mux(frame, video_index)?;
        }

        // 完成写入
        muxer.finish().unwrap();

        //////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////

        // Demuxer 测试视频解码
        let demuxer = Demuxer::new(output_path)?;
        for des in demuxer.streams() {
            println!("{:?}, {:?}", des.stream_index, des.media_type)
        }

        for res in demuxer {
            match res {
                Ok((index, frame)) => {
                    println!(
                        "stream index:{}, {:?}, timebase:{:?}",
                        index, frame, frame.time_base
                    );
                }
                Err(e) => {
                    println!("Error decoding frame: {}", e)
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_mux_demux_audio_aac() -> Result<()> {
        let output_path =
            crate::test_support::test_output_path("mux", "test_mux_demux_audio_aac.aac");
        let sample_rate = 44_100;
        let nb_samples = 1024;
        let channels = 2;

        // 添加音频流
        let audio_encoder = Encoder::new_audio(channels, sample_rate, SampleFormat::FLTP).unwrap();
        let mut muxer = Muxer::new(&output_path)?;

        let encoder_time_base = audio_encoder.time_base();
        let audio_index = muxer.add_encoder(audio_encoder)?;

        // 累积的样本数，用于计算PTS
        let mut total_samples = 0;

        // 生成测试音频帧 // 5秒音频 440Hz
        for _ in 0..(sample_rate * 5 / nb_samples) {
            let mut sine_frame = generate_audio_sine_wave_frame(
                440.0,
                channels as usize,
                nb_samples as usize,
                sample_rate,
            )?;

            // 设置正确的PTS和时间基
            sine_frame.set_pts(total_samples);
            sine_frame.set_time_base(encoder_time_base);

            println!(
                "audio frame: {:?}, time_base={:?}",
                sine_frame, encoder_time_base
            );

            muxer.mux(sine_frame, audio_index)?;

            // 更新累积的样本数
            total_samples += nb_samples as i64;
        }

        muxer.finish().unwrap();

        //////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////

        // Demuxer 测试音频解码
        let demuxer = Demuxer::new(output_path)?;
        for des in demuxer.streams() {
            println!("{:?}, {:?}", des.stream_index, des.media_type)
        }

        for res in demuxer {
            match res {
                Ok((index, frame)) => {
                    println!(
                        "stream index:{}, {:?}, time_base:{:?}",
                        index, frame, frame.time_base
                    )
                }
                Err(e) => {
                    println!("Error decoding frame: {}", e)
                }
            }
        }

        Ok(())
    }

    /// 容器级 metadata 与流级 language 标签的写入与回读验证（多轨场景）。
    #[test]
    fn test_mux_metadata_and_language() -> Result<()> {
        let output_path = crate::test_support::test_output_path("mux", "test_mux_metadata.mp4");

        let (width, height) = (320, 240);
        let sample_rate = 44_100;
        let nb_samples = 1024;

        let video_encoder = Encoder::new_video(width, height)?;
        let video_time_base = video_encoder.time_base();
        let video_frame_rate = video_encoder.frame_rate();
        let audio_encoder = Encoder::new_audio(2, sample_rate, SampleFormat::FLTP).unwrap();
        let audio_time_base = audio_encoder.time_base();

        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;
        let audio_index = muxer.add_encoder(audio_encoder)?;

        // 参数校验：越界索引必须 fail fast（含 NUL 的条目在写头时被跳过并告警）
        assert!(muxer.set_stream_metadata(99, "language", "eng").is_err());

        muxer.set_metadata("title", "rsmedia metadata test")?;
        muxer.set_metadata("artist", "rsmedia")?;
        muxer.set_stream_metadata(video_index, "language", "und")?;
        muxer.set_stream_metadata(audio_index, "language", "chi")?;

        // 1 秒视频 + 1 秒音频（header 在首个 mux 调用时写入，metadata 届时生效）
        for index in 0..video_frame_rate.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * video_time_base.den as i64);
            frame.set_time_base(video_time_base);
            muxer.mux(frame, video_index)?;
        }
        let mut total_samples = 0i64;
        for _ in 0..(sample_rate / nb_samples) {
            let mut frame =
                generate_audio_sine_wave_frame(440.0, 2, nb_samples as usize, sample_rate)?;
            frame.set_pts(total_samples);
            frame.set_time_base(audio_time_base);
            muxer.mux(frame, audio_index)?;
            total_samples += nb_samples as i64;
        }
        muxer.finish()?;

        // 回读验证：容器级 title/artist 与各流 language 标签
        let reader = StreamReader::new(&output_path)?;
        let input = reader.input();

        let get_str = |dict: *mut ffi::AVDictionary, key: &str| -> Option<String> {
            // SAFETY: `dict` points at a live `AVDictionary` owned by the input
            // format context / stream, which outlives this call.
            unsafe { Metadata::from_raw_dict(dict) }
                .get(key)
                .map(String::from)
        };

        let title = get_str(input.metadata, "title");
        assert_eq!(
            title.as_deref(),
            Some("rsmedia metadata test"),
            "container title metadata mismatch"
        );

        let artist = get_str(input.metadata, "artist");
        assert_eq!(artist.as_deref(), Some("rsmedia"));

        let streams = input.streams();
        let raw_or_null = |d: Option<rsmpeg::avutil::AVDictionaryRef>| {
            d.map(|d| d.as_ptr() as *mut _)
                .unwrap_or(std::ptr::null_mut())
        };
        let video_lang = get_str(raw_or_null(streams[video_index].metadata()), "language");
        assert_eq!(video_lang.as_deref(), Some("und"));

        let audio_lang = get_str(raw_or_null(streams[audio_index].metadata()), "language");
        assert_eq!(audio_lang.as_deref(), Some("chi"));

        Ok(())
    }

    #[test]
    fn test_mux_demux_audio_mp3() -> Result<()> {
        let output_path =
            crate::test_support::test_output_path("mux", "test_mux_demux_audio_mp3.mp3");
        let sample_rate = 44_100;
        let bit_rate = 128_000;
        let nb_samples = 1152; // libmp3lame 要求的 frame_size 为 1152
        let channels = 2;

        // 修改音频编码器为MP3
        let audio_encoder =
            EncoderBuilder::new_audio(bit_rate, channels, sample_rate, SampleFormat::FLTP)
                // 使用LAME MP3编码器
                .with_codec_name("libmp3lame".to_string())
                .build()?;

        let mut muxer = Muxer::new(output_path)?;

        let encoder_time_base = audio_encoder.time_base();
        let audio_index = muxer.add_encoder(audio_encoder)?;

        // 累积的样本数，用于计算PTS
        let mut total_samples = 0;

        // 生成测试音频帧 // 5秒音频 440Hz
        for _ in 0..(sample_rate * 5 / nb_samples) {
            let mut sine_frame = generate_audio_sine_wave_frame(
                440.0,
                channels as usize,
                nb_samples as usize,
                sample_rate,
            )?;

            // 设置正确的PTS和时间基
            sine_frame.set_pts(total_samples);
            sine_frame.set_time_base(encoder_time_base);

            println!(
                "audio frame: {:?}, time_base={:?}",
                sine_frame, encoder_time_base
            );

            muxer.mux(sine_frame, audio_index)?;

            // 更新累积的样本数
            total_samples += nb_samples as i64;
        }

        muxer.finish().unwrap();

        Ok(())
    }

    #[test]
    fn test_multiple_streams() -> Result<()> {
        // 视频参数
        pub const VIDEO_WIDTH: usize = 640;
        pub const VIDEO_HEIGHT: usize = 360;
        pub const VIDEO_FPS: f32 = 30f32;
        pub const VIDEO_DURATION_SEC: u32 = 3;

        // 音频参数
        pub const AUDIO_SAMPLE_RATE: i32 = 48_000;
        pub const AUDIO_CHANNELS: i32 = 2;
        pub const SAMPLES_PER_FRAME: u32 = 1024;

        let output_path = crate::test_support::test_output_path("mux", "test_multiple_streams.mp4");

        let video_encoder = EncoderBuilder::new_video(VIDEO_WIDTH, VIDEO_HEIGHT)
            .with_fps(VIDEO_FPS)
            .build()?;

        let audio_encoder =
            Encoder::new_audio(AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, SampleFormat::FLTP)?;

        let mut muxer = Muxer::new(&output_path)?;

        let video_time_base = video_encoder.time_base();
        let audio_time_base = audio_encoder.time_base();

        // 添加视频流 和 音频流
        let video_idx = muxer.add_encoder(video_encoder)?;
        let audio_idx = muxer.add_encoder(audio_encoder)?;

        // 计算总视频帧数
        let total_video_frames = (VIDEO_FPS as u32 * VIDEO_DURATION_SEC) as i64;

        // 计算每个视频帧对应的音频样本数，例如：48000Hz / 30fps = 1600个(样本/视频帧)
        let audio_samples_per_video_frame = (AUDIO_SAMPLE_RATE as f64 / VIDEO_FPS as f64) as usize;

        // 音频的PTS，需要根据视频帧数和音频帧数计算
        let mut audio_pts: i64 = 0;

        for frame_idx in 0..total_video_frames {
            // 生成视频帧
            let mut video_frame = generate_video_frame(VIDEO_WIDTH, VIDEO_HEIGHT, frame_idx);

            // 设置视频帧PTS (以编码器 90kHz 为基准)
            let frame_duration = video_time_base.den as i64 / VIDEO_FPS as i64;
            let video_pts = frame_idx * frame_duration;
            video_frame.set_pts(video_pts);
            video_frame.set_time_base(video_time_base);

            println!(
                "Video frame: {}, pts: {}, timebase: {:?}",
                frame_idx, video_pts, video_time_base
            );
            muxer.mux(video_frame, video_idx)?;

            // 视频和音频帧不一对一写入,为什么需要这样计算？
            // 1. 不同的时间基准 ：
            // - 视频以帧率计算（如30fps）
            // - 音频以采样率计算（如48000Hz）
            // 2. 不同的编码要求 ：
            // - AAC音频编码器要求固定的帧大小（通常是1024个样本）
            // - 视频编码器（如H.264）有不同的帧大小要求
            // 3. 同步需求 ：
            // - 为了保持音视频同步，需要确保每个视频帧对应的音频数据都被正确编码
            //
            // 向上取整除法，计算需要生成的音频帧数
            // 在音频处理中，我们需要知道多少个固定大小的帧能容纳所有样本。如果不向上取整，可能会丢失部分音频数据。
            // 例如，对于1600个样本和1024大小的帧：
            // - 1600 / 1024 = 1.56... ≈ 1（向下取整）
            // - 但1个帧只能容纳1024个样本，剩余576个样本会被丢弃
            // - 使用向上取整：(1600 + 1024 - 1) / 1024 = 2，确保所有样本都被处理
            // 这就是为什么在计算音频帧数时使用这个向上取整除法公式的原因
            let audio_frames_needed =
                (audio_samples_per_video_frame as u32).div_ceil(SAMPLES_PER_FRAME);
            for _ in 0..audio_frames_needed {
                // 使用新的音频帧生成函数
                let mut audio_frame = generate_audio_sine_wave_frame(
                    440.0, // 440Hz的音调
                    AUDIO_CHANNELS as usize,
                    SAMPLES_PER_FRAME as usize,
                    AUDIO_SAMPLE_RATE,
                )?;

                audio_frame.set_pts(audio_pts);
                audio_frame.set_time_base(audio_time_base);

                println!(
                    "Audio frame: pts: {}, samples: {}, timebase: {:?}",
                    audio_pts, SAMPLES_PER_FRAME, audio_time_base
                );

                muxer.mux(audio_frame, audio_idx)?;

                // 更新音频PTS (以采样率为基准)
                audio_pts += SAMPLES_PER_FRAME as i64;
            }
        }

        // 完成写入
        muxer.finish().unwrap();

        /////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////

        // 解封装验证
        let demuxer = Demuxer::new(output_path)?;
        for stream in demuxer.streams() {
            println!("{:?}, {:?}", stream.stream_index, stream.media_type)
        }

        for res in demuxer {
            match res {
                Ok((index, frame)) => {
                    println!("stream index:{}, {:?}", index, frame)
                }
                Err(e) => {
                    println!("Error decoding frame: {}", e)
                }
            }
        }

        Ok(())
    }

    /// transcode from one container format to another
    ///
    /// # Examples
    ///
    /// ```no_run
    /// transcode("input.mp4", "output.mov").unwrap();
    /// ```
    fn transcode(input_path: &str, output_path: &str) -> Result<()> {
        let mut input_reader = StreamReader::new(Path::new(input_path))?;
        let input = input_reader.input();

        // inner output
        use crate::io::Writer as _;
        let mut output_writer = StreamWriter::new(Path::new(output_path))?;
        let output = output_writer.output_mut();

        let stream_mapping: Vec<_> = {
            let mut stream_index = 0usize;
            input
                .streams()
                .iter()
                .map(|stream| {
                    let codec_type = stream.codecpar().codec_type();
                    if !codec_type.is_video() && !codec_type.is_audio() && !codec_type.is_subtitle()
                    {
                        None
                    } else {
                        output.new_stream().set_codecpar(stream.codecpar().clone());
                        stream_index += 1;
                        Some(stream_index - 1)
                    }
                })
                .collect()
        };

        output
            .dump(0, strutils::str_to_cstring(output_path)?.as_c_str())
            .context("Dump output format context failed.")?;

        output
            .write_header(&mut None)
            .context("Writer header failed.")?;

        while let Some((input_stream_index, mut packet)) =
            input_reader.read_packet().context("Read packet failed.")?
        {
            let Some(output_stream_index) = stream_mapping[input_stream_index] else {
                continue;
            };
            {
                let in_stream = &input_reader.input().streams()[input_stream_index];
                let output_stream = &output.streams()[output_stream_index];
                packet.rescale_ts(in_stream.time_base, output_stream.time_base);
                packet.set_stream_index(output_stream_index as i32);
                packet.set_pos(-1);
            }
            output
                .interleaved_write_frame(&mut packet)
                .context("Interleaved write frame failed.")?;
        }

        output.write_trailer().context("Write trailer failed.")
    }

    #[test]
    fn test_transcode() -> Result<()> {
        let output = crate::test_support::test_output_path("mux", "test_transcode.mov");
        transcode("assets/mp4.mp4", output.to_str().unwrap())?;
        Ok(())
    }

    /// 验证两种情况：
    /// 1. 从未 mux 任何数据（header 未写）时，`finish()` 应为空操作返回空累积器，
    ///    不会产生仅含 encode-EOS 包但无 header/trailer 的残缺输出。
    /// 2. 重复调用 `finish()` 是幂等的，不会重复写 trailer。
    #[test]
    fn test_finish_without_mux_and_idempotent() -> Result<()> {
        let output_path =
            crate::test_support::test_output_path("mux", "test_finish_without_mux.mp4");

        let encoder = Encoder::new_video(320, 240)?;
        let mut muxer = Muxer::new(output_path)?;
        muxer.add_encoder(encoder)?;

        // 未 mux 任何帧，直接 finish：应为空操作，不写 header
        muxer.finish()?;
        assert!(!muxer.have_written_header, "header should not be written");

        // 第二次 finish 应幂等，不重复写 trailer
        muxer.finish()?;

        Ok(())
    }

    /// 验证用户“忘记调用 finish()”时，Drop 会自动 flush 编码器延迟缓冲并写 trailer，
    /// 生成的容器文件依旧可以被 Demuxer 正常读取（不损坏）。
    #[test]
    fn test_drop_flush_without_explicit_finish() -> Result<()> {
        let output_path =
            crate::test_support::test_output_path("mux", "test_drop_flush_no_finish.mp4");

        let (width, height) = (320, 240);
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();

        {
            let mut muxer = Muxer::new(&output_path)?;
            let video_index = muxer.add_encoder(video_encoder)?;
            for index in 0..12 {
                let mut frame = generate_video_frame(width, height, index);
                frame.set_pts(index * encoder_time_base.den as i64);
                frame.set_time_base(encoder_time_base);
                muxer.mux(frame, video_index)?;
            }
            // 故意不调用 finish()，直接离开作用域触发 Drop 自动 flush
        }

        // Drop 自动完成 flush + trailer 后，文件应可被读取并解码
        let demuxer = Demuxer::new(&output_path)?;
        // 能成功创建 demuxer 且能读到帧即证明容器完整（有 header + trailer）
        let decoded = demuxer.filter_map(|res| res.ok().map(|_| ())).count();
        assert!(
            decoded > 0,
            "expected at least one decoded frame, got {decoded}"
        );

        Ok(())
    }

    /// 章节写入与回读：2 个章节按毫秒时间基写入 MP4，重开后 `Demuxer::chapters()`
    /// 应还原标题与秒级起止时间；参数校验（start<0、end<=start、内嵌 NUL）须报错。
    /// 字幕通过 [`Muxer::mux_subtitle_segment`] 进入容器：视频 + 字幕两路
    /// 编码流，写 4 条 cue 后用库内字幕解码通道回读，逐字段断言无损。
    #[test]
    fn test_mux_subtitle_segment() -> Result<()> {
        use crate::MediaType;
        use crate::subtitle::SubtitleSegment;

        let output_path =
            crate::test_support::test_output_path("mux", "test_mux_subtitle_segment.mp4");
        crate::test_support::remove_test_output(&output_path);

        let header = "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

        let video_encoder = Encoder::new_video(320, 240)?;
        let video_tb = video_encoder.time_base();
        let subtitle_encoder = EncoderBuilder::new_subtitle()
            .with_codec_name(Some("mov_text".to_string()))
            .with_subtitle_header(header)
            .build()?;

        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;
        let subtitle_index = muxer.add_encoder(subtitle_encoder)?;
        assert_ne!(video_index, subtitle_index, "two encoder streams");
        muxer.set_stream_metadata(subtitle_index, "language", "eng")?;

        let segments = [
            SubtitleSegment::new(0, 400, "First cue, with, commas"),
            SubtitleSegment::new(500, 900, "Second cue"),
            SubtitleSegment::new(1000, 1400, "Third cue"),
            SubtitleSegment::new(1500, 1900, "Last cue"),
        ];

        // 1 秒视频（30fps），pts 走自动编号；字幕逐段编码进容器。
        for i in 0..30i64 {
            let mut frame = generate_video_frame(320, 240, i);
            frame.set_time_base(video_tb);
            muxer.mux(frame, video_index)?;
        }
        for segment in &segments {
            muxer.mux_subtitle_segment(segment, subtitle_index)?;
        }
        muxer.finish()?;

        // trailer 之后再写字幕段必须被拒绝（否则数据被追加到已封闭的容器尾部）
        let err = muxer
            .mux_subtitle_segment(&segments[0], subtitle_index)
            .expect_err("mux_subtitle_segment after finish must fail");
        assert!(
            err.is_invalid_config(),
            "post-trailer write must be an invalid_config error: {err}"
        );

        // 回读：字幕解码通道逐段无损还原。
        let mut reader = StreamReader::new(&output_path)?;
        let mut decoder = DecoderBuilder::new(MediaType::SUBTITLE)
            .with_codec_name(Some("mov_text".to_string()))
            .build_from_reader(&reader)?;
        let mut decoded = Vec::new();
        while let Some(segment) = decoder.decode_subtitle_segment(&mut reader)? {
            decoded.push(segment);
        }
        assert_eq!(decoded.len(), segments.len(), "decoded: {decoded:?}");
        for (got, want) in decoded.iter().zip(segments.iter()) {
            assert_eq!(got.text, want.text, "text mismatch");
            assert_eq!(got.start_ms, want.start_ms, "start_ms mismatch");
            assert_eq!(got.end_ms, want.end_ms, "end_ms mismatch");
        }

        crate::test_support::remove_test_output(&output_path);
        Ok(())
    }

    #[test]
    fn test_mux_chapters() -> Result<()> {
        let output_path = crate::test_support::test_output_path("mux", "test_mux_chapters.mp4");

        let (width, height) = (320, 240);
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();

        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;

        // 参数校验 fail fast（时间非法；含 NUL 的标题在写头时被跳过并告警）
        assert!(muxer.add_chapter(Chapter::new("bad", -1.0, 1.0)).is_err());
        assert!(muxer.add_chapter(Chapter::new("bad", 2.0, 1.0)).is_err());

        muxer.add_chapter(Chapter::new("Intro", 0.0, 1.0))?;
        muxer.add_chapter(Chapter::new("Part Two", 1.0, 2.0))?;
        // 显式 id（`None` 才自动编号）：写进 AVChapter，但容器会自行编号，见下方断言。
        muxer.add_chapter(Chapter {
            id: Some(42),
            ..Chapter::new("Explicit", 2.0, 3.0)
        })?;

        // 1 秒视频（30fps）
        for index in 0..encoder_time_base.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        // 回读：标题 + 秒级起止
        let demuxer = Demuxer::new(&output_path)?;
        let chapters = demuxer.chapters();
        assert_eq!(chapters.len(), 3, "expected 3 chapters, got {chapters:?}");
        assert_eq!(chapters[0].title, "Intro");
        assert!((chapters[0].start - 0.0).abs() < 1e-6);
        assert!((chapters[0].end - 1.0).abs() < 1e-6);
        assert_eq!(chapters[1].title, "Part Two");
        assert!((chapters[1].start - 1.0).abs() < 1e-6);
        assert!((chapters[1].end - 2.0).abs() < 1e-6);
        // id 语义：未指定时自动编号（0, 1, ...）；回读一律是具体值（`Some`）。
        assert_eq!(chapters[0].id, Some(0), "auto-assigned ids start at 0");
        assert_eq!(chapters[1].id, Some(1));
        // MP4（movenc）把章节写成文本轨并自行顺次编号，因此显式 id 不会保留；
        // 回读拿到的仍是具体值（`Some`），只是由容器决定。
        assert_eq!(chapters[2].id, Some(2));
        assert_eq!(chapters[2].title, "Explicit");

        // MKV 使用原生 chapter atom（非 MP4 文本轨），同样必须完整回传
        let mkv_path = crate::test_support::test_output_path("mux", "test_mux_chapters.mkv");
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();
        let mut muxer = Muxer::new(&mkv_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;
        muxer.add_chapter(Chapter::new("MKV Intro", 0.0, 1.0))?;
        muxer.add_chapter(Chapter {
            id: Some(7),
            ..Chapter::new("MKV Explicit", 1.0, 2.0)
        })?;
        for index in 0..(encoder_time_base.den as i64 * 2) {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        let demuxer = Demuxer::new(&mkv_path)?;
        let chapters = demuxer.chapters();
        assert_eq!(
            chapters.len(),
            2,
            "expected 2 mkv chapters, got {chapters:?}"
        );
        assert_eq!(chapters[0].title, "MKV Intro");
        assert!((chapters[0].end - 1.0).abs() < 1e-6);
        // Matroska 同样忽略显式 id，按位置从 1 编号（ffprobe 看到的 uid 就是 1、2）。
        assert_eq!(chapters[0].id, Some(1), "matroska ids start at 1");
        assert_eq!(chapters[1].id, Some(2));
        assert_eq!(chapters[1].title, "MKV Explicit");
        assert!((chapters[1].start - 1.0).abs() < 1e-6);
        assert!((chapters[1].end - 2.0).abs() < 1e-6);

        Ok(())
    }

    /// GIF 调色板管线（palettegen/paletteuse 单遍滤镜图）编码：
    /// 60 帧 @30fps 输入 → fps=10 抽帧 + 调色板量化 → gif 编码器（pal8），
    /// 回读应得到 ~20 帧 GIF（fps 滤镜丢帧），解码器为 gif。
    #[test]
    fn test_encode_gif_palette_pipeline() -> Result<()> {
        let output_path = crate::test_support::test_output_path("mux", "test_gif_palette.gif");
        crate::test_support::remove_test_output(&output_path);

        let (width, height) = (96usize, 64usize);
        let in_fps = 30.0f32;
        let out_fps = 10.0f32;

        // 没有gif_palette
        let gif_palette = crate::filter::video::gif_palette(out_fps, None);
        if crate::filter::get_by_name(gif_palette.name())?.is_none() {
            return Ok(());
        }

        // 编码器按**输入**帧率（30fps）构建，滤镜图内 fps=10 完成抽帧
        let encoder = EncoderBuilder::new_video(width, height)
            .with_codec_name("gif".to_string())
            .with_fps(in_fps)
            .with_filters(vec![gif_palette])
            .build()?;

        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(encoder)?;

        // 2 秒 @30fps 输入（编码器 time_base = 1/30，帧间隔 1 tick）
        let encoder_time_base = ffi::AVRational { num: 1, den: 30 };
        for index in 0..60i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        // 回读验证：帧数 ~20（fps=10 x 2s），codec 为 GIF，尺寸不变
        let demuxer = Demuxer::new(&output_path)?;
        let frames: Vec<_> = demuxer.filter_map(|res| res.ok()).collect();
        assert_eq!(
            frames.len(),
            20,
            "expected 20 gif frames (60 in @30fps -> 10fps), got {}",
            frames.len()
        );
        let (_, frame) = &frames[0];
        assert_eq!(frame.width as usize, width);
        assert_eq!(frame.height as usize, height);

        let frame_count = frames.len() as f64;

        // 容器流信息校验
        let reader = StreamReader::new(&output_path)?;
        let stream = &reader.input().streams()[0];
        assert_eq!(stream.codecpar().codec_id, ffi::AV_CODEC_ID_GIF);

        // 时间戳校验：20 帧 @10fps 时长应为 ~2.0s，容器帧率 10fps。
        // 回归保护：muxer 在 write_header 内强制 GIF 流 time_base=1/100，
        // 若 pts 未按最终流时间基重缩放，时长会缩短 10 倍（0.2s）。
        let tb = stream.time_base;
        let duration_secs = stream.duration as f64 * tb.num as f64 / tb.den as f64;
        assert!(
            (duration_secs - frame_count / out_fps as f64).abs() < 0.1,
            "stream duration should be ~{frame_count} / {out_fps}s, got {duration_secs}s"
        );
        assert_eq!(
            (stream.avg_frame_rate.num, stream.avg_frame_rate.den),
            (out_fps as i32, 1),
            "container frame rate should be {out_fps}"
        );

        Ok(())
    }

    /// 封面图（attached_pic）写入：主视频流 + mjpeg 封面流（RGB24 帧自动转
    /// 编码器协商格式），输出中封面流应带 AV_DISPOSITION_ATTACHED_PIC 标记。
    #[test]
    fn test_mux_cover_art() -> Result<()> {
        use crate::pixel::PixelFormat;

        let output_path = crate::test_support::test_output_path("mux", "test_mux_cover_art.mp4");

        let (width, height) = (320, 240);
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();

        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;

        // 生成一张 RGB24 渐变封面帧（编码器自动协商并转换为 mjpeg 支持的格式）
        let mut cover = AVFrame::new();
        cover.set_width(width as i32);
        cover.set_height(height as i32);
        cover.set_format(PixelFormat::RGB24.into());
        cover.alloc_buffer().context("alloc cover frame")?;
        let rgb = cover.data_mut()[0];
        let linesize = cover.linesize[0];
        for y in 0..height {
            for x in 0..width {
                let index = y * linesize as usize + x * 3;
                unsafe {
                    *rgb.add(index) = (x * 255 / width) as u8;
                    *rgb.add(index + 1) = (y * 255 / height) as u8;
                    *rgb.add(index + 2) = 128;
                }
            }
        }
        let cover_index = muxer.add_cover_art(cover)?;

        // 1 秒主视频
        for index in 0..encoder_time_base.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        // 回读验证：封面流存在且带 attached_pic 标记，编码为 mjpeg
        let reader = StreamReader::new(&output_path)?;
        let streams = reader.input().streams();
        assert_eq!(
            streams.len(),
            2,
            "expected 2 streams (video + cover), got {}",
            streams.len()
        );
        let cover_stream = &streams[cover_index];
        assert!(
            cover_stream.disposition & ffi::AV_DISPOSITION_ATTACHED_PIC as i32 != 0,
            "cover stream must be marked with AV_DISPOSITION_ATTACHED_PIC"
        );
        assert_eq!(
            cover_stream.codecpar().codec_id,
            ffi::AV_CODEC_ID_MJPEG,
            "cover stream must be mjpeg-encoded"
        );

        Ok(())
    }

    /// 包一层 [`BufferWriter`](crate::io::BufferWriter)，把每次输出记为**字节数**
    /// 并累加。
    ///
    /// `Out` 取 `usize` 而不是字节块，是为了能用一条不变量验收"header 与一帧的多个
    /// packet 的输出都被累积返回"：所有 `mux`/`finish` 返回的字节数之和，必须等于
    /// writer 实际写出的总字节数。任何一处"覆盖而非累积"都会让这个等式不成立。
    struct CountingWriter {
        inner: crate::io::BufferWriter,
        total: usize,
    }

    impl CountingWriter {
        fn new(format: &str) -> Result<Self> {
            Ok(Self {
                inner: crate::io::BufferWriter::new(format)?,
                total: 0,
            })
        }

        /// 计入本次写出的字节数，并把同样的大小作为本次的 `Out`。
        fn count(&mut self, bytes: bytes::Bytes) -> usize {
            self.total += bytes.len();
            bytes.len()
        }
    }

    impl Writer for CountingWriter {
        type Out = usize;
        type Accum = usize;

        fn merge_out(acc: &mut usize, out: usize) {
            *acc += out;
        }

        fn merge_accum(acc: &mut usize, other: usize) {
            *acc += other;
        }

        fn write_header(&mut self) -> Result<usize> {
            let bytes = self.inner.write_header()?;
            Ok(self.count(bytes))
        }

        fn write_frame(&mut self, packet: &mut AVPacket) -> Result<usize> {
            let bytes = self.inner.write_frame(packet)?;
            Ok(self.count(bytes))
        }

        fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<usize> {
            let bytes = self.inner.write_interleaved(packet)?;
            Ok(self.count(bytes))
        }

        fn write_trailer(&mut self) -> Result<usize> {
            let bytes = self.inner.write_trailer()?;
            Ok(self.count(bytes))
        }

        fn output(&self) -> &AVFormatContextOutput {
            self.inner.output()
        }

        fn output_mut(&mut self) -> &mut AVFormatContextOutput {
            self.inner.output_mut()
        }

        fn is_header_written(&self) -> bool {
            self.inner.is_header_written()
        }
    }

    /// header 与一帧编出的多个 packet（B 帧重排序、编码器内部缓冲）的输出都必须
    /// 累积进返回值；只留最后一个会让缓冲型 Writer 的调用方拿到缺头/截断的容器。
    ///
    /// 验收方式是一条总量不变量，而不是去数 packet：`mux`/`finish` 返回的字节数
    /// 之和必须等于 writer 实际写出的总字节数。
    #[test]
    fn test_mux_returns_every_byte_it_wrote() -> Result<()> {
        let mut muxer = Muxer::new_from_writer(CountingWriter::new("mp4")?);
        let encoder = Encoder::new_video(64, 64)?;
        let index = muxer.add_encoder(encoder)?;

        let mut returned = 0usize;
        for frame_index in 0..30i64 {
            let mut frame = generate_video_frame(64, 64, frame_index);
            frame.set_pts(frame_index);
            returned += muxer.mux(frame, index)?;
        }
        returned += muxer.finish()?;

        let written = muxer.writer.total;
        assert!(written > 0, "the muxer wrote nothing at all");
        assert_eq!(
            returned, written,
            "mux/finish handed back {returned} of the {written} bytes written: \
             a header or packet output was overwritten instead of accumulated"
        );
        Ok(())
    }

    /// 单流模式只为 `media_type` 的**最佳流**建解码器（与
    /// `Reader::find_best_stream` 同一判据），其余流的 packet 被丢弃。
    #[test]
    fn test_demux_single_stream_selects_the_best_stream() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_demux_single_stream.mp4");

        // 先写一个 video + audio 的文件，好让"选哪条流"真的有得选。
        {
            let mut muxer = Muxer::new(&path)?;
            let video_index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
            let audio_index =
                muxer.add_encoder(Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?)?;
            for frame_index in 0..5i64 {
                let mut frame = generate_video_frame(64, 64, frame_index);
                frame.set_pts(frame_index);
                muxer.mux(frame, video_index)?;
            }
            for index in 0..5i64 {
                let audio = generate_audio_sine_wave_frame(440.0, 2, 1024, 44_100)?;
                let mut audio = audio;
                audio.set_pts(index * 1024);
                muxer.mux(audio, audio_index)?;
            }
            muxer.finish()?;
        }

        let reader = StreamReader::new(&path)?;
        let (expected_index, _) = reader.find_best_stream(MediaType::VIDEO)?;
        let mut demuxer = Demuxer::new_single_stream(reader, MediaType::VIDEO, None, None)?;
        assert_eq!(
            demuxer.streams().len(),
            1,
            "single-stream mode must build exactly one decoder"
        );
        assert_eq!(demuxer.streams()[0].stream_index, expected_index);

        let mut decoded = 0usize;
        while let Some((index, _frame)) = demuxer.demux()? {
            assert_eq!(index, expected_index, "only the selected stream may appear");
            decoded += 1;
        }
        assert!(decoded > 0, "single-stream mode decoded nothing");
        assert!(
            !demuxer.is_passthrough(),
            "single-stream mode decodes, it is not a passthrough demuxer"
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// header 写出之后再 `add_encoder` 必须报错（invalid_config），而不是 SIGSEGV。
    ///
    /// AVFormatContext 的流数组在 `write_header`（首个包 mux 时懒触发）之后固定，
    /// 中途扩张会让 `av_interleaved_write_frame` 访问越界流索引 —— 边界测试实测段错误。
    #[test]
    fn test_add_encoder_after_packets_is_rejected() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_late_add.mp4");
        let mut muxer = Muxer::new(&path)?;
        let index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
        for frame_index in 0..3i64 {
            let mut frame = generate_video_frame(64, 64, frame_index);
            frame.set_pts(frame_index);
            muxer.mux(frame, index)?;
        }

        let late = match muxer.add_encoder(Encoder::new_video(64, 64)?) {
            Ok(_) => panic!("add_encoder after packets must fail, not segfault"),
            Err(e) => e,
        };
        assert!(late.is_invalid_config(), "{late}");

        // 已注册的流仍可继续写、finish 仍正常。
        let mut frame = generate_video_frame(64, 64, 9);
        frame.set_pts(9);
        muxer.mux(frame, index)?;
        muxer.finish()?;
        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// `packets()` 是 `demux_packet()` 的迭代器形式：同一输入上两者产出的
    /// packet 序列必须逐条一致（同样的流索引与 pts）。
    #[test]
    fn test_packets_iterator_matches_demux_packet() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_packets_iterator.mp4");
        {
            let mut muxer = Muxer::new(&path)?;
            let index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
            for frame_index in 0..5i64 {
                let mut frame = generate_video_frame(64, 64, frame_index);
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        let mut via_iterator = Demuxer::new_passthrough(StreamReader::new(&path)?)?;
        let iterator_items: Vec<(usize, i64)> = via_iterator
            .packets()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|(index, packet)| (index, packet.pts))
            .collect();

        let mut via_loop = Demuxer::new_passthrough(StreamReader::new(&path)?)?;
        let mut loop_items = Vec::new();
        while let Some((index, packet)) = via_loop.demux_packet()? {
            loop_items.push((index, packet.pts));
        }

        assert!(!iterator_items.is_empty(), "no packets read back");
        assert_eq!(
            iterator_items, loop_items,
            "packets() and demux_packet() disagree"
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// `set_normalize_timestamps(true)`：epoch 级起点（模拟 `use_wallclock_as_timestamps=1`
    /// 的墙钟 pts）在 FLV 里被平移到 0，帧间间隔与采集卡顿造成的跳变逐 tick 保留。
    /// 默认（false）时相对间隔虽在、但起点被 FLV 的 32 位毫秒字段回绕成垃圾值。
    #[test]
    fn test_normalize_timestamps_shifts_epoch_pts_to_zero() -> Result<()> {
        // 25fps + ultrafast（无 B 帧）：pts/dts 同坐标，容器不做负时间戳平移，
        // 可以把回读的 pts 与注入值逐个精确比对。
        const EPOCH_TICKS: i64 = 1_789_635_572 * 25;
        // 第 5 帧前模拟一次 1s 的采集卡顿（正常间隔 40ms = 1 tick），其后恢复。
        let delta_ticks = |i: i64| if i >= 5 { 25 + (i - 5) } else { i };

        let run = |normalize: bool, name: &str| -> Result<Vec<i64>> {
            let path = crate::test_support::test_output_path("mux", name);
            {
                let mut muxer = Muxer::new(&path)?;
                muxer.set_normalize_timestamps(normalize);
                let mut options = crate::Options::new();
                options.insert("preset", "ultrafast");
                let encoder = EncoderBuilder::new_video(64, 64)
                    .with_fps(25.0)
                    .with_options(options)
                    .build()?;
                let index = muxer.add_encoder(encoder)?;
                for i in 0..10i64 {
                    let mut frame = generate_video_frame(64, 64, i);
                    frame.set_pts(EPOCH_TICKS + delta_ticks(i));
                    muxer.mux(frame, index)?;
                }
                muxer.finish()?;
            }

            let reader = StreamReader::new(&path)?;
            let tb = reader.input().streams()[0].time_base;
            let mut demuxer = Demuxer::new_passthrough(reader)?;
            let mut pts_us: Vec<i64> = demuxer
                .packets()
                .map(|item| {
                    item.map(|(_, packet)| {
                        rsmpeg::avutil::av_rescale_q(packet.pts, tb, ffi::AV_TIME_BASE_Q)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // 包顺序受 dts 交错影响，比较排序后的时间轴。
            pts_us.sort_unstable();
            crate::test_support::remove_test_output(&path);
            Ok(pts_us)
        };

        let normalized = run(true, "test_normalize_ts_on.flv")?;
        let raw = run(false, "test_normalize_ts_off.flv")?;
        assert_eq!(normalized.len(), 10, "帧数不对：{normalized:?}");

        // 归一化后：首个提交单元落在 0，后续间隔（含 1s 卡顿）逐 tick 保留。
        let expected: Vec<i64> = (0..10i64).map(|i| delta_ticks(i) * 40_000).collect();
        assert_eq!(
            normalized, expected,
            "normalize=true 的时间轴应为 0 起点且保留真实间隔"
        );
        // 未开启时：相对间隔仍在，但 FLV 的时间戳字段只有 32 位毫秒，epoch 起点被
        // 回绕成垃圾值（文件 start 落在几十万秒处）——这正是必须归一化的原因。
        let raw_deltas: Vec<i64> = raw.windows(2).map(|w| w[1] - w[0]).collect();
        let expected_deltas: Vec<i64> = (1..10i64)
            .map(|i| (delta_ticks(i) - delta_ticks(i - 1)) * 40_000)
            .collect();
        assert_eq!(raw_deltas, expected_deltas, "回绕不应破坏相对间隔");
        assert_ne!(
            raw.first().copied().unwrap_or(0),
            EPOCH_TICKS * 40_000,
            "normalize=false 时 FLV 承载不了 epoch 起点"
        );
        Ok(())
    }

    /// 编码流与透传流不能混用：两条入口各自拒绝另一类流，并给出可读的错误。
    #[test]
    fn test_mux_rejects_wrong_stream_kind() -> Result<()> {
        // 源文件与两个输出文件必须分开：`Muxer::new` 以写模式打开会截断它。
        let source = crate::test_support::test_output_path("mux", "test_wrong_kind_source.mp4");
        {
            let mut muxer = Muxer::new(&source)?;
            let index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
            for frame_index in 0..3i64 {
                let mut frame = generate_video_frame(64, 64, frame_index);
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        // 编码流 -> mux_packet：必须是"不是透传流"的错误。
        let encoder_side = crate::test_support::test_output_path("mux", "test_wrong_kind_a.mp4");
        let mut muxer = Muxer::new(&encoder_side)?;
        let encoder_index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
        let mut packet = AVPacket::new();
        let err = match muxer.mux_packet(&mut packet, encoder_index) {
            Ok(_) => panic!("mux_packet must reject an encoder stream"),
            Err(e) => e,
        };
        assert!(
            err.is_invalid_config(),
            "mux_packet on an encoder stream is a caller mistake: {err}"
        );
        assert!(err.to_string().contains("not a copy stream"), "{err}");

        // 透传流 -> mux：必须是"是透传流"的错误。透传流的编解码参数取自源流。
        let copy_side = crate::test_support::test_output_path("mux", "test_wrong_kind_b.mp4");
        let mut muxer = Muxer::new(&copy_side)?;
        let reader = StreamReader::new(&source)?;
        let info = StreamInfo::from_reader(&reader, 0)?;
        let copy_index = muxer.add_copy_stream(&info)?;
        let frame = generate_video_frame(64, 64, 0);
        let err = match muxer.mux(frame, copy_index) {
            Ok(_) => panic!("mux must reject a copy stream"),
            Err(e) => e,
        };
        assert!(
            err.is_invalid_config(),
            "mux on a copy stream is a caller mistake: {err}"
        );
        assert!(err.to_string().contains("use mux_packet"), "{err}");

        for path in [&source, &encoder_side, &copy_side] {
            crate::test_support::remove_test_output(path);
        }
        Ok(())
    }

    /// `finish()` 是幂等的：第一次写 trailer，之后每次都是 no-op。
    ///
    /// 它内部对每个流调用 `Encoder::flush`，而 `Encoder::flush` 现在按状态幂等
    /// 短路 —— 否则第二次 finish 会撞上 FFmpeg 的 "encoder is already flushed"，
    /// 与这里承诺的幂等语义矛盾。
    #[test]
    fn test_finish_is_idempotent_with_content() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_finish_idempotent.mp4");
        let mut muxer = Muxer::new(&path)?;
        let index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
        for frame_index in 0..5i64 {
            let mut frame = generate_video_frame(64, 64, frame_index);
            frame.set_pts(frame_index);
            muxer.mux(frame, index)?;
        }

        // 第一次：写 trailer
        muxer.finish()?;
        assert!(
            muxer.have_written_trailer,
            "the first finish() must write the trailer"
        );
        // 之后每次：幂等 no-op，绝不报错
        for round in 2..=3 {
            muxer.finish()?;
            assert!(
                muxer.have_written_trailer,
                "finish() #{round} must stay idempotent"
            );
        }

        // 容器本身仍然完好：能打开、能解出帧
        let demuxer = Demuxer::new(&path)?;
        let mut frames = 0usize;
        for item in demuxer {
            let (_index, _frame) = item?;
            frames += 1;
        }
        assert!(frames > 0, "the finished container decoded no frames");

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// trailer 写出后透传路径同样不能再写包：否则数据会被追加到已封闭的容器
    /// 尾部（`mux_packet` 早先缺这条守卫）。
    #[test]
    fn test_mux_packet_after_finish_is_rejected() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_copy_after_finish.mp4");
        crate::test_support::remove_test_output(&path);

        let mut demuxer = Demuxer::new_passthrough(StreamReader::new("assets/mp4.mp4")?)?;
        let info = demuxer.stream_info(0)?;
        let mut muxer = Muxer::new(&path)?;
        let index = muxer.add_copy_stream(&info)?;

        let (_, mut packet) = demuxer.demux_packet()?.expect("asset has packets");
        muxer.mux_packet(&mut packet, index)?;
        muxer.finish()?;

        let (_, mut packet) = demuxer.demux_packet()?.expect("asset has more packets");
        let err = muxer
            .mux_packet(&mut packet, index)
            .expect_err("mux_packet after finish must fail");
        assert!(
            err.is_invalid_config(),
            "post-trailer write must be an invalid_config error: {err}"
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// flush 之后再编码必须是一个**看得懂**的错误（`is_invalid_config`），而不是
    /// 把 FFmpeg 的 "encoder is already flushed" 原样抛给调用方。
    #[test]
    fn test_encode_after_finish_is_rejected_with_a_clear_error() -> Result<()> {
        let path = crate::test_support::test_output_path("mux", "test_encode_after_finish.mp4");
        let mut muxer = Muxer::new(&path)?;
        let index = muxer.add_encoder(Encoder::new_video(64, 64)?)?;
        for frame_index in 0..5i64 {
            let mut frame = generate_video_frame(64, 64, frame_index);
            frame.set_pts(frame_index);
            muxer.mux(frame, index)?;
        }
        muxer.finish()?;

        let mut frame = generate_video_frame(64, 64, 99);
        frame.set_pts(99);
        let err = match muxer.mux(frame, index) {
            Ok(_) => panic!("encoding into a finished muxer must fail"),
            Err(e) => e,
        };
        assert!(
            err.is_invalid_config(),
            "the phase error must be classified as invalid configuration: {err}"
        );
        assert!(
            err.to_string().contains("cannot encode"),
            "unexpected message: {err}"
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }
}
