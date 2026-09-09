use crate::error::{Context, Result, RsmediaError};
use crate::filter::Filter;
use crate::hwaccel::HWDeviceConfig;
use crate::io::{Reader, Writer};
use crate::stream::MediaType;
use crate::stream::StreamInfo;
use crate::{
    Decoder, DecoderBuilder, Encoder, EncoderBuilder, Location, StreamReader, StreamWriter,
};

use dashmap::DashMap;
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;

/// A container chapter mark (MP4/MKV chapters), with times in seconds.
///
/// Chapter times are stored internally on a millisecond time base (1/1000),
/// matching what `ffmetadata` and most container tooling use.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Unique chapter id; `0` means auto-assign sequential ids (0, 1, 2, ...)
    /// when the header is written.
    pub id: i64,
    /// Human-readable chapter title, stored as the chapter's `title` metadata.
    pub title: String,
    /// Chapter start time in seconds.
    pub start: f64,
    /// Chapter end time in seconds (exclusive).
    pub end: f64,
}

impl Chapter {
    /// Creates a chapter with an auto-assigned id.
    pub fn new(title: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            id: 0,
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
/// ```rust,ignore
/// let reader = Reader::new(Path::new("from_file.mp4")).unwrap();
/// let writer = Writer::new(Path::new("to_file.mkv")).unwrap();
/// let muxer = MuxerBuilder::new(writer)
///     .with_streams(&reader)
///     .unwrap()
///     .build();
/// while let Ok(packet) = reader.read() {
///     muxer.mux(packet).unwrap();
/// }
/// muxer.finish().unwrap();
/// ```
///
/// Mux from file to MP4 and print length of first 100 buffer segments:
///
/// ```rust,ignore
/// let reader = Reader::new(Path::new("my_file.mp4")).unwrap();
/// let writer = BufferWriter::new("mp4").unwrap();
/// let mut muxer = MuxerBuilder::new(writer)
///     .with_streams(&reader)
///     .build()
///     .unwrap();
/// for _ in 0..100 {
///     println!("len: {}", muxer.mux().unwrap().len());
/// }
/// muxer.finish()?;
/// ```
pub struct Muxer<W: Writer> {
    pub writer: W,
    streams: Vec<MuxerStream>,
    interleaved: bool,
    have_written_header: bool,
    have_written_trailer: bool,
    /// Container-level metadata (e.g. "title", "artist"), applied to the
    /// format context right before the header is written.
    metadata: HashMap<String, String>,
    /// Per-stream metadata (e.g. "language"), keyed by the output stream
    /// index returned from [`Muxer::add_stream`], applied right before the
    /// header is written.
    stream_metadata: HashMap<usize, HashMap<String, String>>,
    /// Container chapters, applied right before the header is written.
    chapters: Vec<Chapter>,
}

pub struct MuxerStream {
    pub encoder: Encoder,
    pub stream_info: StreamInfo,
    pub media_type: MediaType,
    pub stream_index: usize,
}

impl MuxerStream {
    pub fn new(encoder: Encoder, stream_info: StreamInfo) -> Self {
        let media_type = encoder.media_type();
        let stream_index = stream_info.index;
        Self {
            encoder,
            media_type,
            stream_info,
            stream_index,
        }
    }
}

impl Muxer<StreamWriter> {
    pub fn new(destination: impl Into<Location>) -> Result<Self> {
        let writer = StreamWriter::new(destination)?;
        Ok(Self::new_from_writer(writer))
    }
}

impl<W: Writer> Muxer<W> {
    pub fn new_from_writer(writer: W) -> Self {
        Self {
            writer,
            streams: Vec::new(),
            interleaved: false,
            have_written_header: false,
            have_written_trailer: false,
            metadata: HashMap::new(),
            stream_metadata: HashMap::new(),
            chapters: Vec::new(),
        }
    }

    pub fn dump(&self, index: usize) -> Result<()> {
        let mux_stream = self.get_stream(index)?;
        println!("{:?}", mux_stream.stream_info);
        Ok(())
    }

    pub fn add_stream(&mut self, encoder: Encoder) -> Result<usize> {
        let stream_idx = self
            .writer
            .add_stream(encoder.codecpar(), encoder.time_base());
        let stream_info = StreamInfo::from_writer(&self.writer, stream_idx)?;
        self.streams.push(MuxerStream::new(encoder, stream_info));
        Ok(stream_idx)
    }

    pub fn get_stream(&self, index: usize) -> Result<&MuxerStream> {
        self.streams
            .iter()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::custom(format!("Stream index: {index} not found")))
    }

    pub fn get_stream_mut(&mut self, index: usize) -> Result<&mut MuxerStream> {
        self.streams
            .iter_mut()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::custom(format!("Stream index: {index} not found")))
    }

    /// Sets a container-level metadata entry, e.g. `title`, `artist`,
    /// `comment`. Applied when the container header is written, i.e. before
    /// the first [`Self::mux`] call; entries set after the header is written
    /// are ignored (with a warning).
    ///
    /// Keys and values must not contain interior NUL bytes.
    pub fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<&mut Self> {
        let (key, value) = validate_metadata_pair(key, value)?;
        if self.have_written_header {
            log::warn!("set_metadata({key:?}) after header write has no effect");
        }
        self.metadata.insert(key, value);
        Ok(self)
    }

    /// Sets a per-stream metadata entry for the stream with index returned
    /// from [`Self::add_stream`]. The common case is `language` with an
    /// ISO 639-2 code ("chi", "eng", "und", ...), which players use to pick
    /// audio/subtitle tracks.
    ///
    /// Applied when the container header is written; keys and values must not
    /// contain interior NUL bytes.
    pub fn set_stream_metadata(
        &mut self,
        stream_index: usize,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<&mut Self> {
        let (key, value) = validate_metadata_pair(key, value)?;
        let nb_streams = self.writer.output().nb_streams as usize;
        if stream_index >= nb_streams {
            return Err(RsmediaError::invalid_config(format!(
                "stream index {stream_index} out of range (nb_streams={nb_streams})"
            )));
        }
        if self.have_written_header {
            log::warn!(
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

        for (key, value) in &self.metadata {
            let (k, v) = (
                crate::strutils::str_to_cstring(key),
                crate::strutils::str_to_cstring(value),
            );
            let ret = unsafe { ffi::av_dict_set(&mut ctx.metadata, k.as_ptr(), v.as_ptr(), 0) };
            if ret < 0 {
                log::warn!("av_dict_set({key:?}) failed: {ret}");
            }
        }

        let streams =
            unsafe { std::slice::from_raw_parts_mut(ctx.streams, ctx.nb_streams as usize) };
        for (idx, entries) in &self.stream_metadata {
            let Some(stream) = streams.get_mut(*idx) else {
                log::warn!(
                    "stream metadata: index {idx} out of range (nb_streams={})",
                    ctx.nb_streams
                );
                continue;
            };
            for (key, value) in entries {
                let (k, v) = (
                    crate::strutils::str_to_cstring(key),
                    crate::strutils::str_to_cstring(value),
                );
                let ret = unsafe {
                    ffi::av_dict_set(&mut (**stream).metadata, k.as_ptr(), v.as_ptr(), 0)
                };
                if ret < 0 {
                    log::warn!("av_dict_set(stream {idx}, {key:?}) failed: {ret}");
                }
            }
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
            log::warn!(
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
        let (_, title) = validate_metadata_pair("title", chapter.title.clone())?;
        self.chapters.push(Chapter {
            id: chapter.id,
            title,
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
    /// stream marked with `AV_DISPOSITION_ATTACHED_PIC`, which MP4 writes as
    /// the `covr` atom and MKV as an attachment.
    ///
    /// The stream is marked and the frame muxed immediately, so this must be
    /// called after all primary streams are added (the header is written on
    /// the first mux) and before any regular [`Self::mux`] call. Returns the
    /// output stream index of the cover stream.
    pub fn add_cover_art_with(&mut self, encoder: Encoder, cover_frame: AVFrame) -> Result<usize> {
        if self.have_written_header {
            return Err(RsmediaError::invalid_config(
                "add_cover_art after header write is not supported",
            ));
        }
        let stream_idx = self.add_stream(encoder)?;

        // Mark the stream as an attached picture. rsmpeg only exposes the
        // output stream array immutably, so the raw stream array is accessed
        // here instead; sound because the writer exclusively owns the context
        // and the disposition must be set before `write_header`.
        let ctx = unsafe { &mut *self.writer.output_mut().as_mut_ptr() };
        let streams =
            unsafe { std::slice::from_raw_parts_mut(ctx.streams, ctx.nb_streams as usize) };
        let Some(stream) = streams.get_mut(stream_idx) else {
            return Err(RsmediaError::custom(format!(
                "cover art stream index {stream_idx} out of range (nb_streams={})",
                ctx.nb_streams
            )));
        };
        unsafe {
            (**stream).disposition |= ffi::AV_DISPOSITION_ATTACHED_PIC as i32;
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
    fn apply_chapters(&mut self) {
        if self.chapters.is_empty() {
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
                log::error!("av_mallocz for chapter {i} failed; chapters are dropped");
                Self::free_chapter_nodes(&mut chapter_nodes);
                return;
            }
            unsafe {
                (*chapter_ptr).id = if chapter.id != 0 {
                    chapter.id
                } else {
                    i as i64
                };
                (*chapter_ptr).time_base = ffi::AVRational { num: 1, den: 1000 };
                (*chapter_ptr).start = start_ms;
                (*chapter_ptr).end = end_ms;
                let title = crate::strutils::str_to_cstring(&chapter.title);
                ffi::av_dict_set(
                    &mut (*chapter_ptr).metadata,
                    c"title".as_ptr(),
                    title.as_ptr(),
                    0,
                );
            }
            chapter_nodes.push(chapter_ptr);
        }

        // 2) 分配连续指针数组。
        let chapters_ptr = unsafe {
            ffi::av_calloc(count, std::mem::size_of::<*mut ffi::AVChapter>())
                as *mut *mut ffi::AVChapter
        };
        if chapters_ptr.is_null() {
            log::error!("av_calloc for {count} chapters failed; chapters are dropped");
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
    }

    /// 释放一组尚未转移所有权的 chapter 节点（`av_free` 对空指针安全）。
    fn free_chapter_nodes(nodes: &mut Vec<*mut ffi::AVChapter>) {
        for node in nodes.drain(..) {
            unsafe {
                ffi::av_free(node as *mut std::os::raw::c_void);
            }
        }
    }

    /// Refreshes cached [`StreamInfo`] for every stream after the header is
    /// written.
    ///
    /// Muxers may adjust stream parameters inside `avformat_write_header` —
    /// most notably the time base (e.g. the GIF muxer forces `1/100`,
    /// MP4 applies its movie timescale). Packet timestamps must be rescaled to
    /// the *post-header* stream time base, so the cached pre-header value
    /// cannot be used for `rescale_ts`.
    fn refresh_stream_info(&mut self) -> Result<()> {
        for mux_stream in self.streams.iter_mut() {
            let stream_info = StreamInfo::from_writer(&self.writer, mux_stream.stream_index)?;
            let tb_changed = stream_info.time_base.num != mux_stream.stream_info.time_base.num
                || stream_info.time_base.den != mux_stream.stream_info.time_base.den;
            if tb_changed {
                log::debug!(
                    "Muxer changed stream {} time_base: {:?} -> {:?}",
                    mux_stream.stream_index,
                    mux_stream.stream_info.time_base,
                    stream_info.time_base
                );
                mux_stream.stream_info = stream_info;
            }
        }
        Ok(())
    }

    /// Mux a single packet. This will mux a single packet.
    ///
    /// # Arguments
    ///
    /// * `packet` - [`Packet`] to mux.
    pub fn mux(&mut self, frame: AVFrame, stream_idx: usize) -> Result<Option<W::Out>> {
        if self.have_written_header {
            let interleaved = self.interleaved;
            let mux_stream = self.get_stream_mut(stream_idx)?;
            let enc_time_base = mux_stream.encoder.time_base();
            let out_time_base = mux_stream.stream_info.time_base;
            let packets = mux_stream.encoder.encode_raw(frame)?;
            // mux_stream 对 self.streams 的借用至此结束，之后可独占使用 self.writer

            let mut last_out = None;
            for mut packet in packets {
                packet.set_pos(-1);
                packet.set_stream_index(stream_idx as i32);
                // 将编码器输出的数据包时间戳，从编码器时间基转换到输出流时间基
                // encode_ctx_timebase => out_stream_time_base
                packet.rescale_ts(enc_time_base, out_time_base);

                last_out = if interleaved {
                    Some(self.writer.write_interleaved(&mut packet)?)
                } else {
                    Some(self.writer.write_frame(&mut packet)?)
                };
            }
            Ok(last_out)
        } else {
            self.have_written_header = true;
            self.apply_metadata();
            self.apply_chapters();
            self.writer.write_header()?;
            self.refresh_stream_info()?;
            self.mux(frame, stream_idx)
        }
    }

    /// Signal to the muxer that writing has finished. This will cause a trailer to be written if
    /// the container format has one.
    pub fn finish(&mut self) -> Result<Option<W::Out>> {
        // 从未 mux 过任何数据（header 也尚未写入）：没有实际内容需要 flush，
        // 直接空操作返回 `None`，不产生“无头”的残缺输出。
        if !self.have_written_header {
            return Ok(None);
        }

        for mux_stream in self.streams.iter_mut() {
            // flush the encoder to ensure all packets are sent to the muxer.
            let out_stream_index = mux_stream.stream_index;
            let out_stream_time_base = mux_stream.stream_info.time_base;
            mux_stream.encoder.flush(
                &mut self.writer,
                self.interleaved,
                out_stream_index,
                out_stream_time_base,
            )?;
        }

        // 已写 header 且未写 trailer 时才写 trailer；header + trailer 均已写说明
        // 是重复调用 finish()，此时幂等返回 None，避免重复写 trailer。
        if !self.have_written_trailer {
            self.have_written_trailer = true;
            self.writer.write_trailer().map(Some)
        } else {
            Ok(None)
        }
    }
}

unsafe impl<W: Writer> Send for Muxer<W> {}
unsafe impl<W: Writer> Sync for Muxer<W> {}

/// Validates a metadata key/value pair: rejects interior NUL bytes, which
/// cannot be represented in the C strings handed to `av_dict_set`.
fn validate_metadata_pair(
    key: impl Into<String>,
    value: impl Into<String>,
) -> Result<(String, String)> {
    let (key, value) = (key.into(), value.into());
    for (what, s) in [("key", &key), ("value", &value)] {
        if s.contains('\0') {
            return Err(RsmediaError::invalid_config(format!(
                "metadata {what} contains interior NUL byte: {s:?}"
            )));
        }
    }
    Ok((key, value))
}

impl<W: Writer> Drop for Muxer<W> {
    fn drop(&mut self) {
        // 用户忘记调用 finish() 时（尤其是错误提前返回/panic），
        // 自动 flush 编码器延迟缓冲并写 trailer，避免生成损坏的容器文件。
        // 仅当已写过 header 时才处理，未 mux 过的空文件不做无意义写入。
        if self.have_written_header
            && !self.have_written_trailer
            && let Err(err) = self.finish()
        {
            log::error!("Failed to auto-flush muxer on drop: {err:#}");
        }
    }
}

/// Demuxer
pub struct Demuxer<R: Reader> {
    pub reader: R,
    streams: Vec<DemuxerStream>,
    states: Arc<DashMap<usize, i32>>,
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

impl Demuxer<StreamReader> {
    pub fn new(source: impl Into<Location>) -> Result<Self> {
        let reader = StreamReader::new(source)?;
        Self::new_from_reader(reader, None, None)
    }
}

impl<R: Reader> Demuxer<R> {
    pub fn new_from_reader(
        reader: R,
        filters: Option<Vec<Filter>>,
        device_config: Option<HWDeviceConfig>,
    ) -> Result<Demuxer<R>> {
        let nb_streams = reader.input().nb_streams as usize;
        let device_type = device_config.as_ref().map(|c| c.device_type);
        let filter_map = filters.unwrap_or_default().into_iter().fold(
            HashMap::<MediaType, Vec<Filter>>::new(),
            |mut map, f| {
                map.entry(f.media_type()).or_default().push(f);
                map
            },
        );

        let mut streams = Vec::new();
        for stream_idx in 0..nb_streams {
            let stream_info = StreamInfo::from_reader(&reader, stream_idx)?;
            let media_type = stream_info.media_type;
            // auto detect hardware acceleration decoder codec
            let Some(codec_name) = stream_info.find_decoder_name(device_type) else {
                // Streams without a registered decoder (chapter tracks,
                // attached pictures, binary data, ...) are skipped instead of
                // failing the whole demuxer.
                log::debug!(
                    "Skipping stream {stream_idx}: no decoder for codec_id {:#x}",
                    stream_info.codec_id
                );
                continue;
            };
            let decoder = match DecoderBuilder::new(media_type)
                .with_codec_name(codec_name.clone())
                .with_hardware_device(device_config.clone())
                .with_filters(filter_map.get(&media_type).cloned())
                .build_from_reader(&reader)
            {
                Ok(decoder) => decoder,
                Err(e) if device_type.is_some() => {
                    // 硬件解码器构建失败（如 hw 初始化失败）：回退软件解码器重试，
                    // 与 find_decoder_name 的回退语义对齐；再失败才让错误上抛。
                    log::warn!(
                        "HW decoder '{codec_name}' failed to build: {e:#}; \
                         falling back to software decoder"
                    );
                    let software_name = stream_info
                        .find_decoder_name(None)
                        .unwrap_or_else(|| codec_name.clone());
                    DecoderBuilder::new(media_type)
                        .with_codec_name(software_name)
                        .with_filters(filter_map.get(&media_type).cloned())
                        .build_from_reader(&reader)
                        .context("Failed to build decoder (hw and software both failed)")?
                }
                Err(e) => {
                    return Err(RsmediaError::custom(format!(
                        "Failed to build decoder: {e:#}"
                    )));
                }
            };

            streams.push(DemuxerStream::new(decoder, stream_info));
        }

        Ok(Self {
            reader,
            streams,
            states: Arc::new(DashMap::new()),
        })
    }

    pub fn streams(&self) -> &[DemuxerStream] {
        &self.streams
    }

    /// Reads back the container chapters (title/start/end in seconds).
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
            unsafe {
                let title_entry =
                    ffi::av_dict_get((*c).metadata, c"title".as_ptr(), ptr::null(), 0);
                let title = if title_entry.is_null() {
                    String::new()
                } else {
                    crate::strutils::cstr_to_string(CStr::from_ptr((*title_entry).value))
                        .unwrap_or_default()
                };
                let tb = (*c).time_base;
                let tb_secs = tb.num as f64 / tb.den as f64;
                chapters.push(Chapter {
                    id: (*c).id,
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
            .ok_or_else(|| RsmediaError::custom(format!("Stream index: {index} not found")))
    }

    pub fn get_stream_mut(&mut self, index: usize) -> Result<&mut DemuxerStream> {
        self.streams
            .iter_mut()
            .find(|s| s.stream_index == index)
            .ok_or_else(|| RsmediaError::custom(format!("Stream index: {index} not found")))
    }

    fn set_flushed(&self, stream_index: usize) {
        self.states.insert(stream_index, 1);
    }

    fn is_flushed(&self, stream_index: usize) -> bool {
        self.states
            .get(&stream_index)
            .map(|v| *v.value() == 1)
            .unwrap_or(false)
    }

    pub fn demux(&mut self) -> Result<Option<(usize, AVFrame)>> {
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
                            // Packets of skipped streams (chapter tracks, ...)
                            // are dropped.
                            log::debug!("Dropping packet of undecodable stream {stream_idx}");
                            continue;
                        };
                        if let Some(frame) = demux_stream.decoder.decode_raw_packet(&packet)? {
                            return Ok(Some((stream_idx, frame)));
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
                for i in 0..self.streams.len() {
                    // 先获取 stream_idx，避免后面重复借用
                    let stream_idx = self.streams[i].stream_index;

                    // 使用实际的 stream_idx 检查状态
                    if self.is_flushed(stream_idx) {
                        continue;
                    }

                    // 然后获取stream的可变引用
                    let demuxer_stream = &mut self.streams[i];
                    match demuxer_stream.decoder.drain_raw() {
                        Ok(Some(frame)) => {
                            return Ok(Some((demuxer_stream.stream_index, frame)));
                        }
                        Ok(None) => {
                            log::debug!("Stream: [{stream_idx}] Decoder flushed. EOF reached.");
                            self.set_flushed(stream_idx);
                            continue;
                        }
                        Err(e) => {
                            log::error!("Stream: [{stream_idx}] Decoder Drain Error: {e}");
                            return Err(e);
                        }
                    }
                }
                return Ok(None);
            }
        }
    }
}

/// Demuxer iterator
///
/// # Examples
///
/// ```rust,ignore
/// let mut demuxer = Demuxer::from_reader(StreamReader::new(Path::new("my_file.mp4"))?)?;
/// for (stream_index, frame) in demuxer {
///     println!("stream_index: {}, frame: {}", stream_index, frame.width());
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

unsafe impl<R: Reader> Send for Demuxer<R> {}
unsafe impl<R: Reader> Sync for Demuxer<R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncoderBuilder, PixelFormat, SampleFormat, StreamReader, StreamWriter, strutils};

    use crate::error::{Context, Result};
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
                    return Err(RsmediaError::custom("Audio data pointer is null"));
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

        let mut muxer = Muxer::new(output_path.as_path())?;

        let encoder_frame_rate = video_encoder.frame_rate();
        let encoder_time_base = video_encoder.time_base();
        let video_index = muxer.add_stream(video_encoder)?;

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
        for des in &demuxer.streams {
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
        let mut muxer = Muxer::new(output_path.as_path())?;

        let encoder_time_base = audio_encoder.time_base();
        let audio_index = muxer.add_stream(audio_encoder)?;

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
        for des in &demuxer.streams {
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
        use std::ffi::CStr;
        use std::ptr;

        let output_path = crate::test_support::test_output_path("mux", "test_mux_metadata.mp4");

        let (width, height) = (320, 240);
        let sample_rate = 44_100;
        let nb_samples = 1024;

        let video_encoder = Encoder::new_video(width, height)?;
        let video_time_base = video_encoder.time_base();
        let video_frame_rate = video_encoder.frame_rate();
        let audio_encoder = Encoder::new_audio(2, sample_rate, SampleFormat::FLTP).unwrap();
        let audio_time_base = audio_encoder.time_base();

        let mut muxer = Muxer::new(output_path.as_path())?;
        let video_index = muxer.add_stream(video_encoder)?;
        let audio_index = muxer.add_stream(audio_encoder)?;

        // 参数校验：内嵌 NUL 与越界索引必须 fail fast
        assert!(muxer.set_metadata("bad\0key", "v").is_err());
        assert!(muxer.set_metadata("title", "bad\0value").is_err());
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
        let reader = StreamReader::new(output_path.as_path())?;
        let input = reader.input();

        let get_str = |dict: *mut ffi::AVDictionary, key: &CStr| -> Option<String> {
            unsafe {
                let entry = ffi::av_dict_get(dict, key.as_ptr(), ptr::null(), 0);
                if entry.is_null() {
                    None
                } else {
                    Some(
                        crate::strutils::cstr_to_string(CStr::from_ptr((*entry).value))
                            .expect("metadata value is UTF8"),
                    )
                }
            }
        };

        let title = get_str(input.metadata, c"title");
        assert_eq!(
            title.as_deref(),
            Some("rsmedia metadata test"),
            "container title metadata mismatch"
        );

        let artist = get_str(input.metadata, c"artist");
        assert_eq!(artist.as_deref(), Some("rsmedia"));

        let streams = input.streams();
        let video_lang = get_str(
            streams[video_index]
                .metadata()
                .map(|d| d.as_ptr() as *mut _)
                .unwrap_or(ptr::null_mut()),
            c"language",
        );
        assert_eq!(video_lang.as_deref(), Some("und"));

        let audio_lang = get_str(
            streams[audio_index]
                .metadata()
                .map(|d| d.as_ptr() as *mut _)
                .unwrap_or(ptr::null_mut()),
            c"language",
        );
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
        let audio_index = muxer.add_stream(audio_encoder)?;

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

        let mut muxer = Muxer::new(output_path.as_path())?;

        let video_time_base = video_encoder.time_base();
        let audio_time_base = audio_encoder.time_base();

        // 添加视频流 和 音频流
        let video_idx = muxer.add_stream(video_encoder)?;
        let audio_idx = muxer.add_stream(audio_encoder)?;

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
        for stream in &demuxer.streams {
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
    /// ```rust,ignore
    /// transcode("input.mp4", "output.mov").unwrap();
    /// ```
    fn transcode(input_path: &str, output_path: &str) -> Result<()> {
        let mut input_reader = StreamReader::new(Path::new(input_path))?;
        let input = input_reader.input();

        // inner output
        use crate::io::private::Output;
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
            .dump(0, strutils::str_to_cstring(output_path).as_c_str())
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
    /// 1. 从未 mux 任何数据（header 未写）时，`finish()` 应为空操作返回 `Ok(None)`，
    ///    不会产生仅含 encode-EOS 包但无 header/trailer 的残缺输出。
    /// 2. 重复调用 `finish()` 是幂等的：第二次返回 `Ok(None)`，不会重复写 trailer。
    #[test]
    fn test_finish_without_mux_and_idempotent() -> Result<()> {
        let output_path =
            crate::test_support::test_output_path("mux", "test_finish_without_mux.mp4");

        let encoder = Encoder::new_video(320, 240)?;
        let mut muxer = Muxer::new(output_path)?;
        muxer.add_stream(encoder)?;

        // 未 mux 任何帧，直接 finish：应为空操作，返回 None
        let first = muxer.finish()?;
        assert!(
            first.is_none(),
            "finish() on an empty muxer should be a no-op"
        );
        assert!(!muxer.have_written_header, "header should not be written");

        // 第二次 finish 应幂等，返回 None，不重复写 trailer
        let second = muxer.finish()?;
        assert!(second.is_none(), "idempotent finish() should return None");

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
            let mut muxer = Muxer::new(output_path.as_path())?;
            let video_index = muxer.add_stream(video_encoder)?;
            for index in 0..12 {
                let mut frame = generate_video_frame(width, height, index);
                frame.set_pts(index * encoder_time_base.den as i64);
                frame.set_time_base(encoder_time_base);
                muxer.mux(frame, video_index)?;
            }
            // 故意不调用 finish()，直接离开作用域触发 Drop 自动 flush
        }

        // Drop 自动完成 flush + trailer 后，文件应可被读取并解码
        let demuxer = Demuxer::new(output_path.as_path())?;
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
    #[test]
    fn test_mux_chapters() -> Result<()> {
        let output_path = crate::test_support::test_output_path("mux", "test_mux_chapters.mp4");

        let (width, height) = (320, 240);
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();

        let mut muxer = Muxer::new(output_path.as_path())?;
        let video_index = muxer.add_stream(video_encoder)?;

        // 参数校验 fail fast
        assert!(muxer.add_chapter(Chapter::new("bad", -1.0, 1.0)).is_err());
        assert!(muxer.add_chapter(Chapter::new("bad", 2.0, 1.0)).is_err());
        assert!(
            muxer
                .add_chapter(Chapter::new("bad\0title", 0.0, 1.0))
                .is_err()
        );

        muxer.add_chapter(Chapter::new("Intro", 0.0, 1.0))?;
        muxer.add_chapter(Chapter::new("Part Two", 1.0, 2.0))?;

        // 1 秒视频（30fps）
        for index in 0..encoder_time_base.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        // 回读：标题 + 秒级起止
        let demuxer = Demuxer::new(output_path.as_path())?;
        let chapters = demuxer.chapters();
        assert_eq!(chapters.len(), 2, "expected 2 chapters, got {chapters:?}");
        assert_eq!(chapters[0].title, "Intro");
        assert!((chapters[0].start - 0.0).abs() < 1e-6);
        assert!((chapters[0].end - 1.0).abs() < 1e-6);
        assert_eq!(chapters[1].title, "Part Two");
        assert!((chapters[1].start - 1.0).abs() < 1e-6);
        assert!((chapters[1].end - 2.0).abs() < 1e-6);

        // MKV 使用原生 chapter atom（非 MP4 文本轨），同样必须完整回传
        let mkv_path = crate::test_support::test_output_path("mux", "test_mux_chapters.mkv");
        let video_encoder = Encoder::new_video(width, height)?;
        let encoder_time_base = video_encoder.time_base();
        let mut muxer = Muxer::new(mkv_path.as_path())?;
        let video_index = muxer.add_stream(video_encoder)?;
        muxer.add_chapter(Chapter::new("MKV Intro", 0.0, 1.0))?;
        for index in 0..encoder_time_base.den as i64 {
            let mut frame = generate_video_frame(width, height, index);
            frame.set_pts(index * encoder_time_base.den as i64);
            frame.set_time_base(encoder_time_base);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        let demuxer = Demuxer::new(mkv_path.as_path())?;
        let chapters = demuxer.chapters();
        assert_eq!(
            chapters.len(),
            1,
            "expected 1 mkv chapter, got {chapters:?}"
        );
        assert_eq!(chapters[0].title, "MKV Intro");
        assert!((chapters[0].end - 1.0).abs() < 1e-6);

        Ok(())
    }

    /// GIF 调色板管线（palettegen/paletteuse 单遍滤镜图）编码：
    /// 60 帧 @30fps 输入 → fps=10 抽帧 + 调色板量化 → gif 编码器（pal8），
    /// 回读应得到 ~20 帧 GIF（fps 滤镜丢帧），解码器为 gif。
    #[test]
    fn test_encode_gif_palette_pipeline() -> Result<()> {
        use crate::filter::video;

        let output_path = crate::test_support::test_output_path("mux", "test_gif_palette.gif");
        crate::test_support::remove_test_output(&output_path);

        let (width, height) = (96usize, 64usize);
        let in_fps = 30.0f32;
        let out_fps = 10.0f32;

        // 编码器按**输入**帧率（30fps）构建，滤镜图内 fps=10 完成抽帧
        let encoder = crate::EncoderBuilder::new_video(width, height)
            .with_codec_name("gif".to_string())
            .with_fps(in_fps)
            .with_filters(vec![video::gif_palette(out_fps, None)])
            .build()?;

        let mut muxer = Muxer::new(output_path.as_path())?;
        let video_index = muxer.add_stream(encoder)?;

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
        let demuxer = Demuxer::new(output_path.as_path())?;
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
        let reader = StreamReader::new(output_path.as_path())?;
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

        let mut muxer = Muxer::new(output_path.as_path())?;
        let video_index = muxer.add_stream(video_encoder)?;

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
        let reader = StreamReader::new(output_path.as_path())?;
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
}
