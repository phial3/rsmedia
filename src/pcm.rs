//! Streaming PCM encode bridge.
//!
//! [`PcmSink`] bridges realtime interleaved PCM sources (e.g. `cpal`/`rodio`
//! recording callbacks, WAV readers) into rsmedia's audio [`Encoder`](crate::encode::Encoder) +
//! [`Muxer`] pipeline:
//!
//! - 任意大小的交错 PCM 块被封装为 packed [`AVFrame`]-（`f32`/`i16`/`u8`），
//!   按块编码，无需调用方对齐编码器 frame_size；
//! - 采样格式/采样率/声道数与编码器不一致时，由编码路径自动重采样转换
//!   （见 [`Encoder`](crate::encode::Encoder) 的 audio `rescale`）；
//! - pts 在编码器时间基（1/编码器采样率）下按样本位置累计，跨块连续；
//! - 编码器内部 FIFO 负责凑满固定帧长（如 AAC 的 1024 样本）；
//! - [`PcmSink::finish`]（或 `Drop`，[`Muxer`] 会自动收尾）冲刷剩余样本并写 trailer。
//!
//! # Example
//!
//! ```no_run
//! use rsmedia::error::Result;
//! fn main() -> Result<()> {
//!     use rsmedia::mux::Muxer;
//!     use rsmedia::pcm::{PcmSink, PcmSpec};
//!     use rsmedia::{Encoder, SampleFormat};
//!     use std::path::Path;
//!
//!     // 1. 按目标规格建编码器（默认 AAC）并加入 Muxer
//!     let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
//!     let mut muxer = Muxer::new("out.m4a")?;
//!     let audio_index = muxer.add_encoder(encoder)?;
//!
//!     // 2. 用 PcmSink 绑定音频流；spec 描述实时源（如麦克风）的采样率/声道数，
//!     //    与编码器规格不一致时由内部持久重采样器自动转换
//!     let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(48_000, 2))?;
//!
//!     // 3. 在 cpal 录音回调里直接投交错 f32 块（O(1) 内存，无需对齐 frame_size）
//!     let mic_chunk = vec![0.0f32; 1024];
//!     sink.write_f32(&mic_chunk)?;
//!
//!     // 4. 冲刷重采样尾样/编码器并写 trailer（Drop 可兜底，显式调用可感知错误）
//!     sink.finish()?;
//!     Ok(())
//! }
//! ```
//!
//! 完整可运行的 cpal 录音示例见 `examples/pcm_recorder.rs`。
//!
//! # Recording into memory and reading it back
//!
//! Bind the sink to a [`BufferWriter`](crate::io::BufferWriter);
//! [`finish`](PcmSink::finish) hands the writer back, and
//! [`BufferWriter::into_bytes`](crate::io::BufferWriter::into_bytes) sees the
//! final image. That matters for formats that patch their header while writing
//! the trailer (mp4, mov, wav): they rewrite bytes that were already handed out
//! by [`take_written`](crate::io::BufferWriter::take_written), so a stream of
//! increments is permanently stale. `mpegts` / `matroska` / `adts` never patch,
//! and either entry works.
//!
//! ```no_run
//! # use rsmedia::error::Result;
//! # use rsmedia::io::{BufferReader, BufferWriter};
//! # use rsmedia::mux::Muxer;
//! # use rsmedia::pcm::{PcmSink, PcmSpec};
//! # use rsmedia::{Encoder, MediaType, SampleFormat, DecoderBuilder};
//! # fn main() -> Result<()> {
//! let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
//! let mut muxer = Muxer::new_from_writer(BufferWriter::new("mp4")?);
//! let idx = muxer.add_encoder(encoder)?;
//! let mut sink = PcmSink::new(muxer, idx, PcmSpec::new(48_000, 2))?;
//! sink.write_f32(&[0.0f32; 4096])?;
//! let bytes: Vec<u8> = sink.finish()?.into_bytes();
//!
//! let reader = BufferReader::new(bytes)?;          // 回灌：解码后再处理
//! # Ok(())
//! # }
//! ```
//!
//! # Pauses, dropouts and the timeline
//!
//! There is no external clock: `pts` is a running sample count, so a gap in the
//! input is *closed up* rather than left as silence. To keep the recording
//! aligned with wall-clock time, **write the silence explicitly** — feed as many
//! zero samples as the pause lasted:
//!
//! ```no_run
//! # use rsmedia::error::Result;
//! # use rsmedia::pcm::{PcmSink, PcmSpec};
//! # fn f(sink: &mut PcmSink<rsmedia::io::StreamWriter>) -> Result<()> {
//! let paused_samples = 3 * 48_000;                 // 3 s at 48 kHz, 1 channel
//! sink.write_f32(&vec![0.0f32; paused_samples])?;  // 2 channels -> `* 2`
//! # Ok(())
//! # }
//! ```
//!
//! # Splitting a long recording into several files
//!
//! [`PcmSink::finish`] consumes the sink, so each segment needs a
//! fresh `Encoder` + `Muxer` + `PcmSink`. That costs nothing but setup: the tail
//! of every segment *is* drained, so no samples are lost at the cut. What is not
//! preserved is the resampler's filter state across the boundary — the first
//! milliseconds of a new segment start from a cold delay line. For a continuous
//! recording that must stay phase-exact, record one file and cut it afterwards,
//! or use [`Muxer::new_segmented`](crate::mux::Muxer::new_segmented).

use crate::error::{Context, Result, RsmediaError};
use crate::io::Writer;
use crate::mux::Muxer;
use crate::resample::Resampler;
use crate::stream::MediaType;
use crate::time::Rational;

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::{AVChannelLayout, AVFrame};
use rsmpeg::ffi;

/// 输入 PCM 规格（来自麦克风/文件等实时源的交错 PCM）。
///
/// 采样格式由写入方法决定（[`PcmSink::write_f32`]/[`PcmSink::write_i16`]/
/// [`PcmSink::write_u8`]/[`PcmSink::write_planar_f32`]），此处只描述采样率、
/// 声道数与（可选的）声道布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmSpec {
    /// 输入采样率（Hz），如 cpal 的 `SampleRate(48_000)`。`i32`，与
    /// [`EncoderBuilder::with_sample_rate`](crate::EncoderBuilder::with_sample_rate)
    /// 和 FFmpeg 的 `int` 字段同宽。
    pub sample_rate: u32,
    /// 声道数（交错布局），如立体声为 2。
    ///
    /// 与 crate 内其余声道数一致用 `i32`（FFmpeg 的 `AVChannelLayout.nb_channels`
    /// 就是 `c_int`）；cpal 的 `SupportedStreamConfig::channels()` 返回 `u16`，
    /// 调用方需 `as i32`。
    pub channels: u32,
    /// 声道布局掩码（`AV_CH_*` 的按位或，如 `AV_CH_LAYOUT_STEREO`），`0` 表示
    /// **不指定** —— 此时按 [`Self::channels`] 取 FFmpeg 的默认布局。
    ///
    /// 用 `u64` 而不是 `ffi::AVChannelLayout`：掩码是个纯值、可 `Copy`，而
    /// `AVChannelLayout` 含裸指针（`opaque`），放进这个 `Copy` 规格里会把它变成
    /// 一个需要生命周期关注的对象。
    ///
    /// 只在声道数 ≥ 3 且**不是**默认布局时才需要设（2.1 的 `FL+FR+LFE` 会被
    /// "按声道数推导"规范化成 `FL+FR+FC`；见 `crate::frame::MediaFrame` 的
    /// 已知限制）。设了但与 [`Self::channels`] 不符会在首次写入时报错。
    pub channel_mask: u64,
}

impl PcmSpec {
    /// 按采样率与声道数建规格（采样格式由写入方法决定，布局取 FFmpeg 默认）。
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            channel_mask: 0,
        }
    }

    /// 指定声道布局掩码（`AV_CH_*` 的按位或）。
    ///
    /// 传 `0` 等价于不指定（回到 [`Self::new`] 的行为）。
    pub const fn with_channel_mask(mut self, channel_mask: u64) -> Self {
        self.channel_mask = channel_mask;
        self
    }
}

/// 单次封装进一个 AVFrame 的最大输入样本数（每声道），限制单帧内存占用；
/// 超长输入会被自动切分为多帧。
const MAX_CHUNK_SAMPLES: usize = 4096;

/// Streaming PCM → [`Encoder`](crate::encode::Encoder) → [`Muxer`] 桥接器。
///
/// 持有 [`Muxer`] 的所有权；`write_*` 按 cpal 回调粒度投放 PCM，`finish`
/// 冲刷编码器并写 trailer。忘记 `finish` 时 `Drop` 仍会经 `Muxer` 自己的
/// `Drop` 收尾（flush 编码器 + 写 trailer），输出容器不会损坏；但重采样器
/// 内部的延迟样本不会被排出、错误也感知不到，详见 [`Self::finish`]。
///
/// 本类型**故意不实现 `Drop`**：它必须能把内部的 `Muxer` 整体 move 出去
/// （[`Self::finish`]），而带 `Drop` 的类型不允许部分移动。收尾
/// 由 `Muxer` 自己的 `Drop` 负责，语义不变。
///
/// 内部持有一个**持久** [`Resampler`]（首次写入时按实际输入格式惰性创建）：
/// 输入块先转成编码器的采样格式/采样率/声道布局，再按实际输出样本数累计
/// pts 送入编码器。持久化上下文是关键 —— 重采样时 swr 内部滤波延迟会把
/// 样本缓存在调用之间，若每次调用重建上下文（如逐帧临时转换），尾部样本
/// 会随各块延迟丢失。
///
/// # Where the bytes go
///
/// `write_*` and [`finish`](Self::finish) return `Result<()>` — the bytes stay
/// inside the muxer's writer and are pulled from it, never handed back by these
/// calls. That means a dropped return value cannot lose data:
///
/// * streaming formats (mpegts, fmp4, …): pull each segment with
///   [`BufferWriter::take_written`](crate::io::BufferWriter::take_written)
///   between writes and send it out;
/// * anything that rewrites its header in the trailer (mp4, mov, wav): take the
///   finished writer with [`finish`](Self::finish) and call
///   [`into_bytes`](crate::io::BufferWriter::into_bytes).
///
/// ```no_run
/// # use rsmedia::error::Result;
/// # use rsmedia::io::BufferWriter;
/// # use rsmedia::mux::Muxer;
/// # use rsmedia::pcm::{PcmSink, PcmSpec};
/// # use rsmedia::{Encoder, SampleFormat};
/// # fn main() -> Result<()> {
/// let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
/// let mut muxer = Muxer::new_from_writer(BufferWriter::new("mpegts")?);
/// let idx = muxer.add_encoder(encoder)?;
/// let mut sink = PcmSink::new(muxer, idx, PcmSpec::new(48_000, 2))?;
/// for _ in 0..10 {
///     let block = [0.0f32; 1024];
///     sink.write_f32(&block)?;
///     let _segment = sink.writer_mut().take_written();   // send it out
/// }
/// let mut writer = sink.finish()?;    // 收尾；字节都在 writer 里
/// let _tail = writer.take_written();  // 最后一段
/// # Ok(())
/// # }
/// ```
///
/// # Not `Send`
///
/// `PcmSink` is **not** `Send` (its `AVChannelLayout` holds a raw pointer), so
/// it cannot be moved to another thread. Feed it from a recording callback over
/// a channel instead of calling `write_*` inside the callback — that is what
/// `examples/pcm_recorder.rs` does, and it is a requirement, not a style choice.
///
/// # Where filters see which format
///
/// Audio processing is attached with
/// [`EncoderBuilder::with_filters`](crate::EncoderBuilder::with_filters), and
/// the graph sits *after* this sink's conversion: it sees the **encoder's**
/// sample format, rate and channel layout, not the microphone's. A filter that
/// needs a different rate must resample inside the graph (`aresample=...`), and
/// a filter that declares an input format must declare the encoder's one.
pub struct PcmSink<W: Writer> {
    muxer: Muxer<W>,
    stream_index: usize,
    /// 输入 PCM 规格。
    spec: PcmSpec,
    /// 编码器采样格式/采样率/声道布局（转换目标）。
    encoder_format: ffi::AVSampleFormat,
    encoder_layout: ffi::AVChannelLayout,
    encoder_sample_rate: i32,
    /// 持久重采样器，按首次写入的输入采样格式惰性创建。
    resampler: Option<(ffi::AVSampleFormat, Resampler)>,
    /// 已写入的输入样本数（每声道）。
    input_samples: u64,
    /// 编码器时间基（1/`encoder_sample_rate`）下的输出样本位置（pts）。
    output_samples: u64,
}

impl<W: Writer> PcmSink<W> {
    /// Wraps an existing [`Muxer`] and binds it to an audio stream.
    ///
    /// # Arguments
    ///
    /// * `muxer` - Muxer that owns the audio [`Encoder`](crate::encode::Encoder)（add_encoder 后传入）。
    /// * `stream_index` - [`Muxer::add_encoder`] 返回的音频流索引。
    /// * `spec` - 输入 PCM 的采样率与声道数（写入方法决定采样格式）。
    pub fn new(muxer: Muxer<W>, stream_index: usize, spec: PcmSpec) -> Result<Self> {
        if spec.sample_rate == 0 || spec.channels == 0 {
            return Err(RsmediaError::invalid_config(format!(
                "invalid PCM spec: sample_rate={}, channels={}",
                spec.sample_rate, spec.channels
            )));
        }
        let (encoder_format, encoder_layout, encoder_sample_rate) = {
            let mux_stream = muxer.get_stream(stream_index)?;
            if mux_stream.media_type != MediaType::AUDIO {
                return Err(RsmediaError::invalid_config(format!(
                    "stream {stream_index} is {:?}, not AUDIO",
                    mux_stream.media_type
                )));
            }
            let encoder = mux_stream.encoder.as_ref().ok_or_else(|| {
                RsmediaError::invalid_config(format!(
                    "stream {stream_index} is a copy stream; PCM playback requires an encoder stream"
                ))
            })?;
            if encoder.sample_rate() == 0 {
                return Err(RsmediaError::invalid_config(
                    "audio encoder has invalid sample rate",
                ));
            }
            (
                encoder.sample_fmt() as _,
                encoder.ch_layout().clone().into_inner(),
                // 编码器采样率转回 FFmpeg 的 `int` 宽度。
                encoder.sample_rate() as i32,
            )
        };
        Ok(Self {
            muxer,
            stream_index,
            spec,
            encoder_format,
            encoder_layout,
            encoder_sample_rate,
            resampler: None,
            input_samples: 0,
            output_samples: 0,
        })
    }

    /// 写入交错 `f32` PCM 块（如 cpal `SampleFormat::F32` 回调数据）。
    ///
    /// 产出的字节留在底层 writer 里，见类型级文档的 "Where the bytes go"。
    pub fn write_f32(&mut self, interleaved: &[f32]) -> Result<()> {
        self.write_chunks(interleaved, ffi::AV_SAMPLE_FMT_FLT)
    }

    /// 写入交错 `i16` PCM 块（如 cpal `SampleFormat::I16` 回调数据）。
    ///
    /// 产出的字节留在底层 writer 里，同 [`Self::write_f32`]。
    pub fn write_i16(&mut self, interleaved: &[i16]) -> Result<()> {
        self.write_chunks(interleaved, ffi::AV_SAMPLE_FMT_S16)
    }

    /// 写入交错 `u8` PCM 块（无符号 8bit，与 `AV_SAMPLE_FMT_U8` 一致）。
    ///
    /// 产出的字节留在底层 writer 里，同 [`Self::write_f32`]。
    pub fn write_u8(&mut self, interleaved: &[u8]) -> Result<()> {
        self.write_chunks(interleaved, ffi::AV_SAMPLE_FMT_U8)
    }

    /// 写入**平面**（planar）`f32` PCM：每个声道一个 slice，`planes[0]` 是
    /// 声道 0 的连续样本。
    ///
    /// 交错源请用 [`Self::write_f32`]；本方法给"手上已经是 planar"（解码器输出、
    /// `MediaFrame` 的平面数据、DSP 链输出）的场景省掉一次交错化。
    ///
    /// 所有 slice 必须等长，slice 个数必须等于 [`PcmSpec::channels`]。
    /// 输入很长时会自动按块切分，因此单次调用的样本数没有上限。
    ///
    /// 产出的字节留在底层 writer 里，同 [`Self::write_f32`]。
    pub fn write_planar_f32(&mut self, planes: &[&[f32]]) -> Result<()> {
        self.write_planar(planes, ffi::AV_SAMPLE_FMT_FLTP)
    }

    /// 冲刷重采样器尾样、编码器剩余样本并写 trailer，然后把 writer 交还给调用方。
    ///
    /// 消耗 sink，因此字节**只能**从返回的 writer 取 —— 这也是为什么本方法返回
    /// `W` 而不是 `()`：落盘型 writer 直接丢弃返回值即可，内存型 writer 拿它调
    /// [`BufferWriter::into_bytes`](crate::io::BufferWriter::into_bytes) 取完整容器。
    ///
    /// 未调用时 `Drop` 只做 `Muxer` 的那部分收尾（flush 编码器 + 写 trailer）：
    /// 容器仍然完整可读，但**不会**冲刷重采样器尾样（最后几十毫秒会丢），错误也
    /// 无法感知。因此显式调用 `finish` 是推荐做法。
    ///
    /// # 落盘 vs 取回内存
    ///
    /// 对会在 trailer 阶段**回写头部**的格式（mp4 / mov / wav），增量字节拼不出
    /// 完整文件 —— 那部分字节在交付之后才被改写，只有 `into_bytes()` 看得到。
    ///
    /// ```no_run
    /// # use rsmedia::error::Result;
    /// # use rsmedia::io::{BufferReader, BufferWriter};
    /// # use rsmedia::mux::Muxer;
    /// # use rsmedia::pcm::{PcmSink, PcmSpec};
    /// # use rsmedia::{Encoder, SampleFormat};
    /// # fn main() -> Result<()> {
    /// let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
    /// let mut muxer = Muxer::new_from_writer(BufferWriter::new("mp4")?);
    /// let idx = muxer.add_encoder(encoder)?;
    /// let mut sink = PcmSink::new(muxer, idx, PcmSpec::new(48_000, 2))?;
    /// sink.write_f32(&[0.0f32; 2048])?;
    /// let bytes: Vec<u8> = sink.finish()?.into_bytes();
    /// let reader = BufferReader::new(bytes)?;        // mp4 完整可读
    /// # Ok(())
    /// # }
    /// ```
    pub fn finish(mut self) -> Result<W> {
        self.drain_resampler()?;
        self.muxer.finish()?;
        Ok(self.muxer.into_writer())
    }

    /// 底层 [`Muxer`] 的 writer（可变借用）。
    ///
    /// 边产边发的场景用它取增量字节：缓冲型 writer 的
    /// [`take_written`](crate::io::BufferWriter::take_written) 要 `&mut`，而
    /// `write_*` 本身不返回字节。写入仍应走 `write_*`（那里面有重采样与 pts 编号），
    /// 本方法只用于取数。
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.muxer.writer
    }

    /// 当前已写入的输入样本数（每声道）。
    pub fn input_samples(&self) -> u64 {
        self.input_samples
    }

    /// 已送进编码器的输出样本数（每声道，编码器采样率下）。
    ///
    /// 与 [`Self::input_samples`] 不同：重采样会改变样本数，所以两者只有在
    /// 输入/输出采样率相同时才相等。这是 [`Self::output_duration`] 的分子。
    pub fn output_samples(&self) -> u64 {
        self.output_samples
    }

    /// 已写入音频在**编码器**时间基下的时长。
    ///
    /// 按 [`Self::output_samples`] / 编码器采样率换算，因此是"已经交付给编码器"
    /// 的时长；编码器的 priming/padding 不在内。
    pub fn output_duration(&self) -> Result<crate::Time> {
        crate::Time::from_units(self.output_samples as i64, self.encoder_sample_rate)
    }

    /// 输入 PCM 规格。
    pub fn spec(&self) -> PcmSpec {
        self.spec
    }

    /// 输入声道布局：显式掩码优先，否则按声道数取 FFmpeg 默认布局。
    fn input_layout(&self) -> Result<ffi::AVChannelLayout> {
        if self.spec.channel_mask == 0 {
            return Ok(AVChannelLayout::from_nb_channels(self.spec.channels as i32).into_inner());
        }
        let layout = AVChannelLayout::from_mask(self.spec.channel_mask).ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "invalid channel mask {:#x}: FFmpeg rejected it",
                self.spec.channel_mask
            ))
        })?;
        let raw = layout.into_inner();
        if raw.nb_channels != self.spec.channels as i32 {
            return Err(RsmediaError::invalid_config(format!(
                "channel mask {:#x} describes {} channels, but the spec declares {}",
                self.spec.channel_mask, raw.nb_channels, self.spec.channels
            )));
        }
        Ok(raw)
    }

    fn write_chunks<T: Copy>(
        &mut self,
        interleaved: &[T],
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        let channels = self.spec.channels as usize;
        if !interleaved.len().is_multiple_of(channels) {
            return Err(RsmediaError::invalid_config(format!(
                "interleaved PCM length {} is not a multiple of {} channels",
                interleaved.len(),
                channels
            )));
        }
        let samples_per_channel = interleaved.len() / channels;
        if samples_per_channel == 0 {
            return Ok(());
        }

        for chunk in interleaved.chunks(MAX_CHUNK_SAMPLES * channels) {
            self.write_chunk(chunk, sample_format)?;
        }
        Ok(())
    }

    fn write_planar<T: Copy>(
        &mut self,
        planes: &[&[T]],
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        let channels = self.spec.channels as usize;
        if planes.len() != channels {
            return Err(RsmediaError::invalid_config(format!(
                "planar input has {} plane(s) but the spec declares {channels} channel(s)",
                planes.len()
            )));
        }
        let nb_samples = planes.first().map_or(0, |p| p.len());
        if nb_samples == 0 {
            return Ok(());
        }
        if let Some(bad) = planes.iter().position(|p| p.len() != nb_samples) {
            return Err(RsmediaError::invalid_config(format!(
                "planar plane {bad} has {} samples, plane 0 has {nb_samples}: planes must be equal-length",
                planes[bad].len()
            )));
        }
        Self::check_element_width::<T>(sample_format)?;

        let mut offset = 0usize;
        while offset < nb_samples {
            let n = (nb_samples - offset).min(MAX_CHUNK_SAMPLES);
            let window: Vec<&[T]> = planes.iter().map(|p| &p[offset..offset + n]).collect();

            // 输入帧：planar，第 i 个声道占 data[i]
            let mut src = AVFrame::new();
            src.set_format(sample_format);
            src.set_nb_samples(n as i32);
            src.set_sample_rate(self.spec.sample_rate as i32);
            src.set_ch_layout(self.input_layout()?);
            src.alloc_buffer()
                .context("Failed to allocate PCM input frame buffer")?;

            // SAFETY: 同上 —— `src` 刚按 `n` 样本 / 输入布局 / `sample_format`
            // 分配过缓冲，planar 时每个声道各占一个 `data[i]`，可写 `n` 个样本；
            // `T` 的宽度已校验，读写范围完全落在各自的平面内。
            unsafe {
                let raw = src.deref_mut();
                for (plane, channel) in window.iter().zip(raw.data.iter()) {
                    let dst = std::slice::from_raw_parts_mut(*channel as *mut T, n);
                    dst.copy_from_slice(plane);
                }
            }
            self.input_samples += n as u64;

            self.convert_and_mux(&mut src, sample_format)?;
            offset += n;
        }
        Ok(())
    }

    fn write_chunk<T: Copy>(
        &mut self,
        interleaved: &[T],
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        let nb_samples = (interleaved.len() / self.spec.channels as usize) as i32;

        // 输入帧：packed（交错）样本，全部位于 data[0]
        let mut src = AVFrame::new();
        src.set_format(sample_format);
        src.set_nb_samples(nb_samples);
        src.set_sample_rate(self.spec.sample_rate as i32);
        src.set_ch_layout(self.input_layout()?);
        src.alloc_buffer()
            .context("Failed to allocate PCM input frame buffer")?;

        Self::check_element_width::<T>(sample_format)?;

        // SAFETY: `src` 刚按 `nb_samples` / `ch_layout`（= `channels`）/
        // `sample_format` 分配过缓冲，因此 `data[0]` 是 packed 格式的样本起点，
        // 可写 `nb_samples * channels == interleaved.len()` 个样本；上面已校验
        // `T` 与 `sample_format` 的元素宽度一致，读写范围完全落在缓冲内。`src`
        // 是本函数的局部独占对象（引用计数 1）。取 `data[0]` 走 `deref_mut`
        // 而非裸指针：这里只读取字段本身，引用足够表达。
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                src.deref_mut().data[0] as *mut T,
                interleaved.len(),
            );
            dst.copy_from_slice(interleaved);
        }
        self.input_samples += nb_samples as u64;

        self.convert_and_mux(&mut src, sample_format)
    }

    /// `T` 必须与 `sample_format` 的元素宽度一致。
    ///
    /// 下面按 `interleaved.len()` 个 `T` 写入按 `sample_format` 分配的缓冲，
    /// 宽度不匹配就是**堆越界写**。这条不变量原先只靠 `write_f32`/`write_i16`/
    /// `write_u8` 三个包装正确配对来维持——新增一个包装时写错格式就是越界，
    /// 所以在这里显式校验。
    fn check_element_width<T: Copy>(sample_format: ffi::AVSampleFormat) -> Result<()> {
        let element_bytes = rsmpeg::avutil::get_bytes_per_sample(sample_format).unwrap_or(0);
        if std::mem::size_of::<T>() != element_bytes {
            return Err(RsmediaError::invalid_config(format!(
                "PCM element type is {} byte(s) but {sample_format} stores {element_bytes}",
                std::mem::size_of::<T>()
            )));
        }
        Ok(())
    }

    /// 把一帧输入 PCM 转到编码器规格、编号并 mux 出去。
    ///
    /// `src` 的采样格式即输入格式（packed 或 planar 都行，FFmpeg 按 `format`
    /// 自行解释），`nb_samples` 已由调用方设好。
    fn convert_and_mux(
        &mut self,
        src: &mut AVFrame,
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        // 转换到编码器规格并按实际输出样本数累计 pts。
        // 容量按本块的输出样本数上界分配（重采样会改变样本数），而不是按 1 秒的
        // 大上界：`swr_convert` 只写到容量为止，容量过剩只是白占内存。
        let capacity = self
            .ensure_resampler(sample_format)?
            .get_out_samples(src.nb_samples);
        let mut dst = self.alloc_encoder_frame(capacity)?;
        self.ensure_resampler(sample_format)?
            .convert_frame(src, &mut dst)?;
        let out_nb = dst.nb_samples;
        if out_nb <= 0 {
            // 重采样器内部缓冲（滤波延迟），随后续输入/flush 输出
            return Ok(());
        }
        dst.set_pts(self.output_samples as i64);
        self.output_samples += out_nb as u64;
        self.muxer.mux(dst, self.stream_index)
    }

    /// 惰性创建持久重采样器；后续写入必须使用同一输入采样格式。
    fn ensure_resampler(&mut self, sample_format: ffi::AVSampleFormat) -> Result<&mut Resampler> {
        if self.resampler.is_none() {
            let in_layout = self.input_layout()?;
            let resampler = Resampler::new(
                in_layout,
                sample_format,
                self.spec.sample_rate as i32,
                self.encoder_layout,
                self.encoder_format,
                self.encoder_sample_rate,
            )?;
            self.resampler = Some((sample_format, resampler));
        }
        let (created_format, resampler) = self.resampler.as_mut().expect("just created");
        if *created_format != sample_format {
            return Err(RsmediaError::invalid_config(format!(
                "input sample format changed mid-stream: started with {}, now {}",
                created_format, sample_format
            )));
        }
        Ok(resampler)
    }

    /// 分配编码器规格的输出帧（容量 `capacity` 样本/声道）。
    fn alloc_encoder_frame(&self, capacity: i32) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_format(self.encoder_format);
        frame.set_nb_samples(capacity.max(1));
        frame.set_sample_rate(self.encoder_sample_rate);
        frame.set_ch_layout(self.encoder_layout);
        frame.set_time_base(
            Rational::new(1, self.encoder_sample_rate)
                .unwrap_or(Rational::ZERO)
                .into(),
        );
        frame
            .alloc_buffer()
            .context("Failed to allocate encoder-format frame buffer")?;
        Ok(frame)
    }

    /// EOF 时冲刷重采样器内部缓冲的尾样，避免丢失最后几十毫秒。
    fn drain_resampler(&mut self) -> Result<()> {
        // take() 取出以避免同时可变/不可变借用 self；冲刷后不再需要
        let Some((_, mut resampler)) = self.resampler.take() else {
            return Ok(());
        };
        // 上界：按 1 秒容量分配，循环取空（swr 滤波延迟通常仅几十毫秒）。
        // **循环次数也要有上限**：`flush` 的实现不在我们控制之内，一旦它持续吐出
        // 非零样本，无界的 `loop` 会让「结束音频流」这个公开 API 永不返回。上限
        // 与解码/编码/滤镜的排空一致（`crate::MAX_DRAIN_ITERATIONS`）。
        for _ in 0..crate::MAX_DRAIN_ITERATIONS {
            let mut dst = self.alloc_encoder_frame(self.encoder_sample_rate)?;
            resampler.flush(&mut dst)?;
            let out_nb = dst.nb_samples;
            if out_nb <= 0 {
                return Ok(());
            }
            dst.set_pts(self.output_samples as i64);
            self.output_samples += out_nb as u64;
            self.muxer.mux(dst, self.stream_index)?;
        }
        Err(RsmediaError::msg(format!(
            "Resampler keeps producing samples while draining ({} iterations); giving up",
            crate::MAX_DRAIN_ITERATIONS
        )))
    }
}

impl<W: Writer> std::fmt::Debug for PcmSink<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcmSink")
            .field("stream_index", &self.stream_index)
            .field("spec", &self.spec)
            .field("encoder_sample_rate", &self.encoder_sample_rate)
            .field("input_samples", &self.input_samples)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncoderBuilder;
    use crate::encode::Encoder;
    use crate::io::{BufferWriter, Reader};
    use crate::mux::Demuxer;
    use crate::{SampleFormat, test_support};

    /// 生成交错 f32 正弦 PCM（amplitude 0.3，单声道/多声道相同相位）。
    fn sine_samples(
        start_sample: u64,
        len_per_channel: usize,
        channels: u32,
        rate: u32,
    ) -> Vec<f32> {
        let mut samples = Vec::with_capacity(len_per_channel * channels as usize);
        for i in 0..len_per_channel {
            let t = (start_sample + i as u64) as f32 / rate as f32;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
            for _ in 0..channels {
                samples.push(v);
            }
        }
        samples
    }

    /// 解码输出文件，返回音频流的总样本数、采样率与声道数。
    fn decode_audio_stream(path: &std::path::Path) -> Result<(usize, u32, u32)> {
        let demuxer = Demuxer::new(path)?;
        let (audio_stream_index, sample_rate, channels) = {
            let s = demuxer
                .streams()
                .iter()
                .find(|s| s.media_type == MediaType::AUDIO)
                .expect("no audio stream in output");
            (
                s.stream_index,
                s.stream_info.sample_rate as u32,
                s.stream_info.channel_layout.nb_channels as u32,
            )
        };

        let mut total_samples = 0usize;
        for res in demuxer {
            let (index, frame) = res?;
            if index == audio_stream_index {
                total_samples += frame.nb_samples as usize;
            }
        }
        Ok((total_samples, sample_rate, channels))
    }

    /// 用 `ffmpeg`-free 的方式校验 codec id：直接读容器流参数。
    fn output_codec_id(path: &std::path::Path) -> ffi::AVCodecID {
        let reader = crate::StreamReader::new(path).expect("reopen output");
        reader.input().streams()[0].codecpar().codec_id
    }

    /// f32 流式写入（含不规则块大小）→ AAC：解码后样本数接近输入、
    /// 采样率/声道数保持、codec 为 aac。
    #[test]
    fn test_pcm_sink_f32_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_f32.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 2u32);
        let total_in = 44_100usize; // 1 秒

        let encoder = Encoder::new_audio(channels, in_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        // 不规则块大小：覆盖 FIFO 凑帧逻辑（1024 样本 AAC 帧长 vs 任意回调块）
        let mut written = 0usize;
        for chunk in [1000usize, 123, 7777, 512, 34_688] {
            if written >= total_in {
                break;
            }
            let n = chunk.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, in_rate))?;
            written += n;
        }
        assert_eq!(sink.input_samples() as usize, total_in);
        sink.finish()?;

        assert_eq!(output_codec_id(&output_path), ffi::AV_CODEC_ID_AAC);
        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        // AAC 编码有 priming/padding，允许 ±一个 frame_size 的误差
        let encoder_frame = 1024usize;
        assert!(
            (samples as i64 - total_in as i64).abs() <= encoder_frame as i64,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// 重采样桥接：输入 48kHz 单声道 f32 → 44.1kHz 立体声编码器，
    /// 解码后应为 44.1kHz、时长接近 1 秒。
    #[test]
    fn test_pcm_sink_resample() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_resample.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, out_rate) = (48_000u32, 44_100u32);
        let (in_channels, out_channels) = (1u32, 2u32);
        let in_total = 48_000usize; // 1 秒

        let encoder = Encoder::new_audio(out_channels, out_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, in_channels))?;

        // 模拟 cpal 典型回调块：512 帧/次
        let mut written = 0usize;
        while written < in_total {
            let n = 512usize.min(in_total - written);
            sink.write_f32(&sine_samples(written as u64, n, in_channels, in_rate))?;
            written += n;
        }
        sink.finish()?;

        let (samples, decoded_rate, decoded_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(decoded_rate, out_rate);
        assert_eq!(decoded_channels, out_channels);
        // 重采样 48k→44.1k 后应得到 ~44100 样本（± 1 帧）
        let expected = in_total as f64 * out_rate as f64 / in_rate as f64;
        assert!(
            (samples as f64 - expected).abs() <= 1024.0,
            "decoded {samples} samples, expected ~{expected}"
        );

        Ok(())
    }

    /// i16 写入路径：交错 S16 → AAC，解码样本数与输入一致（±1 帧）。
    #[test]
    fn test_pcm_sink_i16_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_i16.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 秒

        let encoder = Encoder::new_audio(channels, in_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 2048usize.min(total_in - written);
            let mut chunk = Vec::with_capacity(n * channels as usize);
            for i in 0..n {
                let t = (written + i) as f32 / in_rate as f32;
                let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                for _ in 0..channels {
                    chunk.push((v * i16::MAX as f32 * 0.3) as i16);
                }
            }
            sink.write_i16(&chunk)?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// u8 写入路径：交错 U8（128 为静音中点）→ AAC，解码样本数与输入一致（±1 帧）。
    #[test]
    fn test_pcm_sink_u8_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_u8.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 1u32);
        let total_in = 22_050usize; // 0.5 秒

        let encoder = Encoder::new_audio(channels, in_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 960usize.min(total_in - written);
            let chunk: Vec<u8> = (0..n)
                .map(|i| {
                    let t = (written + i) as f32 / in_rate as f32;
                    (128.0 + (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 40.0) as u8
                })
                .collect();
            sink.write_u8(&chunk)?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// 音频滤镜通用输入格式声明：编码器 FLTP，滤镜链声明输入 FLT（packed），
    /// 链内 `aformat` 转回 FLTP。进图前 swr 预转换 + 图内转换 + 滤镜后 rescale
    /// 三段协同，解码样本数与输入一致（±1 帧）。
    #[test]
    fn test_pcm_sink_filter_input_sample_format() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_filter_input.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 秒

        let filter = crate::filter::Filter::new(
            "aformat",
            MediaType::AUDIO,
            "aformat=sample_fmts=fltp".to_string(),
        )
        .with_input_format(SampleFormat::FLT);

        // 没有这个filter，不用测试
        if crate::filter::get_by_name(filter.name())?.is_none() {
            return Ok(());
        }

        let encoder = EncoderBuilder::new_audio(128_000, channels, in_rate, SampleFormat::FLTP)
            .with_filters(vec![filter])
            .build()?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 1000usize.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, in_rate))?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// mp4 会在 trailer 阶段 seek **回写**头部字节（实测：`BufferReader` 打不开按
    /// 增量拼出来的文件）。因此完整镜像只有 `finish()` 交回的 writer +
    /// `into_bytes()` 能拿到。
    #[test]
    fn test_finish_yields_a_readable_mp4() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_finish.mp4");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 秒

        let encoder = Encoder::new_audio(channels, in_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new_from_writer(BufferWriter::new("mp4")?);
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 1000.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, in_rate))?;
            written += n;
        }
        let bytes: Vec<u8> = sink.finish()?.into_bytes();
        assert!(!bytes.is_empty(), "mp4 output must not be empty");
        std::fs::write(&output_path, &bytes)?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );
        Ok(())
    }

    /// planar 输入：两个声道给不同频率，解码样本数与声道数都对。
    #[test]
    fn test_pcm_sink_planar_f32_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_planar.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 秒

        let encoder = Encoder::new_audio(channels, in_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, channels))?;

        // 左 440Hz / 右 220Hz：若实现把同一份数据抄进两个平面，这里也发现不了，
        // 但至少能证明"两个平面都被写入且长度一致"这条路径是通的。
        let mut written = 0usize;
        while written < total_in {
            let n = 1000.min(total_in - written);
            let mut left = Vec::with_capacity(n);
            let mut right = Vec::with_capacity(n);
            for i in 0..n {
                let t = (written + i) as f32 / in_rate as f32;
                left.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3);
                right.push((2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3);
            }
            sink.write_planar_f32(&[&left, &right])?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, in_rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );
        Ok(())
    }

    /// planar 的校验：平面个数不符、平面长度不等。
    #[test]
    fn test_planar_input_validation() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_planar_invalid.m4a");
        test_support::remove_test_output(&output_path);
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(44_100, 2))?;

        // 平面个数 ≠ 声道数
        assert!(sink.write_planar_f32(&[&[0.0f32; 64]]).is_err());
        // 平面长度不等
        assert!(
            sink.write_planar_f32(&[&[0.0f32; 64], &[0.0f32; 32]])
                .is_err()
        );
        // 空平面是 no-op
        assert!(sink.write_planar_f32(&[&[], &[]]).is_ok());
        Ok(())
    }

    /// 声道布局掩码：与声道数不符要拒；相符时解码声道数不变。
    #[test]
    fn test_channel_mask() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_mask.m4a");
        test_support::remove_test_output(&output_path);

        // 立体声掩码（AV_CH_FRONT_LEFT | AV_CH_FRONT_RIGHT = 0x3）
        const AV_CH_LAYOUT_STEREO: u64 = 0x3;

        // 掩码说 2 个声道，spec 说 1 个 ⇒ 拒绝
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(
            muxer,
            audio_index,
            PcmSpec::new(44_100, 1).with_channel_mask(AV_CH_LAYOUT_STEREO),
        )?;
        assert!(
            sink.write_f32(&[0.0f32; 128]).is_err(),
            "a mask that disagrees with the channel count must be rejected"
        );
        drop(sink);

        // 相符：整段走通
        test_support::remove_test_output(&output_path);
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(
            muxer,
            audio_index,
            PcmSpec::new(44_100, 2).with_channel_mask(AV_CH_LAYOUT_STEREO),
        )?;
        let mut written = 0usize;
        while written < 22_050 {
            let n = 1000.min(22_050 - written);
            sink.write_f32(&sine_samples(written as u64, n, 2, 44_100))?;
            written += n;
        }
        sink.finish()?;
        let (_, _, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_channels, 2);
        Ok(())
    }

    /// 输出侧计数：48k 输入 → 44.1k 编码器，输出样本数少于输入，时长≈1 秒。
    #[test]
    fn test_output_samples_and_duration() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_counters.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, out_rate) = (48_000u32, 44_100u32);
        let in_total = 48_000usize;

        let encoder = Encoder::new_audio(2, out_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, 2))?;

        let mut written = 0usize;
        while written < in_total {
            let n = 1000.min(in_total - written);
            sink.write_f32(&sine_samples(written as u64, n, 2, in_rate))?;
            written += n;
        }
        assert_eq!(sink.input_samples() as usize, in_total);
        assert!(
            sink.output_samples() < sink.input_samples(),
            "48k -> 44.1k must produce fewer output samples ({} vs {})",
            sink.output_samples(),
            sink.input_samples()
        );
        let duration = sink.output_duration()?;
        assert!(
            (duration.as_secs_f64() - 1.0).abs() < 0.05,
            "output duration {:.3}s, expected ~1.0",
            duration.as_secs_f64()
        );
        sink.finish()?;
        Ok(())
    }

    /// 参数与状态校验：非音频流拒绝、非法规格拒绝、声道不对齐的块拒绝。
    #[test]
    fn test_pcm_sink_validation() -> Result<()> {
        // 非法规格：new() 出错时 muxer 被丢弃（未写 header，无副作用）
        let assert_invalid_spec = |spec: PcmSpec| -> Result<()> {
            let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
            test_support::remove_test_output(&output_path);
            let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
            let mut muxer = Muxer::new(&output_path)?;
            let audio_index = muxer.add_encoder(encoder)?;
            assert!(PcmSink::new(muxer, audio_index, spec).is_err());
            Ok(())
        };
        assert_invalid_spec(PcmSpec::new(0, 2))?;
        assert_invalid_spec(PcmSpec::new(44_100, 0))?;

        // 非音频流拒绝
        let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
        test_support::remove_test_output(&output_path);
        let video_encoder = EncoderBuilder::new_video(64, 64).build()?;
        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;
        assert!(PcmSink::new(muxer, video_index, PcmSpec::new(44_100, 2)).is_err());

        // 声道不对齐的交错块拒绝；空块为 no-op
        let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(44_100, 2))?;
        assert!(sink.write_f32(&[0.0f32; 3]).is_err());
        assert!(sink.write_f32(&[]).is_ok());
        Ok(())
    }
}
