use crate::error::{Context, Result, RsmediaError};
use crate::init::{AVLogFlag, AVLogLevel};
use crate::location::Location;
use crate::options::Options;
use crate::stream::MediaType;
use crate::{strutils, time};

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avcodec::{AVCodecParameters, AVPacket};
use rsmpeg::avformat::{
    AVFormatContextInput, AVFormatContextOutput, AVIOContextContainer, AVIOContextCustom,
    AVInputFormat, ReadPacketCallback, SeekCallback, WritePacketCallback,
};
use rsmpeg::avutil::{AVDictionary, AVMem};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

use bytes::{BufMut, Bytes, BytesMut};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// `AVERROR(EIO)`：FFmpeg 以负 errno 报错，即 `AVERROR(e) = -e`。
/// errno 真值由平台头文件提供，不手写数字。
const AVERROR_EIO: i32 = -libc::EIO;

/// avio 内部缓冲大小（读写回调模式的滚动窗口）。
const AVIO_BUFFER_SIZE: usize = 4096;

ffi_enum!(
    /// Flags for [`Seekable::seek_to_frame`] (FFmpeg `AVSEEK_FLAG_*`).
    ///
    /// # Combining flags
    ///
    /// `|` combines flag bits, and the result is the **raw `i32` mask** — a fieldless
    /// enum cannot hold an unnamed combination such as `BACKWARD | ANY` — which
    /// [`Seekable::seek_to_frame`] accepts directly, because its parameter is
    /// `impl Into<i32>`. A single flag converts on its own:
    ///
    /// ```
    /// # use rsmedia::AVSeekFlag;
    /// let mask: i32 = AVSeekFlag::BACKWARD | AVSeekFlag::ANY; // 0b101
    /// let one: i32 = AVSeekFlag::ANY.into();
    /// ```
    ///
    /// Mixing a flag with a raw mask needs an explicit `as i32`: the `AVSEEK_FLAG_*`
    /// constants in rsmpeg are **`u32`**, whereas the operators are generated for the
    /// enum's own `repr` (here `i32`, because `av_seek_frame` takes `int`) and for
    /// `AVSeekFlag` only — a mixed-signedness overload is deliberately not provided:
    ///
    /// ```
    /// # use rsmedia::AVSeekFlag;
    /// let mixed: i32 = AVSeekFlag::FRAME | AVSeekFlag::BYTE;
    /// // `AVSeekFlag::FRAME | ffi::AVSEEK_FLAG_BYTE` does not compile: u32 vs i32.
    /// ```
    ///
    /// A combination is a bare `i32` from then on: no API turns it back into named
    /// flags, so test individual bits against `AVSeekFlag::X.as_raw()`.
    AVSeekFlag, i32 {
        BACKWARD => ffi::AVSEEK_FLAG_BACKWARD;
        BYTE => ffi::AVSEEK_FLAG_BYTE;
        ANY => ffi::AVSEEK_FLAG_ANY;
        FRAME => ffi::AVSEEK_FLAG_FRAME;
    }
);

pub trait Reader {
    fn input(&self) -> &AVFormatContextInput;
    fn input_mut(&mut self) -> &mut AVFormatContextInput;

    /// Read the next packet. Returns `None` on EOF.
    ///
    /// 成功时返回 `(packet 所属流的 index, packet)`；调用方如需更多流信息
    /// （time_base、metadata 等），可通过 `self.input().streams().get(index)`
    /// 直接使用 rsmpeg 的 [`AVStream`](rsmpeg::avformat::AVStream)。
    fn read_packet(&mut self) -> Result<Option<(usize, AVPacket)>> {
        match self.input_mut().read_packet() {
            Ok(Some(pkt)) => Ok(Some((pkt.stream_index as usize, pkt))),
            Ok(None) => Ok(None),
            Err(e) => Err(RsmediaError::FFmpeg(e)),
        }
    }

    /// Find the best stream
    ///
    /// # Arguments
    ///
    /// * `media_type` - MediaType maybe Video, Audio, etc.
    fn find_best_stream(&self, media_type: MediaType) -> Result<(usize, String)> {
        self.input()
            .find_best_stream(media_type as _)?
            .map(|(index, codec)| (index, strutils::cstr_to_string_lossy(codec.name())))
            .ok_or(RsmediaError::msg(format!(
                "No stream found for MediaType:{media_type:?}"
            )))
    }
}

/// Random-access (seek) capability interface.
///
/// This trait expresses "**seek operations are available**", not "a seek is
/// guaranteed to succeed". Whether a seek works depends on the **runtime
/// source**, which is why every method returns [`Result`] and reports the
/// underlying FFmpeg error on failure. A single [`StreamReader`] can read both a
/// local mp4 and an http/rtsp source, so both share this one API; the only
/// difference is whether a seek succeeds at runtime:
///
/// | Source | Seekable | Notes |
/// |--------|----------|-------|
/// | Local file (`file` protocol) | yes | regular files are always byte-seekable |
/// | In-memory buffer ([`BufferReader`]) | yes | a seek callback is installed, so seeking always works |
/// | HTTP(S) | server-dependent | seekable when the server returns `Accept-Ranges: bytes`; chunked/live responses usually are not |
/// | RTSP / RTP | server-dependent | depends on whether the server supports Range / PLAY jumps |
///
/// Use [`is_byte_seekable`](Seekable::is_byte_seekable) for a cheap pre-check
/// of the IO layer; container-level demuxer seek (e.g. RTSP) is opaque in
/// FFmpeg 5.0+ and cannot be probed, so treat the return value of each
/// `seek_*` call as authoritative.
///
/// # Example
///
/// ```no_run
/// use rsmedia::io::Seekable;
/// use rsmedia::StreamReader;
///
/// # fn main() -> rsmedia::error::Result<()> {
/// // Network source: a plain string literal is enough -- anything with a
/// // network scheme becomes a URL, everything else a filesystem path.
/// let mut reader = StreamReader::new("https://example.com/video.mp4")?;
/// if reader.is_byte_seekable() {
///     reader.seek_to_timestamp(10_000)?; // seek to the keyframe near 10s
/// }
/// # Ok(())
/// # }
/// ```
pub trait Seekable: Reader {
    /// Whether the underlying IO is byte-addressable, i.e. FFmpeg's
    /// `AVIOContext::seekable` has the `AVIO_SEEKABLE_NORMAL` bit set.
    ///
    /// - Local files and in-memory buffers ([`BufferReader`]): `true`.
    /// - HTTP(S): `true` when the server returns `Accept-Ranges: bytes`.
    /// - Live streams / pipes / most RTSP: `false`.
    ///
    /// This is the strongest **cheaply detectable** signal for "will a seek
    /// work" (e.g. whether to enable a draggable progress bar without doing an
    /// actual seek, which resets decoders and drops buffered data), but it is
    /// still **neither necessary nor sufficient**:
    ///
    /// - Demuxer-level container seek is opaque since FFmpeg 5.0
    ///   (`AVInputFormat` no longer exposes `read_seek`), so e.g. RTSP reports
    ///   `false` yet may still seek by timestamp through its own `read_seek`.
    /// - Conversely `AVFMT_NO_BYTE_SEEK` (set even by mp4/mov) only forbids
    ///   `AVSEEK_FLAG_BYTE`, not timestamp seeks, so it is deliberately *not*
    ///   consulted here.
    ///
    /// Treat the return value of each `seek_*` call as the authority.
    fn is_byte_seekable(&self) -> bool {
        let pb = self.input().pb;
        unsafe { !pb.is_null() && ((*pb).seekable & ffi::AVIO_SEEKABLE_NORMAL as i32) != 0 }
    }

    /// Seek in reader. This will change the reader head so that it points to a location within one
    /// second of the target timestamp or it will return an error.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ms` - Number of millisecond from start of video to seek to.
    fn seek_to_timestamp(&mut self, timestamp_ms: i64) -> Result<()> {
        // Conversion factor from timestamp in milliseconds to `TIME_BASE` units.
        const CONVERSION_FACTOR: i64 = (time::TIME_BASE.den / 1000) as i64;
        // One second left and right leeway when seeking.
        const LEEWAY: i64 = time::TIME_BASE.den as i64;
        let timestamp = CONVERSION_FACTOR * timestamp_ms;
        // 注意区间必须不对称（max 比 min 更贴近 ts）：`avformat_seek_file` 会
        // 忽略调用方的 BACKWARD 标志，并对不支持 read_seek2 的 demuxer 依据
        // `(ts - min) > (max - ts)` 推导回退方向（见 libavformat/seek.c）。
        // 若区间完全对称则恰好平局，会退化为 FORWARD seek——当 ts 之后没有
        // 关键帧时 seek 直接失败。这里让 min 侧留出 1μs 余量，保证始终
        // BACKWARD（定位到 ts 之前最近的关键帧，语义同 `av_seek_frame`）。
        seek_file(
            self.input_mut(),
            -1,
            timestamp - LEEWAY,
            timestamp,
            timestamp + LEEWAY - 1,
        )
        .context("Failed to seek timestamp in reader")?;
        Ok(())
    }

    /// Seek to start of reader. This function performs best effort seeking to the start of the
    /// file.
    fn seek_to_start(&mut self) -> Result<()> {
        // min=ts=INT64_MIN、max=INT64_MAX：目标即"最早可定位点"，
        // 强制 BACKWARD 回退到容器开头（best effort，不支持 seek 的源会报错）。
        seek_file(self.input_mut(), -1, i64::MIN, i64::MIN, i64::MAX)
            .context("Failed to seek to start of reader")?;
        Ok(())
    }

    /// Seek to a position in a stream: by **timestamp** by default, or by byte offset /
    /// "frame index" when the matching flag is set.
    ///
    /// Wraps FFmpeg's `av_seek_frame`. `frame_ts` is interpreted according to `flags`,
    /// and — this is the part worth reading — **the flags do not all do what their names
    /// promise**. Measured behaviour on FFmpeg 9:
    ///
    /// - **No flag, or `AVSeekFlag::ANY` / `AVSeekFlag::BACKWARD`** — timestamp seek, in
    ///   the stream's time-base units. Works on containers that build an index (MP4/MOV,
    ///   MPEG-TS, MKV, …); fails on raw Annex-B, which has neither index nor timestamps
    ///   (next bullet).
    /// - **`AVSeekFlag::FRAME` is not implemented by FFmpeg.** `avformat.h` documents it
    ///   as "seeking based on frame number", but no code path in libavformat reads it: a
    ///   few niche demuxers *reject* it (`mv`, `cine`, `concat`, subtitles, …) and
    ///   everything else ignores it, so the call falls through to the ordinary timestamp
    ///   path. On MP4, `seek_to_frame(0, 7, AVSeekFlag::FRAME)` therefore does **not**
    ///   land on frame 7 and does **not** fail either: it seeks to timestamp 7 and lands
    ///   on the first keyframe at `ts >= 7` (measured: frame 1 of an all-intra file).
    ///   Do not use it expecting frame indexes.
    /// - **`AVSeekFlag::BYTE`** repositions by byte offset — a plain `avio_seek`.
    ///   Demuxers that index by timestamp and set `AVFMT_NO_BYTE_SEEK` reject it with an
    ///   error (**MP4/MOV** is one; measured), while raw Annex-B accepts it (pure
    ///   byte-offset repositioning; measured).
    /// - **Raw Annex-B (`.h264` / `.hevc`)**: the raw demuxers carry no
    ///   `read_seek`/`read_seek2` and no `read_timestamp`, and decode every packet with
    ///   `pts = AV_NOPTS_VALUE`, so nothing ever fills the index and **every** seek by
    ///   timestamp or by "frame" fails — only `BYTE` works. [`Seekable::seek_to_start`]
    ///   fails there too.
    ///
    /// Support is otherwise source-dependent (network/live inputs may refuse or block),
    /// so treat the returned `Result` as authoritative — propagate it instead of
    /// `.unwrap()`-ing it, and prefer [`Seekable::seek_to_timestamp`] for containers.
    /// Seeking lands on the nearest **keyframe** unless `AVSeekFlag::ANY` is combined,
    /// which allows landing on a non-keyframe.
    ///
    /// A `stream_index` outside the input's streams is rejected here, before FFmpeg
    /// sees it: `av_seek_frame` dereferences the `AVStream` it is given without
    /// bounds-checking, so an out-of-range index is undefined behaviour rather than
    /// an error code.
    ///
    /// # Arguments
    ///
    /// * `stream_index` - The index of the stream to seek to.
    /// * `frame_ts` - The target timestamp (stream time-base units), or a byte offset
    ///   with `AVSeekFlag::BYTE`. "Frame index" only if a demuxer ever honours
    ///   `AVSeekFlag::FRAME`, which none does today — see above.
    /// * `flags` - [`AVSeekFlag`] bits: a single flag, or a raw `i32` mask built with
    ///   `|` (e.g. `AVSeekFlag::BACKWARD | AVSeekFlag::ANY`) — see [`AVSeekFlag`] for
    ///   how to mix in rsmpeg's `u32` `AVSEEK_FLAG_*` constants.
    fn seek_to_frame(
        &mut self,
        stream_index: usize,
        frame_ts: i64,
        flags: impl Into<i32>,
    ) -> Result<()> {
        let flags: i32 = flags.into();
        // 越界的流索引必须先拦下：`av_seek_frame` 会直接按索引取 `AVStream` 并
        // 读它的时间基，越界即未定义行为（实测 segfault），而不是返回错误码。
        let nb_streams = self.input().nb_streams as usize;
        if stream_index >= nb_streams {
            return Err(RsmediaError::invalid_config(format!(
                "Cannot seek stream {stream_index}: the input has {nb_streams} stream(s)"
            )));
        }
        unsafe {
            let res = ffi::av_seek_frame(
                self.input_mut().as_mut_ptr(),
                stream_index as i32,
                frame_ts,
                flags,
            );
            if res < 0 {
                return Err(RsmediaError::av_error(res).with_context(format!(
                    "Failed to seek stream {stream_index} to ts={frame_ts} (flags {flags:#x})"
                )));
            }
            Ok(())
        }
    }
}

/// Thin wrapper over `avformat_seek_file`: `min`/`ts`/`max` bound the target interval.
///
/// # Arguments
///
/// * `input` - The input context.
/// * `stream_index` - Target stream index; `-1` means timestamps are in `AV_TIME_BASE` units.
/// * `min` / `ts` / `max` - Target interval, inclusive on both ends.
///
/// Always called with `flags = 0`: `avformat_seek_file` ignores
/// `AVSEEK_FLAG_BACKWARD` and derives the seek direction from the **asymmetry**
/// of the interval instead (see [`Seekable::seek_to_timestamp`]). Use
/// [`Seekable::seek_to_frame`] when `AVSeekFlag` values such as `FRAME` / `BYTE`
/// are needed.
fn seek_file(
    input: &mut AVFormatContextInput,
    stream_index: i32,
    min: i64,
    ts: i64,
    max: i64,
) -> Result<()> {
    let res = unsafe { ffi::avformat_seek_file(input.as_mut_ptr(), stream_index, min, ts, max, 0) };
    if res < 0 {
        // >=0 on success, error code otherwise
        return Err(RsmediaError::av_error(res).with_context(format!(
            "Failed to seek stream {stream_index} into ts=[{min}, {max}] (target {ts})"
        )));
    }
    Ok(())
}

////////////////////////////////////////
// 内存 IO 的定位计算（BufferReader / BufferWriter 共享）
////////////////////////////////////////

/// 是否为 `AVSEEK_SIZE` 查询：只取数据总长，**不**移动当前位置。
fn is_seek_size_query(whence: i32) -> bool {
    whence & (ffi::AVSEEK_SIZE as i32) != 0
}

/// 按 `whence` 语义计算定位后的偏移，结果夹在 `[0, len]`。
///
/// 返回 `None` 表示 `whence` 非法：调用方应返回 -1 且保持当前位置不变
/// （与 POSIX `lseek` 一致），而不是把 -1 当成新位置存下去。
fn seek_offset(pos: i64, len: usize, offset: i64, whence: i32) -> Option<i64> {
    let len = len as i64;
    // 去掉 AVSEEK_FORCE 位后按 POSIX whence 解释（常量取自 libc，禁止手写）
    let base = whence & !(ffi::AVSEEK_FORCE as i32);
    let target = match base {
        libc::SEEK_SET => offset,
        libc::SEEK_CUR => pos + offset,
        libc::SEEK_END => len + offset,
        _ => return None,
    };
    Some(target.clamp(0, len))
}

////////////////////////////////////////
// Builder 共享 setter
////////////////////////////////////////

/// 为 reader builder 生成 `with_format` / `with_options` / `with_interrupt`，
/// 三者都只写 [`ReaderSpec`]。
///
/// 三个 reader builder 的这三项语义完全相同（含 `impl Into<Option<_>>` 的取值
/// 方式），因此只在这里写一遍：builder 只需持有 `spec: ReaderSpec<'a>`。
macro_rules! impl_reader_builder_setters {
    () => {
        /// Specify a custom format for the reader.
        ///
        /// # Arguments
        ///
        /// * `format` - Container format to use.
        pub fn with_format(mut self, format: &'a str) -> Self {
            self.spec.format = Some(format);
            self
        }

        /// Specify options for the backend.
        ///
        /// # Arguments
        ///
        /// * `options` - Options to pass on to input.
        pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
            self.spec.options = options.into();
            self
        }

        /// Attach an [`Interrupt`] handle to cancel blocked operations.
        ///
        /// 覆盖范围见 [`Interrupt`]：URL 输入（`StreamReader`）的打开与协议层
        /// 阻塞读都能被取消；自定义 IO 输入（`BufferReader` / `IoReader`）只有
        /// 探测循环生效，回调自身的阻塞读不受保护。
        pub fn with_interrupt(mut self, interrupt: impl Into<Option<Interrupt>>) -> Self {
            self.spec.interrupt = interrupt.into();
            self
        }
    };
}

/// 为 writer builder 生成 `with_options`（只写 [`WriterSpec`]）。
///
/// `with_format` 不在其中：`StreamWriter` 的格式可省略（由扩展名推断），
/// 内存 / 自定义 IO writer 则必填，两者的存储类型不同。
macro_rules! impl_writer_builder_setters {
    () => {
        /// Specify options for the backend.
        ///
        /// # Arguments
        ///
        /// * `options` - Options to pass on to output.
        pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
            self.spec.options = options.into();
            self
        }
    };
}

////////////////////////////////////////
// 自定义 AVIO 上下文装配（Reader/Writer 共享）
////////////////////////////////////////

/// 用自定义 AVIO 打开输入上下文并完成格式探测（内部已含
/// `avformat_open_input` + `avformat_find_stream_info`）。
///
/// 文件/URL 输入不走这里：应使用 `AVFormatContextInput::builder().url()`，
/// 由 FFmpeg 协议层（file/http/rtmp...）处理。本函数只服务于
/// `BufferReader`、`IoReader` 等非路径源。
///
/// 返回的 context 已安装 `interrupt`（若有）；调用方须保证 `interrupt`
/// 存活至 context drop。
fn open_input_custom(
    io_context: AVIOContextCustom,
    format: Option<&str>,
    options: Option<Options>,
    interrupt: Option<&Interrupt>,
    dump_name: &std::ffi::CStr,
) -> Result<AVFormatContextInput> {
    // 格式名来自调用者：含 NUL 字节时返回错误而不是 panic。
    let fmt_opt = match format {
        Some(name) => {
            let name_c = strutils::str_to_cstring(name)?;
            AVInputFormat::find(&name_c)
        }
        None => None,
    };
    let mut dict = options.and_then(|opts| opts.into_dict());
    let mut ctx = AVFormatContextInput::builder()
        .maybe_format(fmt_opt.as_deref())
        .options(&mut dict)
        .io_context(AVIOContextContainer::Custom(io_context))
        .open()
        .context("Create input format context with custom IO failed.")?;
    if let Some(interrupt) = interrupt {
        install_interrupt(&mut ctx, interrupt);
    }
    ctx.dump(0, dump_name)
        .context("Dump input format context failed.")?;
    Ok(ctx)
}

/// 用自定义 AVIO 构建输出上下文。
///
/// 构建阶段的 muxer 私有选项不会在此消费（FFmpeg 在
/// `avformat_write_header` 时才读取），由各 Writer 的 `write_header`
/// 经 [`write_header_with_options`] 透传。
fn build_output_custom(
    io_context: AVIOContextCustom,
    format: &str,
) -> Result<AVFormatContextOutput> {
    let format_cstr = strutils::str_to_cstring(format)?;
    AVFormatContextOutput::builder()
        .format_name(&format_cstr)
        .io_context(AVIOContextContainer::Custom(io_context))
        .build()
        .context("Create output format context with custom IO failed.")
}

////////////////////////////////////////
// Interrupt（阻塞操作取消/超时）
////////////////////////////////////////

/// FFmpeg 阻塞操作（网络读、seek 等）的中断控制。
///
/// FFmpeg 在每次可能阻塞的操作前调用 `AVIOInterruptCB`，回调返回非 0 时操作
/// 立即中止并返回错误。将同一个句柄传给 `ReaderBuilder::with_interrupt` 后，
/// 可从任意线程 [`abort`](Interrupt::abort) 或设置 [`timeout`](Interrupt::set_timeout)
/// 来取消卡住的读取。
///
/// 覆盖范围：
/// - 文件/URL 输入（[`StreamReaderBuilder`]）：回调在 `avformat_open_input`
///   **之前**安装，因此打开/探测阶段与协议层阻塞读（含 http/tcp 的重试与轮询
///   等待）均可被取消（原理见 `open_input_with_interrupt`）；
/// - seek：FFmpeg 的 `avio_seek`/`url_seek` 路径本身不做中断检查，能否被打断
///   取决于解复用器内部是否还要读数据（本地文件 seek 无阻塞，不需要中断）；
/// - 自定义 IO 输入（`BufferReader`/`IoReader` 等）不经过协议层，只有 format
///   context 层生效（探测循环），用户回调自身的阻塞读不受保护。
///
/// 输出侧对称性：写出方向（[`StreamWriter`] / [`BufferWriter`] / [`IoWriter`]）
/// 没有对应入口，推流这类阻塞写目前不可取消。
#[derive(Clone)]
pub struct Interrupt {
    data: Arc<InterruptData>,
}

struct InterruptData {
    abort: AtomicBool,
    deadline: Mutex<Option<std::time::Instant>>,
}

impl Interrupt {
    /// 创建一个未触发、无超时的中断句柄。
    pub fn new() -> Self {
        Self {
            data: Arc::new(InterruptData {
                abort: AtomicBool::new(false),
                deadline: Mutex::new(None),
            }),
        }
    }

    /// 立即中止所有阻塞操作（可跨线程调用）。
    pub fn abort(&self) {
        self.data.abort.store(true, Ordering::Relaxed);
    }

    /// 设置超时：从现在起 `timeout` 后自动触发中断。
    /// 对每次"打开 reader → 读取"的生命周期只生效一次，需要复用时重新设置。
    pub fn set_timeout(&self, timeout: std::time::Duration) {
        *self.data.deadline.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(std::time::Instant::now() + timeout);
    }

    /// 是否已触发（abort 或超时到期）。
    pub fn triggered(&self) -> bool {
        if self.data.abort.load(Ordering::Relaxed) {
            return true;
        }
        self.data
            .deadline
            .lock()
            .map(|d| d.is_some_and(|t| std::time::Instant::now() >= t))
            .unwrap_or(false)
    }
}

impl Default for Interrupt {
    fn default() -> Self {
        Self::new()
    }
}

/// 中断回调：返回 1 通知 FFmpeg 中止当前阻塞操作。
///
/// SAFETY: `opaque` 指向由 Reader 持有的 `InterruptData`，其生命周期
/// 覆盖整个 format context（context 先于数据 drop），回调期间指针有效。
unsafe extern "C" fn interrupt_callback(opaque: *mut std::ffi::c_void) -> std::ffi::c_int {
    let data = unsafe { &*(opaque as *const InterruptData) };
    let aborted = data.abort.load(Ordering::Relaxed);
    let timed_out = data
        .deadline
        .lock()
        .map(|d| d.is_some_and(|t| std::time::Instant::now() >= t))
        .unwrap_or(false);
    i32::from(aborted || timed_out)
}

/// 由 [`Interrupt`] 构造 FFmpeg 中断回调结构。
///
/// `opaque` 指向 `interrupt` 内部 `Arc<InterruptData>` 的堆内容；调用方必须
/// 让该 `Interrupt` 存活至回调被移除（Reader 将其作为字段持有，且 context
/// 字段先于 interrupt 字段 drop）。
fn interrupt_cb(interrupt: &Interrupt) -> ffi::AVIOInterruptCB {
    ffi::AVIOInterruptCB {
        callback: Some(interrupt_callback),
        opaque: Arc::as_ptr(&interrupt.data) as *mut std::ffi::c_void,
    }
}

/// 用原始 FFI 打开输入，并在 `avformat_open_input` **之前**安装中断回调。
///
/// 之所以不能用 rsmpeg 的 builder（它在内部 alloc + open，插不进回调）：FFmpeg
/// 不实时读取 `AVFormatContext.interrupt_callback`，而是把回调**按值拷贝**给
/// 用到的每一层：
/// - `io_open_default` → `ffio_open_whitelist(..., s->interrupt_callback, ...)`
///   → `ffurl_alloc` 存入 `URLContext.interrupt_callback`（`avio.c`），协议层每次
///   读只看这份副本（`retry_transfer_wrapper` 里的
///   `ff_check_interrupt(&h->interrupt_callback)`）；内层连接继续按值接力
///   （`tcp.c`: `ffurl_alloc(..., &s->interrupt_callback)`，`http.c` 等同理）；
/// - 网络等待/重试轮询用它（`network.c`: `ff_poll_interrupt`）；
/// - 探测循环用它（`demux.c`: `ff_check_interrupt(&ic->interrupt_callback)`）。
///
/// 因此回调必须在 open 前就位：open 之后补写只能覆盖探测循环，而那时协议层
/// 副本已经形成，阻塞读（网络流的主要场景）拦不住。
///
/// 也正因为回调要先于 open，这里只能写裸指针：`Deref`/`UnsafeDerefMut` 要求
/// 手上已有 [`AVFormatContextInput`]，而 open 前提前包装是不安全的——open 失败
/// 时 FFmpeg 自行释放 context 并把局部指针置空，包装体的 `Drop` 会二次释放。
/// 上下文成功建立之后的字段写（[`install_interrupt`]）则走 rsmpeg 的访问器。
fn open_input_with_interrupt(
    filename: &std::ffi::CStr,
    format: Option<&AVInputFormat>,
    options: &mut Option<AVDictionary>,
    interrupt: &Interrupt,
) -> Result<AVFormatContextInput> {
    let mut ctx = unsafe { ffi::avformat_alloc_context() };
    if ctx.is_null() {
        return Err(RsmediaError::msg("avformat_alloc_context failed"));
    }
    let fmt = format.map(|f| f.as_ptr()).unwrap_or(std::ptr::null());
    let mut opts = options
        .as_mut()
        .map(|d| d.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());
    let ret = unsafe {
        // SAFETY: `ctx` is a non-null pointer to a context from
        // `avformat_alloc_context` (checked just above) that has not been handed
        // to FFmpeg yet, so this write to `interrupt_callback` is exclusive.
        // `avformat_open_input` takes ownership of the context on success and, per
        // its documentation, frees it on failure (leaving `ctx` null) — which is
        // why the error path below must not touch `ctx` again.
        (*ctx).interrupt_callback = interrupt_cb(interrupt);
        ffi::avformat_open_input(&mut ctx, filename.as_ptr(), fmt, &mut opts)
    };
    if ret < 0 {
        // 文档保证 open 失败时用户提供的 context 已被释放、`ctx` 置空。
        return Err(RsmpegError::OpenInputError(ret).into());
    }
    // 与 rsmpeg builder 一致：把 FFmpeg 回写的剩余选项接回 Rust 所有权
    // （旧值已被 FFmpeg 就地消费/释放，必须整体换出后 forget，不能 drop）。
    let mut leftover = unsafe { std::ptr::NonNull::new(opts).map(|p| AVDictionary::from_raw(p)) };
    std::mem::swap(options, &mut leftover);
    std::mem::forget(leftover);

    // SAFETY: ctx 非空（open 成功），所有权交给 RAII 包装（Drop: avformat_close_input，
    // 它会关闭并释放 pb），因此 io_context 留空即可。
    let mut ctx_input =
        unsafe { AVFormatContextInput::from_raw(std::ptr::NonNull::new_unchecked(ctx)) };
    let ret =
        unsafe { ffi::avformat_find_stream_info(ctx_input.as_mut_ptr(), std::ptr::null_mut()) };
    if ret < 0 {
        return Err(RsmpegError::FindStreamInfoError(ret).into());
    }
    Ok(ctx_input)
}

/// 把中断回调装到 format context（探测循环 `avformat_find_stream_info` 会读它）。
///
/// 只服务于自定义 IO 输入（`BufferReader`/`IoReader` 等）：这类输入不经过
/// 协议层，没有 `URLContext` 副本，因此只有探测循环受保护；文件/URL 输入走
/// [`open_input_with_interrupt`]，回调在 open 前就位、协议层一并生效。
fn install_interrupt(ctx: &mut AVFormatContextInput, interrupt: &Interrupt) {
    // SAFETY: context 独占；opaque 指向的 Arc 目标由调用方保活。此处只写
    // `interrupt_callback` 字段，不触碰 FFmpeg 自身的指针/所有权成员，故走
    // rsmpeg 为"改 ffi 结构体成员"提供的 `UnsafeDerefMut` 访问器。
    unsafe {
        ctx.deref_mut().interrupt_callback = interrupt_cb(interrupt);
    }
}

////////////////////////////////////////
// StreamReader（Location/URL 输入）
////////////////////////////////////////

/// 三个 reader builder 的共享后端字段（格式名、options、中断句柄）。
///
/// setter 由 [`impl_reader_builder_setters`] 生成，
/// 因此三个 builder 的公开语义只有一份定义
#[derive(Default)]
struct ReaderSpec<'a> {
    format: Option<&'a str>,
    options: Option<Options>,
    interrupt: Option<Interrupt>,
}

/// Builds a [`StreamReader`].
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use rsmedia::io::StreamReaderBuilder;
/// use rsmedia::Options;
/// let mut options = Options::new();
/// options.insert("rtsp_transport", "tcp");
///
/// let mut reader = StreamReaderBuilder::new("my_file.mp4")
///    .with_options(Some(options))
///    .build()
///    .unwrap();
/// ```
pub struct StreamReaderBuilder<'a> {
    source: Location,
    spec: ReaderSpec<'a>,
}

impl<'a> StreamReaderBuilder<'a> {
    /// Create a new reader with the specified locator.
    ///
    /// # Arguments
    ///
    /// * `source` - Source to read.
    pub fn new(source: impl Into<Location>) -> Self {
        Self {
            source: source.into(),
            spec: ReaderSpec::default(),
        }
    }

    impl_reader_builder_setters!();

    /// Build [`StreamReader`].
    pub fn build(self) -> Result<StreamReader> {
        let Self { source, spec } = self;
        let filename = strutils::path_to_cstring(&source.as_path())?;
        let protocol = unsafe { ffi::avio_find_protocol_name(filename.as_ptr()) };
        if protocol.is_null() {
            return Err(RsmediaError::unsupported(format!(
                "input source protocol: {source}"
            )));
        }
        tracing::debug!(
            "Using input protocol: [{}], source: {}",
            // SAFETY: `avio_find_protocol_name` returns a pointer into FFmpeg's
            // static protocol registry (not a per-call allocation), valid for the
            // process lifetime; the NULL case was rejected above.
            unsafe { strutils::c_char_to_str(protocol) },
            source
        );

        // 格式名来自调用者：含 NUL 字节时返回错误而不是 panic。
        let fmt_opt = match spec.format {
            Some(name) => {
                let name_c = strutils::str_to_cstring(name)?;
                AVInputFormat::find(&name_c)
            }
            None => None,
        };
        let mut dict = spec.options.and_then(|opts| opts.into_dict());
        let mut ctx_input = match &spec.interrupt {
            // 带中断句柄时必须让回调先于 `avformat_open_input` 存在，见
            // `open_input_with_interrupt`；无中断时走 rsmpeg 的常规 builder。
            Some(interrupt) => {
                open_input_with_interrupt(&filename, fmt_opt.as_deref(), &mut dict, interrupt)?
            }
            None => AVFormatContextInput::builder()
                .url(&filename)
                .maybe_format(fmt_opt.as_deref())
                .options(&mut dict)
                .open()
                .context("Create input format context failed.")?,
        };
        ctx_input
            .dump(0, &filename)
            .context("Dump input format context failed.")?;
        Ok(StreamReader {
            source,
            core: ReaderCore::new(ctx_input, spec.interrupt),
        })
    }
}

////////////////////////////////////////
// 内置 Reader 共享状态
////////////////////////////////////////

/// 三个内置 Reader 的共享状态：输入上下文 + 中断句柄的保活。
///
/// `interrupt` 只在构建期被装进 format context，其回调 `opaque` 指向该
/// `Arc` 的堆内容；本字段负责在 context 存活期间保活它（只持有、不读取）。
/// 声明顺序即 drop 顺序：`input` 先释放，回调数据后释放——反过来会让仍在
/// 进行中的阻塞操作触发回调时读到悬挂指针（use-after-free）。
struct ReaderCore {
    input: AVFormatContextInput,
    _interrupt: Option<Interrupt>,
}

impl ReaderCore {
    fn new(input: AVFormatContextInput, interrupt: Option<Interrupt>) -> Self {
        Self {
            input,
            _interrupt: interrupt,
        }
    }
}

impl Reader for ReaderCore {
    fn input(&self) -> &AVFormatContextInput {
        &self.input
    }

    fn input_mut(&mut self) -> &mut AVFormatContextInput {
        &mut self.input
    }
}

/// Video reader that can read from files or URLs.
///
/// Implements [`Seekable`]: local files are always byte-seekable, whereas
/// network sources (http/rtsp) depend on the protocol and the server — see the
/// capability table on [`Seekable`].
pub struct StreamReader {
    source: Location,
    core: ReaderCore,
}

impl StreamReader {
    /// Create a new video file reader on a given source (path, URL, etc.).
    ///
    /// # Arguments
    ///
    /// * `source` - Source to read from.
    #[inline]
    pub fn new(source: impl Into<Location>) -> Result<Self> {
        StreamReaderBuilder::new(source).build()
    }

    /// The source this reader was opened with.
    pub fn source(&self) -> &Location {
        &self.source
    }
}

impl Reader for StreamReader {
    fn input(&self) -> &AVFormatContextInput {
        self.core.input()
    }

    fn input_mut(&mut self) -> &mut AVFormatContextInput {
        self.core.input_mut()
    }
}

impl Seekable for StreamReader {}

/// 线程安全性说明：`AVFormatContext` 本身非线程安全，这里仅承诺可以**移动**
/// 到其他线程独占使用（`Send`），不承诺 `&Self` 跨线程共享（不实现 `Sync`）。
unsafe impl Send for StreamReader {}

////////////////////////////////////////
// BufferReader（内存输入）
////////////////////////////////////////

/// Builds a [`BufferReader`].
///
/// # Example
///
/// ```no_run
/// use rsmedia::io::BufferReaderBuilder;
/// let data = std::fs::read("my_file.mp4").unwrap();
/// let reader = BufferReaderBuilder::new(data).build().unwrap();
/// ```
pub struct BufferReaderBuilder<'a> {
    data: Vec<u8>,
    spec: ReaderSpec<'a>,
}

impl<'a> BufferReaderBuilder<'a> {
    /// Create a new reader over an in-memory buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Media file contents to read.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            spec: ReaderSpec::default(),
        }
    }

    impl_reader_builder_setters!();

    /// Build [`BufferReader`].
    pub fn build(self) -> Result<BufferReader> {
        let Self { data, spec } = self;
        let data = Arc::new(data);
        let pos = Arc::new(AtomicUsize::new(0));

        let read_data = data.clone();
        let read_pos = pos.clone();
        let read_packet: ReadPacketCallback = Box::new(move |_opaque, buf: &mut [u8]| {
            let p = read_pos.load(Ordering::Relaxed);
            let n = read_data.len().saturating_sub(p).min(buf.len());
            buf[..n].copy_from_slice(&read_data[p..p + n]);
            read_pos.store(p + n, Ordering::Relaxed);
            if n == 0 { ffi::AVERROR_EOF } else { n as i32 }
        });

        let seek_data = data.clone();
        let seek_pos = pos.clone();
        let seek: SeekCallback = Box::new(move |_opaque, offset, whence| {
            let len = seek_data.len();
            if is_seek_size_query(whence) {
                return len as i64;
            }
            let pos = seek_pos.load(Ordering::Relaxed) as i64;
            match seek_offset(pos, len, offset, whence) {
                Some(target) => {
                    seek_pos.store(target as usize, Ordering::Relaxed);
                    target
                }
                // 非法 whence：报 -1 且不移动位置。
                None => -1,
            }
        });

        let io_context = AVIOContextCustom::alloc_context(
            AVMem::new(AVIO_BUFFER_SIZE),
            false,
            Vec::new(),
            Some(read_packet),
            None,
            Some(seek),
        );
        let ctx_input = open_input_custom(
            io_context,
            spec.format,
            spec.options,
            spec.interrupt.as_ref(),
            c"memory",
        )?;
        Ok(BufferReader {
            core: ReaderCore::new(ctx_input, spec.interrupt),
        })
    }
}

/// Video reader that reads from an in-memory buffer.
///
/// Implements [`Seekable`]: the whole input lives in memory and the custom AVIO
/// installs a seek callback, so FFmpeg sets `AVIO_SEEKABLE_NORMAL` and
/// [`Seekable::is_byte_seekable`] is always `true`.
pub struct BufferReader {
    core: ReaderCore,
}

impl BufferReader {
    /// Create a new video reader over an in-memory buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Media file contents to read.
    #[inline]
    pub fn new(data: Vec<u8>) -> Result<Self> {
        BufferReaderBuilder::new(data).build()
    }
}

impl Reader for BufferReader {
    fn input(&self) -> &AVFormatContextInput {
        self.core.input()
    }

    fn input_mut(&mut self) -> &mut AVFormatContextInput {
        self.core.input_mut()
    }
}

impl Seekable for BufferReader {}

/// 仅承诺可移动到其他线程独占使用（内部回调均为 `Send`）。
unsafe impl Send for BufferReader {}

////////////////////////////////////////
// IoReader（任意 std::io::Read 输入）
////////////////////////////////////////

/// Builds an [`IoReader`].
pub struct IoReaderBuilder<'a, R> {
    reader: R,
    spec: ReaderSpec<'a>,
}

impl<'a, R: std::io::Read + Send + 'static> IoReaderBuilder<'a, R> {
    /// Create a new reader wrapping any [`std::io::Read`] implementor.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source stream to read from (e.g. socket, pipe, decryptor).
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            spec: ReaderSpec::default(),
        }
    }

    impl_reader_builder_setters!();

    /// Build [`IoReader`].
    pub fn build(self) -> Result<IoReader> {
        let Self { reader, spec } = self;
        let mut reader = reader;
        let read_packet: ReadPacketCallback = Box::new(move |_opaque, buf: &mut [u8]| {
            loop {
                match reader.read(buf) {
                    Ok(0) => return ffi::AVERROR_EOF,
                    Ok(n) => return n as i32,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        tracing::error!("IoReader read error: {e}");
                        return AVERROR_EIO;
                    }
                }
            }
        });

        let io_context = AVIOContextCustom::alloc_context(
            AVMem::new(AVIO_BUFFER_SIZE),
            false,
            Vec::new(),
            Some(read_packet),
            None,
            None,
        );
        let ctx_input = open_input_custom(
            io_context,
            spec.format,
            spec.options,
            spec.interrupt.as_ref(),
            c"stream",
        )?;
        Ok(IoReader {
            core: ReaderCore::new(ctx_input, spec.interrupt),
        })
    }
}

/// Video reader that reads from any [`std::io::Read`] implementor.
///
/// For streaming inputs (sockets, pipes, decryptors, ...). Does **not** implement
/// [`Seekable`]. Because `avformat_open_input` needs to probe, `reader` must be
/// re-readable; no seeking is required since probing only moves forward.
pub struct IoReader {
    core: ReaderCore,
}

impl IoReader {
    /// Create a new video reader wrapping any [`std::io::Read`] implementor.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source stream to read from.
    #[inline]
    pub fn new(reader: impl std::io::Read + Send + 'static) -> Result<Self> {
        IoReaderBuilder::new(reader).build()
    }
}

impl Reader for IoReader {
    fn input(&self) -> &AVFormatContextInput {
        self.core.input()
    }

    fn input_mut(&mut self) -> &mut AVFormatContextInput {
        self.core.input_mut()
    }
}

/// 仅承诺可移动到其他线程独占使用（内部 reader 与回调均为 `Send`）。
unsafe impl Send for IoReader {}

////////////////////////////////////////
// Writer trait
////////////////////////////////////////

/// Any type that implements this can write video packets.
///
/// 该 trait 是公开的扩展点：可以为任意目标（socket、channel、加密流等）
/// 实现自定义 Writer。
///
/// 输出模型分两层：
/// - [`Out`](Self::Out)：单次 `write_*` 调用产生的输出（如一个 [`Bytes`] 块）；
/// - [`Accum`](Self::Accum)：跨多次调用累积输出的容器（如 `Vec<Bytes>`）。
///
/// 二者解耦，单次输出可以是不变类型（`Bytes`，交接/共享零拷贝），而累积走
/// chunk 列表的指针移动（`push`/`extend`），没有任何字节级 memcpy。
pub trait Writer {
    /// 单次 `write_*` 调用产生的输出类型：
    /// [`StreamWriter`] / [`IoWriter`] 为 `()`（数据直接写出），
    /// [`BufferWriter`] 为 [`Bytes`]（本次调用新增的字节块）。
    type Out;

    /// 跨多次 `write_*` 累积 [`Out`](Self::Out) 的容器；空累积器即
    /// `Default::default()`，因此"从零开始累积"无需 `Option` 包装。
    /// [`StreamWriter`] / [`IoWriter`] 为 `()`，[`BufferWriter`] 为 `Vec<Bytes>`。
    type Accum: Default;

    /// Write the container header.
    ///
    /// 容器头**恰好写一次**：再次调用返回 [`RsmediaError::InvalidConfig`]
    /// （FFmpeg 要求 header 先于所有包、且只写一次；二次写会把 muxer 的内部
    /// 状态重置到"刚开始写"，与已落盘的字节、已注册的流冲突）。
    fn write_header(&mut self) -> Result<Self::Out>;

    /// 容器头是否已成功写出过（[`write_header`](Self::write_header) 的调用结果）。
    ///
    /// 这是"输出上下文的流数组是否已固定"的权威判据：header 写出后，
    /// `AVFormatContext` 的流数量由 muxer 接管，此时再
    /// [`add_stream`](Self::add_stream) 会破坏 FFmpeg 内部持有的指针/索引
    /// （实测 SIGSEGV）。库内路径（[`Muxer`](crate::mux::Muxer)、字幕写入）
    /// 都以此为准，自定义实现必须如实记录、不要靠猜——默认的
    /// [`add_stream`](Self::add_stream) 实现就靠它拦截误用。
    fn is_header_written(&self) -> bool;

    /// Write a packet into the container.
    ///
    /// # Arguments
    ///
    /// * `packet` - AVPacket to write.
    fn write_frame(&mut self, packet: &mut AVPacket) -> Result<Self::Out>;

    /// Write a packet into the container and take care of interleaving.
    ///
    /// # Arguments
    ///
    /// * `packet` - AVPacket to write.
    fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<Self::Out>;

    /// Write the container trailer.
    fn write_trailer(&mut self) -> Result<Self::Out>;

    /// Obtain reference to output context.
    fn output(&self) -> &AVFormatContextOutput;

    /// Obtain mutable reference to output context.
    fn output_mut(&mut self) -> &mut AVFormatContextOutput;

    /// new stream
    ///
    /// 只能在 [`write_header`](Self::write_header) **之前**调用：header 写出后
    /// 流数组已固定，再建流会破坏 muxer 内部持有的指针/索引（SIGSEGV），因此
    /// 此时返回 [`RsmediaError::InvalidConfig`] 而不是继续执行。
    ///
    /// 返回新流的 index（写包时作为 stream index 使用）。自定义实现若覆盖本方法，
    /// 必须用 [`is_header_written`](Self::is_header_written) 做同样的拦截——这是
    /// 安全前提，不是风格问题。
    fn add_stream(
        &mut self,
        codecpar: AVCodecParameters,
        timebase: ffi::AVRational,
    ) -> Result<usize> {
        if self.is_header_written() {
            return Err(RsmediaError::invalid_config(format!(
                "Cannot add a stream after the container header has been written \
                 ({} output stream(s) already registered); register all streams first",
                self.output().nb_streams
            )));
        }
        let mut av_stream = self.output_mut().new_stream();
        av_stream.set_codecpar(codecpar);
        av_stream.set_time_base(timebase);
        Ok(av_stream.index as usize)
    }

    /// 获取输出流当前的时间基。
    ///
    /// 注意：`write_header` 之后 muxer 可能调整 stream 的时间基（例如 MP4 的
    /// movenc 会重设 timescale）。因此写包时应**实时获取**，不要缓存 write 前
    /// 的值，否则 packet 的 pts/duration 会按错误的 time_base 解析。
    ///
    /// 流不存在时返回错误：早先这里回退为 [`TIME_BASE`](crate::time::TIME_BASE)
    /// （1/1000000），会让 `rescale_ts` 静默算出完全错误的时间戳。
    fn stream_time_base(&self, stream_index: usize) -> Result<ffi::AVRational> {
        self.output()
            .streams()
            .get(stream_index)
            .map(|stream| stream.time_base)
            .ok_or_else(|| {
                RsmediaError::invalid_config(format!(
                    "Output stream {stream_index} does not exist ({} streams)",
                    self.output().nb_streams
                ))
            })
    }

    /// Folds one more write's output into the accumulator.
    ///
    /// For the buffering writers `Out` is the *incremental* new output of a single
    /// write, so a step that produces several packets must accumulate instead of
    /// overwrite — [keeping only the last chunk would silently hand back a
    /// truncated stream][mux]. Start from an empty accumulator with
    /// `<Self as Writer>::Accum::default()`.
    ///
    /// [mux]: crate::mux::Muxer::mux
    fn merge_out(acc: &mut Self::Accum, out: Self::Out);

    /// Merges one accumulator into another: folding a sub-pipeline's accumulated
    /// output ([`Encoder::flush`](crate::Encoder::flush), PCM writer chunks, …)
    /// into the caller's accumulator.
    ///
    /// Must accumulate, not overwrite — same rationale as [`Self::merge_out`].
    fn merge_accum(acc: &mut Self::Accum, other: Self::Accum);
}

/// 将 builder 阶段未消费的 options 在 `write_header` 时传给 muxer。
///
/// FFmpeg 的 muxer 私有选项（如 `movflags`）在 `avformat_write_header` 时
/// 才被消费，构建阶段传入的 options 若不透传到这里会被静默丢弃。
fn write_header_with_options(
    output: &mut AVFormatContextOutput,
    options: &mut Option<AVDictionary>,
) -> Result<()> {
    output
        .write_header(options)
        .context("Failed to write header")
}

/// Flush avio 内部缓冲，确保字节立即送达 write 回调（内存/流式 Writer 用）。
fn flush_avio(output: &mut AVFormatContextOutput) {
    unsafe {
        let pb = (*output.as_mut_ptr()).pb;
        if !pb.is_null() {
            ffi::avio_flush(pb);
        }
    }
}

////////////////////////////////////////
// 内置 Writer 共享状态
////////////////////////////////////////

/// writer builder 的共享后端字段（options）；setter 由
/// [`impl_writer_builder_setters`] 生成。
///
/// `format` 不在其中：`StreamWriter` 的格式可省略（按扩展名推断），
/// 内存 / 自定义 IO writer 则必填，两者的存储类型不同。
#[derive(Default)]
struct WriterSpec {
    options: Option<Options>,
}

/// 三个内置 Writer 的共享内核：输出上下文、options、header 状态，以及
/// "每次写后是否 flush avio"的策略。
///
/// [`Writer`] 的固定语义都收在这里，只写一遍：
/// - 容器头**恰好写一次**（二次调用在触到 FFmpeg 之前就报错；只在成功后
///   置位 `header_written`，失败时仍是处女态，允许修好参数重试）；
/// - 构建阶段未消费的 options 在 `write_header` 时透传给 muxer；
/// - 内存/流式 Writer 每次写完 `avio_flush`，保证字节立即送达回调；
///   文件/URL Writer 交给 FFmpeg 自己缓冲。
///
/// 因此各 Writer 实现只剩 `Out`/`Accum` 的差异（见各类型自己的 `impl Writer`）。
struct WriterCore {
    output: AVFormatContextOutput,
    options: Option<AVDictionary>,
    /// [`Writer::is_header_written`] 的记录。
    header_written: bool,
    /// 是否在每次 `write_*` 后 flush avio。
    flush_after_write: bool,
}

impl WriterCore {
    fn new(
        output: AVFormatContextOutput,
        options: Option<AVDictionary>,
        flush_after_write: bool,
    ) -> Self {
        Self {
            output,
            options,
            header_written: false,
            flush_after_write,
        }
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(RsmediaError::invalid_config(
                "write_header() was already called on this writer: a container header is written \
                 exactly once, before any packet",
            ));
        }
        let mut dict = self.options.take();
        write_header_with_options(&mut self.output, &mut dict)?;
        self.header_written = true;
        self.flush();
        Ok(())
    }

    fn is_header_written(&self) -> bool {
        self.header_written
    }

    fn write_frame(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.output.write_frame(packet)?;
        self.flush();
        Ok(())
    }

    fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.output.interleaved_write_frame(packet)?;
        self.flush();
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        self.output
            .write_trailer()
            .context("Failed to write trailer")?;
        self.flush();
        Ok(())
    }

    /// flush avio 内部缓冲（`flush_after_write` 为假时是空操作）。
    fn flush(&mut self) {
        if self.flush_after_write {
            flush_avio(&mut self.output);
        }
    }

    fn output(&self) -> &AVFormatContextOutput {
        &self.output
    }

    fn output_mut(&mut self) -> &mut AVFormatContextOutput {
        &mut self.output
    }
}

////////////////////////////////////////
// StreamWriter（Location/URL 输出）
////////////////////////////////////////

/// Build a [`StreamWriter`].
pub struct StreamWriterBuilder<'a> {
    destination: Location,
    format: Option<&'a str>,
    spec: WriterSpec,
}

impl<'a> StreamWriterBuilder<'a> {
    /// Create a new writer with the specified destination.
    ///
    /// # Arguments
    ///
    /// * `destination` - Destination to write to.
    pub fn new(destination: impl Into<Location>) -> Self {
        Self {
            destination: destination.into(),
            format: None,
            spec: WriterSpec::default(),
        }
    }

    /// Specify a container format for the writer.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format to use. eg. `"mp4"`, `"mkv"`, `"mov"`, `"avi"`, `"flv"`.
    ///
    /// reference: <https://trac.ffmpeg.org/wiki/HWAccelIntro>
    ///
    /// | Format                          | Filename Extension | H.264/AVC | H.265/HEVC | AV1   |
    /// |---------------------------------|--------------------|-----------|------------|-------|
    /// | Matroska                        | .mkv               | Y         | Y          | Y     |
    /// | MPEG-4 Part 14 (MP4)            | .mp4               | Y         | Y          | Y     |
    /// | Audio Video Interleave (AVI)    | .avi               | Y         | N          | Y     |
    /// | Material Exchange Format (MXF)  | .mxf               | Y         | n/a        | n/a   |
    /// | MPEG transport stream (TS)      | .ts                | Y         | Y          | N     |
    /// | 3GPP (3GP)                      | .3gp               | Y         | n/a        | n/a   |
    /// | Flash Video (FLV)               | .flv               | Y         | n/a        | n/a   |
    /// | WebM                            | .webm              | n/a       | n/a        | Y     |
    /// | Advanced Systems Format (ASF)   | .asf, .wmv         | Y         | Y          | Y     |
    /// | QuickTime File Format (QTFF)    | .mov               | Y         | Y          | n/a   |
    pub fn with_format(mut self, format: &'a str) -> Self {
        self.format = Some(format);
        self
    }

    impl_writer_builder_setters!();

    /// Build [`StreamWriter`].
    pub fn build(self) -> Result<StreamWriter> {
        let Self {
            destination,
            format,
            spec,
        } = self;
        let filename = strutils::path_to_cstring(&destination.as_path())?;
        // 格式名来自调用者：含 NUL 字节时返回错误而不是 panic。
        let format = match format {
            Some(name) => Some(strutils::str_to_cstring(name)?),
            None => None,
        };
        let mut dict = spec.options.and_then(|opts| opts.into_dict());
        let output_ctx = AVFormatContextOutput::builder()
            .filename(&filename)
            .maybe_format_name(format.as_deref())
            // options 先由 avio/protocol 层消费（avio_open2），未被消费的
            // （如 movflags 等 muxer 私有选项）留在 dict 中，待 write_header
            // 时传给 muxer。
            .options(&mut dict)
            .build()
            .context("Create output format context failed.")?;
        Ok(StreamWriter {
            destination,
            core: WriterCore::new(output_ctx, dict, false),
        })
    }
}

/// File writer for video files.
///
/// # Example
///
/// Create a video writer that produces fragmented MP4:
///
/// ```no_run
/// use rsmedia::io::StreamWriterBuilder;
/// use rsmedia::Options;
///
/// let mut options = Options::new();
/// options.insert("movflags", "frag_keyframe+empty_moov");
///
/// let mut writer = StreamWriterBuilder::new("my_file.mp4")
///     .with_options(Some(options))
///     .build()
///     .unwrap();
/// ```
pub struct StreamWriter {
    destination: Location,
    core: WriterCore,
}

impl StreamWriter {
    /// Create a new file writer for video files.
    ///
    /// # Arguments
    ///
    /// * `dest` - Where to write to.
    #[inline]
    pub fn new(destination: impl Into<Location>) -> Result<Self> {
        StreamWriterBuilder::new(destination).build()
    }

    /// The destination this writer was opened with.
    pub fn destination(&self) -> &Location {
        &self.destination
    }
}

impl Writer for StreamWriter {
    type Out = ();
    type Accum = ();

    fn merge_out(_acc: &mut (), _out: ()) {}
    fn merge_accum(_acc: &mut (), _other: ()) {}

    fn write_header(&mut self) -> Result<()> {
        self.core.write_header()
    }

    fn is_header_written(&self) -> bool {
        self.core.is_header_written()
    }

    fn write_frame(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.core.write_frame(packet)
    }

    fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.core.write_interleaved(packet)
    }

    fn write_trailer(&mut self) -> Result<()> {
        self.core.write_trailer()
    }

    fn output(&self) -> &AVFormatContextOutput {
        self.core.output()
    }

    fn output_mut(&mut self) -> &mut AVFormatContextOutput {
        self.core.output_mut()
    }
}

/// 仅承诺可移动到其他线程独占使用。
unsafe impl Send for StreamWriter {}

////////////////////////////////////////
// BufferWriter（内存输出，持久可 seek 的 custom IO）
////////////////////////////////////////

/// 内存写状态：`data` 为累计输出，`pos` 为 avio 当前写位置（支持 seek 回退
/// 重写），`delivered` 为已通过增量接口返回给调用方的字节数。
///
/// 存储用 [`BytesMut`] 而非 `Vec<u8>`：`into_bytes` 可零拷贝 freeze/转换，
/// 追加路径走 [`BufMut::put_slice`] 免去 `resize` 的 memset + memcpy 双写。
#[derive(Default)]
struct MemWriterState {
    data: BytesMut,
    pos: usize,
    delivered: usize,
}

/// 自定义 IO 写回调共享逻辑。
///
/// 该函数在 FFmpeg 的 `extern "C"` 调用栈中执行，**严禁 panic**
/// （unwind 穿过 FFI 边界是 UB）：锁中毒时返回 `AVERROR(EIO)`。
fn mem_write(state: &Mutex<MemWriterState>, buf: &[u8]) -> i32 {
    let mut st = match state.lock() {
        Ok(guard) => guard,
        Err(_) => return AVERROR_EIO,
    };
    if st.pos == st.data.len() {
        // 追加快路径：顺序写出（绝大多数调用）直接 put，一次写入。
        st.data.put_slice(buf);
    } else {
        // seek 回退路径：重写中部区域，需保证缓冲覆盖到位。
        let pos = st.pos;
        let end = pos + buf.len();
        if end > st.data.len() {
            st.data.resize(end, 0);
        }
        st.data[pos..end].copy_from_slice(buf);
    }
    st.pos += buf.len();
    buf.len() as i32
}

/// 自定义 IO seek 回调共享逻辑（同样禁止 panic；失败返回 -1）。
fn mem_seek(state: &Mutex<MemWriterState>, offset: i64, whence: i32) -> i64 {
    let mut st = match state.lock() {
        Ok(guard) => guard,
        Err(_) => return -1,
    };
    let len = st.data.len();
    if is_seek_size_query(whence) {
        return len as i64;
    }
    match seek_offset(st.pos as i64, len, offset, whence) {
        Some(target) => {
            st.pos = target as usize;
            target
        }
        // 非法 whence：报 -1 且不移动写位置。
        None => -1,
    }
}

/// Build a [`BufferWriter`].
pub struct BufferWriterBuilder<'a> {
    format: &'a str,
    spec: WriterSpec,
}

impl<'a> BufferWriterBuilder<'a> {
    /// Create a new writer that writes to a buffer.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format to use.
    pub fn new(format: &'a str) -> Self {
        Self {
            format,
            spec: WriterSpec::default(),
        }
    }

    impl_writer_builder_setters!();

    /// Build [`BufferWriter`].
    pub fn build(self) -> Result<BufferWriter> {
        let Self { format, spec } = self;
        let state = Arc::new(Mutex::new(MemWriterState::default()));

        let write_state = state.clone();
        let write_packet: WritePacketCallback =
            Box::new(move |_opaque, buf: &[u8]| mem_write(&write_state, buf));

        let seek_state = state.clone();
        let seek: SeekCallback =
            Box::new(move |_opaque, offset, whence| mem_seek(&seek_state, offset, whence));

        // 持久且可 seek 的内存 IO：avio 的 pos 跨 write_* 调用连续，
        // 支持 mp4 等 muxer 在 trailer 阶段回写 header 区域。
        let io_context = AVIOContextCustom::alloc_context(
            AVMem::new(AVIO_BUFFER_SIZE),
            true,
            Vec::new(),
            None,
            Some(write_packet),
            Some(seek),
        );
        let output = build_output_custom(io_context, format)?;
        Ok(BufferWriter {
            core: WriterCore::new(output, spec.options.and_then(|opts| opts.into_dict()), true),
            state,
        })
    }
}

/// Video writer that writes to a buffer.
///
/// 每次写入操作（write_header/write_frame/write_trailer）返回**本次新增**的
/// 字节增量，适合流式格式（mpegts、fmp4 等）的分段发送；对会回写 header 的
/// 格式（普通 mp4），请在 write_trailer 后用 [`Self::into_bytes`] 取完整输出。
///
/// # Example
///
/// ```no_run
/// use rsmedia::io::{BufferWriter, Writer};
///
/// # fn main() -> rsmedia::error::Result<()> {
/// let mut writer = BufferWriter::new("mpegts")?;
/// let _header = writer.write_header()?;
/// # Ok(())
/// # }
/// ```
pub struct BufferWriter {
    core: WriterCore,
    state: Arc<Mutex<MemWriterState>>,
}

impl BufferWriter {
    /// Create a video writer that writes to a buffer and returns the resulting bytes.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format to use.
    #[inline]
    pub fn new(format: &str) -> Result<Self> {
        BufferWriterBuilder::new(format).build()
    }

    /// 取出本次写入操作新增的字节增量。
    ///
    /// 返回 [`Bytes`]：从 FFmpeg 复用的 avio 缓冲中拷出（这层拷贝不可避免），
    /// 但之后的交接、切片、跨线程共享都是引用计数，不再有第二次拷贝。
    fn take_written(&mut self) -> Bytes {
        let mut st = self.state.lock().expect("mem writer state poisoned");
        let delta = Bytes::copy_from_slice(&st.data[st.delivered..]);
        st.delivered = st.data.len();
        delta
    }

    /// 消耗 writer 并返回**完整**的输出字节。
    ///
    /// 对会在 trailer 阶段回写 header 的格式（如普通 mp4），增量接口拿不到
    /// 回写的字节，必须用本方法获取最终完整结果。应在 `write_trailer` 之后
    /// 调用。
    pub fn into_bytes(self) -> Vec<u8> {
        let Self { core, state } = self;
        let WriterCore { output, .. } = core;
        // 先释放 format context（连带 IO 回调释放其持有的 state 引用）
        drop(output);
        match Arc::try_unwrap(state) {
            Ok(st) => Vec::from(st.into_inner().expect("mem writer state poisoned").data),
            // 不可达：output 已 drop，回调持有的 Arc 引用随之释放。
            // 用 panic（fail-fast）而非静默返回空 Vec，避免数据无声丢失。
            Err(_) => panic!("BufferWriter: state still referenced after context drop"),
        }
    }
}

impl Writer for BufferWriter {
    type Out = Bytes;
    type Accum = Vec<Bytes>;

    /// chunk 列表 `push`：只移动 `Bytes` 句柄，零字节拷贝。
    fn merge_out(acc: &mut Vec<Bytes>, out: Bytes) {
        acc.push(out);
    }

    /// chunk 列表 `extend`：只移动 `Bytes` 句柄，零字节拷贝。
    fn merge_accum(acc: &mut Vec<Bytes>, other: Vec<Bytes>) {
        acc.extend(other);
    }

    fn write_header(&mut self) -> Result<Bytes> {
        self.core.write_header()?;
        Ok(self.take_written())
    }

    fn is_header_written(&self) -> bool {
        self.core.is_header_written()
    }

    fn write_frame(&mut self, packet: &mut AVPacket) -> Result<Bytes> {
        self.core.write_frame(packet)?;
        Ok(self.take_written())
    }

    fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<Bytes> {
        self.core.write_interleaved(packet)?;
        Ok(self.take_written())
    }

    fn write_trailer(&mut self) -> Result<Bytes> {
        self.core.write_trailer()?;
        Ok(self.take_written())
    }

    fn output(&self) -> &AVFormatContextOutput {
        self.core.output()
    }

    fn output_mut(&mut self) -> &mut AVFormatContextOutput {
        self.core.output_mut()
    }
}

/// 仅承诺可移动到其他线程独占使用。
unsafe impl Send for BufferWriter {}

////////////////////////////////////////
// IoWriter（任意 std::io::Write 输出）
////////////////////////////////////////

/// Builds a [`IoWriter`].
pub struct IoWriterBuilder<'a, W> {
    writer: W,
    format: &'a str,
    spec: WriterSpec,
}

impl<'a, W: std::io::Write + Send + 'static> IoWriterBuilder<'a, W> {
    /// Create a new writer wrapping any [`std::io::Write`] implementor.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format to use.
    /// * `writer` - Destination stream to write to (e.g. socket, pipe, encryptor).
    pub fn new(format: &'a str, writer: W) -> Self {
        Self {
            writer,
            format,
            spec: WriterSpec::default(),
        }
    }

    impl_writer_builder_setters!();

    /// Build [`IoWriter`].
    pub fn build(self) -> Result<IoWriter<W>> {
        let Self {
            writer,
            format,
            spec,
        } = self;
        let inner = Arc::new(Mutex::new(writer));

        let write_inner = inner.clone();
        // 回调在 FFI 调用栈中执行，严禁 panic：锁中毒或写入失败均返回 AVERROR(EIO)。
        let write_packet: WritePacketCallback = Box::new(move |_opaque, buf: &[u8]| {
            let mut w = match write_inner.lock() {
                Ok(guard) => guard,
                Err(_) => return AVERROR_EIO,
            };
            match w.write_all(buf) {
                Ok(()) => buf.len() as i32,
                Err(e) => {
                    tracing::error!("IoWriter write error: {e}");
                    AVERROR_EIO
                }
            }
        });

        let io_context = AVIOContextCustom::alloc_context(
            AVMem::new(AVIO_BUFFER_SIZE),
            true,
            Vec::new(),
            None,
            Some(write_packet),
            None,
        );
        let output = build_output_custom(io_context, format)?;
        Ok(IoWriter {
            core: WriterCore::new(output, spec.options.and_then(|opts| opts.into_dict()), true),
            inner,
        })
    }
}

/// Video writer that writes to any [`std::io::Write`] implementor.
///
/// 每次 `write_*` 调用后会 flush avio 缓冲，字节及时送达底层流。
pub struct IoWriter<W: std::io::Write + Send + 'static> {
    core: WriterCore,
    inner: Arc<Mutex<W>>,
}

impl<W: std::io::Write + Send + 'static> IoWriter<W> {
    /// Create a video writer wrapping any [`std::io::Write`] implementor.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format to use.
    /// * `writer` - Destination stream to write to.
    #[inline]
    pub fn new(format: &str, writer: W) -> Result<Self> {
        IoWriterBuilder::new(format, writer).build()
    }

    /// 消耗 writer 并取回底层 [`std::io::Write`] 实现（应在 `write_trailer` 之后调用）。
    pub fn into_inner(self) -> std::io::Result<W> {
        let Self { core, inner } = self;
        let WriterCore { output, .. } = core;
        // 先释放 format context（连带 IO 回调释放其持有的 inner 引用）
        drop(output);
        match Arc::try_unwrap(inner) {
            // 锁中毒（回调 panic 过）时仍取回 writer 本体
            Ok(w) => Ok(w
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())),
            // 理论不可达：output 已释放，唯一引用在此
            Err(_) => Err(std::io::Error::other(
                "custom io writer still has live references",
            )),
        }
    }
}

impl<W: std::io::Write + Send + 'static> Writer for IoWriter<W> {
    type Out = ();
    type Accum = ();

    fn merge_out(_acc: &mut (), _out: ()) {}
    fn merge_accum(_acc: &mut (), _other: ()) {}

    fn write_header(&mut self) -> Result<()> {
        self.core.write_header()
    }

    fn is_header_written(&self) -> bool {
        self.core.is_header_written()
    }

    fn write_frame(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.core.write_frame(packet)
    }

    fn write_interleaved(&mut self, packet: &mut AVPacket) -> Result<()> {
        self.core.write_interleaved(packet)
    }

    fn write_trailer(&mut self) -> Result<()> {
        self.core.write_trailer()
    }

    fn output(&self) -> &AVFormatContextOutput {
        self.core.output()
    }

    fn output_mut(&mut self) -> &mut AVFormatContextOutput {
        self.core.output_mut()
    }
}

/// 仅承诺可移动到其他线程独占使用（内部 writer 与回调均为 `Send`）。
unsafe impl<W: std::io::Write + Send + 'static> Send for IoWriter<W> {}

////////////////////////////////////////
// Logging
////////////////////////////////////////

/// Initialize the logging handler. This will redirect all ffmpeg logging to the Rust `tracing`
/// crate and any subscribers to it.
///
/// `level` 也被记下来供回调自行过滤：FFmpeg 的 `av_log_set_level` 只作用于它
/// 自带的默认回调，安装自定义回调后**所有**级别的消息都会送进回调，因此
/// [`AVLogLevel::QUIET`] 必须由回调自己实现（见私有 `log_callback`）。
pub fn init_logging(level: AVLogLevel, flag: AVLogFlag) {
    LOG_LEVEL.store(level as i32, Ordering::Relaxed);
    unsafe {
        ffi::av_log_set_callback(Some(log_callback));
        ffi::av_log_set_level(level as _);
        ffi::av_log_set_flags(flag as _);
    }
}

/// [`init_logging`] 配置的最大输出级别（FFmpeg 数值语义：越大越啰嗦）。
///
/// 默认 `INFO`，与 FFmpeg 自带默认回调的过滤器行为一致。
static LOG_LEVEL: AtomicI32 = AtomicI32::new(AVLogLevel::INFO as i32);

/// Internal function with C-style callback behavior that receives all log messages from ffmpeg and
/// handles them with the `tracing` crate, the Rust way.
///
/// 这里是 `catch_unwind` 边界：`tracing` 的订阅者可能 panic，而 panic 逸出
/// `extern "C"` 边界会 abort 整个进程；宁可丢一条日志，也不能拖垮宿主程序。
///
/// # Arguments
///
/// * `avcl` - Internal struct with log message data.
/// * `level_no` - Log message level integer.
/// * `fmt` - Log message format string.
/// * `vl` - Variable list with format string items.
unsafe extern "C" fn log_callback(
    avcl: *mut std::ffi::c_void,
    level_no: std::ffi::c_int,
    fmt: *const std::ffi::c_char,
    #[cfg(all(target_arch = "x86_64", target_family = "unix"))] vl: *mut ffi::__va_list_tag,
    #[cfg(not(all(target_arch = "x86_64", target_family = "unix")))] vl: ffi::va_list,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: 参数由 FFmpeg 传入，在回调期间保持有效；实现只读取它们。
        unsafe { log_callback_impl(avcl, level_no, fmt, vl) };
    }));
}

/// [`log_callback`] 的实现体，运行在 `catch_unwind` 边界内。
unsafe fn log_callback_impl(
    avcl: *mut std::ffi::c_void,
    level_no: std::ffi::c_int,
    fmt: *const std::ffi::c_char,
    #[cfg(all(target_arch = "x86_64", target_family = "unix"))] vl: *mut ffi::__va_list_tag,
    #[cfg(not(all(target_arch = "x86_64", target_family = "unix")))] vl: ffi::va_list,
) {
    // 先按 `init_logging` 的级别过滤（FFmpeg 装了自定义回调后不再自己过滤，
    // 这一步是 `AVLogLevel::QUIET` 能真正静音的唯一保证）。
    if level_no > LOG_LEVEL.load(Ordering::Relaxed) {
        return;
    }

    // Check whether or not the message would be printed at all.
    let event_would_log = match level_no as u32 {
        // These are all error states.
        ffi::AV_LOG_PANIC | ffi::AV_LOG_FATAL | ffi::AV_LOG_ERROR => {
            tracing::enabled!(tracing::Level::ERROR)
        }
        ffi::AV_LOG_WARNING => tracing::enabled!(tracing::Level::WARN),
        ffi::AV_LOG_INFO => tracing::enabled!(tracing::Level::INFO),
        // There is no "verbose" in `log`, so we just put it in the "debug" category.
        ffi::AV_LOG_VERBOSE | ffi::AV_LOG_DEBUG => tracing::enabled!(tracing::Level::DEBUG),
        ffi::AV_LOG_TRACE => tracing::enabled!(tracing::Level::TRACE),
        _ => {
            return;
        }
    };

    if event_would_log {
        // Allocate some memory for the log line (might be truncated). 1024 bytes is the number used
        // by ffmpeg itself, so it should be mostly fine.
        let mut line = [0; 1024];
        let mut print_prefix: std::ffi::c_int = 1;
        // Use the ffmpeg default formatting.
        let ret = unsafe {
            ffi::av_log_format_line2(
                avcl,
                level_no,
                fmt,
                vl,
                line.as_mut_ptr(),
                (line.len()) as std::ffi::c_int,
                (&mut print_prefix) as *mut std::ffi::c_int,
            )
        };
        // Simply discard the log message if formatting fails.
        if ret > 0
            && let Ok(line) = unsafe { std::ffi::CStr::from_ptr(line.as_mut_ptr()) }.to_str()
        {
            let line = line.trim();
            if log_filter_hacks(line) {
                match level_no as u32 {
                    // These are all error states.
                    ffi::AV_LOG_PANIC | ffi::AV_LOG_FATAL | ffi::AV_LOG_ERROR => {
                        tracing::error!(target: "rsmedia", "{}", line)
                    }
                    ffi::AV_LOG_WARNING => tracing::warn!(target: "rsmedia", "{}", line),
                    ffi::AV_LOG_INFO => tracing::info!(target: "rsmedia", "{}", line),
                    // There is no "verbose" in `log`, so we just put it in the "debug" category.
                    ffi::AV_LOG_VERBOSE | ffi::AV_LOG_DEBUG => {
                        tracing::debug!(target: "rsmedia", "{}", line)
                    }
                    ffi::AV_LOG_TRACE => tracing::trace!(target: "rsmedia", "{}", line),
                    _ => {}
                };
            }
        } else if ret > 0 {
            // 格式化成功但不是合法 UTF-8：`tracing` 的 message 只能是 UTF-8 字符串，
            // 这行字节无法承载。取有损转换会掩盖"日志内容被改写"，因此丢弃并留痕。
            tracing::debug!(
                target: "rsmedia",
                "discarded a non-UTF-8 FFmpeg log line (level {level_no}, {ret} bytes)"
            );
        }
    }
}

/// Helper function to filter out any lines that we don't want to log because they contaminate.
/// Currently, it includes the following log line hacks:
///
/// * **Pelco H264 encoding issue**. Pelco cameras and encoders have a problem with their SEI NALs
///   that causes ffmpeg to complain but does not hurt the stream. It does cause continuous error
///   messages though which we filter out here.
fn log_filter_hacks(line: &str) -> bool {
    /* Hack 1 */
    const HACK_1_PELCO_NEEDLE_1: &str = "SEI type 5 size";
    const HACK_1_PELCO_NEEDLE_2: &str = "truncated at";
    !(line.contains(HACK_1_PELCO_NEEDLE_1) && line.contains(HACK_1_PELCO_NEEDLE_2))
}

/// 当前 FFmpeg 构建支持的**输入**协议名（`file`、`http`、`rtsp`、`rtp`、`tcp`…）。
///
/// 可用于构建前探测能力：URL 的协议不在列表中时，`avformat_open_input` 必然
/// 失败。列表内容取决于 FFmpeg 编译配置。
pub fn input_protocols() -> Vec<String> {
    rsmpeg::avformat::AVIOProtocol::inputs()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// 当前 FFmpeg 构建支持的**输出**协议名。
pub fn output_protocols() -> Vec<String> {
    rsmpeg::avformat::AVIOProtocol::outputs()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// 返回将处理该 URL 的协议名（如 `"file"`、`"http"`），无匹配协议时为 `None`。
pub fn find_protocol_name(url: &str) -> Option<String> {
    // URL 来自调用者：含 NUL 字节的 URL 不可能匹配任何协议，返回 None 不 panic。
    let url_c = strutils::str_to_cstring(url).ok()?;
    rsmpeg::avformat::AVIOProtocol::find_protocol_name(&url_c)
        .map(|p| p.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncoderBuilder;
    use crate::mux::{Demuxer, Muxer};
    use crate::options::Options;
    use crate::pixel::PixelFormat;
    use crate::{DecoderBuilder, MediaType};
    use rsmpeg::avutil::AVFrame;

    /// `ffi_enum!` 生成的位集合能力：`|` 组合与 `Into<i32>` 转换。
    #[test]
    fn test_avseek_flag_bitops() {
        // 注意：rsmpeg 侧的 AVSEEK_FLAG_* 常量类型为 u32，断言时归一化到 i32。
        let combined = AVSeekFlag::BACKWARD | AVSeekFlag::ANY;
        assert_eq!(
            combined,
            ffi::AVSEEK_FLAG_BACKWARD as i32 | ffi::AVSEEK_FLAG_ANY as i32
        );

        // 与 rsmpeg 的 u32 常量混合时需显式 `as i32`（宏只实现 BitOr<i32>，
        // 以免 repr=u32 的旗标枚举生成重复 impl）。
        let mixed = AVSeekFlag::FRAME | (ffi::AVSEEK_FLAG_BYTE as i32);
        assert_eq!(
            mixed,
            ffi::AVSEEK_FLAG_FRAME as i32 | ffi::AVSEEK_FLAG_BYTE as i32
        );

        let raw: i32 = AVSeekFlag::FRAME.into();
        assert_eq!(raw, ffi::AVSEEK_FLAG_FRAME as i32);
        // as_raw 与 Into 结果一致
        assert_eq!(AVSeekFlag::BYTE.as_raw(), i32::from(AVSeekFlag::BYTE));
    }

    /// 生成 RGB24 渐变测试帧（image2 序列写入用）。
    fn generate_rgb_frame(width: usize, height: usize, index: i64) -> AVFrame {
        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(PixelFormat::RGB24.into());
        frame.alloc_buffer().expect("alloc rgb frame buffer");

        let plane = frame.data_mut()[0];
        let linesize = frame.linesize[0];
        for y in 0..height {
            for x in 0..width {
                let i = y * linesize as usize + x * 3;
                unsafe {
                    *plane.add(i) = (x * 255 / width) as u8;
                    *plane.add(i + 1) = (y * 255 / height) as u8;
                    *plane.add(i + 2) = ((index * 25) % 256) as u8;
                }
            }
        }
        frame
    }

    /// 图片序列写入（image2 muxer + png 编码器）：写入 N 帧 => 磁盘上生成
    /// N 个按 `%03d` 模式编号的 PNG 文件。
    ///
    /// 输出目录独占（`images_write`）：image2 会按编号截断重建文件，与其它
    /// 测试共用目录时，并行调度下会互相把对方已写完的文件截成 0 字节。
    #[test]
    fn test_write_image_sequence() -> Result<()> {
        let pattern = crate::test_support::test_output_path("images_write", "img_%03d.png");
        let n_frames = 8;

        let writer = StreamWriterBuilder::new(&pattern)
            .with_format("image2")
            .build()?;
        let mut muxer = Muxer::new_from_writer(writer);

        let encoder = EncoderBuilder::new_video(64, 48)
            .with_codec_name("png".to_string())
            .build()?;
        let video_index = muxer.add_encoder(encoder)?;

        for i in 0..n_frames {
            let mut frame = generate_rgb_frame(64, 48, i);
            frame.set_pts(i);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;

        // 校验每个编号文件都存在且非空（image2 从 start_number=1 开始编号）
        let dir = pattern.parent().unwrap();
        for i in 1..=n_frames {
            let file = dir.join(format!("img_{i:03}.png"));
            let meta = std::fs::metadata(&file)
                .context(format!("expected sequence file {}", file.display()))?;
            assert!(meta.len() > 0, "sequence file {} is empty", file.display());
        }
        // 未写入的下一个编号不应存在
        assert!(!dir.join(format!("img_{:03}.png", n_frames + 1)).exists());

        Ok(())
    }

    /// 图片序列读取（image2 demuxer）：按 `%03d` 模式打开序列，解码帧数应与
    /// 写入帧数一致，且尺寸正确。
    ///
    /// 序列在自己独占的 `images_read` 目录中现场生成：不复用写入测试的产物
    /// —— 那样两个测试会争用同一组文件名（见 [`test_write_image_sequence`]）。
    #[test]
    fn test_read_image_sequence() -> Result<()> {
        let pattern = crate::test_support::test_output_path("images_read", "img_%03d.png");
        {
            let writer = StreamWriterBuilder::new(&pattern)
                .with_format("image2")
                .build()?;
            let mut muxer = Muxer::new_from_writer(writer);
            let encoder = EncoderBuilder::new_video(64, 48)
                .with_codec_name("png".to_string())
                .build()?;
            let video_index = muxer.add_encoder(encoder)?;
            for i in 0..8 {
                let mut frame = generate_rgb_frame(64, 48, i);
                frame.set_pts(i);
                muxer.mux(frame, video_index)?;
            }
            muxer.finish()?;
        }

        let reader = StreamReaderBuilder::new(&pattern)
            .with_format("image2")
            .with_options(Options::from_iter([(
                "framerate".to_string(),
                "5".to_string(),
            )]))
            .build()?;

        let demuxer = Demuxer::new_from_reader(reader, None, None)?;
        let decoded: Vec<_> = demuxer.filter_map(|res| res.ok()).collect();
        assert_eq!(
            decoded.len(),
            8,
            "expected 8 decoded frames from image sequence"
        );
        for (_, frame) in &decoded {
            assert_eq!(frame.width, 64);
            assert_eq!(frame.height, 48);
        }

        Ok(())
    }

    /// 单张图片读取：jpg 由 FFmpeg 自动探测（image2/mjpeg demuxer），应能
    /// 解码出**恰好一帧**，尺寸与非空白像素都对得上源图。
    ///
    /// 断言钉死真值（`assets/cat.jpg` 是 2000×1333）而不是只查 `> 0`：
    /// 只查非零时，任何尺寸的错解码都能过，等于没断言。
    #[test]
    fn test_read_single_image() -> Result<()> {
        let demuxer = Demuxer::new("assets/cat.jpg")?;
        let decoded: Vec<_> = demuxer.filter_map(|res| res.ok()).collect();
        assert_eq!(
            decoded.len(),
            1,
            "a single image must decode to exactly one frame"
        );

        let (_, frame) = &decoded[0];
        assert_eq!(
            (frame.width, frame.height),
            (2000, 1333),
            "decoded geometry must match assets/cat.jpg"
        );

        // 布局无关：平面格式的第 0 个平面是亮度，packed 格式就是整块像素。
        // `Demuxer` 给的是裸 `AVFrame`，第 0 个平面按"行宽 × 高度"取即可满足
        // "非空白"这一个判据（不做逐像素语义断言）。
        // SAFETY: `frame` 刚从解码器取出且未被 unref，`data[0]` 指向该平面
        // `linesize[0] * height` 字节的有效缓冲，切片不越过这块缓冲。
        let plane_len = frame.linesize[0] as usize * frame.height as usize;
        let plane = unsafe { std::slice::from_raw_parts(frame.data[0], plane_len) };
        let min = *plane.iter().min().expect("non-empty frame");
        let max = *plane.iter().max().expect("non-empty frame");
        assert!(
            max - min > 32,
            "the decoded image looks blank: {min}..{max}"
        );

        Ok(())
    }

    /// header 写出后的状态守卫：二次 `write_header` 与 `add_stream` 都必须报
    /// [`RsmediaError::InvalidConfig`]（header 后流数组已固定，再建流是 SIGSEGV 级
    /// 的未定义行为），而不是继续执行。
    #[test]
    fn test_writer_rejects_header_and_add_stream_after_header_written() -> Result<()> {
        let mut writer = BufferWriter::new("mp4")?;
        assert!(!writer.is_header_written());

        let encoder = EncoderBuilder::new_video(64, 48).build()?;
        let index = writer.add_stream(encoder.codecpar(), encoder.time_base())?;
        assert_eq!(index, 0, "first stream must get index 0");

        writer.write_header()?;
        assert!(writer.is_header_written(), "header state must be recorded");

        let err = writer
            .write_header()
            .expect_err("second write_header must be rejected");
        assert!(
            matches!(err, RsmediaError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );

        let err = writer
            .add_stream(encoder.codecpar(), encoder.time_base())
            .expect_err("add_stream after header must be rejected");
        assert!(
            matches!(err, RsmediaError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );
        Ok(())
    }

    /// 内存写入 -> 内存读取 -> 解码 + seek 的完整往返：
    /// 验证 BufferWriter（持久 custom IO）、BufferReader、Seekable 实现协同工作。
    #[test]
    fn test_buffer_writer_reader_roundtrip() -> Result<()> {
        // 1. 编码 mpegts 到 BufferWriter（流式格式）
        let writer = BufferWriter::new("mpegts")?;
        let mut muxer = Muxer::new_from_writer(writer);
        let encoder = EncoderBuilder::new_video(64, 48).build()?;
        let tb = encoder.time_base();
        let video_index = muxer.add_encoder(encoder)?;
        let mut total = 0usize;
        for i in 0..8 {
            let mut frame = generate_rgb_frame(64, 48, i);
            frame.set_pts(i);
            frame.set_time_base(tb);
            let chunks = muxer.mux(frame, video_index)?;
            total += chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
        }
        muxer.finish()?;
        let bytes = muxer.into_writer().into_bytes();
        assert!(!bytes.is_empty(), "buffer writer produced no output");
        assert!(
            bytes.len() >= total,
            "into_bytes should contain at least all incremental chunks"
        );

        // 2. BufferReader 读回并解码全部帧
        let mut reader = BufferReader::new(bytes)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut frames = 0;
        while decoder.decode_raw(&mut reader)?.is_some() {
            frames += 1;
        }
        assert_eq!(frames, 8, "expected 8 decoded frames from buffer reader");

        // 3. seek 到起点后再解一帧（验证 BufferReader 的 Seekable 实现）
        let source = {
            let writer = BufferWriter::new("mpegts")?;
            let mut muxer = Muxer::new_from_writer(writer);
            let encoder = EncoderBuilder::new_video(64, 48).build()?;
            let tb = encoder.time_base();
            let video_index = muxer.add_encoder(encoder)?;
            for i in 0..8 {
                let mut frame = generate_rgb_frame(64, 48, i);
                frame.set_pts(i);
                frame.set_time_base(tb);
                muxer.mux(frame, video_index)?;
            }
            muxer.finish()?;
            muxer.into_writer().into_bytes()
        };
        let mut reader = BufferReader::new(source)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        reader.seek_to_start()?;
        let frame = decoder
            .decode_raw(&mut reader)?
            .expect("expected a frame after seek to start");
        assert!(frame.width > 0 && frame.height > 0);

        Ok(())
    }

    /// Local mp4: `StreamReader`'s underlying IO is byte-seekable, so
    /// `is_byte_seekable` must be `true` and `seek_to_start` must succeed.
    #[test]
    fn test_stream_reader_local_is_byte_seekable() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        assert!(
            reader.is_byte_seekable(),
            "local file IO should report byte-seekable"
        );
        reader.seek_to_start()?;
        Ok(())
    }

    /// In-memory input: `BufferReader` installs a seek callback, so FFmpeg sets
    /// `AVIO_SEEKABLE_NORMAL` and `is_byte_seekable` is always `true`.
    #[test]
    fn test_buffer_reader_is_byte_seekable() -> Result<()> {
        let data = std::fs::read("assets/mp4.mp4")?;
        let mut reader = BufferReader::new(data)?;
        assert!(
            reader.is_byte_seekable(),
            "in-memory reader with a seek callback should report byte-seekable"
        );
        reader.seek_to_start()?;
        Ok(())
    }

    /// IoReader：任意 std::io::Read 流读取媒体。
    ///
    /// 使用 mpegts（流式容器，demux 无需 seek）；moov 在文件尾部的 mp4
    /// 不适用于非 seek 输入。
    #[test]
    fn test_io_reader() -> Result<()> {
        // 先生成一个 mpegts 测试文件
        let path = crate::test_support::test_output_path("io", "rsmedia_io_reader.ts");
        crate::test_support::remove_test_output(&path);
        {
            let mut muxer = Muxer::new(&path)?;
            let encoder = EncoderBuilder::new_video(64, 48).build()?;
            let tb = encoder.time_base();
            let video_index = muxer.add_encoder(encoder)?;
            for i in 0..4 {
                let mut frame = generate_rgb_frame(64, 48, i);
                frame.set_pts(i);
                frame.set_time_base(tb);
                muxer.mux(frame, video_index)?;
            }
            muxer.finish()?;
        }

        let file = std::fs::File::open(&path)?;
        let reader = IoReader::new(std::io::BufReader::new(file))?;
        let demuxer = Demuxer::new_from_reader(reader, None, None)?;
        let n = demuxer.filter_map(|res| res.ok()).count();
        assert!(n > 0, "expected packets from IoReader");

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// IoWriter：写任意 std::io::Write 汇（Vec<u8>），结束后取回数据。
    #[test]
    fn test_io_writer() -> Result<()> {
        let writer = IoWriter::new("mpegts", Vec::new())?;
        let mut muxer = Muxer::new_from_writer(writer);
        let encoder = EncoderBuilder::new_video(64, 48).build()?;
        let tb = encoder.time_base();
        let video_index = muxer.add_encoder(encoder)?;
        for i in 0..4 {
            let mut frame = generate_rgb_frame(64, 48, i);
            frame.set_pts(i);
            frame.set_time_base(tb);
            muxer.mux(frame, video_index)?;
        }
        muxer.finish()?;
        let sink = muxer.into_writer().into_inner()?;
        assert!(!sink.is_empty(), "custom io writer produced no output");

        Ok(())
    }

    /// `file` 协议在任何 FFmpeg 构建中都存在。
    #[test]
    fn test_input_protocols_contain_file() {
        let protocols = input_protocols();
        assert!(
            protocols.iter().any(|p| p == "file"),
            "protocols: {protocols:?}"
        );
    }

    /// URL 的协议识别与枚举结果一致。
    #[test]
    fn test_find_protocol_name() {
        assert_eq!(find_protocol_name("/tmp/a.mp4").as_deref(), Some("file"));
        assert_eq!(
            find_protocol_name("http://example.com/a.mp4").as_deref(),
            Some("http")
        );
    }

    /// `seek_to_timestamp` 真的把读位置前移了：从 2s 处续读得到的帧数，必须明显
    /// 少于从头读完整段（seek 自己"成功"却什么都没移动是最难查的一类问题）。
    ///
    /// 这里必须**自己造一个 GOP 已知的文件**：`assets/mp4.mp4` 的 166 帧只有一个
    /// 关键帧，向前的 BACKWARD seek 无论请求哪个时间点都只能落回第 0 帧。
    #[test]
    fn test_seek_to_timestamp_advances_the_read_position() -> Result<()> {
        let path = crate::test_support::test_output_path("io", "test_seek_advances.mp4");
        {
            let mut muxer = crate::Muxer::new(&path)?;
            let encoder = EncoderBuilder::new_video(64, 64)
                .with_fps(25.0)
                .with_gop_size(10)
                .build()?;
            let index = muxer.add_encoder(encoder)?;
            for frame_index in 0..100i64 {
                let mut frame = AVFrame::new();
                frame.set_width(64);
                frame.set_height(64);
                frame.set_format(PixelFormat::YUV420P.into());
                frame
                    .alloc_buffer()
                    .context("Failed to allocate frame buffer")?;
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        let count_frames = |seek_ms: Option<i64>| -> Result<usize> {
            let mut reader = StreamReader::new(&path)?;
            if let Some(ms) = seek_ms {
                reader.seek_to_timestamp(ms)?;
            }
            let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
            let mut frames = 0usize;
            while decoder.decode::<u8>(&mut reader)?.is_some() {
                frames += 1;
            }
            Ok(frames)
        };

        let all = count_frames(None)?;
        let from_two_seconds = count_frames(Some(2_000))?;
        assert_eq!(all, 100, "the whole 4s stream should decode 100 frames");
        assert!(
            from_two_seconds < all,
            "seeking to 2s then reading gave the same frame count as reading from the start \
             ({from_two_seconds} vs {all}): the seek did not move the read position"
        );
        assert!(
            from_two_seconds > 0,
            "the second half of the stream must still decode"
        );

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// `seek_to_timestamp` 的**落点精度**：全 I 帧文件（gop=1，每帧都是关键帧，
    /// BACKWARD seek 无回退余地）上，seek 到任意帧边界必须精确落在该帧。
    ///
    /// 上面的 `test_seek_to_timestamp_advances_the_read_position` 只断言
    /// "位置移动了"，抓不住落点漂移类回归（曾因此漏检）。
    #[test]
    fn test_seek_to_timestamp_lands_exactly_on_all_intra() -> Result<()> {
        const FPS: f64 = 25.0;
        const FRAMES: i64 = 60;
        let path = crate::test_support::test_output_path("io", "test_seek_exact.mp4");
        {
            let mut muxer = crate::Muxer::new(&path)?;
            let encoder = EncoderBuilder::new_video(64, 64)
                .with_fps(FPS as f32)
                .with_gop_size(1)
                .build()?;
            let index = muxer.add_encoder(encoder)?;
            for frame_index in 0..FRAMES {
                let mut frame = AVFrame::new();
                frame.set_width(64);
                frame.set_height(64);
                frame.set_format(PixelFormat::YUV420P.into());
                frame
                    .alloc_buffer()
                    .context("Failed to allocate frame buffer")?;
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        // 步长采样覆盖首/中/尾帧边界；目标时间 = 第 n 帧起点。
        for n in (0..FRAMES).step_by(7).chain(std::iter::once(FRAMES - 1)) {
            let target_ms = (n as f64 / FPS * 1000.0).round() as i64;
            let mut reader = StreamReader::new(&path)?;
            reader.seek_to_timestamp(target_ms)?;
            let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
            let frame = decoder
                .decode::<u8>(&mut reader)?
                .ok_or_else(|| RsmediaError::msg("no frame decoded after the seek"))?;
            let tb = reader.input().streams()[decoder.stream_index()].time_base;
            let landed_secs = frame.pts as f64 * tb.num as f64 / tb.den as f64;
            let expected_secs = n as f64 / FPS;
            assert!(
                (landed_secs - expected_secs).abs() <= 0.5 / FPS + 1e-6,
                "seek to {target_ms}ms landed at {landed_secs:.4}s, \
                 expected frame {n} at {expected_secs:.4}s"
            );
        }

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// `seek_to_frame` 的参数是 `impl Into<i32>`：既要吃单个 `AVSeekFlag`，也要吃
    /// `|` 组合出来的**裸掩码**（字段枚举装不下 `BACKWARD | ANY` 这种无名组合）。
    ///
    /// 目标正好是帧边界，所以 `BACKWARD | ANY` 与两者单独使用都落在同一帧。
    #[test]
    fn test_seek_to_frame_accepts_flag_masks() -> Result<()> {
        const FPS: f64 = 25.0;
        const FRAMES: i64 = 60;
        const TARGET: i64 = 30;
        let path = crate::test_support::test_output_path("io", "test_seek_flags.mp4");
        {
            let mut muxer = crate::Muxer::new(&path)?;
            let encoder = EncoderBuilder::new_video(64, 64)
                .with_fps(FPS as f32)
                .with_gop_size(1)
                .build()?;
            let index = muxer.add_encoder(encoder)?;
            for frame_index in 0..FRAMES {
                let mut frame = AVFrame::new();
                frame.set_width(64);
                frame.set_height(64);
                frame.set_format(PixelFormat::YUV420P.into());
                frame
                    .alloc_buffer()
                    .context("Failed to allocate frame buffer")?;
                frame.set_pts(frame_index);
                muxer.mux(frame, index)?;
            }
            muxer.finish()?;
        }

        // 单位：`seek_to_frame` 的 ts 是**流时间基**（`seek_to_timestamp` 才是
        // AV_TIME_BASE 微秒），所以先读流的 tb 再换算，别拿 TIME_BASE 直接算。
        let tb = StreamReader::new(&path)?.input().streams()[0].time_base;
        let target_ts =
            (TARGET as f64 / FPS * f64::from(tb.den) / f64::from(tb.num)).round() as i64;
        assert!(target_ts > 0, "target_ts 计算异常：{target_ts}");
        // 组合掩码、单个旗标、裸 i32 掩码三条路径，都必须落在第 TARGET 帧。
        let cases: [(&str, i32); 3] = [
            (
                "AVSeekFlag::BACKWARD | AVSeekFlag::ANY",
                AVSeekFlag::BACKWARD | AVSeekFlag::ANY,
            ),
            ("AVSeekFlag::ANY", AVSeekFlag::ANY.into()),
            (
                "裸 i32 掩码（AVSeekFlag::ANY.as_raw()）",
                AVSeekFlag::ANY.as_raw(),
            ),
        ];
        for (what, flags) in cases {
            let mut reader = StreamReader::new(&path)?;
            reader
                .seek_to_frame(0, target_ts, flags)
                .with_context(|| format!("seek with {what}"))?;
            let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
            let frame = decoder
                .decode::<u8>(&mut reader)?
                .ok_or_else(|| RsmediaError::msg("no frame decoded after the seek"))?;
            let tb = reader.input().streams()[decoder.stream_index()].time_base;
            let landed = frame.pts as f64 * tb.num as f64 / tb.den as f64;
            let expected = TARGET as f64 / FPS;
            assert!(
                (landed - expected).abs() <= 0.5 / FPS + 1e-6,
                "{what}: 落在 {landed:.4}s, 期望第 {TARGET} 帧 {expected:.4}s"
            );
        }

        crate::test_support::remove_test_output(&path);
        Ok(())
    }

    /// `seek_to_frame` 是**可失败**的：不存在的流索引必须报错，而不是静默不动。
    /// （`av_seek_frame` 会越界解引用 `AVStream`，所以这层校验不是可选的。）
    #[test]
    fn test_seek_to_frame_rejects_an_unknown_stream() -> Result<()> {
        let mut reader = StreamReader::new("assets/mp4.mp4")?;
        assert!(
            reader.seek_to_frame(99, 0, AVSeekFlag::BACKWARD).is_err(),
            "seeking a stream that does not exist must fail"
        );
        Ok(())
    }

    /// 中断句柄：新建时未触发，`abort` 立即触发，零超时也立即触发；
    /// 且**已触发的句柄必须真的拦得下打开与读包**。
    ///
    /// 曾经的实现只在 `avformat_open_input` 之后把回调写到 format context 上，
    /// 而 FFmpeg 在打开时会把回调**按值拷贝**进协议层（`ffurl_alloc`），
    /// `av_read_frame` 自己也不查回调，于是协议层的阻塞读永远看不到它——本测试
    /// 当时只能断言"装配成功，无法验证中止效果"。现在带 `with_interrupt` 的 URL
    /// 输入走 `open_input_with_interrupt`（回调先于 open 安装），已触发的句柄会在
    /// 打开阶段的探测读上直接拿到 AVERROR_EXIT。
    #[test]
    fn test_interrupt_abort_and_timeout() -> Result<()> {
        let fresh = Interrupt::new();
        assert!(!fresh.triggered(), "a fresh handle must not be triggered");

        let expired = Interrupt::new();
        expired.set_timeout(std::time::Duration::ZERO);
        assert!(expired.triggered(), "a zero timeout fires immediately");

        // 未触发的句柄对读取完全透明。
        let mut reader = StreamReaderBuilder::new("assets/mp4.mp4")
            .with_interrupt(fresh)
            .build()?;
        assert!(
            reader.read_packet()?.is_some(),
            "a fresh interrupt must not disturb reading"
        );

        // 已触发的句柄：打开阶段的探测读就会被协议层拦下。
        let aborted = Interrupt::new();
        aborted.abort();
        assert!(aborted.triggered(), "abort must trigger");

        match StreamReaderBuilder::new("assets/mp4.mp4")
            .with_interrupt(aborted)
            .build()
        {
            Err(RsmediaError::FFmpeg(RsmpegError::OpenInputError(_))) => {}
            Err(other) => {
                panic!("an aborted interrupt must fail opening with OpenInputError, got {other:?}")
            }
            Ok(mut reader) => {
                // 兜底：若某天该输入在打开阶段不经过协议层读，那么读包必须立刻被拦。
                let mut read = 0usize;
                loop {
                    match reader.read_packet() {
                        Ok(Some(_)) => read += 1,
                        Ok(None) => panic!(
                            "an aborted interrupt took no effect: {read} packets read through"
                        ),
                        Err(_) => break,
                    }
                }
            }
        }
        Ok(())
    }

    /// 输出协议枚举与输入协议对称（至少都包含 `file`）。
    #[test]
    fn test_output_protocols_contain_file() {
        assert!(
            output_protocols().iter().any(|name| name == "file"),
            "the `file` output protocol must be available"
        );
    }
}
