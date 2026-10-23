//! Global FFmpeg initialization and logging configuration.
//!
//! Logging configuration is resolved with the following priority (first match
//! wins), so that `rsmedia::init()` works with zero configuration yet stays
//! fully controllable:
//!
//! 1. **`RUST_LOG`** (env_logger syntax) — selects the FFmpeg log level:
//!    a directive targeting `rsmedia` (`rsmedia=debug`) wins over a bare
//!    global level (`debug`); unknown directives are ignored.
//! 2. **`RUST_BACKTRACE`** — set to `1`/`full` while debugging, the level is
//!    floored at [`AVLogLevel::DEBUG`] so FFmpeg internals show up too.
//! 3. **Default** — [`AVLogLevel::INFO`] with [`AVLogFlag::SKIP_REPEATED`],
//!    matching FFmpeg's own out-of-the-box behavior.
//!
//! Callers who want full control without consulting the environment use
//! [`init_with_level`] / [`init_with`]. All entry points are idempotent: the
//! first call wins, later calls are no-ops (the AVLog callback is global).
//!
//! Every entry point also registers all libavdevice devices
//! (`avdevice_register_all`), so input/output devices such as `lavfi` are
//! discoverable without any extra FFI on the caller's side.

use crate::error::{Result, RsmediaError};
use crate::flags::FlagSet;
use std::sync::atomic::{AtomicI32, Ordering};

use once_cell::sync::OnceCell;
use rsmpeg::ffi;

static INIT: OnceCell<()> = OnceCell::new();

impl Default for AVLogLevel {
    /// FFmpeg 自身的默认级别：普通信息可见，噪音不可见。
    fn default() -> Self {
        Self::INFO
    }
}

impl Default for AVLogFlag {
    /// The crate's default logging *policy*: skip repeated log lines, matching FFmpeg's own
    /// out-of-the-box behaviour.
    ///
    /// Note this is a policy for one specific flag, not "the empty set" — a freshly created
    /// [`FlagSet<AVLogFlag>`](crate::FlagSet) is empty by [`Default`] as any set would be, and
    /// is therefore *not* the same value. Use this impl when you want FFmpeg's default
    /// behaviour; use [`FlagSet::EMPTY`](crate::FlagSet::EMPTY) when you want no flags at all.
    fn default() -> Self {
        Self::SKIP_REPEATED
    }
}

/// Initialize global FFmpeg settings with the most user-friendly resolution.
///
/// Configuration priority: `RUST_LOG` → `RUST_BACKTRACE` floor → defaults
/// (see the [module docs](self) for the exact mapping). Equivalent to
/// `init_with` with whatever the environment resolves to. Idempotent.
///
/// ```no_run
/// rsmedia::init().unwrap();
/// // RUST_LOG=debug cargo run … 即可看到 FFmpeg 内部日志
/// ```
pub fn init() -> Result<()> {
    let mut level = level_from_env().unwrap_or_default();
    // 调试姿态：开了 RUST_BACKTRACE 的会话里，FFmpeg 内部日志至少到 Debug，
    // 让 `RUST_BACKTRACE=1 cargo run` 一个变量同时拿到回溯与内部日志。
    if backtrace_enabled() && level < AVLogLevel::DEBUG {
        level = AVLogLevel::DEBUG;
    }
    init_with(level, AVLogFlag::default())
}

/// Initialize with an explicit level; flags default to [`AVLogFlag::SKIP_REPEATED`].
///
/// Does **not** consult `RUST_LOG`/`RUST_BACKTRACE` — the argument wins.
/// Idempotent.
pub fn init_with_level(level: AVLogLevel) -> Result<()> {
    init_with(level, AVLogFlag::default())
}

/// Initialize with full control over level and flags; the environment is not
/// consulted. Idempotent — the first call wins.
///
/// 位集按 [`FlagSet<AVLogFlag>`](crate::FlagSet) 给出：单个标志
/// （[`AVLogFlag::SKIP_REPEATED`]）或用 `|` 组合出的位集
/// （`AVLogFlag::SKIP_REPEATED | AVLogFlag::PRINT_LEVEL`）都能直接传。
///
/// Also registers all libavdevice devices (see the [module docs](self)).
pub fn init_with(level: AVLogLevel, flag: impl Into<FlagSet<AVLogFlag>>) -> Result<()> {
    INIT.get_or_try_init(|| {
        // 1. Register all libavdevice devices.
        // SAFETY: `avdevice_register_all` has no preconditions and is idempotent/thread-safe
        // on the FFmpeg side (`avpriv_register_devices` stores the static lists with atomic
        // stores); keeping it inside the once-guard keeps all init work under the documented
        // "first call wins" semantics.
        unsafe { ffi::avdevice_register_all() };

        // 2. Redirect logging to the Rust logging facade.
        init_logging(level, flag);
        Ok::<(), RsmediaError>(())
    })?;

    Ok(())
}

/// FFmpeg log level from the `RUST_LOG` environment variable, if it carries
/// anything this library can use. Malformed values never fail init — they are
/// just ignored, falling through to the next source.
fn level_from_env() -> Option<AVLogLevel> {
    let rust_log = std::env::var_os("RUST_LOG")?;
    level_from_directives(rust_log.to_str()?)
}

/// Parse env_logger-style directives into the FFmpeg level that applies to
/// this library's single `rsmedia` log channel.
///
/// * `rsmedia=LEVEL` (or `rsmedia::…=LEVEL`) beats a bare global level;
/// * the last matching directive wins (env_logger semantics);
/// * directives for other targets and unknown levels are ignored.
fn level_from_directives(directives: &str) -> Option<AVLogLevel> {
    let mut global: Option<AVLogLevel> = None;
    let mut rsmedia: Option<AVLogLevel> = None;
    for directive in directives
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
    {
        let (target, level_token) = match directive.split_once('=') {
            Some((target, level)) => (Some(target.trim()), level.trim()),
            None => (None, directive),
        };
        let Some(level) = parse_level(level_token) else {
            continue;
        };
        match target {
            Some(target) if target == "rsmedia" || target.starts_with("rsmedia::") => {
                rsmedia = Some(level);
            }
            None => global = Some(level),
            _ => {}
        }
    }
    rsmedia.or(global)
}

/// `RUST_BACKTRACE`-style value: anything non-empty other than `"0"` means
/// "the user is debugging".
fn backtrace_enabled() -> bool {
    match std::env::var_os("RUST_BACKTRACE") {
        Some(value) => match value.to_str() {
            Some(value) => !value.is_empty() && value != "0",
            None => false,
        },
        None => false,
    }
}

/// Map an env_logger level token onto the FFmpeg level.
fn parse_level(token: &str) -> Option<AVLogLevel> {
    let level = match token.to_ascii_lowercase().as_str() {
        "off" | "quiet" => AVLogLevel::QUIET,
        "error" => AVLogLevel::ERROR,
        "warn" | "warning" => AVLogLevel::WARNING,
        "info" => AVLogLevel::INFO,
        "debug" => AVLogLevel::DEBUG,
        "trace" => AVLogLevel::TRACE,
        _ => return None,
    };
    Some(level)
}

ffi_enum!(
    #[allow(non_camel_case_types)]
    AVLogLevel, i32 {
        QUIET => ffi::AV_LOG_QUIET;
        PANIC => ffi::AV_LOG_PANIC;
        FATAL => ffi::AV_LOG_FATAL;
        ERROR => ffi::AV_LOG_ERROR;
        WARNING => ffi::AV_LOG_WARNING;
        INFO => ffi::AV_LOG_INFO;
        VERBOSE => ffi::AV_LOG_VERBOSE;
        DEBUG => ffi::AV_LOG_DEBUG;
        TRACE => ffi::AV_LOG_TRACE;
        MAX_OFFSET => ffi::AV_LOG_MAX_OFFSET;
    }
);

impl AVLogLevel {
    /// FFmpeg 数值语义的啰嗦程度：级别值随啰嗦程度单调递增
    /// （`AV_LOG_QUIET = -8` 最安静，`AV_LOG_TRACE` 最啰嗦）。
    fn severity(self) -> i32 {
        self as i32
    }
}

/// 啰嗦程度排序：`QUIET < PANIC < … < DEBUG < TRACE`。
///
/// 手写而非宏派生：宏派生在本工具链上按判别值比较，对含负值的级别不可靠。
impl Ord for AVLogLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.severity().cmp(&other.severity())
    }
}

impl PartialOrd for AVLogLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

ffi_enum!(
     #[allow(non_camel_case_types)]
    AVLogFlag, u32 {
        SKIP_REPEATED => ffi::AV_LOG_SKIP_REPEATED;
        PRINT_LEVEL => ffi::AV_LOG_PRINT_LEVEL;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        PRINT_TIME => ffi::AV_LOG_PRINT_TIME;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        PRINT_DATETIME => ffi::AV_LOG_PRINT_DATETIME;
    }
);

////////////////////////////////////////
// Logging
////////////////////////////////////////

/// Initialize the logging handler. This will redirect all ffmpeg logging to the Rust `tracing`
/// crate and any subscribers to it.
///
/// `level` 是会被真正执行的过滤门槛：FFmpeg 的 `av_log_set_level` 只作用于它
/// 自带的默认回调，装上自定义回调后**所有**级别的消息都会送进来，因此本函数
/// 记下 `level` 并在回调里自行丢弃更啰嗦的消息——包括 [`AVLogLevel::QUIET`]。
///
/// `flag` 是 [`AVLogFlag`] 位集，按 [`FlagSet<AVLogFlag>`](crate::FlagSet) 给出（单个标志或 `|` 组合）。
pub fn init_logging(level: AVLogLevel, flag: impl Into<FlagSet<AVLogFlag>>) {
    let flag = flag.into();
    LOG_LEVEL.store(level as i32, Ordering::Relaxed);
    // SAFETY: all three are process-global setters with no preconditions; `log_callback` is an
    // `extern "C"` fn whose signature is exactly the one FFmpeg calls it through.
    unsafe {
        ffi::av_log_set_callback(Some(log_callback));
        ffi::av_log_set_level(level as _);
        ffi::av_log_set_flags(flag.bits() as _);
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
        // SAFETY: FFmpeg passes `avcl`/`fmt`/`vl` per its `av_log` callback contract; `line` is
        // a live 1024-byte buffer and `print_prefix` a live local, both written by the call and
        // neither retained.
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
        // SAFETY: on success (`ret > 0`) FFmpeg NUL-terminated `line`, so reading it as a C
        // string is in-bounds.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_level() {
        assert_eq!(parse_level("debug"), Some(AVLogLevel::DEBUG));
        assert_eq!(parse_level("WARN"), Some(AVLogLevel::WARNING));
        assert_eq!(parse_level("warning"), Some(AVLogLevel::WARNING));
        assert_eq!(parse_level("off"), Some(AVLogLevel::QUIET));
        assert_eq!(parse_level("verbose"), None);
        assert_eq!(parse_level(""), None);
    }

    #[test]
    fn test_bare_global_level() {
        assert_eq!(level_from_directives("debug"), Some(AVLogLevel::DEBUG));
        assert_eq!(
            level_from_directives(" warn , foo "),
            Some(AVLogLevel::WARNING)
        );
    }

    #[test]
    fn test_rsmedia_directive_beats_global() {
        assert_eq!(
            level_from_directives("warn,rsmedia=trace"),
            Some(AVLogLevel::TRACE)
        );
        assert_eq!(
            level_from_directives("rsmedia::io=error,debug"),
            Some(AVLogLevel::ERROR)
        );
    }

    #[test]
    fn test_last_matching_directive_wins() {
        assert_eq!(
            level_from_directives("rsmedia=error,rsmedia=trace"),
            Some(AVLogLevel::TRACE)
        );
        assert_eq!(level_from_directives("debug,info"), Some(AVLogLevel::INFO));
    }

    #[test]
    fn test_other_targets_and_garbage_are_ignored() {
        assert_eq!(
            level_from_directives("tokio=trace,hyper=debug"),
            None,
            "other targets must not leak into rsmedia's channel"
        );
        assert_eq!(level_from_directives("nonsense"), None);
        assert_eq!(level_from_directives("rsmedia=nonsense"), None);
        assert_eq!(level_from_directives(""), None);
    }

    #[test]
    fn test_defaults_match_ffmpeg() {
        assert_eq!(AVLogLevel::default(), AVLogLevel::INFO);
        assert_eq!(AVLogFlag::default(), AVLogFlag::SKIP_REPEATED);
    }

    #[test]
    fn test_level_ordering_matches_verbosity() {
        // 数值语义：级别越大越啰嗦，set_level 打印 <= 自身的消息。
        // 关键回归：QUIET 的 FFmpeg 值是 -8（负值），排序必须保留负号。
        assert_eq!(AVLogLevel::QUIET.severity(), -8);
        assert!(AVLogLevel::QUIET < AVLogLevel::PANIC);
        assert!(AVLogLevel::QUIET < AVLogLevel::INFO);
        assert!(AVLogLevel::INFO < AVLogLevel::DEBUG);
        assert!(AVLogLevel::DEBUG < AVLogLevel::TRACE);
        assert!(AVLogLevel::TRACE < AVLogLevel::MAX_OFFSET);
    }
}
