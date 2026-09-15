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

use crate::error::{Result, RsmediaError};
use crate::io::init_logging;

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
    /// FFmpeg 自身的默认行为：跳过重复的日志行。
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
pub fn init_with(level: AVLogLevel, flag: AVLogFlag) -> Result<()> {
    INIT.get_or_try_init(|| {
        // Redirect logging to the Rust logging facade.
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
    /// FFmpeg 数值语义的啰嗦程度。
    ///
    /// 级别值随啰嗦程度单调递增（`AV_LOG_QUIET = -8` 最安静，`AV_LOG_TRACE`
    /// 最啰嗦）；用 `i32` 作判别类型后判别值本身就是正确的排序键。
    fn severity(self) -> i32 {
        self as i32
    }
}

/// 啰嗦程度排序：`QUIET < PANIC < … < DEBUG < TRACE`。
///
/// 手写而非宏派生：位标志类枚举（[`AVLogFlag`]）根本不该有顺序，宏派生在
/// 本工具链上按判别值比较，对负级别并不可靠——这正是本枚举选择 `i32` 的原因。
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
        // 关键回归：QUIET 的 FFmpeg 值是 -8，i32 判别值直接保留负号；
        // 若误用 u32 判别（或按回绕值派生），QUIET 会变成"最啰嗦"。
        assert_eq!(AVLogLevel::QUIET.severity(), -8);
        assert!(AVLogLevel::QUIET < AVLogLevel::PANIC);
        assert!(AVLogLevel::QUIET < AVLogLevel::INFO);
        assert!(AVLogLevel::INFO < AVLogLevel::DEBUG);
        assert!(AVLogLevel::DEBUG < AVLogLevel::TRACE);
        assert!(AVLogLevel::TRACE < AVLogLevel::MAX_OFFSET);
    }
}
