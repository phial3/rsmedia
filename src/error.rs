//! Unified error type for the rsmedia library.
//!
//! All public APIs return [`Result`] with [`RsmediaError`], so downstream code
//! can match on error variants instead of string-matching `anyhow` messages.
//!
//! ```no_run
//! use rsmedia::{RsmediaError, Result};
//!
//! fn handle(res: Result<()>) {
//!     match res {
//!         Err(RsmediaError::CodecNotFound(name)) => eprintln!("codec unavailable: {name}"),
//!         Err(e) => eprintln!("error: {e}"),
//!         Ok(()) => {}
//!     }
//! }
//! ```

use std::error::Error as StdError;

use rsmpeg::error::RsmpegError;

/// Unified error type for all rsmedia APIs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RsmediaError {
    /// Underlying FFmpeg (rsmpeg) error.
    #[error("FFmpeg error: {0}")]
    FFmpeg(#[from] RsmpegError),
    /// I/O error, typically from custom AVIO callbacks or file access.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The requested codec/encoder/decoder does not exist in this FFmpeg build.
    #[error("codec not found in this FFmpeg build: '{0}'")]
    CodecNotFound(String),
    /// The requested container format does not exist in this FFmpeg build.
    #[error("format not found in this FFmpeg build: '{0}'")]
    FormatNotFound(String),
    /// The operation is not supported on this platform, build or codec.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// Invalid or contradictory configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Any other error with a human readable message.
    #[error("{0}")]
    Other(String),
    /// An opaque error bubbled up from a third-party dependency (ndarray,
    /// yuv, …) with its original source preserved.
    ///
    /// Attach human-readable context on top with [`Context::context`]; the
    /// root cause stays downcastable (`error.root().source()`) for anyhow /
    /// eyre users.
    #[error("{0}")]
    External(#[source] Box<dyn StdError + Send + Sync + 'static>),
    /// An error with additional context attached (produced by [`Context`]).
    #[error("{context}: {source}")]
    Context {
        context: String,
        #[source]
        source: Box<RsmediaError>,
    },
}

impl RsmediaError {
    /// Attach a context message to this error, preserving the source chain.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        // Nested contexts collapse into a single message chain: keep the
        // innermost error and join the context strings outermost-first.
        if let RsmediaError::Context {
            context: inner,
            source,
        } = self
        {
            self = *source;
            self.with_context(format!("{}: {inner}", context.into()))
        } else {
            RsmediaError::Context {
                context: context.into(),
                source: Box::new(self),
            }
        }
    }

    /// Build an [`RsmediaError::Other`] from any displayable value.
    pub fn msg(msg: impl Into<String>) -> Self {
        RsmediaError::Other(msg.into())
    }

    /// Build an [`RsmediaError::CodecNotFound`].
    pub fn codec_not_found(name: impl Into<String>) -> Self {
        RsmediaError::CodecNotFound(name.into())
    }

    /// Build an [`RsmediaError::FormatNotFound`].
    pub fn format_not_found(name: impl Into<String>) -> Self {
        RsmediaError::FormatNotFound(name.into())
    }

    /// Build an [`RsmediaError::Unsupported`].
    pub fn unsupported(reason: impl Into<String>) -> Self {
        RsmediaError::Unsupported(reason.into())
    }

    /// Build an [`RsmediaError::InvalidConfig`].
    pub fn invalid_config(reason: impl Into<String>) -> Self {
        RsmediaError::InvalidConfig(reason.into())
    }

    /// Peel off all [`Context`] wrappers and return the root error, so callers
    /// can match on the originating variant even when the error passed through
    /// several `context(...)` layers.
    pub fn root(&self) -> &Self {
        let mut current = self;
        while let RsmediaError::Context { source, .. } = current {
            current = source;
        }
        current
    }

    /// Whether the root cause is a codec/encoder/decoder missing from this
    /// FFmpeg build ([`RsmediaError::CodecNotFound`]) — e.g. a distro build
    /// without libx264. Use this to skip gracefully instead of matching
    /// error strings.
    pub fn is_codec_not_found(&self) -> bool {
        matches!(self.root(), RsmediaError::CodecNotFound(_))
    }

    /// Whether the root cause is a container format missing from this FFmpeg
    /// build ([`RsmediaError::FormatNotFound`]).
    pub fn is_format_not_found(&self) -> bool {
        matches!(self.root(), RsmediaError::FormatNotFound(_))
    }

    /// Whether the root cause is an operation unsupported on this platform,
    /// build or environment ([`RsmediaError::Unsupported`]) — e.g. no GPU
    /// device is available.
    pub fn is_unsupported(&self) -> bool {
        matches!(self.root(), RsmediaError::Unsupported(_))
    }

    /// Whether the root cause is an invalid or contradictory configuration
    /// ([`RsmediaError::InvalidConfig`]).
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.root(), RsmediaError::InvalidConfig(_))
    }
}

impl From<std::ffi::NulError> for RsmediaError {
    fn from(e: std::ffi::NulError) -> Self {
        RsmediaError::InvalidConfig(format!(
            "string passed to FFmpeg contains interior NUL byte: {e}"
        ))
    }
}

impl From<std::str::Utf8Error> for RsmediaError {
    fn from(e: std::str::Utf8Error) -> Self {
        RsmediaError::Other(format!("FFmpeg returned a non-UTF8 string: {e}"))
    }
}

impl From<ndarray::ShapeError> for RsmediaError {
    fn from(e: ndarray::ShapeError) -> Self {
        RsmediaError::External(Box::new(e))
    }
}

impl From<yuv::YuvError> for RsmediaError {
    fn from(e: yuv::YuvError) -> Self {
        RsmediaError::External(Box::new(e))
    }
}

#[cfg(feature = "image")]
impl From<image::ImageError> for RsmediaError {
    fn from(e: image::ImageError) -> Self {
        RsmediaError::Other(format!("image processing error: {e}"))
    }
}

/// Library-level `Result` alias used by all public APIs.
pub type Result<T> = std::result::Result<T, RsmediaError>;

/// Internal convenience macro: build an [`RsmediaError::Other`] with
/// `format!`-style arguments (migrates `anyhow!` call sites).
macro_rules! format_err {
    ($($arg:tt)*) => {
        $crate::error::RsmediaError::Other(format!($($arg)*))
    };
}
pub(crate) use format_err;

/// A drop-in replacement for `anyhow::Context`, implemented for `Option` and
/// `Result` whose error converts into [`RsmediaError`].
pub trait Context<T> {
    /// Attach a context message to the error (or to a missing `Option` value).
    fn context<C: Into<String>>(self, context: C) -> Result<T>;

    /// Lazily attach a context message to the error (or to a missing value).
    fn with_context<C: Into<String>, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    fn context<C: Into<String>>(self, context: C) -> Result<T> {
        self.ok_or_else(|| RsmediaError::Other(context.into()))
    }

    fn with_context<C: Into<String>, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| RsmediaError::Other(f().into()))
    }
}

impl<T, E> Context<T> for std::result::Result<T, E>
where
    E: Into<RsmediaError>,
{
    fn context<C: Into<String>>(self, context: C) -> Result<T> {
        self.map_err(|e| e.into().with_context(context))
    }

    fn with_context<C: Into<String>, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| e.into().with_context(f()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_variants() {
        assert_eq!(
            RsmediaError::codec_not_found("libx265").to_string(),
            "codec not found in this FFmpeg build: 'libx265'"
        );
        assert_eq!(
            RsmediaError::format_not_found("mkv").to_string(),
            "format not found in this FFmpeg build: 'mkv'"
        );
        assert_eq!(
            RsmediaError::unsupported("qsv on this platform").to_string(),
            "unsupported operation: qsv on this platform"
        );
    }

    #[test]
    fn test_context_chain() {
        let err: std::result::Result<(), RsmpegError> = Err(RsmpegError::DecoderDrainError);
        let wrapped = err.context("Failed to drain decoder").unwrap_err();
        assert_eq!(
            wrapped.to_string(),
            "Failed to drain decoder: FFmpeg error: Decoder have no frame currently, Try send new input."
        );
        // source() must walk down to the rsmpeg error.
        let src = wrapped.source().expect("context has a source");
        assert_eq!(
            src.to_string(),
            "FFmpeg error: Decoder have no frame currently, Try send new input."
        );
    }

    #[test]
    fn test_context_option() {
        let missing: Option<u8> = None;
        assert_eq!(
            missing.context("no device found").unwrap_err().to_string(),
            "no device found"
        );
    }

    #[test]
    fn test_nested_context_joins() {
        let err = RsmediaError::msg("inner failure");
        let wrapped = err.with_context("outer").with_context("outermost");
        assert_eq!(wrapped.to_string(), "outermost: outer: inner failure");
    }

    /// 每个 `is_*` 谓词都要认出自己的变体，且**穿透 context 链**识别根因——
    /// 新增变体时这张表会跟着漏掉，所以逐条对着变体列出来。
    #[test]
    fn test_is_predicates_track_their_variants() {
        for (error, test) in [
            (
                RsmediaError::format_not_found("nope"),
                RsmediaError::is_format_not_found as fn(&RsmediaError) -> bool,
            ),
            (
                RsmediaError::codec_not_found("nope"),
                RsmediaError::is_codec_not_found as fn(&RsmediaError) -> bool,
            ),
            (
                RsmediaError::unsupported("nope"),
                RsmediaError::is_unsupported as fn(&RsmediaError) -> bool,
            ),
            (
                RsmediaError::invalid_config("nope"),
                RsmediaError::is_invalid_config as fn(&RsmediaError) -> bool,
            ),
        ] {
            assert!(
                test(&error),
                "{error:?} must be recognised by its predicate"
            );
            // 加了 context 之后根因仍是同一个，谓词必须继续生效。
            let wrapped = error.with_context("outer");
            assert!(
                test(&wrapped),
                "context must not hide the root cause: {wrapped:?}"
            );
        }
    }

    /// `RsmediaError` 必须保持 `Send + Sync + 'static`，否则无法用 `?`
    /// 转进 `anyhow::Error` / `eyre::Report`，也无法跨线程传递。
    #[test]
    fn test_error_is_send_sync_static() {
        fn assert_bounds<T: Send + Sync + 'static>() {}
        assert_bounds::<RsmediaError>();
    }

    /// anyhow 互操作：`?` 自动转换 + 源链完整可下钻到原始错误。
    #[test]
    fn test_anyhow_interop_preserves_chain() {
        fn fallible() -> anyhow::Result<()> {
            // 元素数与形状不符 → 稳定触发 ShapeError。
            let shape_error: std::result::Result<(), ndarray::ShapeError> =
                ndarray::Array2::<u8>::from_shape_vec((2, 2), vec![0; 3]).map(|_| ());
            shape_error.context("building a frame")?;
            Ok(())
        }

        let report = fallible().unwrap_err();
        // 逐层下钻：anyhow → Context → External → ndarray::ShapeError。
        let rsmedia_err = report
            .downcast_ref::<RsmediaError>()
            .expect("auto-converted");
        let external = match rsmedia_err.root() {
            RsmediaError::External(source) => source,
            other => panic!("unexpected root: {other:?}"),
        };
        external
            .downcast_ref::<ndarray::ShapeError>()
            .expect("ndarray source preserved");
        assert!(report.to_string().contains("building a frame"));
    }

    /// 反向边界：`anyhow::Error` 装箱进 [`RsmediaError::External`]。
    ///
    /// 库不依赖 anyhow，所以没有 `From<anyhow::Error>`；调用方用
    /// `RsmediaError::External(err.into())`（anyhow 官方支持转
    /// `Box<dyn Error + Send + Sync>`），源链在 Display/source() 中保留。
    #[test]
    fn test_anyhow_error_wraps_into_external() {
        let anyhow_err = anyhow::Error::msg("upstream failure").context("while demuxing input"); // anyhow 自己的链
        let boxed: Box<dyn StdError + Send + Sync + 'static> = anyhow_err.into();
        let err = RsmediaError::External(boxed);
        // 注意：anyhow 的 Display 只显示最外层 context；内层消息经 source() 可达。
        assert!(err.to_string().contains("while demuxing input"));

        let wrapped = err.with_context("rsmedia boundary");
        assert!(wrapped.to_string().contains("rsmedia boundary"));
        // root() 落在 External；其 source() 即 anyhow 的 StdError 包装，
        // 再往下是原始的 "upstream failure" —— 链完整活着。
        assert!(matches!(wrapped.root(), RsmediaError::External(_)));
        let anyhow_wrapper = wrapped.root().source().expect("external has a source");
        let inner = anyhow_wrapper
            .source()
            .expect("anyhow keeps its own causes via source()");
        assert_eq!(inner.to_string(), "upstream failure");
    }
}
