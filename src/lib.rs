//! Safe Rust bindings over the FFmpeg `libav*` family (through
//! [rsmpeg](https://github.com/larksuite/rsmpeg)), covering the common media
//! pipeline: demux/decode → filter/scale/resample → encode → mux.
//!
//! # Mental model
//!
//! Four verbs, one type each:
//!
//! | Direction | Type | Reads / writes |
//! |---|---|---|
//! | container → frames | [`Demuxer`] / [`Decoder`] | packets, decoded [`MediaFrame`]s |
//! | frames → container | [`Encoder`] / [`Muxer`] | frames in, packets out |
//! | bytes ↔ container | [`StreamReader`] / [`StreamWriter`] | any [`Reader`] / [`Writer`] |
//! | frame data | [`MediaFrame`] | ndarray-backed, convertible across pixel formats |
//!
//! [`Demuxer`]: mux::Demuxer
//! [`Decoder`]: decode::Decoder
//! [`Encoder`]: encode::Encoder
//! [`Muxer`]: mux::Muxer
//! [`StreamReader`]: io::StreamReader
//! [`StreamWriter`]: io::StreamWriter
//! [`Reader`]: io::Reader
//! [`Writer`]: io::Writer
//! [`MediaFrame`]: frame::MediaFrame
//!
//! Sources and destinations are named through one abstraction,
//! [`Location`]: a local path, a `PathBuf` or a URL are the
//! same call, so `StreamReader::new("/tmp/a.mp4")` and
//! `StreamReader::new("http://host/a.mp4")` are interchangeable.
//!
//! # No setup call
//!
//! Regular files, buffers and network sources work without any initialization —
//! reach for the type you need. The one exception is **capture devices**
//! (webcam, microphone, screen grab): those need [`init()`] once before the
//! first `Decoder`, because FFmpeg's device registry is not populated
//! otherwise and the failure looks like an unopenable input.
//!
//! # Choosing an FFmpeg version
//!
//! Exactly one `ffmpegN` cargo feature must be enabled (default: `ffmpeg9`);
//! see the [feature table in the README](https://github.com/dromara/rsmedia#cargo-features).
//!
//! # Errors and logging
//!
//! Fallible APIs return [`Result<T>`](error::Result) with
//! [`RsmediaError`], which distinguishes *unsupported*
//! (this platform/build cannot do it), *invalid config* (caller mistake) and
//! *other* (I/O and runtime failures). FFmpeg's own diagnostics go through the
//! [`tracing`] facade — install a subscriber (or set `RUST_LOG`) to see them.

#[macro_use]
mod macros;

pub mod bsf;
pub mod codec;
pub mod colors;
pub mod decode;
pub mod encode;
pub mod error;
pub mod filter;
pub mod flags;
pub mod fmt;
pub mod frame;
pub mod hwaccel;
pub mod imgutils;
pub mod init;
pub mod io;
pub mod location;
pub mod mux;
pub mod options;
pub mod pcm;
pub mod pixel;
pub mod resample;
pub mod resize;
pub mod scale;
pub mod state;
pub mod stream;
pub mod strutils;
pub mod subtitle;
pub mod time;

pub use bsf::Bsf;
pub use codec::{AVCodecFlag, AVCodecFlag2, CodecConfig, FormatInfo, Profile, ThreadType};
pub use colors::Color;
pub use decode::{Decoder, DecoderBuilder, ErrRecognition, SkipFrame};
pub use encode::{Encoder, EncoderBuilder};
pub use error::{Result, RsmediaError};
pub use filter::{Expr, Filter, FilterGraph, FilterGraphBuilder, FilterNode};
pub use flags::FlagSet;
pub use fmt::{DataLayout, FrameFormat, SampleFormat};
pub use frame::{AVFrameFlag, ElementType, FrameData, FrameSideData, MediaFrame};
pub use hwaccel::{HWDeviceConfig, HWDeviceType, ProbeDepth};
pub use init::{AVLogFlag, AVLogLevel, init, init_with, init_with_level};
pub use io::{AVSeekFlag, DynWriter, Reader, Seekable, Writer};
pub use io::{StreamReader, StreamReaderBuilder, StreamWriter, StreamWriterBuilder};
pub use location::{Location, Url};
pub use mux::{Chapter, Demuxer, Muxer};
pub use options::{AVOptions, Metadata, Quality, VideoProfile};
pub use pcm::{PcmSink, PcmSpec};
pub use pixel::PixelFormat;
pub use resample::Resampler;
pub use resize::Resize;
pub use scale::{ScaleAlgorithm, ScaleQuality, Scaler, VideoSpec};
pub use stream::MediaType;
pub use subtitle::SubtitleSegment;
pub use time::{Rational, Time};

/// Upper bound on consecutive "not now" answers before waiting stops.
///
/// FFmpeg's send/receive API answers `EAGAIN` both mid-stream (the unit's input
/// or output queue is full) and while draining (the unit still holds buffered
/// data). For a while either is legitimate — a B-frame lookahead or a
/// frame-threaded decoder can answer `EAGAIN` many times — but neither may last
/// forever, or a damaged stream would turn a public call into a hang. Every
/// retry path here (decoder, encoder, filter graph, demuxer) gives up with an
/// error after this many attempts.
///
/// This bounds *waiting*, not output: a loop that resets its counter whenever
/// something is produced may legitimately run longer than this. See
/// `MAX_DRAIN_OUTPUTS` for the other bound.
pub(crate) const MAX_EAGAIN_RETRIES: usize = 1_000;

/// Upper bound on how many items a single drain call may hand back.
///
/// [`Resampler::flush_frames`] and the subtitle drain keep pulling until the
/// source reports it has nothing left, so this caps how much buffered data one
/// call accepts — a source that never empties is a broken one, not a productive
/// one. It says nothing about waiting: those sources answer at once, which is
/// why this is kept separate from `MAX_EAGAIN_RETRIES`.
pub(crate) const MAX_DRAIN_OUTPUTS: usize = 1_000;

/// re-exported under the name `ffmpeg`
pub use rsmpeg as ffmpeg;

/// Unit-test helpers
#[cfg(test)]
pub mod test_support;
