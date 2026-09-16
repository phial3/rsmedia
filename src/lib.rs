#[macro_use]
mod macros;

pub mod bsf;
pub mod codec;
pub mod colors;
pub mod decode;
pub mod encode;
pub mod error;
pub mod filter;
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
pub mod stream;
pub mod strutils;
pub mod subtitle;
pub mod time;

pub use bsf::Bsf;
pub use codec::{CodecConfig, FormatInfo, Profile};
pub use colors::Color;
pub use decode::{Decoder, DecoderBuilder};
pub use encode::{Encoder, EncoderBuilder};
pub use error::{Result, RsmediaError};
pub use filter::Filter;
pub use fmt::{DataLayout, FrameFormat, SampleFormat};
pub use frame::{ElementType, FrameData, FrameSideData, MediaFrame};
pub use hwaccel::{HWDeviceConfig, HWDeviceType};
pub use init::{AVLogFlag, AVLogLevel, init, init_with, init_with_level};
pub use io::{AVSeekFlag, Reader, Seekable, Writer};
pub use io::{StreamReader, StreamReaderBuilder, StreamWriter, StreamWriterBuilder};
pub use location::{Location, Url};
pub use mux::{Chapter, Demuxer, Muxer};
pub use options::{Metadata, Options, Quality, VideoProfile};
pub use pcm::{PcmSink, PcmSpec};
pub use pixel::PixelFormat;
pub use resample::Resampler;
pub use resize::Resize;
pub use scale::{ScaleAlgorithm, ScaleQuality, Scaler};
pub use stream::MediaType;
pub use subtitle::SubtitleSegment;
pub use time::Time;

pub(crate) const MAX_DRAIN_ITERATIONS: usize = 1_000;

#[cfg(feature = "image")]
pub use imgutils::thumbnail;

/// re-exported under the name `ffmpeg`
pub use rsmpeg as ffmpeg;

/// Unit-test helpers
#[cfg(test)]
pub mod test_support;
