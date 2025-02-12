pub mod decode;
pub mod encode;
pub mod error;
#[cfg(feature = "ndarray")]
pub mod frame;
#[cfg(feature = "ndarray")]
pub use frame::FrameArray;
pub mod hwaccel;
pub mod init;
pub mod io;
pub mod location;
pub mod options;
pub mod packet;
pub mod rational;
pub mod resize;
pub mod stream;
pub mod time;

mod ffi_hwaccel;
mod flags;

pub use decode::{Decoder, DecoderBuilder};
pub use encode::{Encoder, EncoderBuilder};
pub use init::init;
pub use io::{Reader, ReaderBuilder, Writer, WriterBuilder};
pub use location::{Location, Url};
pub use options::Options;
pub use packet::Packet;
pub use rational::Rational;
pub use resize::Resize;
pub use time::Time;

/// Re-export internal `AvPixel` as `PixelFormat` for callers.
pub type PixelFormat = rsmpeg::avutil::AVPixelFormat;

/// Re-export internal `AvFrame` for caller to use.
pub type RawFrame = rsmpeg::avutil::AVFrame;

/// Default frame pixel format.
pub(crate) const FRAME_PIXEL_FORMAT_RGB24: rsmpeg::avutil::AVPixelFormat = rsmpeg::ffi::AV_PIX_FMT_RGB24;
