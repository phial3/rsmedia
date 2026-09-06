pub mod decode;
pub mod encode;
#[cfg(feature = "ndarray")]
pub mod frame;
#[cfg(feature = "ndarray")]
pub use frame::{MediaFrame, MediaFrameFormat, MediaFrameType};
pub mod codec;
pub mod colors;
pub mod filter;
pub mod flags;
pub mod hwaccel;
pub mod imgutils;
pub mod init;
pub mod io;
pub mod location;
pub mod mux;
pub mod options;
pub mod pixel;
pub mod resize;
pub mod stream;
pub mod swctx;
pub mod time;
pub mod utils;

pub use swctx::ScaleAlgorithm;

pub use decode::{Decoder, DecoderBuilder};
pub use encode::{Encoder, EncoderBuilder};
pub use flags::{MediaType, SampleFormat};
pub use init::init;
pub use io::{Reader, Writer};
pub use io::{StreamReader, StreamReaderBuilder, StreamWriter, StreamWriterBuilder};
pub use location::{Location, Url};
pub use options::Options;
pub use pixel::PixelFormat;
pub use resize::Resize;
pub use time::Time;

/// Re-export internal definition for caller to use.
pub use rsmpeg::avutil;

/// Test utilities - compiled only for library unit tests, so they never
/// pollute the shipped binary. Integration tests get the same helpers from
/// `tests/common/mod.rs` (see that file for the single source of truth).
#[cfg(test)]
pub mod test_utils {
    // Single source of truth for test-output helpers; shared with integration
    // tests via `include!` so the path logic is not duplicated.
    include!("../tests/common/mod.rs");
}
