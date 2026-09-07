pub mod decode;
pub mod encode;
#[cfg(feature = "ndarray")]
pub mod frame;
#[cfg(feature = "ndarray")]
pub use frame::{MediaFrame, MediaFrameType};
pub mod codec;
pub mod colors;
pub mod error;
pub mod fmt;
pub mod filter;
pub mod hwaccel;
pub mod imgutils;
pub mod init;
pub mod io;
pub mod location;
pub mod mux;
pub mod options;
pub mod pcm;
pub mod pixel;
pub mod resize;
pub mod stream;
pub mod strutils;
pub mod subtitle;
pub mod swctx;
pub mod time;

pub use codec::{CodecConfig, FormatInfo, Profile};
pub use decode::{Decoder, DecoderBuilder};
pub use encode::{Encoder, EncoderBuilder};
pub use error::{Error, Result, RsmediaError};
pub use fmt::{FrameFormat, SampleFormat};
pub use init::init;
pub use io::{Reader, Writer};
pub use io::{StreamReader, StreamReaderBuilder, StreamWriter, StreamWriterBuilder};
pub use location::{Location, Url};
pub use options::{Options, Quality, VideoProfile};
pub use pcm::{PcmSink, PcmSpec};
pub use pixel::PixelFormat;
pub use resize::Resize;
pub use stream::MediaType;
pub use subtitle::SubtitleSegment;
pub use swctx::ScaleAlgorithm;
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
    //
    // `tests/common/mod.rs` refers to this crate as `rsmedia::...` (it is also
    // compiled standalone by integration tests). Alias the crate root so the
    // same paths resolve inside the library too.
    use crate as rsmedia;
    include!("../tests/common/mod.rs");
}
