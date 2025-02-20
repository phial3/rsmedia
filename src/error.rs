use rsmpeg::error::RsmpegError;

/// Represents video I/O Errors. Some errors are generated by the ffmpeg backend, and are wrapped in
/// `BackendError`.
#[derive(Debug)]
pub enum MediaError {
    ReadExhausted,
    DecodeExhausted,
    WriteRetryLimitReached,
    InvalidFrameFormat,
    InvalidExtraData,
    InvalidPixelFormat,
    UninitializedCodec,
    InvalidCodecParameters,
    InvalidResizeParameters,
    UnsupportedCodecParameterSets,
    UnsupportedCodecHWDeviceType,
    TranscodeError(String),
    BackendError(RsmpegError),
}

impl std::error::Error for MediaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            MediaError::TranscodeError(_) => None,
            MediaError::ReadExhausted => None,
            MediaError::DecodeExhausted => None,
            MediaError::WriteRetryLimitReached => None,
            MediaError::InvalidFrameFormat => None,
            MediaError::InvalidPixelFormat => None,
            MediaError::InvalidExtraData => None,
            MediaError::InvalidCodecParameters => None,
            MediaError::UnsupportedCodecParameterSets => None,
            MediaError::InvalidResizeParameters => None,
            MediaError::UninitializedCodec => None,
            MediaError::UnsupportedCodecHWDeviceType => None,
            MediaError::BackendError(ref internal) => Some(internal),
        }
    }
}

impl std::fmt::Display for MediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            MediaError::TranscodeError(ref msg) => write!(f, "transcode error: {}", msg),
            MediaError::ReadExhausted => write!(f, "stream exhausted"),
            MediaError::DecodeExhausted => write!(f, "stream exhausted"),
            MediaError::WriteRetryLimitReached => {
                write!(f, "cannot write to video stream, even after multiple tries")
            }
            MediaError::InvalidFrameFormat => write!(
                f,
                "provided frame does not match expected dimensions and/or pixel format"
            ),
            MediaError::InvalidPixelFormat => write!(f, "invalid pixel format"),
            MediaError::InvalidExtraData => write!(f, "codec parameters extradata is corrupted"),
            MediaError::InvalidCodecParameters => write!(f, "invalid codec parameters"),
            MediaError::UnsupportedCodecParameterSets => write!(
                f,
                "extracting parameter sets for this codec is not suppored"
            ),
            MediaError::InvalidResizeParameters => {
                write!(f, "cannot resize frame into provided dimensions")
            }
            MediaError::UninitializedCodec => {
                write!(f, "codec context is not initialized properly")
            }
            MediaError::UnsupportedCodecHWDeviceType => {
                write!(f, "codec does not supported hardware acceleration device")
            }
            MediaError::BackendError(ref internal) => internal.fmt(f),
        }
    }
}

impl From<RsmpegError> for MediaError {
    fn from(internal: RsmpegError) -> MediaError {
        MediaError::BackendError(internal)
    }
}
