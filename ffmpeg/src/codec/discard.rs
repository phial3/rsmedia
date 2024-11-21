use rsmpeg::ffi;

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum Discard {
    None,
    Default,
    NonReference,
    Bidirectional,
    NonIntra,
    NonKey,
    All,
}

impl From<ffi::AVDiscard> for Discard {
    fn from(value: ffi::AVDiscard) -> Self {
        match value {
            ffi::AVDISCARD_NONE => Discard::None,
            ffi::AVDISCARD_DEFAULT => Discard::Default,
            ffi::AVDISCARD_NONREF => Discard::NonReference,
            ffi::AVDISCARD_BIDIR => Discard::Bidirectional,
            ffi::AVDISCARD_NONINTRA => Discard::NonIntra,
            ffi::AVDISCARD_NONKEY => Discard::NonKey,
            ffi::AVDISCARD_ALL => Discard::All,

            // non-exhaustive patterns: `i32::MIN..=-17_i32`, `-15_i32..=-1_i32`, `1_i32..=7_i32` and 5 more not covered
            _ => todo!("unknown discard value {}", value),
        }
    }
}

impl From<Discard> for ffi::AVDiscard {
    fn from(value: Discard) -> ffi::AVDiscard {
        match value {
            Discard::None => ffi::AVDISCARD_NONE,
            Discard::Default => ffi::AVDISCARD_DEFAULT,
            Discard::NonReference => ffi::AVDISCARD_NONREF,
            Discard::Bidirectional => ffi::AVDISCARD_BIDIR,
            Discard::NonIntra => ffi::AVDISCARD_NONINTRA,
            Discard::NonKey => ffi::AVDISCARD_NONKEY,
            Discard::All => ffi::AVDISCARD_ALL,
        }
    }
}