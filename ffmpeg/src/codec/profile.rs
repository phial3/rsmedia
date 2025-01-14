use super::Id;
use libc::c_int;
use sys::ffi;

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum Profile {
    Unknown,
    Reserved,

    AAC(AAC),
    MPEG2(MPEG2),
    DTS(DTS),
    H264(H264),
    VC1(VC1),
    MPEG4(MPEG4),
    JPEG2000(JPEG2000),
    HEVC(HEVC),
    VP9(VP9),
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum AAC {
    Main,
    Low,
    SSR,
    LTP,
    HE,
    HEv2,
    LD,
    ELD,

    MPEG2Low,
    MPEG2HE,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum DTS {
    Default,
    ES,
    _96_24,
    HD_HRA,
    HD_MA,
    Express,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum MPEG2 {
    _422,
    High,
    SS,
    SNRScalable,
    Main,
    Simple,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum H264 {
    Constrained,
    Intra,
    Baseline,
    ConstrainedBaseline,
    Main,
    Extended,
    High,
    High10,
    High10Intra,
    High422,
    High422Intra,
    High444,
    High444Predictive,
    High444Intra,
    CAVLC444,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum VC1 {
    Simple,
    Main,
    Complex,
    Advanced,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum MPEG4 {
    Simple,
    SimpleScalable,
    Core,
    Main,
    NBit,
    ScalableTexture,
    SimpleFaceAnimation,
    BasicAnimatedTexture,
    Hybrid,
    AdvancedRealTime,
    CoreScalable,
    AdvancedCoding,
    AdvancedCore,
    AdvancedScalableTexture,
    SimpleStudio,
    AdvancedSimple,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum JPEG2000 {
    CStreamRestriction0,
    CStreamRestriction1,
    CStreamNoRestriction,
    DCinema2K,
    DCinema4K,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum HEVC {
    Main,
    Main10,
    MainStillPicture,
    Rext,
}

#[allow(non_camel_case_types)]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum VP9 {
    _0,
    _1,
    _2,
    _3,
}

impl From<(Id, c_int)> for Profile {
    fn from((id, value): (Id, c_int)) -> Profile {
        if value == ffi::FF_PROFILE_UNKNOWN {
            return Profile::Unknown;
        }

        if value == ffi::FF_PROFILE_RESERVED {
            return Profile::Reserved;
        }

        let val_u32 = value as u32;
        match id {
            Id::AAC => match val_u32 {
                ffi::FF_PROFILE_AAC_MAIN => Profile::AAC(AAC::Main),
                ffi::FF_PROFILE_AAC_LOW => Profile::AAC(AAC::Low),
                ffi::FF_PROFILE_AAC_SSR => Profile::AAC(AAC::SSR),
                ffi::FF_PROFILE_AAC_LTP => Profile::AAC(AAC::LTP),
                ffi::FF_PROFILE_AAC_HE => Profile::AAC(AAC::HE),
                ffi::FF_PROFILE_AAC_HE_V2 => Profile::AAC(AAC::HEv2),
                ffi::FF_PROFILE_AAC_LD => Profile::AAC(AAC::LD),
                ffi::FF_PROFILE_AAC_ELD => Profile::AAC(AAC::ELD),

                ffi::FF_PROFILE_MPEG2_AAC_LOW => Profile::AAC(AAC::MPEG2Low),
                ffi::FF_PROFILE_MPEG2_AAC_HE => Profile::AAC(AAC::MPEG2HE),

                _ => Profile::Unknown,
            },

            Id::DTS => match val_u32 {
                ffi::FF_PROFILE_DTS => Profile::DTS(DTS::Default),
                ffi::FF_PROFILE_DTS_ES => Profile::DTS(DTS::ES),
                ffi::FF_PROFILE_DTS_96_24 => Profile::DTS(DTS::_96_24),
                ffi::FF_PROFILE_DTS_HD_HRA => Profile::DTS(DTS::HD_HRA),
                ffi::FF_PROFILE_DTS_HD_MA => Profile::DTS(DTS::HD_MA),
                ffi::FF_PROFILE_DTS_EXPRESS => Profile::DTS(DTS::Express),

                _ => Profile::Unknown,
            },

            Id::MPEG2VIDEO => match val_u32 {
                ffi::FF_PROFILE_MPEG2_422 => Profile::MPEG2(MPEG2::_422),
                ffi::FF_PROFILE_MPEG2_HIGH => Profile::MPEG2(MPEG2::High),
                ffi::FF_PROFILE_MPEG2_SS => Profile::MPEG2(MPEG2::SS),
                ffi::FF_PROFILE_MPEG2_SNR_SCALABLE => Profile::MPEG2(MPEG2::SNRScalable),
                ffi::FF_PROFILE_MPEG2_MAIN => Profile::MPEG2(MPEG2::Main),
                ffi::FF_PROFILE_MPEG2_SIMPLE => Profile::MPEG2(MPEG2::Simple),

                _ => Profile::Unknown,
            },

            Id::H264 => match val_u32 {
                ffi::FF_PROFILE_H264_CONSTRAINED => Profile::H264(H264::Constrained),
                ffi::FF_PROFILE_H264_INTRA => Profile::H264(H264::Intra),
                ffi::FF_PROFILE_H264_BASELINE => Profile::H264(H264::Baseline),
                ffi::FF_PROFILE_H264_CONSTRAINED_BASELINE => {
                    Profile::H264(H264::ConstrainedBaseline)
                }
                ffi::FF_PROFILE_H264_MAIN => Profile::H264(H264::Main),
                ffi::FF_PROFILE_H264_EXTENDED => Profile::H264(H264::Extended),
                ffi::FF_PROFILE_H264_HIGH => Profile::H264(H264::High),
                ffi::FF_PROFILE_H264_HIGH_10 => Profile::H264(H264::High10),
                ffi::FF_PROFILE_H264_HIGH_10_INTRA => Profile::H264(H264::High10Intra),
                ffi::FF_PROFILE_H264_HIGH_422 => Profile::H264(H264::High422),
                ffi::FF_PROFILE_H264_HIGH_422_INTRA => Profile::H264(H264::High422Intra),
                ffi::FF_PROFILE_H264_HIGH_444 => Profile::H264(H264::High444),
                ffi::FF_PROFILE_H264_HIGH_444_PREDICTIVE => Profile::H264(H264::High444Predictive),
                ffi::FF_PROFILE_H264_HIGH_444_INTRA => Profile::H264(H264::High444Intra),
                ffi::FF_PROFILE_H264_CAVLC_444 => Profile::H264(H264::CAVLC444),

                _ => Profile::Unknown,
            },

            Id::VC1 => match val_u32 {
                ffi::FF_PROFILE_VC1_SIMPLE => Profile::VC1(VC1::Simple),
                ffi::FF_PROFILE_VC1_MAIN => Profile::VC1(VC1::Main),
                ffi::FF_PROFILE_VC1_COMPLEX => Profile::VC1(VC1::Complex),
                ffi::FF_PROFILE_VC1_ADVANCED => Profile::VC1(VC1::Advanced),

                _ => Profile::Unknown,
            },

            Id::MPEG4 => match val_u32 {
                ffi::FF_PROFILE_MPEG4_SIMPLE => Profile::MPEG4(MPEG4::Simple),
                ffi::FF_PROFILE_MPEG4_SIMPLE_SCALABLE => Profile::MPEG4(MPEG4::SimpleScalable),
                ffi::FF_PROFILE_MPEG4_CORE => Profile::MPEG4(MPEG4::Core),
                ffi::FF_PROFILE_MPEG4_MAIN => Profile::MPEG4(MPEG4::Main),
                ffi::FF_PROFILE_MPEG4_N_BIT => Profile::MPEG4(MPEG4::NBit),
                ffi::FF_PROFILE_MPEG4_SCALABLE_TEXTURE => Profile::MPEG4(MPEG4::ScalableTexture),
                ffi::FF_PROFILE_MPEG4_SIMPLE_FACE_ANIMATION => {
                    Profile::MPEG4(MPEG4::SimpleFaceAnimation)
                }
                ffi::FF_PROFILE_MPEG4_BASIC_ANIMATED_TEXTURE => {
                    Profile::MPEG4(MPEG4::BasicAnimatedTexture)
                }
                ffi::FF_PROFILE_MPEG4_HYBRID => Profile::MPEG4(MPEG4::Hybrid),
                ffi::FF_PROFILE_MPEG4_ADVANCED_REAL_TIME => Profile::MPEG4(MPEG4::AdvancedRealTime),
                ffi::FF_PROFILE_MPEG4_CORE_SCALABLE => Profile::MPEG4(MPEG4::CoreScalable),
                ffi::FF_PROFILE_MPEG4_ADVANCED_CODING => Profile::MPEG4(MPEG4::AdvancedCoding),
                ffi::FF_PROFILE_MPEG4_ADVANCED_CORE => Profile::MPEG4(MPEG4::AdvancedCore),
                ffi::FF_PROFILE_MPEG4_ADVANCED_SCALABLE_TEXTURE => {
                    Profile::MPEG4(MPEG4::AdvancedScalableTexture)
                }
                ffi::FF_PROFILE_MPEG4_SIMPLE_STUDIO => Profile::MPEG4(MPEG4::SimpleStudio),
                ffi::FF_PROFILE_MPEG4_ADVANCED_SIMPLE => Profile::MPEG4(MPEG4::AdvancedSimple),

                _ => Profile::Unknown,
            },

            Id::JPEG2000 => match val_u32 {
                ffi::FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_0 => {
                    Profile::JPEG2000(JPEG2000::CStreamRestriction0)
                }
                ffi::FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_1 => {
                    Profile::JPEG2000(JPEG2000::CStreamRestriction1)
                }
                ffi::FF_PROFILE_JPEG2000_CSTREAM_NO_RESTRICTION => {
                    Profile::JPEG2000(JPEG2000::CStreamNoRestriction)
                }
                ffi::FF_PROFILE_JPEG2000_DCINEMA_2K => Profile::JPEG2000(JPEG2000::DCinema2K),
                ffi::FF_PROFILE_JPEG2000_DCINEMA_4K => Profile::JPEG2000(JPEG2000::DCinema4K),

                _ => Profile::Unknown,
            },

            Id::HEVC => match val_u32 {
                ffi::FF_PROFILE_HEVC_MAIN => Profile::HEVC(HEVC::Main),
                ffi::FF_PROFILE_HEVC_MAIN_10 => Profile::HEVC(HEVC::Main10),
                ffi::FF_PROFILE_HEVC_MAIN_STILL_PICTURE => Profile::HEVC(HEVC::MainStillPicture),
                ffi::FF_PROFILE_HEVC_REXT => Profile::HEVC(HEVC::Rext),

                _ => Profile::Unknown,
            },

            Id::VP9 => match val_u32 {
                ffi::FF_PROFILE_VP9_0 => Profile::VP9(VP9::_0),
                ffi::FF_PROFILE_VP9_1 => Profile::VP9(VP9::_1),
                ffi::FF_PROFILE_VP9_2 => Profile::VP9(VP9::_2),
                ffi::FF_PROFILE_VP9_3 => Profile::VP9(VP9::_3),

                _ => Profile::Unknown,
            },

            _ => Profile::Unknown,
        }
    }
}

impl From<Profile> for c_int {
    fn from(value: Profile) -> c_int {
        match value {
            Profile::Unknown => ffi::FF_PROFILE_UNKNOWN,
            Profile::Reserved => ffi::FF_PROFILE_RESERVED,

            Profile::AAC(AAC::Main) => ffi::FF_PROFILE_AAC_MAIN as i32,
            Profile::AAC(AAC::Low) => ffi::FF_PROFILE_AAC_LOW as i32,
            Profile::AAC(AAC::SSR) => ffi::FF_PROFILE_AAC_SSR as i32,
            Profile::AAC(AAC::LTP) => ffi::FF_PROFILE_AAC_LTP as i32,
            Profile::AAC(AAC::HE) => ffi::FF_PROFILE_AAC_HE as i32,
            Profile::AAC(AAC::HEv2) => ffi::FF_PROFILE_AAC_HE_V2 as i32,
            Profile::AAC(AAC::LD) => ffi::FF_PROFILE_AAC_LD as i32,
            Profile::AAC(AAC::ELD) => ffi::FF_PROFILE_AAC_ELD as i32,

            Profile::AAC(AAC::MPEG2Low) => ffi::FF_PROFILE_MPEG2_AAC_LOW as i32,
            Profile::AAC(AAC::MPEG2HE) => ffi::FF_PROFILE_MPEG2_AAC_HE as i32,

            Profile::DTS(DTS::Default) => ffi::FF_PROFILE_DTS as i32,
            Profile::DTS(DTS::ES) => ffi::FF_PROFILE_DTS_ES as i32,
            Profile::DTS(DTS::_96_24) => ffi::FF_PROFILE_DTS_96_24 as i32,
            Profile::DTS(DTS::HD_HRA) => ffi::FF_PROFILE_DTS_HD_HRA as i32,
            Profile::DTS(DTS::HD_MA) => ffi::FF_PROFILE_DTS_HD_MA as i32,
            Profile::DTS(DTS::Express) => ffi::FF_PROFILE_DTS_EXPRESS as i32,

            Profile::MPEG2(MPEG2::_422) => ffi::FF_PROFILE_MPEG2_422 as i32,
            Profile::MPEG2(MPEG2::High) => ffi::FF_PROFILE_MPEG2_HIGH as i32,
            Profile::MPEG2(MPEG2::SS) => ffi::FF_PROFILE_MPEG2_SS as i32,
            Profile::MPEG2(MPEG2::SNRScalable) => ffi::FF_PROFILE_MPEG2_SNR_SCALABLE as i32,
            Profile::MPEG2(MPEG2::Main) => ffi::FF_PROFILE_MPEG2_MAIN as i32,
            Profile::MPEG2(MPEG2::Simple) => ffi::FF_PROFILE_MPEG2_SIMPLE as i32,

            Profile::H264(H264::Constrained) => ffi::FF_PROFILE_H264_CONSTRAINED as i32,
            Profile::H264(H264::Intra) => ffi::FF_PROFILE_H264_INTRA as i32,
            Profile::H264(H264::Baseline) => ffi::FF_PROFILE_H264_BASELINE as i32,
            Profile::H264(H264::ConstrainedBaseline) => {
                ffi::FF_PROFILE_H264_CONSTRAINED_BASELINE as i32
            }
            Profile::H264(H264::Main) => ffi::FF_PROFILE_H264_MAIN as i32,
            Profile::H264(H264::Extended) => ffi::FF_PROFILE_H264_EXTENDED as i32,
            Profile::H264(H264::High) => ffi::FF_PROFILE_H264_HIGH as i32,
            Profile::H264(H264::High10) => ffi::FF_PROFILE_H264_HIGH_10 as i32,
            Profile::H264(H264::High10Intra) => ffi::FF_PROFILE_H264_HIGH_10_INTRA as i32,
            Profile::H264(H264::High422) => ffi::FF_PROFILE_H264_HIGH_422 as i32,
            Profile::H264(H264::High422Intra) => ffi::FF_PROFILE_H264_HIGH_422_INTRA as i32,
            Profile::H264(H264::High444) => ffi::FF_PROFILE_H264_HIGH_444 as i32,
            Profile::H264(H264::High444Predictive) => {
                ffi::FF_PROFILE_H264_HIGH_444_PREDICTIVE as i32
            }
            Profile::H264(H264::High444Intra) => ffi::FF_PROFILE_H264_HIGH_444_INTRA as i32,
            Profile::H264(H264::CAVLC444) => ffi::FF_PROFILE_H264_CAVLC_444 as i32,

            Profile::VC1(VC1::Simple) => ffi::FF_PROFILE_VC1_SIMPLE as i32,
            Profile::VC1(VC1::Main) => ffi::FF_PROFILE_VC1_MAIN as i32,
            Profile::VC1(VC1::Complex) => ffi::FF_PROFILE_VC1_COMPLEX as i32,
            Profile::VC1(VC1::Advanced) => ffi::FF_PROFILE_VC1_ADVANCED as i32,

            Profile::MPEG4(MPEG4::Simple) => ffi::FF_PROFILE_MPEG4_SIMPLE as i32,
            Profile::MPEG4(MPEG4::SimpleScalable) => ffi::FF_PROFILE_MPEG4_SIMPLE_SCALABLE as i32,
            Profile::MPEG4(MPEG4::Core) => ffi::FF_PROFILE_MPEG4_CORE as i32,
            Profile::MPEG4(MPEG4::Main) => ffi::FF_PROFILE_MPEG4_MAIN as i32,
            Profile::MPEG4(MPEG4::NBit) => ffi::FF_PROFILE_MPEG4_N_BIT as i32,
            Profile::MPEG4(MPEG4::ScalableTexture) => ffi::FF_PROFILE_MPEG4_SCALABLE_TEXTURE as i32,
            Profile::MPEG4(MPEG4::SimpleFaceAnimation) => {
                ffi::FF_PROFILE_MPEG4_SIMPLE_FACE_ANIMATION as i32
            }
            Profile::MPEG4(MPEG4::BasicAnimatedTexture) => {
                ffi::FF_PROFILE_MPEG4_BASIC_ANIMATED_TEXTURE as i32
            }
            Profile::MPEG4(MPEG4::Hybrid) => ffi::FF_PROFILE_MPEG4_HYBRID as i32,
            Profile::MPEG4(MPEG4::AdvancedRealTime) => {
                ffi::FF_PROFILE_MPEG4_ADVANCED_REAL_TIME as i32
            }
            Profile::MPEG4(MPEG4::CoreScalable) => ffi::FF_PROFILE_MPEG4_CORE_SCALABLE as i32,
            Profile::MPEG4(MPEG4::AdvancedCoding) => ffi::FF_PROFILE_MPEG4_ADVANCED_CODING as i32,
            Profile::MPEG4(MPEG4::AdvancedCore) => ffi::FF_PROFILE_MPEG4_ADVANCED_CORE as i32,
            Profile::MPEG4(MPEG4::AdvancedScalableTexture) => {
                ffi::FF_PROFILE_MPEG4_ADVANCED_SCALABLE_TEXTURE as i32
            }
            Profile::MPEG4(MPEG4::SimpleStudio) => ffi::FF_PROFILE_MPEG4_SIMPLE_STUDIO as i32,
            Profile::MPEG4(MPEG4::AdvancedSimple) => ffi::FF_PROFILE_MPEG4_ADVANCED_SIMPLE as i32,

            Profile::JPEG2000(JPEG2000::CStreamRestriction0) => {
                ffi::FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_0 as i32
            }
            Profile::JPEG2000(JPEG2000::CStreamRestriction1) => {
                ffi::FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_1 as i32
            }
            Profile::JPEG2000(JPEG2000::CStreamNoRestriction) => {
                ffi::FF_PROFILE_JPEG2000_CSTREAM_NO_RESTRICTION as i32
            }
            Profile::JPEG2000(JPEG2000::DCinema2K) => ffi::FF_PROFILE_JPEG2000_DCINEMA_2K as i32,
            Profile::JPEG2000(JPEG2000::DCinema4K) => ffi::FF_PROFILE_JPEG2000_DCINEMA_4K as i32,

            Profile::HEVC(HEVC::Main) => ffi::FF_PROFILE_HEVC_MAIN as i32,
            Profile::HEVC(HEVC::Main10) => ffi::FF_PROFILE_HEVC_MAIN_10 as i32,
            Profile::HEVC(HEVC::MainStillPicture) => ffi::FF_PROFILE_HEVC_MAIN_STILL_PICTURE as i32,
            Profile::HEVC(HEVC::Rext) => ffi::FF_PROFILE_HEVC_REXT as i32,

            Profile::VP9(VP9::_0) => ffi::FF_PROFILE_VP9_0 as i32,
            Profile::VP9(VP9::_1) => ffi::FF_PROFILE_VP9_1 as i32,
            Profile::VP9(VP9::_2) => ffi::FF_PROFILE_VP9_2 as i32,
            Profile::VP9(VP9::_3) => ffi::FF_PROFILE_VP9_3 as i32,
        }
    }
}
