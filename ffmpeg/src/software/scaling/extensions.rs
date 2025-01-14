use super::{Context, Flags};
use crate::{decoder, frame, util::format, Error};

// #[cfg(not(feature = "ffmpeg7"))]
// use crate::Picture;

// #[cfg(not(feature = "ffmpeg7"))]
// impl Picture<'_> {
//     #[inline]
//     pub fn scaler(&self, width: u32, height: u32, flags: Flags) -> Result<Context, Error> {
//         Context::get(
//             self.format(),
//             self.width(),
//             self.height(),
//             self.format(),
//             width,
//             height,
//             flags,
//         )
//     }
//
//     #[inline]
//     pub fn converter(&self, format: format::Pixel) -> Result<Context, Error> {
//         Context::get(
//             self.format(),
//             self.width(),
//             self.height(),
//             format,
//             self.width(),
//             self.height(),
//             Flags::FAST_BILINEAR,
//         )
//     }
// }

impl frame::Video {
    #[inline]
    pub fn scaler(&self, width: u32, height: u32, flags: Flags) -> Result<Context, Error> {
        Context::get(
            self.format(),
            self.width(),
            self.height(),
            self.format(),
            width,
            height,
            flags,
        )
    }

    #[inline]
    pub fn converter(&self, format: format::Pixel) -> Result<Context, Error> {
        Context::get(
            self.format(),
            self.width(),
            self.height(),
            format,
            self.width(),
            self.height(),
            Flags::FAST_BILINEAR,
        )
    }
}

impl decoder::Video {
    #[inline]
    pub fn scaler(&self, width: u32, height: u32, flags: Flags) -> Result<Context, Error> {
        Context::get(
            self.format(),
            self.width(),
            self.height(),
            self.format(),
            width,
            height,
            flags,
        )
    }

    #[inline]
    pub fn converter(&self, format: format::Pixel) -> Result<Context, Error> {
        Context::get(
            self.format(),
            self.width(),
            self.height(),
            format,
            self.width(),
            self.height(),
            Flags::FAST_BILINEAR,
        )
    }
}
