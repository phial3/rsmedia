//! NOTE: this will be much better once specialization comes

use std::ffi::CString;
use std::mem;

use libc::{c_int, c_void};
use rsmpeg::ffi;

use crate::{
    util::format,
    ChannelLayout, Error, Rational,
};

macro_rules! check {
    ($expr:expr) => {
        match $expr {
            0 => Ok(()),
            e => Err(Error::from(e)),
        }
    };
}

pub unsafe trait Target {
    fn as_ptr(&self) -> *const c_void;
    fn as_mut_ptr(&mut self) -> *mut c_void;
}

pub trait Settable: Target {
    fn set<T: 'static>(&mut self, name: &str, value: &T) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_bin(
                self.as_mut_ptr(),
                name.as_ptr(),
                value as *const _ as *const _,
                mem::size_of::<T>() as c_int,
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_str(&mut self, name: &str, value: &str) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();
            let value = CString::new(value).unwrap();

            check!(ffi::av_opt_set(
                self.as_mut_ptr(),
                name.as_ptr(),
                value.as_ptr(),
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_int(&mut self, name: &str, value: i64) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_int(
                self.as_mut_ptr(),
                name.as_ptr(),
                value,
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_double(&mut self, name: &str, value: f64) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_double(
                self.as_mut_ptr(),
                name.as_ptr(),
                value,
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_rational<T: Into<Rational>>(&mut self, name: &str, value: T) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_q(
                self.as_mut_ptr(),
                name.as_ptr(),
                value.into().into(),
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_image_size(&mut self, name: &str, w: u32, h: u32) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_image_size(
                self.as_mut_ptr(),
                name.as_ptr(),
                w as c_int,
                h as c_int,
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_pixel_format(&mut self, name: &str, format: format::Pixel) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_pixel_fmt(
                self.as_mut_ptr(),
                name.as_ptr(),
                format.into(),
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_sample_format(&mut self, name: &str, format: format::Sample) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            check!(ffi::av_opt_set_sample_fmt(
                self.as_mut_ptr(),
                name.as_ptr(),
                format.into(),
                ffi::AV_OPT_SEARCH_CHILDREN as i32,
            ))
        }
    }

    fn set_channel_layout(&mut self, name: &str, layout: ChannelLayout) -> Result<(), Error> {
        unsafe {
            let name = CString::new(name).unwrap();

            #[cfg(not(feature = "ffmpeg7"))]
            {
                check!(av_opt_set_channel_layout(
                    self.as_mut_ptr(),
                    name.as_ptr(),
                    layout.bits() as i64,
                    AV_OPT_SEARCH_CHILDREN
                ))
            }

            #[cfg(feature = "ffmpeg7")]
            {
                check!(ffi::av_opt_set_chlayout(
                    self.as_mut_ptr(),
                    name.as_ptr(),
                    &layout.into(),
                    ffi::AV_OPT_SEARCH_CHILDREN as i32
                ))
            }
        }
    }
}

pub trait Gettable: Target {}

pub trait Iterable: Target {}