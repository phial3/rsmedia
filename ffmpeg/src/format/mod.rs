pub use crate::util::format::{pixel, Pixel, sample, Sample};
use crate::{
    util::interrupt,
    Dictionary,
    Error,
    Format,
};

pub mod stream;

pub mod chapter;

pub mod context;
pub use self::context::Context;

pub mod format;
#[cfg(not(feature = "ffmpeg7"))]
pub use self::format::list;
pub use self::format::{flag, Flags};
pub use self::format::{Input, Output};

pub mod network;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::str::from_utf8_unchecked;

use rsmpeg::ffi;

// #[cfg(not(feature = "ffmpeg_5_0"))]
pub fn register_all() {
    // unsafe {
    //     ffi::av_register_all();
    // }
}

// #[cfg(not(feature = "ffmpeg_5_0"))]
// pub fn register(format: &Format) {
//     match *format {
//         Format::Input(ref format) => unsafe {
//             ffi::av_register_input_format(format.as_ptr() as *mut _);
//         },
//
//         Format::Output(ref format) => unsafe {
//             ffi::av_register_output_format(format.as_ptr() as *mut _);
//         },
//     }
// }

pub fn version() -> u32 {
    unsafe { ffi::avformat_version() }
}

pub fn configuration() -> &'static str {
    unsafe { from_utf8_unchecked(CStr::from_ptr(ffi::avformat_configuration()).to_bytes()) }
}

pub fn license() -> &'static str {
    unsafe { from_utf8_unchecked(CStr::from_ptr(ffi::avformat_license()).to_bytes()) }
}

// XXX: use to_cstring when stable
fn from_path<P: AsRef<Path> + ?Sized>(path: &P) -> CString {
    CString::new(path.as_ref().as_os_str().to_str().unwrap()).unwrap()
}

// NOTE: this will be better with specialization or anonymous return types
pub fn open<P: AsRef<Path> + ?Sized>(path: &P, format: &Format) -> Result<Context, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);

        match *format {
            Format::Input(ref format) => match ffi::avformat_open_input(
                &mut ps,
                path.as_ptr(),
                format.as_ptr() as *mut _,
                ptr::null_mut(),
            ) {
                0 => match ffi::avformat_find_stream_info(ps, ptr::null_mut()) {
                    r if r >= 0 => Ok(Context::Input(context::Input::wrap(ps))),
                    e => Err(Error::from(e)),
                },

                e => Err(Error::from(e)),
            },

            Format::Output(ref format) => match ffi::avformat_alloc_output_context2(
                &mut ps,
                format.as_ptr() as *mut _,
                ptr::null(),
                path.as_ptr(),
            ) {
                0 => match ffi::avio_open(&mut (*ps).pb, path.as_ptr(), ffi::AVIO_FLAG_WRITE as i32) {
                    0 => Ok(Context::Output(context::Output::wrap(ps))),
                    e => Err(Error::from(e)),
                },

                e => Err(Error::from(e)),
            },
        }
    }
}

pub fn open_with<P: AsRef<Path> + ?Sized>(
    path: &P,
    format: &Format,
    options: Dictionary,
) -> Result<Context, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);
        let mut opts = options.disown();

        match *format {
            Format::Input(ref format) => {
                let res = ffi::avformat_open_input(
                    &mut ps,
                    path.as_ptr(),
                    format.as_ptr() as *mut _,
                    &mut opts,
                );

                Dictionary::own(opts);

                match res {
                    0 => match ffi::avformat_find_stream_info(ps, ptr::null_mut()) {
                        r if r >= 0 => Ok(Context::Input(context::Input::wrap(ps))),
                        e => Err(Error::from(e)),
                    },

                    e => Err(Error::from(e)),
                }
            }

            Format::Output(ref format) => match ffi::avformat_alloc_output_context2(
                &mut ps,
                format.as_ptr() as *mut _,
                ptr::null(),
                path.as_ptr(),
            ) {
                0 => match ffi::avio_open(&mut (*ps).pb, path.as_ptr(), ffi::AVIO_FLAG_WRITE as i32) {
                    0 => Ok(Context::Output(context::Output::wrap(ps))),
                    e => Err(Error::from(e)),
                },

                e => Err(Error::from(e)),
            },
        }
    }
}

pub fn input<P: AsRef<Path> + ?Sized>(path: &P) -> Result<context::Input, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);

        match ffi::avformat_open_input(&mut ps, path.as_ptr(), ptr::null_mut(), ptr::null_mut()) {
            0 => match ffi::avformat_find_stream_info(ps, ptr::null_mut()) {
                r if r >= 0 => Ok(context::Input::wrap(ps)),
                e => {
                    ffi::avformat_close_input(&mut ps);
                    Err(Error::from(e))
                }
            },

            e => Err(Error::from(e)),
        }
    }
}

pub fn input_with_dictionary<P: AsRef<Path> + ?Sized>(
    path: &P,
    options: Dictionary,
) -> Result<context::Input, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);
        let mut opts = options.disown();
        let res = ffi::avformat_open_input(&mut ps, path.as_ptr(), ptr::null_mut(), &mut opts);

        Dictionary::own(opts);

        match res {
            0 => match ffi::avformat_find_stream_info(ps, ptr::null_mut()) {
                r if r >= 0 => Ok(context::Input::wrap(ps)),
                e => {
                    ffi::avformat_close_input(&mut ps);
                    Err(Error::from(e))
                }
            },

            e => Err(Error::from(e)),
        }
    }
}

pub fn input_with_interrupt<P: AsRef<Path> + ?Sized, F>(
    path: &P,
    closure: F,
) -> Result<context::Input, Error>
where
    F: FnMut() -> bool,
{
    unsafe {
        let mut ps = ffi::avformat_alloc_context();
        let path = from_path(path);
        (*ps).interrupt_callback = interrupt::new(Box::new(closure)).interrupt;

        match ffi::avformat_open_input(&mut ps, path.as_ptr(), ptr::null_mut(), ptr::null_mut()) {
            0 => match ffi::avformat_find_stream_info(ps, ptr::null_mut()) {
                r if r >= 0 => Ok(context::Input::wrap(ps)),
                e => {
                    ffi::avformat_close_input(&mut ps);
                    Err(Error::from(e))
                }
            },

            e => Err(Error::from(e)),
        }
    }
}

pub fn output<P: AsRef<Path> + ?Sized>(path: &P) -> Result<context::Output, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);

        match ffi::avformat_alloc_output_context2(&mut ps, ptr::null_mut(), ptr::null(), path.as_ptr()) {
            0 => match ffi::avio_open(&mut (*ps).pb, path.as_ptr(), ffi::AVIO_FLAG_WRITE as i32) {
                0 => Ok(context::Output::wrap(ps)),
                e => Err(Error::from(e)),
            },

            e => Err(Error::from(e)),
        }
    }
}

pub fn output_with<P: AsRef<Path> + ?Sized>(
    path: &P,
    options: Dictionary,
) -> Result<context::Output, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);
        let mut opts = options.disown();

        match ffi::avformat_alloc_output_context2(&mut ps, ptr::null_mut(), ptr::null(), path.as_ptr()) {
            0 => {
                let res = ffi::avio_open2(
                    &mut (*ps).pb,
                    path.as_ptr(),
                    ffi::AVIO_FLAG_WRITE as i32,
                    ptr::null(),
                    &mut opts,
                );

                Dictionary::own(opts);

                match res {
                    0 => Ok(context::Output::wrap(ps)),
                    e => Err(Error::from(e)),
                }
            }

            e => Err(Error::from(e)),
        }
    }
}

pub fn output_as<P: AsRef<Path> + ?Sized>(
    path: &P,
    format: &str,
) -> Result<context::Output, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);
        let format = CString::new(format).unwrap();

        match ffi::avformat_alloc_output_context2(
            &mut ps,
            ptr::null_mut(),
            format.as_ptr(),
            path.as_ptr(),
        ) {
            0 => match ffi::avio_open(&mut (*ps).pb, path.as_ptr(), ffi::AVIO_FLAG_WRITE as i32) {
                0 => Ok(context::Output::wrap(ps)),
                e => Err(Error::from(e)),
            },

            e => Err(Error::from(e)),
        }
    }
}

pub fn output_as_with<P: AsRef<Path> + ?Sized>(
    path: &P,
    format: &str,
    options: Dictionary,
) -> Result<context::Output, Error> {
    unsafe {
        let mut ps = ptr::null_mut();
        let path = from_path(path);
        let format = CString::new(format).unwrap();
        let mut opts = options.disown();

        match ffi::avformat_alloc_output_context2(
            &mut ps,
            ptr::null_mut(),
            format.as_ptr(),
            path.as_ptr(),
        ) {
            0 => {
                let res = ffi::avio_open2(
                    &mut (*ps).pb,
                    path.as_ptr(),
                    ffi::AVIO_FLAG_WRITE as i32,
                    ptr::null(),
                    &mut opts,
                );

                Dictionary::own(opts);

                match res {
                    0 => Ok(context::Output::wrap(ps)),
                    e => Err(Error::from(e)),
                }
            }

            e => Err(Error::from(e)),
        }
    }
}