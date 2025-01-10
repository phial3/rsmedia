use libc::c_int;
use sys::ffi;

use super::Context;
use crate::{Error, Frame, Rational};

pub struct Sink<'a> {
    ctx: &'a mut Context,
}

impl<'a> Sink<'a> {
    pub unsafe fn wrap(ctx: &'a mut Context) -> Self {
        Self { ctx }
    }
}

impl Sink<'_> {
    pub fn frame(&mut self, frame: &mut Frame) -> Result<(), Error> {
        unsafe {
            match ffi::av_buffersink_get_frame(self.ctx.as_mut_ptr(), frame.as_mut_ptr()) {
                n if n >= 0 => Ok(()),
                e => Err(Error::from(e)),
            }
        }
    }

    pub fn samples(&mut self, frame: &mut Frame, samples: usize) -> Result<(), Error> {
        unsafe {
            match ffi::av_buffersink_get_samples(
                self.ctx.as_mut_ptr(),
                frame.as_mut_ptr(),
                samples as c_int,
            ) {
                n if n >= 0 => Ok(()),
                e => Err(Error::from(e)),
            }
        }
    }

    pub fn set_frame_size(&mut self, value: u32) {
        unsafe {
            ffi::av_buffersink_set_frame_size(self.ctx.as_mut_ptr(), value);
        }
    }

    pub fn time_base(&self) -> Rational {
        unsafe { ffi::av_buffersink_get_time_base(self.ctx.as_ptr()) }.into()
    }
}
