use std::ptr;

use sys::ffi;

use super::Context;
use crate::{Error, Frame};

pub struct Source<'a> {
    ctx: &'a mut Context,
}

impl<'a> Source<'a> {
    pub unsafe fn wrap(ctx: &'a mut Context) -> Self {
        Self { ctx }
    }
}

impl Source<'_> {
    pub fn failed_requests(&self) -> usize {
        unsafe { ffi::av_buffersrc_get_nb_failed_requests(self.ctx.as_ptr() as *mut _) as usize }
    }

    pub fn add(&mut self, frame: &Frame) -> Result<(), Error> {
        unsafe {
            match ffi::av_buffersrc_add_frame(self.ctx.as_mut_ptr(), frame.as_ptr() as *mut _) {
                0 => Ok(()),
                e => Err(Error::from(e)),
            }
        }
    }

    pub fn flush(&mut self) -> Result<(), Error> {
        unsafe { self.add(&Frame::wrap(ptr::null_mut())) }
    }

    pub fn close(&mut self, pts: i64) -> Result<(), Error> {
        unsafe {
            match ffi::av_buffersrc_close(self.ctx.as_mut_ptr(), pts, 0) {
                0 => Ok(()),
                e => Err(Error::from(e)),
            }
        }
    }
}
