use std::ops::Deref;

use sys::ffi;

use super::Stream;
use crate::{codec, format::context::common::Context, Dictionary, Rational};

pub struct StreamMut<'a> {
    context: &'a mut Context,
    index: usize,

    immutable: Stream<'a>,
}

impl StreamMut<'_> {
    pub unsafe fn wrap(context: &mut Context, index: usize) -> StreamMut {
        StreamMut {
            context: std::mem::transmute_copy(&context),
            index,

            immutable: Stream::wrap(std::mem::transmute_copy(&context), index),
        }
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut ffi::AVStream {
        *(*self.context.as_mut_ptr()).streams.add(self.index)
    }
}

impl StreamMut<'_> {
    pub fn set_time_base<R: Into<Rational>>(&mut self, value: R) {
        unsafe {
            (*self.as_mut_ptr()).time_base = value.into().into();
        }
    }

    pub fn set_rate<R: Into<Rational>>(&mut self, value: R) {
        unsafe {
            (*self.as_mut_ptr()).r_frame_rate = value.into().into();
        }
    }

    pub fn set_avg_frame_rate<R: Into<Rational>>(&mut self, value: R) {
        unsafe {
            (*self.as_mut_ptr()).avg_frame_rate = value.into().into();
        }
    }

    pub fn set_parameters<P: Into<codec::Parameters>>(&mut self, parameters: P) {
        let parameters = parameters.into();

        unsafe {
            ffi::avcodec_parameters_copy((*self.as_mut_ptr()).codecpar, parameters.as_ptr());
        }
    }

    pub fn set_metadata(&mut self, metadata: Dictionary) {
        unsafe {
            let metadata = metadata.disown();
            (*self.as_mut_ptr()).metadata = metadata;
        }
    }
}

impl<'a> Deref for StreamMut<'a> {
    type Target = Stream<'a>;

    fn deref(&self) -> &Self::Target {
        &self.immutable
    }
}
