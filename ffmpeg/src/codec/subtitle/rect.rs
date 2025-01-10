use std::ffi::CStr;
use std::marker::PhantomData;
use std::str::from_utf8_unchecked;

use super::{Flags, Type};
use sys::ffi::*;

// #[cfg(not(feature = "ffmpeg5"))]
// use crate::{format, Picture};

pub enum Rect<'a> {
    None(*const AVSubtitleRect),
    Bitmap(Bitmap<'a>),
    Text(Text<'a>),
    Ass(Ass<'a>),
}

impl Rect<'_> {
    pub unsafe fn wrap(ptr: *const AVSubtitleRect) -> Self {
        match Type::from((*ptr).type_) {
            Type::None => Rect::None(ptr),
            Type::Bitmap => Rect::Bitmap(Bitmap::wrap(ptr)),
            Type::Text => Rect::Text(Text::wrap(ptr)),
            Type::Ass => Rect::Ass(Ass::wrap(ptr)),
        }
    }

    pub unsafe fn as_ptr(&self) -> *const AVSubtitleRect {
        match *self {
            Rect::None(ptr) => ptr,
            Rect::Bitmap(ref b) => b.as_ptr(),
            Rect::Text(ref t) => t.as_ptr(),
            Rect::Ass(ref a) => a.as_ptr(),
        }
    }
}

impl Rect<'_> {
    pub fn flags(&self) -> Flags {
        unsafe {
            Flags::from_bits_truncate(match *self {
                Rect::None(ptr) => (*ptr).flags,
                Rect::Bitmap(ref b) => (*b.as_ptr()).flags,
                Rect::Text(ref t) => (*t.as_ptr()).flags,
                Rect::Ass(ref a) => (*a.as_ptr()).flags,
            })
        }
    }
}

pub struct Bitmap<'a> {
    ptr: *const AVSubtitleRect,

    _marker: PhantomData<&'a ()>,
}

impl Bitmap<'_> {
    pub unsafe fn wrap(ptr: *const AVSubtitleRect) -> Self {
        Bitmap {
            ptr,
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_ptr(&self) -> *const AVSubtitleRect {
        self.ptr
    }
}

impl Bitmap<'_> {
    pub fn x(&self) -> usize {
        unsafe { (*self.as_ptr()).x as usize }
    }

    pub fn y(&self) -> usize {
        unsafe { (*self.as_ptr()).y as usize }
    }

    pub fn width(&self) -> u32 {
        unsafe { (*self.as_ptr()).w as u32 }
    }

    pub fn height(&self) -> u32 {
        unsafe { (*self.as_ptr()).h as u32 }
    }

    pub fn colors(&self) -> usize {
        unsafe { (*self.as_ptr()).nb_colors as usize }
    }

    // XXX: must split Picture and PictureMut
    // #[cfg(not(feature = "ffmpeg5"))]
    // pub fn picture(&self, format: crate::format::Pixel) -> Picture<'a> {
    //     unsafe {
    //         Picture::wrap(
    //             &(*self.as_ptr()).pict as *const _ as *mut _,
    //             format,
    //             (*self.as_ptr()).w as u32,
    //             (*self.as_ptr()).h as u32,
    //         )
    //     }
    // }
}

pub struct Text<'a> {
    ptr: *const AVSubtitleRect,

    _marker: PhantomData<&'a ()>,
}

impl Text<'_> {
    pub unsafe fn wrap(ptr: *const AVSubtitleRect) -> Self {
        Text {
            ptr,
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_ptr(&self) -> *const AVSubtitleRect {
        self.ptr
    }
}

impl Text<'_> {
    pub fn get(&self) -> &str {
        unsafe { from_utf8_unchecked(CStr::from_ptr((*self.as_ptr()).text).to_bytes()) }
    }
}

pub struct Ass<'a> {
    ptr: *const AVSubtitleRect,

    _marker: PhantomData<&'a ()>,
}

impl Ass<'_> {
    pub unsafe fn wrap(ptr: *const AVSubtitleRect) -> Self {
        Ass {
            ptr,
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_ptr(&self) -> *const AVSubtitleRect {
        self.ptr
    }
}

impl Ass<'_> {
    pub fn get(&self) -> &str {
        unsafe { from_utf8_unchecked(CStr::from_ptr((*self.as_ptr()).ass).to_bytes()) }
    }
}
