use std::ffi::{CStr, CString};
use std::ops::Index;
use std::ptr;
use std::slice;

use libc::{c_int, c_void};
use rsmpeg::ffi;

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum Sample {
    None,

    U8(Type),
    I16(Type),
    I32(Type),
    I64(Type),
    F32(Type),
    F64(Type),
}

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum Type {
    Packed,
    Planar,
}

impl Sample {
    #[inline]
    pub fn name(&self) -> &'static str {
        unsafe {
            std::str::from_utf8_unchecked(CStr::from_ptr(ffi::av_get_sample_fmt_name((*self).into())).to_bytes())
        }
    }

    #[inline]
    pub fn packed(&self) -> Self {
        unsafe { Sample::from(ffi::av_get_packed_sample_fmt((*self).into())) }
    }

    #[inline]
    pub fn planar(&self) -> Self {
        unsafe { Sample::from(ffi::av_get_planar_sample_fmt((*self).into())) }
    }

    #[inline]
    pub fn is_planar(&self) -> bool {
        unsafe { ffi::av_sample_fmt_is_planar((*self).into()) == 1 }
    }

    #[inline]
    pub fn is_packed(&self) -> bool {
        !self.is_planar()
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        unsafe { ffi::av_get_bytes_per_sample((*self).into()) as usize }
    }

    #[inline]
    pub fn buffer(&self, channels: u16, samples: usize, align: bool) -> Buffer {
        Buffer::new(*self, channels, samples, align)
    }
}

impl From<ffi::AVSampleFormat> for Sample {
    #[inline]
    fn from(value: ffi::AVSampleFormat) -> Self {
        match value {
            ffi::AV_SAMPLE_FMT_NONE => Sample::None,

            ffi::AV_SAMPLE_FMT_U8 => Sample::U8(Type::Packed),
            ffi::AV_SAMPLE_FMT_S16 => Sample::I16(Type::Packed),
            ffi::AV_SAMPLE_FMT_S32 => Sample::I32(Type::Packed),
            ffi::AV_SAMPLE_FMT_S64 => Sample::I64(Type::Packed),
            ffi::AV_SAMPLE_FMT_FLT => Sample::F32(Type::Packed),
            ffi::AV_SAMPLE_FMT_DBL => Sample::F64(Type::Packed),

            ffi::AV_SAMPLE_FMT_U8P => Sample::U8(Type::Planar),
            ffi::AV_SAMPLE_FMT_S16P => Sample::I16(Type::Planar),
            ffi::AV_SAMPLE_FMT_S32P => Sample::I32(Type::Planar),
            ffi::AV_SAMPLE_FMT_S64P => Sample::I64(Type::Planar),
            ffi::AV_SAMPLE_FMT_FLTP => Sample::F32(Type::Planar),
            ffi::AV_SAMPLE_FMT_DBLP => Sample::F64(Type::Planar),

            ffi::AV_SAMPLE_FMT_NB => Sample::None,

            i32::MIN..=-2_i32 | 13_i32..=i32::MAX => todo!(),
        }
    }
}

impl From<&'static str> for Sample {
    #[inline]
    fn from(value: &'static str) -> Self {
        unsafe {
            let value = CString::new(value).unwrap();

            Sample::from(ffi::av_get_sample_fmt(value.as_ptr()))
        }
    }
}

impl From<Sample> for ffi::AVSampleFormat {
    #[inline]
    fn from(value: Sample) -> ffi::AVSampleFormat {
        match value {
            Sample::None => ffi::AV_SAMPLE_FMT_NONE,

            Sample::U8(Type::Packed) => ffi::AV_SAMPLE_FMT_U8,
            Sample::I16(Type::Packed) => ffi::AV_SAMPLE_FMT_S16,
            Sample::I32(Type::Packed) => ffi::AV_SAMPLE_FMT_S32,
            Sample::I64(Type::Packed) => ffi::AV_SAMPLE_FMT_S64,
            Sample::F32(Type::Packed) => ffi::AV_SAMPLE_FMT_FLT,
            Sample::F64(Type::Packed) => ffi::AV_SAMPLE_FMT_DBL,

            Sample::U8(Type::Planar) => ffi::AV_SAMPLE_FMT_U8P,
            Sample::I16(Type::Planar) => ffi::AV_SAMPLE_FMT_S16P,
            Sample::I32(Type::Planar) => ffi::AV_SAMPLE_FMT_S32P,
            Sample::I64(Type::Planar) => ffi::AV_SAMPLE_FMT_S64P,
            Sample::F32(Type::Planar) => ffi::AV_SAMPLE_FMT_FLTP,
            Sample::F64(Type::Planar) => ffi::AV_SAMPLE_FMT_DBLP,
        }
    }
}

pub struct Buffer {
    pub format: Sample,
    pub channels: u16,
    pub samples: usize,
    pub align: bool,

    buffer: *mut *mut u8,
    size: c_int,
}

impl Buffer {
    #[inline]
    pub fn size(format: Sample, channels: u16, samples: usize, align: bool) -> usize {
        unsafe {
            ffi::av_samples_get_buffer_size(
                ptr::null_mut(),
                i32::from(channels),
                samples as c_int,
                format.into(),
                !align as c_int,
            ) as usize
        }
    }

    #[inline]
    pub fn new(format: Sample, channels: u16, samples: usize, align: bool) -> Self {
        unsafe {
            let mut buf = Buffer {
                format,
                channels,
                samples,
                align,

                buffer: ptr::null_mut(),
                size: 0,
            };

            ffi::av_samples_alloc_array_and_samples(
                &mut buf.buffer,
                &mut buf.size,
                i32::from(channels),
                samples as c_int,
                format.into(),
                !align as c_int,
            );

            buf
        }
    }
}

impl Index<usize> for Buffer {
    type Output = [u8];

    #[inline]
    fn index(&self, index: usize) -> &[u8] {
        if index >= self.samples {
            panic!("out of bounds");
        }

        unsafe { slice::from_raw_parts(*self.buffer.add(index), self.size as usize) }
    }
}

impl Clone for Buffer {
    #[inline]
    fn clone(&self) -> Self {
        let mut buf = Buffer::new(self.format, self.channels, self.samples, self.align);
        buf.clone_from(self);

        buf
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        unsafe {
            ffi::av_samples_copy(
                self.buffer,
                source.buffer as *const *mut u8,
                0,
                0,
                source.samples as c_int,
                i32::from(source.channels),
                source.format.into(),
            );
        }
    }
}

impl Drop for Buffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            ffi::av_freep(self.buffer as *mut c_void);
        }
    }
}