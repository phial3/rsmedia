use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use super::immutable;
use sys::ffi;

pub struct Ref<'a> {
    ptr: *mut ffi::AVDictionary,
    imm: immutable::Ref<'a>,

    _marker: PhantomData<&'a ()>,
}

impl Ref<'_> {
    pub unsafe fn wrap(ptr: *mut ffi::AVDictionary) -> Self {
        Ref {
            ptr,
            imm: immutable::Ref::wrap(ptr),
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_mut_ptr(&self) -> *mut ffi::AVDictionary {
        self.ptr
    }
}

impl Ref<'_> {
    pub fn set(&mut self, key: &str, value: &str) {
        unsafe {
            let key = CString::new(key).unwrap();
            let value = CString::new(value).unwrap();
            let mut ptr = self.as_mut_ptr();

            if ffi::av_dict_set(&mut ptr, key.as_ptr(), value.as_ptr(), 0) < 0 {
                panic!("out of memory");
            }

            self.ptr = ptr;
            self.imm = immutable::Ref::wrap(ptr);
        }
    }
}

impl<'a> Deref for Ref<'a> {
    type Target = immutable::Ref<'a>;

    fn deref(&self) -> &Self::Target {
        &self.imm
    }
}

impl fmt::Debug for Ref<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.imm.fmt(fmt)
    }
}
