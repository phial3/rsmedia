use crate::flags::AvDispositionFlags;
use crate::io::Reader;
use crate::options::{Dictionary, DictionaryRef};
use crate::packet::Packet;
use crate::Rational;

use rsmpeg::avcodec::{AVCodec, AVCodecParameters};
use rsmpeg::avformat::AVFormatContextInput;
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

use anyhow::Result;
use libc::{c_int, c_uint};
use std::marker::PhantomData;
use std::ops::Deref;

/// Holds transferable stream information. This can be used to duplicate stream settings for the
/// purpose of transmuxing or transcoding.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub index: usize,
    codec_parameters: AVCodecParameters,
    time_base: Rational,
}

impl StreamInfo {
    /// Fetch stream information from a reader by stream index.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader to find stream information from.
    /// * `stream_index` - Index of stream in reader.
    pub(crate) fn from_reader(reader: &Reader, stream_index: usize) -> Result<Self> {
        let stream =
            reader
                .input
                .streams()
                .get(stream_index)
                .ok_or(RsmpegError::FindStreamInfoError(
                    ffi::AVERROR_STREAM_NOT_FOUND,
                ))?;

        Self::from_params(
            stream.codecpar().to_owned(),
            stream.time_base.into(),
            stream_index,
        )
    }

    pub fn from_params(
        codecpar: AVCodecParameters,
        timebase: Rational,
        stream_index: usize,
    ) -> Result<Self> {
        Ok(Self {
            index: stream_index,
            codec_parameters: codecpar,
            time_base: timebase,
        })
    }

    /// Turn information back into parts for usage.
    ///
    /// Note: Consumes stream information object.
    ///
    /// # Return value
    ///
    /// A tuple consisting of:
    /// * The stream index.
    /// * Codec parameters.
    /// * Original stream time base.
    #[allow(unused)]
    pub(crate) fn into_parts(self) -> (usize, AVCodecParameters, Rational) {
        (self.index, self.codec_parameters, self.time_base)
    }
}

impl std::fmt::Display for StreamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let codec = {
            let mut c = AVCodec::find_encoder(self.codec_parameters.codec_id);
            if c.is_none() {
                c = AVCodec::find_decoder(self.codec_parameters.codec_id);
            }
            c.unwrap()
        };
        write!(
            f,
            "StreamInfo {{ index: {}, codec: {}:{}, time_base: {} }}",
            self.index,
            codec.name().to_str().unwrap(),
            codec.long_name().to_str().unwrap(),
            self.time_base
        )
    }
}

unsafe impl Send for StreamInfo {}
unsafe impl Sync for StreamInfo {}

/////////////////////////////////////////
pub struct StreamSideData<'a> {
    ptr: *mut ffi::AVPacketSideData,
    _marker: PhantomData<&'a Packet>,
}

impl StreamSideData<'_> {
    pub fn wrap(ptr: *mut ffi::AVPacketSideData) -> Self {
        StreamSideData {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const ffi::AVPacketSideData {
        self.ptr as *const _
    }
}

impl StreamSideData<'_> {
    pub fn kind(&self) -> ffi::AVPacketSideDataType {
        unsafe { ffi::AVPacketSideDataType::from((*self.as_ptr()).type_) }
    }

    pub fn data(&self) -> &[u8] {
        #[allow(clippy::unnecessary_cast)]
        unsafe {
            std::slice::from_raw_parts((*self.as_ptr()).data, (*self.as_ptr()).size as usize)
        }
    }
}

////////////////////////////////////////
// #[derive(Debug)]
pub struct Stream<'a> {
    context: &'a AVFormatContextInput,
    index: usize,
}

impl Stream<'_> {
    pub fn wrap(context: &AVFormatContextInput, index: usize) -> Stream {
        Stream { context, index }
    }

    pub fn as_ptr(&self) -> *const ffi::AVStream {
        unsafe { *(*self.context.as_ptr()).streams.add(self.index) }
    }
}

impl Stream<'_> {
    pub fn id(&self) -> i32 {
        unsafe { (*self.as_ptr()).id }
    }

    // pub fn parameters(&self) -> codec::Parameters {
    //     unsafe {
    //         codec::Parameters::wrap((*self.as_ptr()).codecpar, Some(self.context.destructor()))
    //     }
    // }

    pub fn index(&self) -> usize {
        unsafe { (*self.as_ptr()).index as usize }
    }

    pub fn time_base(&self) -> Rational {
        unsafe { Rational::from((*self.as_ptr()).time_base) }
    }

    pub fn start_time(&self) -> i64 {
        unsafe { (*self.as_ptr()).start_time }
    }

    pub fn duration(&self) -> i64 {
        unsafe { (*self.as_ptr()).duration }
    }

    pub fn frames(&self) -> i64 {
        unsafe { (*self.as_ptr()).nb_frames }
    }

    pub fn disposition(&self) -> AvDispositionFlags {
        unsafe { AvDispositionFlags::from_bits_truncate((*self.as_ptr()).disposition as c_uint) }
    }

    pub fn discard(&self) -> ffi::AVDiscard {
        unsafe { ffi::AVDiscard::from((*self.as_ptr()).discard) }
    }

    pub fn side_data(&self) -> StreamSideDataIter {
        StreamSideDataIter::new(self)
    }

    pub fn rate(&self) -> Rational {
        unsafe { Rational::from((*self.as_ptr()).r_frame_rate) }
    }

    pub fn avg_frame_rate(&self) -> Rational {
        unsafe { Rational::from((*self.as_ptr()).avg_frame_rate) }
    }

    pub fn metadata(&self) -> DictionaryRef {
        unsafe { DictionaryRef::wrap((*self.as_ptr()).metadata) }
    }
}

impl PartialEq for Stream<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

impl Eq for Stream<'_> {}

pub struct StreamSideDataIter<'a> {
    stream: &'a Stream<'a>,
    current: c_int,
}

impl StreamSideDataIter<'_> {
    pub fn new<'sd, 's: 'sd>(stream: &'s Stream) -> StreamSideDataIter<'sd> {
        StreamSideDataIter { stream, current: 0 }
    }
}

impl<'a> Iterator for StreamSideDataIter<'a> {
    type Item = StreamSideData<'a>;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        unsafe {
            if self.current >= (*self.stream.as_ptr()).nb_side_data {
                return None;
            }

            self.current += 1;

            Some(StreamSideData::wrap(
                (*self.stream.as_ptr())
                    .side_data
                    .offset((self.current - 1) as isize),
            ))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let length = (*self.stream.as_ptr()).nb_side_data as usize;

            (
                length - self.current as usize,
                Some(length - self.current as usize),
            )
        }
    }
}

impl ExactSizeIterator for StreamSideDataIter<'_> {}

/////////////////////////////////////////
pub struct StreamMut<'a> {
    context: &'a mut AVFormatContextInput,
    index: usize,
    immutable: Stream<'a>,
}

impl StreamMut<'_> {
    /// Wraps a mutable reference to `AVFormatContextInput` and an index into a `StreamMut`.
    pub fn wrap(context: &mut AVFormatContextInput, index: usize) -> StreamMut {
        unsafe {
            StreamMut {
                context: std::mem::transmute_copy(&context),
                index,
                immutable: Stream::wrap(std::mem::transmute_copy(&context), index),
            }
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::AVStream {
        unsafe { *(*self.context.as_mut_ptr()).streams.add(self.index) }
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

    // pub fn set_parameters<P: Into<codec::Parameters>>(&mut self, parameters: P) {
    //     let parameters = parameters.into();
    //
    //     unsafe {
    //         avcodec_parameters_copy((*self.as_mut_ptr()).codecpar, parameters.as_ptr());
    //     }
    // }

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
