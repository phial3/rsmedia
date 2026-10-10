//! Streaming PCM capture into an audio encoder.
//!
//! [`PcmSink`] bridges a realtime interleaved PCM source (a `cpal` capture
//! callback, a WAV reader, a DSP chain) into rsmedia's audio
//! [`Encoder`](crate::encode::Encoder) + [`Muxer`] pipeline:
//!
//! - interleaved blocks of any size and of `f32` / `i16` / `u8` samples are
//!   wrapped into packed [`AVFrame`]s, so the caller never aligns chunks to the
//!   encoder's `frame_size` and never touches `unsafe`;
//! - a block whose rate, channel layout or sample format differs from the
//!   encoder's is converted by the encoder itself — see
//!   [What the encoder already does](#what-the-encoder-already-does);
//! - the encoder's own FIFO accumulates the encoder's fixed frame size (1024
//!   samples for AAC) and numbers output packets;
//! - [`PcmSink::finish`] flushes the encoder and writes the trailer, then hands
//!   the writer back.
//!
//! # Example
//!
//! ```no_run
//! use rsmedia::error::Result;
//! fn main() -> Result<()> {
//!     use rsmedia::mux::Muxer;
//!     use rsmedia::pcm::{PcmSink, PcmSpec};
//!     use rsmedia::{Encoder, SampleFormat};
//!
//!     // 1. build the encoder for the *output* spec (AAC by default) and mux it
//!     let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
//!     let mut muxer = Muxer::new("out.m4a")?;
//!     let audio_index = muxer.add_encoder(encoder)?;
//!
//!     // 2. describe the *source* (e.g. a microphone); differences are converted
//!     let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(48_000, 2))?;
//!
//!     // 3. hand it whatever the capture callback produced
//!     sink.write_f32(&vec![0.0f32; 1024])?;
//!
//!     // 4. flush and write the trailer (Drop is a fallback, but errors are lost)
//!     sink.finish()?;
//!     Ok(())
//! }
//! ```
//!
//! A complete runnable `cpal` recording example lives in `examples/pcm_recorder.rs`.
//!
//! # What the encoder already does
//!
//! [`PcmSink`] deliberately owns **no** resampler and assigns **no** timestamps.
//! [`Encoder::encode_raw`](crate::encode::Encoder::encode_raw) already:
//!
//! - resamples to its own rate / layout / sample format, keeping one persistent
//!   `SwrContext` so filter delay is not lost between blocks;
//! - accumulates fixed-size audio frames in an `AVAudioFifo`;
//! - numbers output packets by samples actually emitted.
//!
//! Doing any of that here would be a second implementation of the same policy:
//! this sink's frames already match the encoder's spec by the time they are
//! muxed, so the encoder's `resample_if_needed` short-circuits and this type's
//! numbering would be discarded. Passing frames straight through keeps one owner
//! per concern, and leaves the encoder's filter graph seeing the source format.
//!
//! # Reading the bytes back
//!
//! `write_*` and [`finish`](PcmSink::finish) return `Result<()>`, never bytes:
//! output stays in the muxer's writer and is pulled from there. Which pull to
//! use depends on the container:
//!
//! * streaming formats (mpegts, matroska, …): take each segment with
//!   [`BufferWriter::take_written`](crate::io::BufferWriter::take_written)
//!   between writes and send it out;
//! * formats that rewrite their header while writing the trailer (mp4, mov,
//!   wav): the complete image is only visible through
//!   [`finish`](PcmSink::finish) → [`into_bytes`](crate::io::BufferWriter::into_bytes),
//!   because the trailer rewrites bytes already handed out by `take_written`.
//!
//! ```no_run
//! # use rsmedia::error::Result;
//! # use rsmedia::io::{BufferReader, BufferWriter};
//! # use rsmedia::mux::Muxer;
//! # use rsmedia::pcm::{PcmSink, PcmSpec};
//! # use rsmedia::{Encoder, SampleFormat};
//! # fn main() -> Result<()> {
//! let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
//! let mut muxer = Muxer::new_from_writer(BufferWriter::new("mpegts")?);
//! let idx = muxer.add_encoder(encoder)?;
//! let mut sink = PcmSink::new(muxer, idx, PcmSpec::new(48_000, 2))?;
//! for _ in 0..10 {
//!     sink.write_f32(&[0.0f32; 1024])?;
//!     let _segment = sink.writer_mut().take_written();   // send it out
//! }
//! let mut writer = sink.finish()?;
//! let _tail = writer.take_written();
//! # Ok(())
//! # }
//! ```
//!
//! # Pauses and the timeline
//!
//! There is no external clock: output timing is a running sample count, so a gap
//! in the input is *closed up* rather than left as silence. To keep a recording
//! aligned with wall-clock time, **write the silence explicitly** — feed as many
//! zero samples as the pause lasted:
//!
//! ```no_run
//! # use rsmedia::error::Result;
//! # use rsmedia::pcm::{PcmSink, PcmSpec};
//! # fn f(sink: &mut PcmSink<rsmedia::io::StreamWriter>) -> Result<()> {
//! let paused_samples = 3 * 48_000;                 // 3 s at 48 kHz, 1 channel
//! sink.write_f32(&vec![0.0f32; paused_samples])?;  // 2 channels -> `* 2`
//! # Ok(())
//! # }
//! ```
//!
//! # Splitting a long recording
//!
//! [`finish`](PcmSink::finish) consumes the sink, so each segment needs a fresh
//! `Encoder` + `Muxer` + `PcmSink`. Setup is cheap and no samples are lost at the
//! cut (the tail is drained), but the *resampler state* is not carried across:
//! since the encoder owns the resampler, a new segment starts from a cold delay
//! line. For a recording that must stay phase-exact end to end, record one file
//! and cut it afterwards, or use
//! [`Muxer::new_segmented`](crate::mux::Muxer::new_segmented).
//!
//! # Not `Send`
//!
//! `PcmSink` is **not** `Send`, so it cannot be moved to another thread. Feed it
//! from a capture callback over a channel instead of calling `write_*` inside the
//! callback — that is what `examples/pcm_recorder.rs` does, and it is a
//! requirement rather than a style choice.

use crate::error::{Context, Result, RsmediaError};
use crate::io::Writer;
use crate::mux::Muxer;
use crate::stream::MediaType;

use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::{AVChannelLayout, AVFrame};
use rsmpeg::ffi;

/// Source PCM specification for a realtime interleaved capture device.
///
/// The sample *format* is chosen by the method used to write
/// ([`write_f32`](PcmSink::write_f32) / [`write_i16`](PcmSink::write_i16) /
/// [`write_u8`](PcmSink::write_u8) / [`write_planar_f32`](PcmSink::write_planar_f32));
/// this type describes the rate, the channel count and an optional channel mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmSpec {
    /// Source sample rate in Hz, e.g. cpal's `SampleRate(48_000)`.
    pub sample_rate: u32,
    /// Channel count; `2` for interleaved stereo.
    pub channels: u32,
    /// Channel layout mask (a bitwise or of `AV_CH_*`), `0` for **unspecified** —
    /// FFmpeg's default layout for [`Self::channels`] is then used.
    ///
    /// A plain value rather than `ffi::AVChannelLayout`, which holds raw pointers
    /// and would make this `Copy` spec a lifetime-bearing object.
    ///
    /// Only needed for 3+ channels whose layout is *not* the default one: a 2.1
    /// `FL+FR+LFE` is normalized to `FL+FR+FC` by the default-layout rule. A mask
    /// that disagrees with [`Self::channels`] is rejected on the first write.
    pub channel_mask: u64,
}

impl PcmSpec {
    /// Build a spec from a sample rate and channel count, leaving the layout at
    /// FFmpeg's default for that many channels.
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            channel_mask: 0,
        }
    }

    /// Set an explicit channel layout mask (a bitwise or of `AV_CH_*`).
    ///
    /// `0` means unspecified, which is the same as [`Self::new`].
    pub const fn with_channel_mask(mut self, channel_mask: u64) -> Self {
        self.channel_mask = channel_mask;
        self
    }
}

/// Streaming PCM → [`Encoder`](crate::encode::Encoder) → [`Muxer`] bridge.
///
/// Owns the [`Muxer`]; `write_*` takes blocks at capture-callback granularity and
/// [`finish`](Self::finish) flushes and writes the trailer. Forgetting `finish`
/// still yields a valid container — the muxer's own `Drop` flushes the encoder
/// and writes the trailer — but the error is unobservable, so calling `finish`
/// is the recommended path.
///
/// This type intentionally implements **no** `Drop`: it must be able to move the
/// inner [`Muxer`] out wholesale ([`finish`](Self::finish)), which a type with a
/// `Drop` impl cannot do. Finalization is the muxer's own `Drop`, so the
/// semantics are unchanged.
pub struct PcmSink<W: Writer> {
    muxer: Muxer<W>,
    stream_index: usize,
    spec: PcmSpec,
    /// Input samples written so far, per channel.
    input_samples: u64,
}

impl<W: Writer> PcmSink<W> {
    /// Wraps an existing [`Muxer`] and binds it to one of its audio streams.
    ///
    /// # Arguments
    ///
    /// * `muxer` - muxer owning the audio [`Encoder`](crate::encode::Encoder),
    ///   as returned by [`Muxer::add_encoder`].
    /// * `stream_index` - index of that audio stream.
    /// * `spec` - the source's sample rate and channel count; the write method
    ///   chooses the sample format. Anything that differs from the encoder's own
    ///   spec is converted by the encoder.
    ///
    /// # Errors
    ///
    /// [`RsmediaError::InvalidConfig`] if the spec is degenerate
    /// (`sample_rate == 0` or `channels == 0`), if `stream_index` does not exist,
    /// if it names a non-audio stream, or if it names a bit-exact copy stream —
    /// which has no encoder to feed.
    pub fn new(muxer: Muxer<W>, stream_index: usize, spec: PcmSpec) -> Result<Self> {
        if spec.sample_rate == 0 || spec.channels == 0 {
            return Err(RsmediaError::invalid_config(format!(
                "invalid PCM spec: sample_rate={}, channels={}",
                spec.sample_rate, spec.channels
            )));
        }
        let stream = muxer.get_stream(stream_index)?;
        if stream.media_type != MediaType::AUDIO {
            return Err(RsmediaError::invalid_config(format!(
                "stream {stream_index} is {:?}, not AUDIO",
                stream.media_type
            )));
        }
        if stream.encoder.is_none() {
            return Err(RsmediaError::invalid_config(format!(
                "stream {stream_index} is a copy stream; PCM capture requires an encoder stream"
            )));
        }
        Ok(Self {
            muxer,
            stream_index,
            spec,
            input_samples: 0,
        })
    }

    /// Write an interleaved `f32` block, as delivered by cpal's
    /// `SampleFormat::F32` capture callback.
    ///
    /// The bytes stay in the underlying writer; see
    /// [Reading the bytes back](Self#reading-the-bytes-back).
    pub fn write_f32(&mut self, interleaved: &[f32]) -> Result<()> {
        self.write_interleaved(interleaved, ffi::AV_SAMPLE_FMT_FLT)
    }

    /// Write an interleaved `i16` block (cpal's `SampleFormat::I16`).
    ///
    /// The bytes stay in the underlying writer, as in [`Self::write_f32`].
    pub fn write_i16(&mut self, interleaved: &[i16]) -> Result<()> {
        self.write_interleaved(interleaved, ffi::AV_SAMPLE_FMT_S16)
    }

    /// Write an interleaved `u8` block (unsigned 8-bit, as in `AV_SAMPLE_FMT_U8`).
    ///
    /// The bytes stay in the underlying writer, as in [`Self::write_f32`].
    pub fn write_u8(&mut self, interleaved: &[u8]) -> Result<()> {
        self.write_interleaved(interleaved, ffi::AV_SAMPLE_FMT_U8)
    }

    /// Write **planar** `f32` PCM: one slice per channel, `planes[0]` being
    /// channel 0's contiguous samples.
    ///
    /// Use [`Self::write_f32`] for interleaved input; this variant spares a
    /// de-interleaving pass when the data is already planar (a decoder's output,
    /// a `MediaFrame`'s planes, a DSP chain's).
    ///
    /// All slices must be equal length, and there must be exactly
    /// [`PcmSpec::channels`] of them. A misaligned length is an error rather than
    /// a partial write.
    ///
    /// The bytes stay in the underlying writer, as in [`Self::write_f32`].
    pub fn write_planar_f32(&mut self, planes: &[&[f32]]) -> Result<()> {
        self.write_planar(planes, ffi::AV_SAMPLE_FMT_FLTP)
    }

    /// Flush the encoder, write the trailer, and hand the writer back.
    ///
    /// Consumes the sink, so the bytes can *only* be obtained from the returned
    /// writer: a file-backed writer can drop it, while a buffered one exposes the
    /// complete container through
    /// [`BufferWriter::into_bytes`](crate::io::BufferWriter::into_bytes).
    ///
    /// Without this call, `Drop` still leaves a readable container but discards
    /// the error, so calling it explicitly is recommended.
    pub fn finish(mut self) -> Result<W> {
        self.muxer.finish()?;
        Ok(self.muxer.into_writer())
    }

    /// The underlying [`Muxer`]'s writer, mutably borrowed.
    ///
    /// Used to pull incremental bytes for a stream-as-you-go sink; writing must
    /// still go through `write_*`, which is where frame construction and
    /// validation live. This is only for reading bytes out.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.muxer.writer
    }

    /// Source samples written so far, per channel.
    pub fn input_samples(&self) -> u64 {
        self.input_samples
    }

    /// The source PCM specification this sink was built with.
    pub fn spec(&self) -> PcmSpec {
        self.spec
    }

    /// Source channel layout: an explicit mask wins, otherwise FFmpeg's default
    /// for the declared channel count.
    fn input_layout(&self) -> Result<ffi::AVChannelLayout> {
        if self.spec.channel_mask == 0 {
            // FFmpeg's channel count is an `int`; `channels` is validated non-zero.
            return Ok(AVChannelLayout::from_nb_channels(self.spec.channels as i32).into_inner());
        }
        let layout = AVChannelLayout::from_mask(self.spec.channel_mask).ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "invalid channel mask {:#x}: FFmpeg rejected it",
                self.spec.channel_mask
            ))
        })?;
        let raw = layout.into_inner();
        if raw.nb_channels != self.spec.channels as i32 {
            return Err(RsmediaError::invalid_config(format!(
                "channel mask {:#x} describes {} channels, but the spec declares {}",
                self.spec.channel_mask, raw.nb_channels, self.spec.channels
            )));
        }
        Ok(raw)
    }

    /// Allocate an input frame in the *source* spec, ready to be filled and muxed.
    ///
    /// No pts is set: the encoder numbers its output by the samples it actually
    /// emitted, which is the only count that stays correct when it resamples.
    fn alloc_input_frame(
        &self,
        nb_samples: i32,
        sample_format: ffi::AVSampleFormat,
    ) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_format(sample_format);
        frame.set_nb_samples(nb_samples);
        // FFmpeg's sample rate is an `int`; the rate is validated non-zero.
        frame.set_sample_rate(self.spec.sample_rate as i32);
        frame.set_ch_layout(self.input_layout()?);
        frame
            .alloc_buffer()
            .context("Failed to allocate PCM input frame buffer")?;
        Ok(frame)
    }

    fn write_interleaved<T: Copy>(
        &mut self,
        interleaved: &[T],
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        let channels = self.spec.channels as usize;
        if !interleaved.len().is_multiple_of(channels) {
            return Err(RsmediaError::invalid_config(format!(
                "interleaved PCM length {} is not a multiple of {channels} channels",
                interleaved.len(),
            )));
        }
        let nb_samples = interleaved.len() / channels;
        if nb_samples == 0 {
            return Ok(());
        }
        Self::check_element_width::<T>(sample_format)?;

        let mut frame = self.alloc_input_frame(nb_samples as i32, sample_format)?;

        // SAFETY: `frame` was just allocated for `nb_samples` samples of the
        // source layout (exactly `self.spec.channels` channels) in `sample_format`,
        // so `data[0]` is the start of the packed samples and holds
        // `nb_samples * channels == interleaved.len()` of them. `check_element_width`
        // above established that `T` matches the format's element width, and the
        // copy is exactly the slice's length, so it stays inside the buffer.
        // `frame` is a local owned by this function (reference count 1).
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                frame.deref_mut().data[0] as *mut T,
                interleaved.len(),
            );
            dst.copy_from_slice(interleaved);
        }
        self.input_samples += nb_samples as u64;
        self.muxer.mux(frame, self.stream_index)
    }

    fn write_planar<T: Copy>(
        &mut self,
        planes: &[&[T]],
        sample_format: ffi::AVSampleFormat,
    ) -> Result<()> {
        let channels = self.spec.channels as usize;
        if planes.len() != channels {
            return Err(RsmediaError::invalid_config(format!(
                "planar input has {} plane(s) but the spec declares {channels} channel(s)",
                planes.len()
            )));
        }
        let nb_samples = planes.first().map_or(0, |p| p.len());
        if nb_samples == 0 {
            return Ok(());
        }
        if let Some(bad) = planes.iter().position(|p| p.len() != nb_samples) {
            return Err(RsmediaError::invalid_config(format!(
                "planar plane {bad} has {} samples, plane 0 has {nb_samples}: \
                 planes must be equal-length",
                planes[bad].len()
            )));
        }
        Self::check_element_width::<T>(sample_format)?;

        let mut frame = self.alloc_input_frame(nb_samples as i32, sample_format)?;

        // SAFETY: `frame` was just allocated for `nb_samples` samples across the
        // source layout's channels, and planar formats give channel `i` its own
        // `data[i]` plane, so each of the `channels` planes holds exactly
        // `nb_samples` elements of width `size_of::<T>()` (checked above).
        // `zip` over the frame's data pointers stops at the shorter side, and
        // every copy is exactly `nb_samples` long, so all writes stay in bounds.
        // `frame` is a local owned by this function (reference count 1).
        unsafe {
            let raw = frame.deref_mut();
            for (plane, channel) in planes.iter().zip(raw.data.iter()) {
                let dst = std::slice::from_raw_parts_mut(*channel as *mut T, nb_samples);
                dst.copy_from_slice(plane);
            }
        }
        self.input_samples += nb_samples as u64;
        self.muxer.mux(frame, self.stream_index)
    }

    /// `T` must match the element width of `sample_format`.
    ///
    /// Both write paths fill a buffer allocated for `sample_format` with
    /// `interleaved.len()` (resp. `nb_samples` per plane) elements of `T`; a
    /// width mismatch is therefore a **heap out-of-bounds write**. This invariant
    /// previously rested only on `write_f32`/`write_i16`/`write_u8` pairing each
    /// `T` with the right format, so a new wrapper pairing them wrongly would
    /// overrun. Check it explicitly instead.
    fn check_element_width<T: Copy>(sample_format: ffi::AVSampleFormat) -> Result<()> {
        let element_bytes = rsmpeg::avutil::get_bytes_per_sample(sample_format).unwrap_or(0);
        if std::mem::size_of::<T>() != element_bytes {
            return Err(RsmediaError::invalid_config(format!(
                "PCM element type is {} byte(s) but {sample_format} stores {element_bytes}",
                std::mem::size_of::<T>()
            )));
        }
        Ok(())
    }
}

impl<W: Writer> std::fmt::Debug for PcmSink<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcmSink")
            .field("stream_index", &self.stream_index)
            .field("spec", &self.spec)
            .field("input_samples", &self.input_samples)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{BufferWriter, Reader};
    use crate::mux::Demuxer;
    use crate::{Encoder, EncoderBuilder, SampleFormat, test_support};

    /// Interleaved f32 sine (amplitude 0.3), same phase on every channel.
    fn sine_samples(
        start_sample: u64,
        len_per_channel: usize,
        channels: u32,
        rate: u32,
    ) -> Vec<f32> {
        let mut samples = Vec::with_capacity(len_per_channel * channels as usize);
        for i in 0..len_per_channel {
            let t = (start_sample + i as u64) as f32 / rate as f32;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
            for _ in 0..channels {
                samples.push(v);
            }
        }
        samples
    }

    /// Decode a file, returning the audio stream's (samples, rate, channels).
    /// Rate and channel count come back as `u32` to match the public API.
    fn decode_audio_stream(path: &std::path::Path) -> Result<(usize, u32, u32)> {
        let demuxer = Demuxer::new(path)?;
        let (audio_stream_index, sample_rate, channels) = {
            let s = demuxer
                .streams()
                .iter()
                .find(|s| s.media_type == MediaType::AUDIO)
                .expect("no audio stream in output");
            (
                s.stream_index,
                s.stream_info.sample_rate,
                s.stream_info.channel_layout.nb_channels,
            )
        };

        let mut total_samples = 0usize;
        for res in demuxer {
            let (index, frame) = res?;
            if index == audio_stream_index {
                total_samples += frame.nb_samples as usize;
            }
        }
        let channels = u32::try_from(channels)
            .map_err(|_| RsmediaError::invalid_config("decoded channel count out of range"))?;
        Ok((total_samples, sample_rate as u32, channels))
    }

    /// Read the codec id straight from the container's stream parameters.
    fn output_codec_id(path: &std::path::Path) -> ffi::AVCodecID {
        let reader = crate::StreamReader::new(path).expect("reopen output");
        reader.input().streams()[0].codecpar().codec_id
    }

    /// f32 blocks of irregular size → AAC. The encoder's FIFO does the framing,
    /// so the decoded length must track the input regardless of block size.
    #[test]
    fn test_pcm_sink_f32_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_f32.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 44_100usize; // 1 second

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        // irregular block sizes: covers the FIFO accumulating to AAC's 1024
        let mut written = 0usize;
        for chunk in [1000usize, 123, 7777, 512, 34_688] {
            if written >= total_in {
                break;
            }
            let n = chunk.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, rate))?;
            written += n;
        }
        assert_eq!(sink.input_samples() as usize, total_in);
        sink.finish()?;

        assert_eq!(output_codec_id(&output_path), ffi::AV_CODEC_ID_AAC);
        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        // AAC has priming/padding; allow one frame_size of slack
        let encoder_frame = 1024usize;
        assert!(
            (samples as i64 - total_in as i64).abs() <= encoder_frame as i64,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// The sink hands frames over in the *source* spec; rate and channel-count
    /// conversion is the encoder's job. 48kHz mono into a 44.1kHz stereo encoder
    /// must still land at 44.1kHz stereo and ~1s of audio.
    #[test]
    fn test_encoder_resamples_source_spec() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_resample.m4a");
        test_support::remove_test_output(&output_path);

        let (in_rate, in_channels) = (48_000u32, 1u32);
        let (out_rate, out_channels) = (44_100u32, 2u32);
        let in_total = 48_000usize; // 1 second

        let encoder = Encoder::new_audio(out_channels, out_rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(in_rate, in_channels))?;

        // a typical cpal callback granularity
        let mut written = 0usize;
        while written < in_total {
            let n = 512.min(in_total - written);
            sink.write_f32(&sine_samples(written as u64, n, in_channels, in_rate))?;
            written += n;
        }
        // the source counter tracks what was written, not what was encoded
        assert_eq!(sink.input_samples() as usize, in_total);
        sink.finish()?;

        let (samples, decoded_rate, decoded_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(decoded_rate, out_rate);
        assert_eq!(decoded_channels, out_channels);
        let expected = in_total as f64 * out_rate as f64 / in_rate as f64;
        assert!(
            (samples as f64 - expected).abs() <= 1024.0,
            "decoded {samples} samples, expected ~{expected}"
        );

        Ok(())
    }

    /// i16 path: interleaved S16 → AAC, decoded length matches the input (±1 frame).
    #[test]
    fn test_pcm_sink_i16_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_i16.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 second

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 2048.min(total_in - written);
            let mut chunk = Vec::with_capacity(n * channels as usize);
            for i in 0..n {
                let t = (written + i) as f32 / rate as f32;
                let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                for _ in 0..channels {
                    chunk.push((v * i16::MAX as f32 * 0.3) as i16);
                }
            }
            sink.write_i16(&chunk)?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// u8 path: interleaved U8 (128 is the silence midpoint) → AAC.
    #[test]
    fn test_pcm_sink_u8_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_u8.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 1u32);
        let total_in = 22_050usize; // 0.5 second

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 960.min(total_in - written);
            let chunk: Vec<u8> = (0..n)
                .map(|i| {
                    let t = (written + i) as f32 / rate as f32;
                    (128.0 + (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 40.0) as u8
                })
                .collect();
            sink.write_u8(&chunk)?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// A filter graph declared on the encoder sees the *source* format, so a
    /// chain declaring packed `FLT` input while the source is packed `f32` needs
    /// no extra `aformat` round trip inside the graph.
    #[test]
    fn test_pcm_sink_with_encoder_filter() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_filter_input.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 second

        let filter = crate::filter::Filter::new(
            "aformat",
            MediaType::AUDIO,
            "aformat=sample_fmts=fltp".to_string(),
        )
        .with_input_format(SampleFormat::FLT);

        // the filter may be absent from this FFmpeg build; then there is
        // nothing to exercise
        if crate::filter::get_by_name(filter.name())?.is_none() {
            return Ok(());
        }

        let encoder = EncoderBuilder::new_audio(128_000, channels, rate, SampleFormat::FLTP)
            .with_filters(vec![filter])
            .build()?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 1000.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, rate))?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );

        Ok(())
    }

    /// mp4 seeks back to rewrite its header while writing the trailer, so a file
    /// assembled from incremental reads does not open; only the writer that
    /// `finish` hands back holds the complete image.
    #[test]
    fn test_finish_yields_a_readable_mp4() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_finish.mp4");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 second

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new_from_writer(BufferWriter::new("mp4")?);
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        let mut written = 0usize;
        while written < total_in {
            let n = 1000.min(total_in - written);
            sink.write_f32(&sine_samples(written as u64, n, channels, rate))?;
            written += n;
        }
        let bytes: Vec<u8> = sink.finish()?.into_bytes();
        assert!(!bytes.is_empty(), "mp4 output must not be empty");
        std::fs::write(&output_path, &bytes)?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );
        Ok(())
    }

    /// Planar input: two channels carrying different frequencies. Proves both
    /// planes are written and that per-plane lengths are honoured.
    #[test]
    fn test_pcm_sink_planar_f32_to_aac() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_planar.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 22_050usize; // 0.5 second

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        // 440 Hz left / 220 Hz right
        let mut written = 0usize;
        while written < total_in {
            let n = 1000.min(total_in - written);
            let mut left = Vec::with_capacity(n);
            let mut right = Vec::with_capacity(n);
            for i in 0..n {
                let t = (written + i) as f32 / rate as f32;
                left.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3);
                right.push((2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3);
            }
            sink.write_planar_f32(&[&left, &right])?;
            written += n;
        }
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );
        Ok(())
    }

    /// Planar validation: wrong plane count and unequal plane lengths are errors.
    #[test]
    fn test_planar_input_validation() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_planar_invalid.m4a");
        test_support::remove_test_output(&output_path);
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(44_100, 2))?;

        // plane count != channel count
        assert!(sink.write_planar_f32(&[&[0.0f32; 64]]).is_err());
        // unequal plane lengths
        assert!(
            sink.write_planar_f32(&[&[0.0f32; 64], &[0.0f32; 32]])
                .is_err()
        );
        // empty planes are a no-op
        assert!(sink.write_planar_f32(&[&[], &[]]).is_ok());
        Ok(())
    }

    /// Channel mask: rejected when it disagrees with the channel count, and the
    /// decoded channel count is preserved when it agrees.
    #[test]
    fn test_channel_mask() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_mask.m4a");
        test_support::remove_test_output(&output_path);

        // stereo mask (AV_CH_FRONT_LEFT | AV_CH_FRONT_RIGHT = 0x3)
        const AV_CH_LAYOUT_STEREO: u64 = 0x3;

        // the mask says 2 channels, the spec says 1 => reject
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(
            muxer,
            audio_index,
            PcmSpec::new(44_100, 1).with_channel_mask(AV_CH_LAYOUT_STEREO),
        )?;
        assert!(
            sink.write_f32(&[0.0f32; 128]).is_err(),
            "a mask that disagrees with the channel count must be rejected"
        );
        drop(sink);

        // agreeing mask: the whole path runs through
        test_support::remove_test_output(&output_path);
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(
            muxer,
            audio_index,
            PcmSpec::new(44_100, 2).with_channel_mask(AV_CH_LAYOUT_STEREO),
        )?;
        let mut written = 0usize;
        while written < 22_050 {
            let n = 1000.min(22_050 - written);
            sink.write_f32(&sine_samples(written as u64, n, 2, 44_100))?;
            written += n;
        }
        sink.finish()?;
        let (_, _, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_channels, 2);
        Ok(())
    }

    /// A channel count the default-layout rule cannot express (2.1) needs an
    /// explicit mask, and it must survive into the output.
    #[test]
    fn test_explicit_surround_mask_is_honored() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_surround.m4a");
        test_support::remove_test_output(&output_path);

        // FL+FR+LFE = 0x4 | 0x8 | 0x10000
        const AV_CH_LAYOUT_2_1: u64 = 0x4 | 0x8 | 0x1_0000;

        let encoder = Encoder::new_audio(3, 48_000, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(
            muxer,
            audio_index,
            PcmSpec::new(48_000, 3).with_channel_mask(AV_CH_LAYOUT_2_1),
        )?;

        let mut written = 0usize;
        while written < 24_000 {
            let n = 1024.min(24_000 - written);
            sink.write_f32(&sine_samples(written as u64, n, 3, 48_000))?;
            written += n;
        }
        sink.finish()?;

        let (_, _, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(
            out_channels, 3,
            "the explicit 2.1 layout must reach the output"
        );
        Ok(())
    }

    /// Spec and state validation: non-audio streams, out-of-range indices,
    /// degenerate specs and channel-misaligned blocks are all rejected; empty
    /// blocks are a no-op.
    #[test]
    fn test_pcm_sink_validation() -> Result<()> {
        // a degenerate spec fails before the muxer writes anything
        let assert_invalid_spec = |spec: PcmSpec| -> Result<()> {
            let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
            test_support::remove_test_output(&output_path);
            let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
            let mut muxer = Muxer::new(&output_path)?;
            let audio_index = muxer.add_encoder(encoder)?;
            assert!(PcmSink::new(muxer, audio_index, spec).is_err());
            Ok(())
        };
        assert_invalid_spec(PcmSpec::new(0, 2))?;
        assert_invalid_spec(PcmSpec::new(44_100, 0))?;

        // a non-audio stream is refused
        let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
        test_support::remove_test_output(&output_path);
        let video_encoder = EncoderBuilder::new_video(64, 64).build()?;
        let mut muxer = Muxer::new(&output_path)?;
        let video_index = muxer.add_encoder(video_encoder)?;
        assert!(PcmSink::new(muxer, video_index, PcmSpec::new(44_100, 2)).is_err());

        // an out-of-range stream index is refused
        let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        assert!(PcmSink::new(muxer, audio_index + 10, PcmSpec::new(44_100, 2)).is_err());

        // a channel-misaligned interleaved block is refused; empty is a no-op
        let output_path = test_support::test_output_path("pcm", "test_pcm_invalid.m4a");
        let encoder = Encoder::new_audio(2, 44_100, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(44_100, 2))?;
        assert!(sink.write_f32(&[0.0f32; 3]).is_err());
        assert!(sink.write_f32(&[]).is_ok());
        assert_eq!(sink.input_samples(), 0);
        assert_eq!(sink.spec(), PcmSpec::new(44_100, 2));
        Ok(())
    }

    /// A block far larger than any encoder frame size is handed over in one
    /// piece; the encoder's FIFO is what splits it, so no sample may be lost.
    #[test]
    fn test_large_single_block() -> Result<()> {
        let output_path = test_support::test_output_path("pcm", "test_pcm_large.m4a");
        test_support::remove_test_output(&output_path);

        let (rate, channels) = (44_100u32, 2u32);
        let total_in = 44_100usize; // 1 second, written in a single call

        let encoder = Encoder::new_audio(channels, rate, SampleFormat::FLTP)?;
        let mut muxer = Muxer::new(&output_path)?;
        let audio_index = muxer.add_encoder(encoder)?;
        let mut sink = PcmSink::new(muxer, audio_index, PcmSpec::new(rate, channels))?;

        sink.write_f32(&sine_samples(0, total_in, channels, rate))?;
        assert_eq!(sink.input_samples() as usize, total_in);
        sink.finish()?;

        let (samples, out_rate, out_channels) = decode_audio_stream(&output_path)?;
        assert_eq!(out_rate, rate);
        assert_eq!(out_channels, channels);
        assert!(
            (samples as i64 - total_in as i64).abs() <= 1024,
            "decoded {samples} samples, expected ~{total_in}"
        );
        Ok(())
    }
}
