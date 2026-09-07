//! Subtitle stream handling: passthrough and encoding helpers.
//!
//! Subtitle encoding is provided by the generic [`crate::encode::Encoder`]
//! (see [`EncoderBuilder::new_subtitle`](crate::encode::EncoderBuilder::new_subtitle)),
//! which uses the standard `avcodec_open2` + `avcodec_encode_subtitle` path
//! with an auto-generated ASS script header. This module provides:
//!
//! * [`copy_subtitle_stream`] — passthrough: copy subtitle packets from a
//!   reader to a writer without decode/encode (same codec, e.g. subrip→subrip).
//! * [`SubtitleSegment`] — a plain text segment used as the input to the
//!   generic encoder's `encode_subtitle_segment`.
//!
//! # Note on subtitle encoders
//!
//! Subtitle encoders (mov_text, subrip) require `subtitle_header` (an ASS
//! `[Script Info]`/`[V4+ Styles]` header) to be set before `avcodec_open2`,
//! otherwise init fails with `AVERROR_INVALIDDATA`. The generic encoder
//! generates a default header automatically.
//!
//! FFmpeg Documentation: <https://ffmpeg.org/doxygen/trunk/group__lavc__subtitle.html>

use crate::error::Result;
use crate::io::{Reader, Writer};

use rsmpeg::ffi;

/// Copy subtitle packets from a reader stream to a writer stream (passthrough).
///
/// Reads packets from `reader`, filters for `src_index`, rescales timestamps
/// to the output stream's time_base, and writes them to `writer` at
/// `out_index`. No decode/encode — the codec is preserved as-is.
///
/// Returns the number of packets copied.
pub fn copy_subtitle_stream<R: Reader, W: Writer>(
    reader: &mut R,
    writer: &mut W,
    src_index: usize,
    out_index: usize,
) -> Result<usize> {
    let src_tb = reader
        .input()
        .streams()
        .get(src_index)
        .map(|s| s.time_base)
        .unwrap_or(ffi::AVRational { num: 1, den: 1000 });

    let out_tb = writer
        .output()
        .streams()
        .get(out_index)
        .map(|s| s.time_base)
        .unwrap_or(src_tb);

    let mut count = 0usize;
    while let Some((stream_index, mut packet)) = reader.read_packet()? {
        if stream_index != src_index {
            continue;
        }
        packet.rescale_ts(src_tb, out_tb);
        packet.set_stream_index(out_index as i32);
        packet.set_pos(-1);
        writer.write_interleaved(&mut packet)?;
        count += 1;
    }
    Ok(count)
}

/// A single subtitle text segment with start/end times.
#[derive(Debug, Clone)]
pub struct SubtitleSegment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Subtitle text (plain UTF-8 text).
    pub text: String,
}

impl SubtitleSegment {
    /// Create a new subtitle segment.
    pub fn new(start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::EncoderBuilder;
    use crate::error::RsmediaError;
    use crate::io::StreamReader;
    use crate::io::private::{Output, Write};
    use crate::test_support;
    use crate::time::Rescale;

    /// 测试用完整 ASS 脚本头：字幕编码器（ass/subrip/mov_text）open 时以此
    /// 初始化样式解析（ff_ass_split），缺失或残缺会导致 open 失败或
    /// Dialogue 字段错位。真实转码流程应从解码侧传递 header。
    /// `subtitle_header` 必须包含完整的 `[Script Info]` 和 `[V4+ Styles]` 段
    const ASS_HEADER: &str = "[Script Info]\n\
         ScriptType: v4.00+\n\
         \n\
         [V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, \
         OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, \
         ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
         Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n\
         \n\
         [Events]\n\
         Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

    /// Create a few sample subtitle segments for testing.
    fn sample_segments() -> Vec<SubtitleSegment> {
        vec![
            SubtitleSegment::new(0, 2000, "Hello World"),
            SubtitleSegment::new(2000, 4000, "Second subtitle"),
            SubtitleSegment::new(4000, 6000, "Third subtitle"),
        ]
    }

    /// Encode subtitles as mov_text into an MP4 file via the generic Encoder,
    /// then read back and **strictly verify** via decode roundtrip:
    /// 1. subtitle stream exists with codec id `AV_CODEC_ID_MOV_TEXT`;
    /// 2. every segment survives encode → mux → demux → decode (text intact);
    /// 3. packet pts, rescaled from the output stream time_base to the
    ///    encoder's 1/1000 time base, equals the segment start time (ms).
    #[test]
    fn test_mov_text_encode_and_readback() -> Result<()> {
        use rsmpeg::avcodec::{AVCodec, AVCodecContext};

        let path = test_support::test_output_path("subtitle", "rsmedia_mov_text.mp4");
        test_support::remove_test_output(&path);

        // 1) Write: create an MP4 with a mov_text subtitle stream
        let segments = sample_segments();
        let mut encoder = EncoderBuilder::new_subtitle()
            .with_codec_name(Some("mov_text".to_string()))
            .with_subtitle_header(ASS_HEADER)
            .build_wrapped(path.as_path())?;
        encoder.encode_subtitle_segments(&segments)?;
        encoder.finish()?;

        // 2) Read back: verify the subtitle stream exists
        let mut reader = StreamReader::new(path.as_path())?;
        let (index, _) = reader.find_best_stream(crate::MediaType::SUBTITLE)?;
        let (codec_id, stream_tb) = {
            let stream = reader.input().streams().get(index).unwrap();
            (stream.codecpar().codec_id, stream.time_base)
        };
        assert_eq!(
            codec_id,
            ffi::AV_CODEC_ID_MOV_TEXT,
            "subtitle codec should be mov_text, got codec_id={codec_id}"
        );

        // 3) Decode roundtrip: demux packets -> decode_subtitle -> rect payload
        let decoder = AVCodec::find_decoder(codec_id)
            .ok_or_else(|| RsmediaError::custom("mov_text decoder not available"))?;
        let mut dctx = AVCodecContext::new(&decoder);
        dctx.open(None)?;

        let mut texts: Vec<String> = Vec::new();
        let mut start_ms: Vec<i64> = Vec::new();
        while let Some((stream_index, mut packet)) = reader.read_packet()? {
            if stream_index != index {
                continue;
            }
            // pts (stream time_base) -> milliseconds (encoder time base 1/1000)
            start_ms.push(
                packet
                    .pts
                    .rescale(stream_tb, crate::time::new_rational(1, 1000)),
            );
            if let Some(subtitle) = dctx.decode_subtitle(Some(&mut packet))? {
                for rect in subtitle.rect_iter() {
                    if let Some(ass) = rect.ass() {
                        texts.push(ass.to_string_lossy().to_string());
                    }
                }
            }
        }

        assert_eq!(
            start_ms,
            vec![0, 2000, 4000],
            "packet pts must round-trip to the segment start times"
        );
        let all = texts.join("\n");
        assert!(all.contains("Hello World"), "decoded text: {all:?}");
        assert!(all.contains("Second subtitle"), "decoded text: {all:?}");
        assert!(all.contains("Third subtitle"), "decoded text: {all:?}");

        test_support::remove_test_output(&path);
        Ok(())
    }

    /// Encode subtitles as subrip into a .srt file via the generic Encoder,
    /// and **strictly verify** the complete SRT structure: serial numbers,
    /// timestamp lines (`HH:MM:SS,mmm --> HH:MM:SS,mmm`, generated end-to-end
    /// from packet pts/duration) and payload text.
    #[test]
    fn test_subrip_encode() -> Result<()> {
        let path = test_support::test_output_path("subtitle", "rsmedia_subrip.srt");
        test_support::remove_test_output(&path);

        let segments = sample_segments();
        let mut encoder = EncoderBuilder::new_subtitle()
            .with_subtitle_header(ASS_HEADER)
            .build_wrapped(path.as_path())?;
        encoder.encode_subtitle_segments(&segments)?;
        encoder.finish()?;

        let content = std::fs::read_to_string(&path)?;
        // 时间戳行：由 packet pts/duration（毫秒）端到端生成，验证整条时间戳链路
        assert!(
            content.contains("00:00:00,000 --> 00:00:02,000"),
            "missing first timestamp line, content:\n{content}"
        );
        assert!(
            content.contains("00:00:02,000 --> 00:00:04,000"),
            "missing second timestamp line, content:\n{content}"
        );
        assert!(
            content.contains("00:00:04,000 --> 00:00:06,000"),
            "missing third timestamp line, content:\n{content}"
        );
        // 文本行
        assert!(content.contains("Hello World"), "content:\n{content}");
        assert!(content.contains("Second subtitle"), "content:\n{content}");
        assert!(content.contains("Third subtitle"), "content:\n{content}");
        // 序号行（SRT 序号 1..N）
        for serial in ["1", "2", "3"] {
            assert!(
                content.lines().any(|l| l.trim() == serial),
                "missing SRT serial {serial}, content:\n{content}"
            );
        }

        test_support::remove_test_output(&path);
        Ok(())
    }

    /// Passthrough test: encode subrip → MKV via the generic Encoder, then
    /// copy the subtitle stream to another MKV.
    #[test]
    fn test_subtitle_passthrough() -> Result<()> {
        // 1) Create an MKV with subrip subtitles
        let input_path = test_support::test_output_path("subtitle", "rsmedia_passthrough_in.mkv");
        test_support::remove_test_output(&input_path);

        let segments = sample_segments();
        let mut encoder = EncoderBuilder::new_subtitle()
            .with_subtitle_header(ASS_HEADER)
            .build_wrapped(input_path.as_path())?;
        encoder.encode_subtitle_segments(&segments)?;
        encoder.finish()?;

        // 2) Read the MKV and copy the subtitle stream to another MKV
        let output_path = test_support::test_output_path("subtitle", "rsmedia_passthrough_out.mkv");
        test_support::remove_test_output(&output_path);

        let mut reader = StreamReader::new(input_path.as_path())?;
        let mut out_writer = crate::io::StreamWriter::new(output_path.as_path())?;

        // Find subtitle stream in input
        let (src_index, _) = reader.find_best_stream(crate::MediaType::SUBTITLE)?;

        // Copy codec parameters to output stream (clone to release the borrow
        // on reader before calling copy_subtitle_stream which needs &mut reader).
        let (codecpar, src_tb) = {
            let src_stream = reader.input().streams().get(src_index).unwrap();
            (src_stream.codecpar().clone(), src_stream.time_base)
        };
        let out_index = out_writer.add_stream(codecpar, src_tb);

        // Write header, copy packets, write trailer
        out_writer.write_header()?;
        let count = copy_subtitle_stream(&mut reader, &mut out_writer, src_index, out_index)?;
        out_writer.write_trailer()?;

        assert_eq!(count, segments.len(), "should copy all subtitle packets");

        // 3) Verify output has the subtitle stream with correct codec
        let out_reader = StreamReader::new(output_path.as_path())?;
        let subtitle_streams: Vec<_> = out_reader
            .input()
            .streams()
            .iter()
            .filter(|s| s.codecpar().codec_type().is_subtitle())
            .collect();
        assert!(
            !subtitle_streams.is_empty(),
            "output should have subtitle stream"
        );

        test_support::remove_test_output(&input_path);
        test_support::remove_test_output(&output_path);
        Ok(())
    }

    /// Encode subtitles with the **ass** encoder into an MKV
    /// (S_TEXT/ASS), then strictly verify via decode roundtrip, mirroring the
    /// mov_text test: codec id, text payload, and pts round-trip.
    #[test]
    fn test_ass_encode_and_readback() -> Result<()> {
        use rsmpeg::avcodec::{AVCodec, AVCodecContext};

        let path = test_support::test_output_path("subtitle", "rsmedia_ass.mkv");
        test_support::remove_test_output(&path);

        // 1) Write: create an MKV with an ASS subtitle stream
        let segments = sample_segments();
        let mut encoder = EncoderBuilder::new_subtitle()
            .with_codec_name(Some("ass".to_string()))
            .with_subtitle_header(ASS_HEADER)
            .build_wrapped(path.as_path())?;
        encoder.encode_subtitle_segments(&segments)?;
        encoder.finish()?;

        // 2) Read back: verify the subtitle stream exists
        let mut reader = StreamReader::new(path.as_path())?;
        let (index, _) = reader.find_best_stream(crate::MediaType::SUBTITLE)?;
        let (codec_id, stream_tb) = {
            let stream = reader.input().streams().get(index).unwrap();
            (stream.codecpar().codec_id, stream.time_base)
        };
        assert_eq!(
            codec_id,
            ffi::AV_CODEC_ID_ASS,
            "subtitle codec should be ass, got codec_id={codec_id}"
        );

        // 3) Decode roundtrip: demux packets -> decode_subtitle -> rect payload
        let decoder = AVCodec::find_decoder(codec_id)
            .ok_or_else(|| RsmediaError::custom("ass decoder not available"))?;
        let mut dctx = AVCodecContext::new(&decoder);
        dctx.open(None)?;

        let mut texts: Vec<String> = Vec::new();
        let mut start_ms: Vec<i64> = Vec::new();
        while let Some((stream_index, mut packet)) = reader.read_packet()? {
            if stream_index != index {
                continue;
            }
            // pts (stream time_base) -> milliseconds (encoder time base 1/1000)
            start_ms.push(
                packet
                    .pts
                    .rescale(stream_tb, crate::time::new_rational(1, 1000)),
            );
            if let Some(subtitle) = dctx.decode_subtitle(Some(&mut packet))? {
                for rect in subtitle.rect_iter() {
                    if let Some(ass) = rect.ass() {
                        texts.push(ass.to_string_lossy().to_string());
                    }
                }
            }
        }

        assert_eq!(
            start_ms,
            vec![0, 2000, 4000],
            "packet pts must round-trip to the segment start times"
        );
        let all = texts.join("\n");
        assert!(all.contains("Hello World"), "decoded text: {all:?}");
        assert!(all.contains("Second subtitle"), "decoded text: {all:?}");
        assert!(all.contains("Third subtitle"), "decoded text: {all:?}");

        test_support::remove_test_output(&path);
        Ok(())
    }

    /// 编码器类型不匹配时应报错而非 panic。
    #[test]
    fn test_encode_subtitle_segment_wrong_media_type() -> Result<()> {
        let mut encoder = crate::encode::Encoder::new_video(64, 64)?;
        let err = encoder
            .encode_subtitle_segment(&SubtitleSegment::new(0, 1000, "x"))
            .unwrap_err();
        assert!(err.to_string().contains("subtitle encoder"));
        Ok(())
    }

    /// 未提供 ASS header 时 build() 应报明确错误（而非 avcodec_open2 的
    /// AVERROR_INVALIDDATA）。
    #[test]
    fn test_missing_subtitle_header_is_rejected() -> Result<()> {
        let err = match EncoderBuilder::new_subtitle().build() {
            Err(e) => e,
            Ok(_) => return Err(RsmediaError::custom("build should fail without header")),
        };
        assert!(err.to_string().contains("header"));
        Ok(())
    }
}
