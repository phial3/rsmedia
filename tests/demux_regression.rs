//! Regression suite for two `Demuxer` defects found by the 2026-10 architecture
//! review and reproduced end-to-end (SIGABRT / silent stream misbinding) before
//! the fix:
//!
//! 1. **Decode-all mode × subtitle streams.** A subtitle decoder was built for
//!    the subtitle stream, but subtitle codecs only support the
//!    `avcodec_decode_subtitle2` channel — not the send/receive *frame* API
//!    that `demux()` drives. Routing a subtitle packet through the frame API
//!    tripped a libavcodec assertion and `abort()`ed the process (no Rust
//!    `Err` to catch, no panic message, exit 134). The fix skips subtitle
//!    streams in decode-all mode; the dedicated subtitle decode path
//!    (`Decoder::new_subtitle` + `decode_subtitle_segment`) and the
//!    copy/remux path (`new_passthrough`) keep working.
//! 2. **Per-stream decoder binding.** `DecoderBuilder::build_from_reader` used
//!    `find_best_stream(media_type)` for codecpar/time-base wiring while the
//!    `Demuxer` explicitly passed only the codec name. In containers with two
//!    streams of the same media type (here: 44.1 kHz + 8 kHz AAC) the second
//!    stream's decoder was configured against the *best* stream's parameters
//!    while being fed the second stream's packets — the 8 kHz track decoded
//!    into frames claiming 44.1 kHz. The fix adds
//!    `DecoderBuilder::with_stream_index`, and `Demuxer` binds each selected
//!    stream's decoder to its own index.
//!
//! Asset generation mirrors `tests/container_roundtrip.rs` (high-level Muxer +
//! subtitle segments; generated in-memory, no external files needed).

mod common;

use common::{gradient_video_frame, sine_audio_frame, test_output_path};
use rsmedia::io::Reader;
use rsmedia::mux::Demuxer;
use rsmedia::{EncoderBuilder, MediaType, Muxer, StreamReader, SubtitleSegment};

/// ASS header the subtitle encoders require before `avcodec_open2`
/// (same minimal script used by `tests/container_roundtrip.rs`).
const SUBTITLE_HEADER: &str = "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const FPS: f32 = 25.0;
const VIDEO_FRAMES: usize = 12;

/// Muxes a video + `mov_text` track into `path` (MP4).
fn mux_video_with_subtitle(path: &std::path::Path) -> anyhow::Result<()> {
    let mut muxer = Muxer::new(path)?;

    let video = EncoderBuilder::new_video(WIDTH, HEIGHT)
        .with_fps(FPS)
        .with_codec_name("libx264")
        .build()?;
    let vtb = video.time_base();

    let subtitle = EncoderBuilder::new_subtitle()
        .with_codec_name("mov_text")
        .with_subtitle_header(SUBTITLE_HEADER)
        .build()?;

    let v_idx = muxer.add_encoder(video)?;
    let s_idx = muxer.add_encoder(subtitle)?;

    for (start, end, text) in [(0, 1500, "第一个 cue"), (1500, 3000, "第二个 cue")] {
        muxer.mux_subtitle_segment(&SubtitleSegment::new(start, end, text), s_idx)?;
    }
    for i in 0..VIDEO_FRAMES {
        let mut av =
            gradient_video_frame(WIDTH, HEIGHT, i as f32 / VIDEO_FRAMES as f32).to_avframe()?;
        av.set_pts(i as i64); // encoder time base = 1/FPS
        av.set_time_base(vtb.into());
        muxer.mux(av, v_idx)?;
    }
    muxer.finish()?;
    Ok(())
}

/// Muxes a video + two AAC streams (44.1 kHz stereo, 8 kHz mono) into `path`.
fn mux_video_two_audio(path: &std::path::Path) -> anyhow::Result<()> {
    let mut muxer = Muxer::new(path)?;

    let video = EncoderBuilder::new_video(WIDTH, HEIGHT)
        .with_fps(30.0)
        .with_codec_name("libx264")
        .build()?;
    let audio_44 = EncoderBuilder::new_audio(96_000, 2, 44_100, rsmedia::SampleFormat::FLTP)
        .with_codec_name("aac")
        .build()?;
    let audio_8k = EncoderBuilder::new_audio(32_000, 1, 8_000, rsmedia::SampleFormat::FLTP)
        .with_codec_name("aac")
        .build()?;

    // Each encoder's input time base (= its own 1/sample_rate for audio,
    // 1/fps for video) before the encoders are moved into the muxer.
    let vtb = video.time_base();
    let tb44 = audio_44.time_base();
    let tb8k = audio_8k.time_base();
    let v_idx = muxer.add_encoder(video)?;
    let a44_idx = muxer.add_encoder(audio_44)?;
    let a8k_idx = muxer.add_encoder(audio_8k)?;

    let video_frames = (3.0f64 * 30.0) as usize;
    let frames_44 = (3.0f64 * 44_100.0 / 1024.0).ceil() as usize;
    let frames_8k = (3.0f64 * 8_000.0 / 1024.0).ceil() as usize;
    for i in 0..frames_44 {
        if i < video_frames {
            let mut av =
                gradient_video_frame(WIDTH, HEIGHT, i as f32 / video_frames as f32).to_avframe()?;
            av.set_pts(i as i64);
            av.set_time_base(vtb.into());
            muxer.mux(av, v_idx)?;
        }
        let mut a = sine_audio_frame(440.0, 2, 1024, 44_100).to_avframe()?;
        a.set_pts((i * 1024) as i64);
        a.set_time_base(tb44.into());
        muxer.mux(a, a44_idx)?;

        if i < frames_8k {
            let mut a = sine_audio_frame(440.0, 1, 1024, 8_000).to_avframe()?;
            a.set_pts((i * 1024) as i64);
            a.set_time_base(tb8k.into());
            muxer.mux(a, a8k_idx)?;
        }
    }
    muxer.finish()?;
    Ok(())
}

// ===========================================================================
// 1. Decode-all × subtitle
// ===========================================================================

#[test]
fn demux_decode_all_skips_subtitle_stream() -> anyhow::Result<()> {
    let path = test_output_path("demux_regression", "sub.mp4");
    let _ = std::fs::remove_file(&path);
    mux_video_with_subtitle(&path)?;

    let mut demuxer = Demuxer::new(&path)?;

    // Decode-all must not hand out a decoder for the subtitle stream: the
    // frame API that `demux()` drives aborts (SIGABRT) on the first routed
    // subtitle packet.
    let selected: Vec<(usize, MediaType)> = demuxer
        .streams()
        .iter()
        .map(|s| (s.stream_index, s.media_type))
        .collect();
    assert!(
        selected.iter().all(|(_, t)| *t != MediaType::SUBTITLE),
        "decode-all mode must skip subtitle streams, got {selected:?}"
    );

    // Full iteration completes without error; the video frames survive.
    let mut video_frames = 0usize;
    while let Some((idx, _frame)) = demuxer.demux()? {
        assert_eq!(idx, 0, "only the video stream should yield frames");
        video_frames += 1;
    }
    assert_eq!(
        video_frames, VIDEO_FRAMES,
        "video frame count must survive the subtitle-skip policy"
    );

    // The dedicated subtitle channel still reads the full cue set back.
    let mut cursor = StreamReader::new(&path)?;
    let mut decoder = rsmedia::Decoder::new_subtitle(&path)?;
    let mut cues = Vec::new();
    while let Some(seg) = decoder.decode_subtitle_segment(&mut cursor)? {
        cues.push(seg);
    }
    assert_eq!(
        cues.len(),
        2,
        "subtitles must still be decodable, got {cues:?}"
    );
    assert!(cues[0].text.contains("第一个 cue"), "cues: {cues:?}");
    assert!(cues[1].text.contains("第二个 cue"), "cues: {cues:?}");

    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// Explicitly naming a subtitle stream in `new_with_streams` must fail fast
/// with the honest `Unsupported` answer (never a later decode-time abort).
#[test]
fn demux_explicit_subtitle_stream_is_rejected() -> anyhow::Result<()> {
    let path = test_output_path("demux_regression", "sub_reject.mp4");
    let _ = std::fs::remove_file(&path);
    mux_video_with_subtitle(&path)?;
    let reader = StreamReader::new(&path)?;

    // Stream 1 is the subtitle track (0 = video). Any subtitle index must be
    // rejected up front.
    let (stream_index, _codec) = reader.find_best_stream(MediaType::SUBTITLE)?;
    let err = match Demuxer::new_with_streams(reader, [stream_index], None, None) {
        Ok(_) => anyhow::bail!("explicitly requested subtitle stream must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("cannot be driven by the frame-API demux() path"),
        "the rejection must explain the alternative paths, got: {err:#}"
    );

    let _ = std::fs::remove_file(&path);
    Ok(())
}

// ===========================================================================
// 2. Per-stream decoder binding (two same-type audio streams)
// ===========================================================================

#[test]
fn demux_binds_each_same_type_stream_to_its_own_codecpar() -> anyhow::Result<()> {
    let path = test_output_path("demux_regression", "twoaudio.mp4");
    let _ = std::fs::remove_file(&path);
    mux_video_two_audio(&path)?;

    let mut demuxer = Demuxer::new(&path)?;
    assert!(
        demuxer.streams().len() >= 3,
        "streams() = {}",
        demuxer.streams().len()
    );

    // Binding check: each audio stream's decoder must carry ITS OWN codecpar
    // (sample rate + stream index). Before the fix, the second audio decoder
    // reported the best stream's 44.1 kHz while the container codecpar said
    // 8 kHz.
    for stream in demuxer.streams() {
        if stream.media_type != MediaType::AUDIO {
            continue;
        }
        let want_rate = stream.stream_info.sample_rate;
        let want_idx = stream.stream_info.index;
        assert_eq!(
            stream.decoder.sample_rate(),
            want_rate,
            "stream {want_idx}: decoder must be configured from its own codecpar"
        );
        assert_eq!(
            stream.decoder.stream_index(),
            want_idx,
            "stream {want_idx}: decoder must be bound to its own stream"
        );
    }

    // Decode check: whatever frames a stream yields must claim the sample rate
    // of that stream (before the fix, the 8 kHz track decoded into frames
    // claiming 44.1 kHz).
    let mut per_stream: std::collections::HashMap<usize, (usize, i32)> =
        std::collections::HashMap::new(); // idx → (frame count, sample rate)
    while let Some((idx, frame)) = demuxer.demux()? {
        let entry = per_stream.entry(idx).or_insert((0, frame.sample_rate));
        assert_eq!(
            entry.1, frame.sample_rate,
            "stream {idx}: per-frame sample rate must be constant"
        );
        entry.0 += 1;
    }
    for stream in demuxer.streams() {
        if stream.media_type != MediaType::AUDIO {
            continue;
        }
        let idx = stream.stream_info.index;
        let (frames, claimed) = per_stream.get(&idx).copied().unwrap_or_default();
        assert!(
            frames > 0,
            "stream {idx}: decode-all must yield frames for every selected stream"
        );
        assert_eq!(
            claimed, stream.stream_info.sample_rate,
            "stream {idx}: decoded frames must claim the container codecpar's sample rate \
             ({frames} frames checked)"
        );
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}
