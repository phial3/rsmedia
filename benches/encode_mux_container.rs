//! Full-container encoding benchmark (criterion): one iteration builds a
//! complete MP4 carrying **video + audio + subtitle streams together**, with
//! every encoder's thread count set to the machine's CPU core count.
//!
//! Subtitle packets cannot go through [`Muxer::mux`] (the subtitle encoder is a
//! synchronous API, not `send_frame`/`receive_packet`), so this bench drives
//! all three encoders through the [`StreamWriter`] directly — the same pattern
//! as `subtitle::encode_subtitle_segments_to_file`, extended to video/audio.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench encode_mux_container
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use rsmedia::encode::EncoderBuilder;
use rsmedia::frame::MediaFrame;
use rsmedia::io::StreamWriter;
use rsmedia::pixel::PixelFormat;
use rsmedia::subtitle::SubtitleSegment;
use rsmedia::{SampleFormat, init};
use rsmedia::{Writer, time};

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::Duration;

mod common;
use common::{
    AUDIO_FRAMES, CHANNELS, FPS, HEIGHT, SAMPLE_RATE, SEGMENTS_PER_JOB, VIDEO_FRAMES, WIDTH,
    bench_dir, cores,
};

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

/// One iteration: a complete 3-stream MP4 (2 s video + ~3 s audio + 30
/// subtitle cues), every encoder using `enc_threads` threads.
fn encode_container(path: &Path, enc_threads: usize) -> Result<()> {
    let mut v_enc = EncoderBuilder::new_video(WIDTH, HEIGHT)
        .with_fps(FPS)
        .with_thread_count(enc_threads)
        .build()?;
    let mut a_enc = EncoderBuilder::new_audio(
        128_000,
        CHANNELS as i32,
        SAMPLE_RATE as i32,
        SampleFormat::FLTP,
    )
    .with_thread_count(enc_threads)
    .build()?;
    let mut s_enc = EncoderBuilder::new_subtitle()
        .with_codec_name(Some("mov_text".to_string()))
        .with_subtitle_header(ASS_HEADER)
        .with_thread_count(enc_threads)
        .build()?;

    let mut writer = StreamWriter::new(path)?;
    let v_idx = writer.add_stream(v_enc.codecpar(), v_enc.time_base());
    let a_idx = writer.add_stream(a_enc.codecpar(), a_enc.time_base());
    let s_idx = writer.add_stream(s_enc.codecpar(), s_enc.time_base());
    writer.write_header()?;
    let (v_tb, a_tb, s_tb) = (
        writer.stream_time_base(v_idx),
        writer.stream_time_base(a_idx),
        writer.stream_time_base(s_idx),
    );

    // Video frames (60): pts left unset, the encoder numbers them automatically.
    let v_frame_ticks = rsmpeg::avutil::av_rescale_q(1, time::new_rational(1, FPS as i32), v_tb);
    for i in 0..VIDEO_FRAMES {
        let frame = rainbow_frame(i as f32 / VIDEO_FRAMES as f32).to_avframe()?;
        for mut pkt in v_enc.encode_raw(frame)? {
            if pkt.duration <= 0 {
                pkt.set_duration(v_frame_ticks);
            }
            write_packet(&mut writer, &mut pkt, v_idx, v_enc.time_base(), v_tb)?;
        }
    }

    // Audio frames (variable sizes; the encoder's sample FIFO re-frames them).
    let a_frame_ticks = rsmpeg::avutil::av_rescale_q(
        a_enc.frame_size() as i64,
        time::new_rational(1, SAMPLE_RATE as i32),
        a_tb,
    );
    for i in 0..AUDIO_FRAMES {
        let nb = 700u32 + ((i as u32 * 173) % 1200);
        let frame = sine_audio_frame(nb).to_avframe()?;
        for mut pkt in a_enc.encode_raw(frame)? {
            if pkt.duration <= 0 {
                pkt.set_duration(a_frame_ticks);
            }
            write_packet(&mut writer, &mut pkt, a_idx, a_enc.time_base(), a_tb)?;
        }
    }

    // Subtitle segments (synchronous encode API; no drain/flush buffering).
    let segments: Vec<SubtitleSegment> = (0..SEGMENTS_PER_JOB)
        .map(|i| {
            SubtitleSegment::new(
                (i * 2000) as i64,
                (i * 2000 + 1800) as i64,
                format!("Line {i} of the container benchmark"),
            )
        })
        .collect();
    for segment in &segments {
        for mut pkt in s_enc.encode_subtitle_segment(segment)? {
            write_packet(&mut writer, &mut pkt, s_idx, s_enc.time_base(), s_tb)?;
        }
    }

    // Drain all three encoders (video/audio may buffer frames), then close.
    v_enc.flush(&mut writer, true, v_idx, v_tb)?;
    a_enc.flush(&mut writer, true, a_idx, a_tb)?;
    s_enc.flush(&mut writer, true, s_idx, s_tb)?;
    writer.write_trailer()?;
    Ok(())
}

/// Timestamp-normalize and write one encoded packet.
fn write_packet(
    writer: &mut StreamWriter,
    pkt: &mut rsmpeg::avcodec::AVPacket,
    stream_idx: usize,
    enc_tb: rsmpeg::ffi::AVRational,
    out_tb: rsmpeg::ffi::AVRational,
) -> Result<()> {
    pkt.set_pos(-1);
    pkt.set_stream_index(stream_idx as i32);
    pkt.rescale_ts(enc_tb, out_tb);
    writer.write_interleaved(pkt)?;
    Ok(())
}

fn rainbow_frame(p: f32) -> MediaFrame<u8> {
    let rgb = rsmedia::colors::hsv_to_rgb(p * 360.0, 100.0, 100.0);
    let mut frame = MediaFrame::<u8>::new_video_frame(
        WIDTH,
        HEIGHT,
        PixelFormat::RGB24,
        time::new_rational(1, FPS as i32),
    )
    .unwrap();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            frame.data[[y, x, 0]] = rgb[0];
            frame.data[[y, x, 1]] = rgb[1];
            frame.data[[y, x, 2]] = rgb[2];
        }
    }
    frame
}

fn sine_audio_frame(nb_samples: u32) -> MediaFrame<f32> {
    let mut frame = MediaFrame::<f32>::new_audio_frame(
        SampleFormat::FLT,
        CHANNELS,
        nb_samples,
        SAMPLE_RATE,
        time::new_rational(1, SAMPLE_RATE as i32),
    )
    .unwrap();
    for i in 0..nb_samples as usize {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        for c in 0..CHANNELS as usize {
            frame.data[[0, i, c]] = v;
        }
    }
    frame
}

fn bench_container(c: &mut Criterion) {
    init().expect("rsmedia init failed");
    let cores = cores();
    let dir = bench_dir();
    let path: PathBuf = dir.join("container_bench.mp4");

    // 65 s of media per iteration (2 s video + 3 s audio + 60 s of subtitle
    // cues), encoded into a single MP4.
    let mut group = c.benchmark_group("encode_container");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_millis(500));
    group.throughput(Throughput::Elements(65));
    group.bench_function(
        format!("mp4 video+audio+subtitle, {cores} threads/encoder"),
        |b| b.iter(|| encode_container(&path, cores).unwrap()),
    );
    remove_tmp(&path);
    group.finish();
}

fn remove_tmp(path: &Path) {
    let _ = std::fs::remove_file(path);
}

criterion_group!(benches, bench_container);
criterion_main!(benches);
