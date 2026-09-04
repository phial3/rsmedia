//! RIIR: https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/decode_video.c
use anyhow::{Context, Result};
use camino::Utf8Path as Path;
use rsmpeg::{
    avcodec::{AVCodecContext, AVPacket},
    avformat::AVFormatContextInput,
    avutil::AVFrame,
    error::RsmpegError,
    ffi,
};
use std::{ffi::CString, fs, io::prelude::*, slice};

/// Save a `AVFrame` as pgm file.
fn pgm_save(frame: &AVFrame, filename: &str) -> Result<()> {
    // Here we only capture the first layer of frame.
    let data = frame.data[0];
    let linesize = frame.linesize[0] as usize;

    let width = frame.width as usize;
    let height = frame.height as usize;

    let buffer = unsafe { slice::from_raw_parts(data, linesize * height) };

    // Create pgm file
    let mut pgm_file = fs::File::create(filename)?;

    // Write pgm header
    pgm_file.write_all(&format!("P5\n{} {}\n{}\n", width, height, 255).into_bytes())?;

    // Write pgm data
    for i in 0..height {
        pgm_file.write_all(&buffer[i * linesize..i * linesize + width])?;
    }

    pgm_file.flush()?;

    Ok(())
}

/// Push packet to `decode_context`, then save the output frames(fetched from the
/// `decode_context`) as pgm files.
fn decode(
    decode_context: &mut AVCodecContext,
    packet: Option<&AVPacket>,
    out_dir: &str,
    out_filename: &str,
) -> Result<()> {
    decode_context.send_packet(packet)?;
    loop {
        let frame = match decode_context.receive_frame() {
            Ok(frame) => frame,
            Err(RsmpegError::DecoderDrainError) | Err(RsmpegError::DecoderFlushedError) => break,
            Err(e) => Err(e).context("Error during decoding")?,
        };
        println!("saving frame {}", decode_context.frame_num);
        pgm_save(
            &frame,
            &format!(
                "{}/{}-{}.pgm",
                out_dir, out_filename, decode_context.frame_num
            ),
        )?;
    }
    Ok(())
}

/// This function extracts video frames from any container file supported by
/// FFmpeg, then saves them to `out_dir` as pgm.
fn decode_video(video_path: &str, out_dir: &str) -> Result<()> {
    let video_path = Path::new(video_path);
    let out_filename = video_path.file_stem().unwrap();
    fs::create_dir_all(out_dir).unwrap();

    // &str ensures no internal null bytes.
    let video_path_c = CString::new(video_path.as_str()).unwrap();
    let mut input_format_context =
        AVFormatContextInput::open(&video_path_c).context("Open video file failed.")?;
    let (stream_index, decoder) = input_format_context
        .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)
        .context("Find best stream failed.")?
        .context("Cannot find video stream in this file.")?;
    let mut decode_context = AVCodecContext::new(&decoder);
    decode_context
        .apply_codecpar(&input_format_context.streams()[stream_index].codecpar())
        .context("Apply codecpar failed.")?;
    decode_context.open(None).context("Could not open codec")?;
    input_format_context.dump(stream_index, &video_path_c)?;

    while let Some(packet) = input_format_context
        .read_packet()
        .context("Read packet failed.")?
    {
        if packet.stream_index == stream_index as i32 {
            decode(&mut decode_context, Some(&packet), out_dir, out_filename)?;
        }
    }

    // Flush decoder
    decode(&mut decode_context, None, out_dir, out_filename)?;

    Ok(())
}

#[test]
fn decode_video_test() {
    decode_video("assets/mp4.mp4", "tests/output/decode_video").unwrap();
}
