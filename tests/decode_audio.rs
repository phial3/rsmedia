//! RIIR: https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/decode_audio.c
use anyhow::{Context, Result};
use rsmpeg::{
    avcodec::{AVCodecContext, AVPacket},
    avformat::AVFormatContextInput,
    avutil::{
        AVFrame, AVSampleFormat, get_bytes_per_sample, get_packed_sample_fmt, get_sample_fmt_name,
        sample_fmt_is_planar,
    },
    error::RsmpegError,
    ffi,
};

use std::{
    ffi::CString,
    fs::{self, File},
    io::Write,
    path::Path,
    slice::from_raw_parts,
};

fn get_format_from_sample_fmt(sample_fmt: AVSampleFormat) -> Option<&'static str> {
    let sample_fmt_entries = [
        (ffi::AV_SAMPLE_FMT_U8, "u8"),
        (ffi::AV_SAMPLE_FMT_S16, "s16le"),
        (ffi::AV_SAMPLE_FMT_S32, "s32le"),
        (ffi::AV_SAMPLE_FMT_FLT, "f32le"),
        (ffi::AV_SAMPLE_FMT_DBL, "f64le"),
    ];
    sample_fmt_entries
        .iter()
        .find(|(fmt, _)| *fmt == sample_fmt)
        .map(|(_, fmt)| *fmt)
}

fn frame_save(frame: &AVFrame, channels: usize, data_size: usize, mut file: &File) -> Result<()> {
    let nb_samples: usize = frame.nb_samples.try_into().context("nb_samples overflow")?;
    // ATTENTION: This is only valid for planar sample formats.
    for i in 0..nb_samples {
        for plane in frame.data.iter().take(channels) {
            let data = unsafe { from_raw_parts(plane.add(data_size * i), data_size) };
            file.write_all(data).context("Write data failed.")?;
        }
    }
    Ok(())
}

fn decode(
    decode_context: &mut AVCodecContext,
    packet: Option<&AVPacket>,
    out_file: &File,
) -> Result<()> {
    decode_context
        .send_packet(packet)
        .context("Send packet failed.")?;
    let channels = decode_context
        .ch_layout
        .nb_channels
        .try_into()
        .context("channels overflow")?;
    let sample_fmt = decode_context.sample_fmt;
    loop {
        let frame = match decode_context.receive_frame() {
            Ok(frame) => frame,
            Err(RsmpegError::DecoderDrainError) | Err(RsmpegError::DecoderFlushedError) => break,
            Err(e) => return Err(e).context("Receive frame failed."),
        };
        let data_size = get_bytes_per_sample(sample_fmt).context("Unknown sample fmt")?;
        frame_save(&frame, channels, data_size, out_file)?;
    }
    Ok(())
}

fn decode_audio(audio_path: &str, out_file_path: &str) -> Result<()> {
    // safety, &str ensures no internal null bytes.
    let audio_path_c = CString::new(audio_path).unwrap();
    let mut input_format_context =
        AVFormatContextInput::open(&audio_path_c).context("Open audio file failed.")?;
    let (stream_index, decoder) = input_format_context
        .find_best_stream(ffi::AVMEDIA_TYPE_AUDIO)
        .context("Find best stream failed.")?
        .context("Cannot find audio stream in this file.")?;
    let mut decode_context = AVCodecContext::new(&decoder);
    decode_context
        .apply_codecpar(&input_format_context.streams()[stream_index].codecpar())
        .context("Apply codecpar failed.")?;
    decode_context.open(None).context("Could not open codec")?;
    input_format_context.dump(stream_index, &audio_path_c)?;

    fs::create_dir_all(Path::new(out_file_path).parent().unwrap()).unwrap();
    let out_file = File::create(out_file_path).context("Open out file failed.")?;

    while let Some(packet) = input_format_context
        .read_packet()
        .context("Read packet failed.")?
    {
        if packet.stream_index == stream_index as i32 {
            decode(&mut decode_context, Some(&packet), &out_file)?;
        }
    }

    // Flush decoder
    decode(&mut decode_context, None, &out_file)?;

    let mut sample_fmt = decode_context.sample_fmt;

    if sample_fmt_is_planar(sample_fmt) {
        let name = get_sample_fmt_name(sample_fmt);
        println!(
            "Warning: the sample format the decoder produced is planar \
            ({}). This example will output the first channel only.",
            name.map(|x| x.to_str().unwrap()).unwrap_or("?")
        );
        sample_fmt = get_packed_sample_fmt(sample_fmt).context("Cannot get packed sample fmt")?;
    }

    let fmt = get_format_from_sample_fmt(sample_fmt).context("Unsupported sample fmt")?;

    println!("Play the output audio file with the command:");
    println!(
        "ffplay -f {} -ac {} -ar {} {}",
        fmt, decode_context.ch_layout.nb_channels, decode_context.sample_rate, out_file_path
    );
    Ok(())
}

#[test]
fn decode_audio_test() {
    decode_audio("assets/wav.wav", "tests/output/decode_audio/wav.pcm").unwrap();
}
