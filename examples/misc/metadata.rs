use anyhow::{Context, Result};
use rsmpeg::{avcodec::AVCodecContext, avformat::AVFormatContextInput, avutil, ffi};
use std::ffi::CString;

/// Get metadata key-value pair form a video file.
pub fn metadata(file: &str) -> Result<Vec<(String, String)>> {
    let mut result = vec![];
    result.push(("file_path".into(), file.to_string()));

    let file = CString::new(file).unwrap();
    let input_format_context = AVFormatContextInput::open(&file)?;

    // Get `duration` and `bit_rate` from `input_format_context`.
    result.push(("duration".into(), input_format_context.duration.to_string()));
    result.push(("bit_rate".into(), input_format_context.bit_rate.to_string()));

    // Get additional info from `input_format_context.metadata()`
    if let Some(metadata) = input_format_context.metadata() {
        for entry in metadata.iter() {
            result.push((
                entry.key().to_str().unwrap().to_string(),
                entry.value().to_str().unwrap().to_string(),
            ));
        }
    }

    {
        // Get `frame_rate` from `video_stream`
        let (video_stream_index, decoder) = input_format_context
            .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)?
            .context("Failed to find video stream")?;

        let video_stream = &input_format_context.streams()[video_stream_index];

        result.push((
            "frame_rate".into(),
            avutil::av_q2d(video_stream.r_frame_rate).to_string(),
        ));

        // Get `width` and `height` from `decode_context`
        let mut decode_context = AVCodecContext::new(&decoder);
        decode_context
            .apply_codecpar(&video_stream.codecpar())
            .unwrap();
        decode_context.open(None).unwrap();
        result.push(("width".into(), decode_context.width.to_string()));
        result.push(("height".into(), decode_context.height.to_string()));
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::metadata;

    #[test]
    fn metadata_test0() {
        assert_eq!(
            metadata("assets/mp4.mp4").unwrap(),
            vec![
                ("file_path".into(), "assets/mp4.mp4".into()),
                ("duration".into(), "5568000".into()),
                ("bit_rate".into(), "551193".into()),
                ("major_brand".into(), "mp42".into()),
                ("minor_version".into(), "0".into()),
                ("compatible_brands".into(), "mp42isomavc1".into()),
                ("creation_time".into(), "2010-03-20T21:29:11.000000Z".into()),
                ("encoder".into(), "HandBrake 0.9.4 2009112300".into()),
                ("frame_rate".into(), "30".into()),
                ("width".into(), "560".into()),
                ("height".into(), "320".into()),
            ]
        );
    }
}
