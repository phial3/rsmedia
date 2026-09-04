use rsmpeg::avformat::*;
use std::ffi::CStr;

/// Dump video/audio/image info to stdout.
fn image_dump(image_path: &CStr) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_format_context = AVFormatContextInput::open(image_path)?;
    input_format_context.dump(0, image_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_dump_test() {
        image_dump(c"assets/cat.jpg").unwrap();
        image_dump(c"assets/mp4.mp4").unwrap();
        image_dump(c"assets/wav.wav").unwrap();
    }
}
