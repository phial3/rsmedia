use rsmedia::h264::{assert_h264_frame, assert_image};

#[test]
fn good() {
    let actual = image::ImageReader::open("assets/dog1.png")
        .unwrap()
        .decode()
        .unwrap();
    assert_image("assets/dog1.png", &actual, 1.0);
}

#[test]
#[should_panic]
fn bad() {
    let actual = image::ImageReader::open("assets/dog1.png")
        .unwrap()
        .decode()
        .unwrap();
    assert_image("assets/dog2.png", &actual, 1.0);
}

#[test]
fn good_h264() {
    let actual = std::fs::read("assets/initial-grid.h264").unwrap();
    assert_h264_frame("assets/initial-grid.png", &actual, 0.999);
}

#[test]
fn good_h264_multiple_frames() {
    let actual = std::fs::read("assets/multiple-frames.h264").unwrap();
    assert_h264_frame("assets/multiple-frames.png", &actual, 0.999);
}

#[test]
#[should_panic]
fn bad_h264() {
    let actual = std::fs::read("assets/multiple-frames.h264").unwrap();
    assert_h264_frame("assets/initial-grid.png", &actual, 0.999);
}

#[test]
#[should_panic]
fn bad_h264_png_compare() {
    let actual = image::ImageReader::open("assets/multiple-frames.png")
        .unwrap()
        .decode()
        .unwrap();
    assert_image("assets/initial-grid.png", &actual, 1.0);
}
