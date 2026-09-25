use colorous::*;
use image::ImageBuffer;
use std::process;

const GRADIENTS: [(Gradient, &str); 38] = [
    // Sequential (multi-hue)
    (TURBO, "TURBO"),
    (VIRIDIS, "VIRIDIS"),
    (INFERNO, "INFERNO"),
    (MAGMA, "MAGMA"),
    (PLASMA, "PLASMA"),
    (CIVIDIS, "CIVIDIS"),
    (WARM, "WARM"),
    (COOL, "COOL"),
    (CUBEHELIX, "CUBEHELIX"),
    (BLUE_GREEN, "BLUE_GREEN"),
    (BLUE_PURPLE, "BLUE_PURPLE"),
    (GREEN_BLUE, "GREEN_BLUE"),
    (ORANGE_RED, "ORANGE_RED"),
    (PURPLE_BLUE_GREEN, "PURPLE_BLUE_GREEN"),
    (PURPLE_BLUE, "PURPLE_BLUE"),
    (PURPLE_RED, "PURPLE_RED"),
    (RED_PURPLE, "RED_PURPLE"),
    (YELLOW_GREEN_BLUE, "YELLOW_GREEN_BLUE"),
    (YELLOW_GREEN, "YELLOW_GREEN"),
    (YELLOW_ORANGE_BROWN, "YELLOW_ORANGE_BROWN"),
    (YELLOW_ORANGE_RED, "YELLOW_ORANGE_RED"),
    // Sequential (single-hue)
    (BLUES, "BLUES"),
    (GREENS, "GREENS"),
    (GREYS, "GREYS"),
    (ORANGES, "ORANGES"),
    (PURPLES, "PURPLES"),
    (REDS, "REDS"),
    // Diverging
    (BROWN_GREEN, "BROWN_GREEN"),
    (PURPLE_GREEN, "PURPLE_GREEN"),
    (PINK_GREEN, "PINK_GREEN"),
    (PURPLE_ORANGE, "PURPLE_ORANGE"),
    (RED_BLUE, "RED_BLUE"),
    (RED_GREY, "RED_GREY"),
    (RED_YELLOW_BLUE, "RED_YELLOW_BLUE"),
    (RED_YELLOW_GREEN, "RED_YELLOW_GREEN"),
    (SPECTRAL, "SPECTRAL"),
    // Cyclical
    (RAINBOW, "RAINBOW"),
    (SINEBOW, "SINEBOW"),
];

const CATEGORICALS: [(&[Color], &str); 10] = [
    (&CATEGORY10, "CATEGORY10"),
    (&ACCENT, "ACCENT"),
    (&DARK2, "DARK2"),
    (&PAIRED, "PAIRED"),
    (&PASTEL1, "PASTEL1"),
    (&PASTEL2, "PASTEL2"),
    (&SET1, "SET1"),
    (&SET2, "SET2"),
    (&SET3, "SET3"),
    (&TABLEAU10, "TABLEAU10"),
];

fn main() {
    let rows = GRADIENTS.len() + CATEGORICALS.len();
    let margin = 2;
    let grid = 80;
    let width = 1800;
    let height = rows * grid - margin;
    let mut imgbuf = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let (x, y) = (x as usize, y as usize);
        let row = y / grid;
        let col = x / grid;
        let border = y % grid >= grid - margin;
        *pixel = if let Some((gradient, _)) = GRADIENTS.get(row) {
            if border {
                image::Rgb([0, 0, 0])
            } else {
                let i = x.saturating_sub(10);
                let n = width - 20;
                let Color { r, g, b } = gradient.eval_rational(i, n);
                image::Rgb([r, g, b])
            }
        } else if let Some((scheme, _)) = CATEGORICALS.get(row - GRADIENTS.len()) {
            if col >= scheme.len() {
                let ch = ((x + y) / 20 % 2 * 15) as u8 + 10;
                image::Rgb([ch, ch, ch])
            } else if border {
                image::Rgb([0, 0, 0])
            } else {
                let Color { r, g, b } = scheme[col];
                image::Rgb([r, g, b])
            }
        } else {
            image::Rgb([0, 0, 0])
        };
    }

    let buf = std::fs::read("fonts/Arial.ttf").expect("Failed to read font file");
    let font = fontdue::Font::from_bytes(&buf[..], fontdue::FontSettings::default())
        .expect("Failed to load font");

    for row in 0..rows {
        let name = if let Some((_, name)) = GRADIENTS.get(row) {
            name
        } else if let Some((_, name)) = CATEGORICALS.get(row - GRADIENTS.len()) {
            name
        } else {
            continue;
        };
        // 文本盒左上角在 (10, row * grid + 10)，24px 字号的基线落在盒顶下方 24px 处。
        draw_text(
            &mut imgbuf,
            [100, 100, 100],
            10,
            (row * grid + 10) as i32 + 24,
            24.0,
            &font,
            name,
        );
    }

    if let Err(err) = imgbuf.save("colorous.png") {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}

/// 用 fontdue 把 `text` 光栅化后叠加到图像上。
///
/// fontdue 的坐标系 y 轴向上，而图像 y 轴向下：基线 `baseline_y` 之上的
/// `ymin + height` 才是字形位图的顶边；每个字形按 `advance_width` 向右前进。
fn draw_text(
    img: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    color: [u8; 3],
    x: i32,
    baseline_y: i32,
    px: f32,
    font: &fontdue::Font,
    text: &str,
) {
    let mut pen_x = x as f32;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, px);
        // 空字形（空格等）的 `width` 为 0、位图也为空，下面的循环体不会执行。
        let left = pen_x as i32 + metrics.xmin;
        let top = baseline_y - metrics.ymin - metrics.height as i32;
        pen_x += metrics.advance_width;
        for (i, coverage) in bitmap.iter().enumerate() {
            if *coverage == 0 {
                continue;
            }
            let dst_x = left + (i % metrics.width) as i32;
            let dst_y = top + (i / metrics.width) as i32;
            if dst_x < 0 || dst_y < 0 || dst_x >= img.width() as i32 || dst_y >= img.height() as i32
            {
                continue;
            }
            let alpha = *coverage as f32 / 255.0;
            let dst = img.get_pixel_mut(dst_x as u32, dst_y as u32);
            for (dst_c, &src_c) in dst.0.iter_mut().zip(color.iter()) {
                *dst_c = (*dst_c as f32 * (1.0 - alpha) + src_c as f32 * alpha).round() as u8;
            }
        }
    }
}
