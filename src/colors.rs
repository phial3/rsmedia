use crate::error::{Result, format_err};

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct Color(pub u32);

/// base
impl Color {
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32))
    }

    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::from_rgba(r, g, b, 0xff)
    }

    pub fn rgba(&self) -> (u8, u8, u8, u8) {
        let r = ((self.0 >> 24) & 0xff) as u8;
        let g = ((self.0 >> 16) & 0xff) as u8;
        let b = ((self.0 >> 8) & 0xff) as u8;
        let a = (self.0 & 0xff) as u8;
        (r, g, b, a)
    }

    pub fn rgb(&self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.rgba();
        (r, g, b)
    }

    pub fn bgr(&self) -> (u8, u8, u8) {
        let (r, g, b) = self.rgb();
        (b, g, r)
    }

    pub fn r(&self) -> u8 {
        self.rgba().0
    }

    pub fn g(&self) -> u8 {
        self.rgba().1
    }

    pub fn b(&self) -> u8 {
        self.rgba().2
    }

    pub fn a(&self) -> u8 {
        self.rgba().3
    }

    pub fn hex(&self, uppercase: bool) -> String {
        if uppercase {
            format!("#{:08X}", self.0)
        } else {
            format!("#{:08x}", self.0)
        }
    }

    pub fn with_alpha(self, a: u8) -> Self {
        let (r, g, b) = self.rgb();
        (r, g, b, a).into()
    }

    /// Create a palette from a slice of colors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 从 u32 数组创建
    /// let colors = Color::create_palette(&[0xFF0000FF, 0x00FF00FF, 0x0000FFFF]);
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 从 RGB 元组数组创建
    /// let rgb_colors = Color::create_palette(&[
    ///     (255, 0, 0),
    ///     (0, 255, 0),
    ///     (0, 0, 255),
    /// ]);
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 从 RGB 数组创建
    /// let array_colors = Color::create_palette(&[
    ///     [255, 128, 0],
    ///     [128, 0, 128],
    ///     [0, 128, 128],
    /// ]);
    /// ```
    pub fn create_palette<A: Into<Self> + Copy>(xs: &[A]) -> Vec<Self> {
        xs.iter().copied().map(Into::into).collect()
    }

    /// Create a palette from a slice of colors, with error handling.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rsmedia::Color;
    /// let colors = Color::try_create_palette(&[0xFF0000FF, 0x00FF00FF, 0x0000FFFF]).unwrap();
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 从十六进制字符串数组创建（可能失败）
    /// let hex_colors = Color::try_create_palette(&["#FF0000", "#00FF00", "#0000FF"])
    ///     .expect("Invalid hex color");
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 混合有效和无效的字符串会返回错误
    /// let result = Color::try_create_palette(&["#FF0000", "invalid", "#0000FF"]);
    /// assert!(result.is_err());
    /// ```
    pub fn try_create_palette<A: TryInto<Self> + Copy>(xs: &[A]) -> Result<Vec<Self>>
    where
        <A as TryInto<Self>>::Error: std::fmt::Debug,
    {
        xs.iter()
            .copied()
            .map(|x| {
                x.try_into()
                    .map_err(|e| format_err!("Failed to convert: {:?}", e))
            })
            .collect()
    }

    /// Generate visually distinct colors by evenly distributing hues in HSL space.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 生成 5 个在 HSL 空间中均匀分布的颜色（色相 0-360 度）
    /// let distinct_colors = Color::palette_distinct(5);
    /// assert_eq!(distinct_colors.len(), 5);
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 生成 10 个不同的颜色
    /// let palette = Color::palette_distinct(10);
    /// assert_eq!(palette.len(), 10);
    /// ```
    ///
    /// ```
    /// # use rsmedia::Color;
    /// // 生成 0 个颜色返回空数组
    /// let empty = Color::palette_distinct(0);
    /// assert!(empty.is_empty());
    /// ```
    pub fn palette_distinct(count: usize) -> Vec<Color> {
        if count == 0 {
            return Vec::new();
        }
        (0..count)
            .map(|i| {
                // 均匀分布色相(0-360度)
                let hue = (i as f32 * 360.0 / count as f32) % 360.0;
                // 固定饱和度和亮度为适中值；from_hsl 的 s/l 取值范围为 0-100
                let saturation = 0.7;
                let lightness = 0.5;
                Color::from_hsl(
                    hue as u16,
                    (saturation * 100.0) as u16,
                    (lightness * 100.0) as u16,
                )
            })
            .collect()
    }
}

/// extend
impl Color {
    /// HSV
    pub fn from_hsv(h: u16, s: u16, v: u16) -> Self {
        let rgb = colorutils_rs::Hsv::new(h, s, v).to_rgb8();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_hsv(&self) -> colorutils_rs::Hsv {
        let (r, g, b) = self.rgb();
        colorutils_rs::Rgb::<u8>::new(r, g, b).to_hsv()
    }

    /// HSL
    pub fn from_hsl(h: u16, s: u16, l: u16) -> Self {
        let rgb = colorutils_rs::Hsl::new(h, s, l).to_rgb8();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_hsl(&self) -> colorutils_rs::Hsl {
        let (r, g, b) = self.rgb();
        colorutils_rs::Rgb::<u8>::new(r, g, b).to_hsl()
    }

    /// LAB
    pub fn from_lab(l: f32, a: f32, b: f32) -> Self {
        let rgb = colorutils_rs::Lab::new(l, a, b).to_rgb8();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_lab(&self) -> colorutils_rs::Lab {
        let (r, g, b) = self.rgb();
        colorutils_rs::Rgb::<u8>::new(r, g, b).to_lab()
    }

    /// XYB
    pub fn from_xyb(x: f32, y: f32, b: f32) -> Self {
        let rgb = colorutils_rs::Xyb::new(x, y, b).to_rgb(colorutils_rs::TransferFunction::Srgb);
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_xyb(&self) -> colorutils_rs::Xyb {
        let (r, g, b) = self.rgb();
        let rgb = colorutils_rs::Rgb::<u8>::new(r, g, b);
        colorutils_rs::Xyb::from_rgb(rgb, colorutils_rs::TransferFunction::Srgb)
    }

    /// XYZ
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        let rgb = colorutils_rs::Xyz::new(x, y, z).to_srgb();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_xyz(&self) -> colorutils_rs::Xyz {
        let (r, g, b) = self.rgb();
        let rgb = colorutils_rs::Rgb::<u8>::new(r, g, b);
        colorutils_rs::Xyz::from_srgb(rgb)
    }
}

/// convert
impl From<u32> for Color {
    fn from(x: u32) -> Self {
        Self(x)
    }
}

impl From<(u8, u8, u8)> for Color {
    fn from((r, g, b): (u8, u8, u8)) -> Self {
        Self::from_rgb(r, g, b)
    }
}

impl From<[u8; 3]> for Color {
    fn from(c: [u8; 3]) -> Self {
        Self::from((c[0], c[1], c[2]))
    }
}

impl From<(u8, u8, u8, u8)> for Color {
    fn from((r, g, b, a): (u8, u8, u8, u8)) -> Self {
        Self::from_rgba(r, g, b, a)
    }
}

impl From<[u8; 4]> for Color {
    fn from(c: [u8; 4]) -> Self {
        Self::from((c[0], c[1], c[2], c[3]))
    }
}

impl From<Color> for (u8, u8, u8, u8) {
    fn from(color: Color) -> Self {
        color.rgba()
    }
}

impl From<Color> for [u8; 4] {
    fn from(color: Color) -> Self {
        let (r, g, b, a) = color.rgba();
        [r, g, b, a]
    }
}

impl From<Color> for (u8, u8, u8) {
    fn from(color: Color) -> Self {
        color.rgb()
    }
}

impl From<Color> for [u8; 3] {
    fn from(color: Color) -> Self {
        let (r, g, b) = color.rgb();
        [r, g, b]
    }
}

impl TryFrom<&str> for Color {
    type Error = &'static str;

    fn try_from(x: &str) -> std::result::Result<Self, Self::Error> {
        let hex = x.trim_start_matches('#');
        let hex = match hex.len() {
            6 => format!("{hex}ff"),
            8 => hex.to_string(),
            _ => return Err("Failed to convert `Color` from str: invalid length"),
        };

        u32::from_str_radix(&hex, 16)
            .map(Self)
            .map_err(|_| "Failed to convert `Color` from str: invalid hex")
    }
}

/// 彩虹色
pub const RED: Color = Color::from_rgb(255, 0, 0);
pub const ORANGE: Color = Color::from_rgb(255, 165, 0);
pub const YELLOW: Color = Color::from_rgb(255, 255, 0);
pub const GREEN: Color = Color::from_rgb(0, 128, 0);
pub const BLUE: Color = Color::from_rgb(0, 0, 255);
pub const INDIGO: Color = Color::from_rgb(75, 0, 130);
pub const VIOLET: Color = Color::from_rgb(238, 130, 238);

/// 基本颜色
pub const PURPLE: Color = Color::from_rgb(128, 0, 128);
pub const MAGENTA: Color = Color::from_rgb(255, 0, 255);
pub const CYAN: Color = Color::from_rgb(0, 255, 255);
pub const LIME: Color = Color::from_rgb(0, 255, 0);
pub const TEAL: Color = Color::from_rgb(0, 128, 128);
pub const BLACK: Color = Color::from_rgb(0, 0, 0);
pub const WHITE: Color = Color::from_rgb(255, 255, 255);
pub const GRAY: Color = Color::from_rgb(128, 128, 128);
pub const SILVER: Color = Color::from_rgb(192, 192, 192);
pub const MAROON: Color = Color::from_rgb(128, 0, 0);
pub const OLIVE: Color = Color::from_rgb(128, 128, 0);
pub const NAVY: Color = Color::from_rgb(0, 0, 128);

/// 扩展颜色
pub const PINK: Color = Color::from_rgb(255, 192, 203);
pub const BROWN: Color = Color::from_rgb(165, 42, 42);
pub const GOLD: Color = Color::from_rgb(255, 215, 0);
pub const TURQUOISE: Color = Color::from_rgb(64, 224, 208);
pub const LAVENDER: Color = Color::from_rgb(230, 230, 250);
pub const CORAL: Color = Color::from_rgb(255, 127, 80);
pub const SALMON: Color = Color::from_rgb(250, 128, 114);
pub const CRIMSON: Color = Color::from_rgb(220, 20, 60);
pub const KHAKI: Color = Color::from_rgb(240, 230, 140);
pub const PLUM: Color = Color::from_rgb(221, 160, 221);

/// Convert RGB to HSV color space.
///
/// 返回 `[h, s, v]`，其中 `h` 为 0-360 度，`s`/`v` 为 0-100
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> [f32; 3] {
    let hsv = colorutils_rs::Rgb::<u8>::new(r, g, b).to_hsv();
    [hsv.h, hsv.s * 100.0, hsv.v * 100.0]
}

/// Convert HSV to RGB color space.
///
/// 输入 `h` 为 0-360 度，`s`/`v` 为 0-100
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let rgb = colorutils_rs::Hsv::new(h as u16, s as u16, v as u16).to_rgb8();
    [rgb.r, rgb.g, rgb.b]
}

/// Calculate the Euclidean distance between two colors in RGB space.
pub fn color_distance(c1: &Color, c2: Color) -> f32 {
    let dr = (c1.r() as f32 - c2.r() as f32).powi(2);
    let dg = (c1.g() as f32 - c2.g() as f32).powi(2);
    let db = (c1.b() as f32 - c2.b() as f32).powi(2);
    (dr + dg + db).sqrt()
}

/// Calculate the perceptual color difference (CIEDE2000) in CIE Lab space.
///
/// Unlike the plain RGB [`color_distance`], CIEDE2000 is aligned with human
/// color perception, making it suitable for quantitative video-quality
/// assessment, watermark-removal evaluation, and palette discrimination.
///
/// Reference thresholds: `<1` imperceptible, `1~2` subtle, `2~10` noticeable,
/// `>10` very different.
pub fn color_delta_e(c1: &Color, c2: Color) -> f32 {
    use palette::color_difference::Ciede2000;
    use palette::{IntoColor, Lab, Srgb};

    let lab1: Lab = Srgb::new(
        c1.r() as f32 / 255.0,
        c1.g() as f32 / 255.0,
        c1.b() as f32 / 255.0,
    )
    .into_color();
    let lab2: Lab = Srgb::new(
        c2.r() as f32 / 255.0,
        c2.g() as f32 / 255.0,
        c2.b() as f32 / 255.0,
    )
    .into_color();
    lab1.difference(lab2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use palette::chromatic_adaptation::AdaptIntoUnclamped;
    use palette::convert::{FromColorUnclamped, IntoColorUnclamped};
    use palette::{Hsl, IntoColor, Lab, LinSrgb, Oklab, Oklch, Srgb, Xyz};

    #[test]
    fn test_color_to_hex() {
        let color = Color::from_rgb(255, 0, 0);
        assert_eq!(color.hex(true), "#FF0000FF");

        let color = Color::from_rgb(0, 255, 0);
        assert_eq!(color.hex(true), "#00FF00FF");

        let color = Color::from_rgb(0, 0, 255);
        assert_eq!(color.hex(true), "#0000FFFF");

        let color = Color::from_rgb(255, 255, 0);
        assert_eq!(color.hex(false), "#ffff00ff");
    }

    #[test]
    fn test_hex_to_color() {
        assert_eq!(
            Color::try_from("#FF0000").unwrap(),
            Color::from_rgb(255, 0, 0)
        );
        assert_eq!(
            Color::try_from("00FF00").unwrap(),
            Color::from_rgb(0, 255, 0)
        );
        assert_eq!(
            Color::try_from("#0000FF").unwrap(),
            Color::from_rgb(0, 0, 255)
        );
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(RED, Color::from_rgb(255, 0, 0));
        assert_eq!(GREEN, Color::from_rgb(0, 128, 0));
        assert_eq!(BLUE, Color::from_rgb(0, 0, 255));
    }

    #[test]
    fn test_rgb_hsv_known_values() {
        // (r, g, b, 期望 h/s/v)
        let cases: &[(u8, u8, u8, f32, f32, f32)] = &[
            (255, 0, 0, 0.0, 100.0, 100.0),     // 纯红
            (0, 255, 0, 120.0, 100.0, 100.0),   // 纯绿
            (0, 0, 255, 240.0, 100.0, 100.0),   // 纯蓝
            (255, 255, 0, 60.0, 100.0, 100.0),  // 黄
            (255, 0, 255, 300.0, 100.0, 100.0), // 品红
            (0, 255, 255, 180.0, 100.0, 100.0), // 青
            (0, 0, 0, 0.0, 0.0, 0.0),           // 纯黑
            (255, 255, 255, 0.0, 0.0, 100.0),   // 纯白
            (128, 128, 128, 0.0, 0.0, 50.0),    // 中灰
        ];

        for &(r, g, b, eh, es, ev) in cases {
            let [h, s, v] = rgb_to_hsv(r, g, b);
            assert!(
                (h - eh).abs() < 2.0,
                "Hue mismatch for RGB({r},{g},{b}): {h} vs {eh}"
            );
            assert!(
                (s - es).abs() <= 2.0,
                "S mismatch for RGB({r},{g},{b}): {s} vs {es}"
            );
            assert!(
                (v - ev).abs() <= 2.0,
                "V mismatch for RGB({r},{g},{b}): {v} vs {ev}"
            );

            // 往返转换允许 ±2 的取整误差
            let [r2, g2, b2] = hsv_to_rgb(h, s, v);
            assert!(
                (r as i32 - r2 as i32).abs() <= 2
                    && (g as i32 - g2 as i32).abs() <= 2
                    && (b as i32 - b2 as i32).abs() <= 2,
                "roundtrip mismatch for RGB({r},{g},{b}): [{r2},{g2},{b2}]"
            );
        }
    }

    #[test]
    fn test_round_trip() {
        let test_colors = [
            [123, 45, 67],  // 随机颜色
            [255, 128, 0],  // 橙色
            [75, 200, 220], // 青色系
            [30, 150, 80],  // 绿色系
        ];

        for &[r, g, b] in &test_colors {
            let [h, s, v] = rgb_to_hsv(r, g, b);
            let [r2, g2, b2] = hsv_to_rgb(h, s, v);

            // 允许 ±3 的误差（浮点舍入 + u8 整型截断）
            assert!((r as i32 - r2 as i32).abs() <= 3);
            assert!((g as i32 - g2 as i32).abs() <= 3);
            assert!((b as i32 - b2 as i32).abs() <= 3);
        }
    }

    #[test]
    fn test_precision() {
        let test_cases = [
            (255, 0, 0),     // 纯红
            (0, 255, 0),     // 纯绿
            (0, 0, 255),     // 纯蓝
            (128, 128, 128), // 灰色
            (255, 255, 0),   // 黄色
            (255, 0, 255),   // 洋红
            (0, 255, 255),   // 青色
        ];

        for (r, g, b) in test_cases.iter() {
            let hsv_f32 = rgb_to_hsv(*r, *g, *b);
            // HSV 输出必须落在合法范围内
            assert!(
                (0.0..360.0).contains(&hsv_f32[0]),
                "Hue out of range for RGB({}, {}, {}): {}",
                r,
                g,
                b,
                hsv_f32[0]
            );
            assert!(
                (0.0..=100.0).contains(&hsv_f32[1]),
                "Saturation out of range for RGB({}, {}, {}): {}",
                r,
                g,
                b,
                hsv_f32[1]
            );
            assert!(
                (0.0..=100.0).contains(&hsv_f32[2]),
                "Value out of range for RGB({}, {}, {}): {}",
                r,
                g,
                b,
                hsv_f32[2]
            );
        }
    }

    #[test]
    fn test_boundary_conditions() {
        // V=0 时必须输出黑色（无论色相/饱和度）
        assert_eq!(hsv_to_rgb(180.0, 50.0, 0.0), [0, 0, 0]);
        assert_eq!(hsv_to_rgb(0.0, 100.0, 0.0), [0, 0, 0]);

        // 接近零的值，饱和度应为正且不越界
        let hsv = rgb_to_hsv(1, 0, 0);
        assert!(hsv[1] > 0.0 && hsv[1] <= 100.0);

        // 近似相等的值，饱和度应落在 [0, 100]
        let hsv = rgb_to_hsv(128, 128, 127);
        assert!(hsv[0] >= 0.0 && hsv[0] <= 360.0);
        assert!(hsv[1] >= 0.0 && hsv[1] <= 100.0);
    }

    #[test]
    fn test_palette_color_conversion() {
        // Example 1: SRGB to HSL
        let srgb_color: Srgb<f32> = Srgb::new(0.8, 0.2, 0.3);
        let hsl_color = Hsl::from_color_unclamped(srgb_color);
        println!("SRGB: {:?} -> HSL: {:?}", srgb_color, hsl_color);

        // Example 2: HSL back to SRGB
        let srgb_again: Srgb<f32> = hsl_color.into_color();
        println!("HSL: {:?} -> SRGB: {:?}", hsl_color, srgb_again);

        // Example 3: SRGB to Oklab (perceptually uniform)
        let oklab_color: Oklab<f32> = srgb_color.into_color();
        println!("SRGB: {:?} -> Oklab: {:?}", srgb_color, oklab_color);

        // Example 4: Oklab to Oklch (polar version of Oklab)
        let oklch_color: Oklch<f32> = oklab_color.into_color();
        println!("Oklab: {:?} -> Oklch: {:?}", oklab_color, oklch_color);

        // Example 5: Oklch back to SRGB
        let srgb_from_oklch: Srgb<f32> = oklch_color.into_color();
        println!("Oklch: {:?} -> SRGB: {:?}", oklch_color, srgb_from_oklch);

        // Example 6: Linear SRGB to standard SRGB
        let linear_srgb: LinSrgb<f32> = LinSrgb::new(0.5, 0.5, 0.5);
        let standard_srgb: Srgb<f32> = linear_srgb.into_color();
        println!(
            "Linear SRGB: {:?} -> SRGB: {:?}",
            linear_srgb, standard_srgb
        );

        println!("\n--- Component Type Conversions ---");

        // Example 7: SRGB f32 [0.0, 1.0] to SRGB u8 [0, 255]
        let srgb_f32: Srgb<f32> = Srgb::new(0.0, 0.5, 1.0);
        let rgb_u8 = Srgb::from_color_unclamped(srgb_f32);
        println!("SRGB f32: {:?} -> SRGB u8: {:?}", srgb_f32, rgb_u8);

        // Example 8: SRGB u8 [0, 255] to HSL f32 [0.0, 1.0] / [0.0, 360.0]
        // Note: Palette handles the u8 -> f32 scaling internally during conversion
        let hsl_f32_from_u8 = Hsl::from_color_unclamped(rgb_u8);
        println!("SRGB u8: {:?} -> HSL f32: {:?}", rgb_u8, hsl_f32_from_u8);

        // Example 9: SRGB u8 to Oklab f32
        let oklab_f32_from_u8: Oklab<f32> = rgb_u8.into_color();
        println!(
            "SRGB u8: {:?} -> Oklab f32: {:?}",
            rgb_u8, oklab_f32_from_u8
        );

        println!("\n--- Using Specific Encodings/White Points (Advanced) ---");
        // For Lab/Lch/Xyz, you might need to specify the white point if not using the default (D65)
        use palette::white_point::{D50, D65};

        // Convert SRGB (implicitly D65) to Lab with a D50 whitepoint via XYZ adaptation
        //  Step 1: Srgb (D65) -> Xyz (D65)
        let xyz_d65: Xyz<D65, f32> = srgb_color.into_color_unclamped();
        println!("SRGB (D65): {:?} -> Lab (D50): {:?}", srgb_color, xyz_d65);
        //  Step 2: Xyz (D65) -> Xyz (D50)
        let xyz_d50: Xyz<D50, f32> = xyz_d65.adapt_into_unclamped();
        println!("Xyz (D65): {:?} -> Lab (D50): {:?}", xyz_d65, xyz_d50);
        // Step 3: Xyz (D50) -> Lab (D50)
        let lab_d50: Lab<D50, f32> = xyz_d50.into_color_unclamped();
        println!("Xyz (D50): {:?} -> Lab (D50): {:?}", xyz_d50, lab_d50);

        // Convert Lab D50 back to SRGB f32 (implicitly D65)
        // 1. Lab(D50) -> Xyz(D50)
        let xyz_d50_from_lab: Xyz<D50, f32> = lab_d50.into_color_unclamped();
        println!(
            "Lab (D50): {:?} -> Xyz (D50): {:?}",
            lab_d50, xyz_d50_from_lab
        );
        // 2. Xyz(D50) -> Xyz(D65) (Adapt back)
        let xyz_d65_from_d50: Xyz<D65, f32> = xyz_d50_from_lab.adapt_into_unclamped();
        println!(
            "Xyz (D50): {:?} -> Xyz (D65): {:?}",
            xyz_d50_from_lab, xyz_d65_from_d50
        );
        // 3. Xyz(D65) -> Srgb(D65)
        let srgb_from_lab_d50: Srgb<f32> = xyz_d65_from_d50.into_color_unclamped();
        println!(
            "Xyz (D65): {:?} -> SRGB (D65): {:?}",
            xyz_d65_from_d50, srgb_from_lab_d50
        );
    }

    #[test]
    fn test_color_delta_e() {
        // 相同颜色 CIEDE2000 色差近似为 0
        let a = Color::from_rgb(120, 80, 200);
        assert!(color_delta_e(&a, a) < 0.01);

        // 明显不同的颜色（黑 vs 白）色差应非常大
        let black = Color::from_rgb(0, 0, 0);
        let white = Color::from_rgb(255, 255, 255);
        let de = color_delta_e(&black, white);
        assert!(de > 50.0, "black vs white deltaE should be large, got {de}");
    }
}
