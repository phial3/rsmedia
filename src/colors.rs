//! Color utilities built on top of [`colorous`] and [`colorutils_rs`].
//!
//! [`colorous::Color`] is a plain `{ r, g, b: u8 }` struct with no alpha and no
//! room for helper methods (it is a foreign type, so inherent impls and orphan
//! trait impls are rejected by the compiler). This module therefore defines its
//! own `Color` — same channel fields **plus** an alpha channel — and bridges to
//! colorous with lossless `From` conversions, so gradient/palette output
//! converts with a single `.into()`.
//!
//! # Converting between colour spaces
//!
//! RGB, HSV and HSL convert into each other in both directions:
//!
//! | source → target | RGB | HSV | HSL |
//! |---|---|---|---|
//! | **RGB** | — | [`rgb_to_hsv`] | [`rgb_to_hsl`] |
//! | **HSV** | [`hsv_to_rgb`] | — | [`hsv_to_hsl`] |
//! | **HSL** | [`hsl_to_rgb`] | [`hsl_to_hsv`] | — |
//!
//! Hue is always an angle in **degrees**; every other channel is a
//! **percentage**. Every function here both takes and returns its values inside
//! the standard domain of those quantities — hue normalised into `[0, 360)`,
//! saturation / value / lightness clamped to `[0, 100]` — so an out-of-range
//! argument is brought back in instead of being left to produce nonsense.
//!
//! That is not a nicety. A hue *is* an angle, so `-90`, `270` and `630` all name
//! one and the same colour, but `colorutils_rs` reads only the non-negative half
//! of that circle: it hands back **black** for a negative hue and the wrong
//! sector for one at or past a full turn, where `palette` wraps on its own. The
//! two libraries likewise pass an out-of-range percentage into their arithmetic
//! untouched, so a saturation of `150` comes back out of [`hsv_to_hsl`] as
//! `300`, and `-50` as `-200`. Normalising and clamping up front makes all six
//! conversions agree on one domain and answer the same thing for the same
//! colour.
//!
//! Beware that [`Color::to_hsv`] and [`Color::to_hsl`] hand back
//! `colorutils_rs`' own types, whose saturation / lightness / value are
//! *fractions* (0-1) — multiply them by 100 before passing them on, or stay on
//! the array-based functions above, which keep one convention throughout.
//!
//! Every conversion delegates to a vetted library — `colorutils_rs` for
//! RGB → HSV and RGB ↔ HSL, `palette` for HSV → RGB and HSV ↔ HSL, which also
//! owns the CIEDE2000 difference at the bottom of this module. Nothing here is
//! hand-rolled maths, and no eight-bit colour is lost going out and back:
//! `rgb_to_hsv` → `hsv_to_rgb` and `rgb_to_hsl` → `hsl_to_rgb` both return every
//! one of the 2^24 8-bit colours unchanged. HSV → RGB is `palette`'s rather than
//! `colorutils_rs`'s for exactly that reason — see [`hsv_to_rgb`].
//!
//! HSV ↔ HSL in particular is solved directly rather than by detouring through
//! 8-bit RGB: the detour costs up to a couple of degrees of hue at mid-range
//! saturations and much more for colours near black or white, where 8 bits
//! cannot represent the difference at all.

use std::fmt::{LowerHex, UpperHex};
use std::str::FromStr;

/// An RGBA color.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// extend
impl Color {
    /// Creates a color from the three channels (alpha = `0xFF`).
    #[inline]
    #[must_use]
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 0xFF }
    }

    /// Creates a color from all four channels.
    #[inline]
    #[must_use]
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Builds a colour from a hex literal at compile time — the `const`
    /// counterpart of the [`FromStr`] grammar, so the two spellings agree:
    /// `#RRGGBB`, or `#RRGGBBAA` when the alpha channel matters (the leading
    /// `#` is optional, digits may be upper or lower case, and a 6-digit value
    /// is treated as fully opaque).
    ///
    /// Use it for colour tables, where a typo should not survive to runtime:
    ///
    /// ```
    /// # use rsmedia::colors::Color;
    /// assert_eq!(Color::from_hex_str("#f0f8ff"), Color::from_rgb(0xF0, 0xF8, 0xFF));
    /// assert_eq!(Color::from_hex_str("#f0f8ff80").a, 0x80);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics unless the literal is exactly 6 or 8 hex digits behind an
    /// optional `#`. In a `const` context — which is the intended use — that
    /// panic is reported as a compile error.
    #[must_use]
    pub const fn from_hex_str(hex: &str) -> Self {
        let digits = match hex.as_bytes() {
            [b'#', rest @ ..] => rest,
            rest => rest,
        };

        match digits.len() {
            6 => Self::from_rgb(
                Self::hex_byte(digits, 0),
                Self::hex_byte(digits, 2),
                Self::hex_byte(digits, 4),
            ),
            8 => Self::from_rgba(
                Self::hex_byte(digits, 0),
                Self::hex_byte(digits, 2),
                Self::hex_byte(digits, 4),
                Self::hex_byte(digits, 6),
            ),
            _ => panic!("colour literal must hold 6 or 8 hex digits, as in \"#f0f8ff\""),
        }
    }

    /// Decodes the pair of hex digits at `at..at + 2`.
    const fn hex_byte(digits: &[u8], at: usize) -> u8 {
        (Self::hex_nibble(digits[at]) << 4) | Self::hex_nibble(digits[at + 1])
    }

    /// Decodes a single hex digit.
    const fn hex_nibble(digit: u8) -> u8 {
        match digit {
            b'0'..=b'9' => digit - b'0',
            b'a'..=b'f' => digit - b'a' + 10,
            b'A'..=b'F' => digit - b'A' + 10,
            _ => panic!("colour literal holds a character that is not a hex digit"),
        }
    }

    /// Returns `[r, g, b, a]`.
    #[inline]
    #[must_use]
    pub const fn as_array(&self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Returns `(r, g, b, a)`.
    #[inline]
    #[must_use]
    pub const fn as_tuple(&self) -> (u8, u8, u8, u8) {
        (self.r, self.g, self.b, self.a)
    }

    /// Returns `(r, g, b, a)`.
    #[inline]
    #[must_use]
    pub const fn rgba(&self) -> (u8, u8, u8, u8) {
        self.as_tuple()
    }

    /// Returns `(r, g, b)`.
    #[inline]
    #[must_use]
    pub const fn rgb(&self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    /// Red channel accessor (kept for API compatibility with the previous
    /// tuple-based implementation).
    #[inline]
    #[must_use]
    pub const fn r(&self) -> u8 {
        self.r
    }

    /// Green channel accessor.
    #[inline]
    #[must_use]
    pub const fn g(&self) -> u8 {
        self.g
    }

    /// Blue channel accessor.
    #[inline]
    #[must_use]
    pub const fn b(&self) -> u8 {
        self.b
    }

    /// Alpha channel accessor.
    #[inline]
    #[must_use]
    pub const fn a(&self) -> u8 {
        self.a
    }

    /// Returns the same color with a new alpha value.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, a: u8) -> Self {
        self.a = a;
        self
    }

    /// Formats the color as `#RRGGBBAA` (`uppercase` selects the letter case).
    #[inline]
    #[must_use]
    pub fn hex(&self, uppercase: bool) -> String {
        if uppercase {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }

    /// Builds a colour from HSV.
    ///
    /// `h` is a hue in degrees and `s` / `v` are **percentages**, matching
    /// [`rgb_to_hsv`] and [`hsv_to_rgb`] — and sharing their contract: a hue
    /// outside `[0, 360)` is wrapped into it and a percentage outside
    /// `[0, 100]` is clamped into it, rather than being passed on raw. Note the
    /// mismatch with [`Self::to_hsv`]: that returns a [`colorutils_rs::Hsv`],
    /// whose own convention is fractions (0..=1) — multiply its `s` / `v` by 100
    /// to feed them back here.
    ///
    /// ```
    /// # use rsmedia::colors::{hsv_to_rgb, Color};
    /// assert_eq!(Color::from_hsv(0, 100, 100), Color::from_rgb(255, 0, 0));
    /// // A full turn is no turn...
    /// assert_eq!(Color::from_hsv(360, 100, 100), Color::from_hsv(0, 100, 100));
    /// // ...and the builder answers exactly as the array-based function does.
    /// assert_eq!(Color::from_hsv(210, 60, 80).as_tuple(), {
    ///     let [r, g, b] = hsv_to_rgb(210.0, 60.0, 80.0);
    ///     (r, g, b, 255)
    /// });
    /// ```
    pub fn from_hsv(h: u16, s: u16, v: u16) -> Self {
        let [r, g, b] = hsv_to_rgb(f32::from(h), f32::from(s), f32::from(v));
        Self::from_rgb(r, g, b)
    }

    /// Converts this colour to HSV.
    ///
    /// The result keeps [`colorutils_rs::Hsv`]'s own units — hue in degrees,
    /// saturation and value as **fractions** (0..=1) — which is *not* what
    /// [`Self::from_hsv`] and [`rgb_to_hsv`] use (percentages). Scale by 100 when
    /// passing the values on to them.
    ///
    /// Built on [`rgb_to_hsv`], so the domain guarantees documented there hold
    /// here too.
    pub fn to_hsv(&self) -> colorutils_rs::Hsv {
        let [h, s, v] = rgb_to_hsv(self.r, self.g, self.b);
        colorutils_rs::Hsv::from_components(h, s / 100.0, v / 100.0)
    }

    /// Builds a colour from HSL; `h` a hue in degrees, `s` / `l` as
    /// **percentages**.
    ///
    /// Like [`Self::from_hsv`], this follows [`hsl_to_rgb`]'s contract: an
    /// out-of-range hue is wrapped into `[0, 360)` and an out-of-range
    /// percentage clamped into `[0, 100]`. As with HSV, [`Self::to_hsl`] returns
    /// fractional `s` / `l` (0..=1).
    ///
    /// ```
    /// # use rsmedia::colors::{hsl_to_rgb, Color};
    /// assert_eq!(Color::from_hsl(120, 100, 25), Color::from_rgb(0, 128, 0));
    /// assert_eq!(Color::from_hsl(480, 100, 25), Color::from_hsl(120, 100, 25));
    /// assert_eq!(Color::from_hsl(210, 60, 40).as_tuple(), {
    ///     let [r, g, b] = hsl_to_rgb(210.0, 60.0, 40.0);
    ///     (r, g, b, 255)
    /// });
    /// ```
    pub fn from_hsl(h: u16, s: u16, l: u16) -> Self {
        let [r, g, b] = hsl_to_rgb(f32::from(h), f32::from(s), f32::from(l));
        Self::from_rgb(r, g, b)
    }

    /// Converts this colour to HSL; see [`Self::to_hsv`] for the units.
    ///
    /// Routed through [`rgb_to_hsl`] rather than `colorutils_rs`' own conversion,
    /// so that the `0 / 0` saturation it produces for pure black and pure white
    /// (see the "Upstream quirk" section on [`rgb_to_hsl`]) reads `0.0` here
    /// instead of NaN — and so that the domain guarantees hold exactly as they do
    /// for [`Self::to_hsv`].
    pub fn to_hsl(&self) -> colorutils_rs::Hsl {
        let [h, s, l] = rgb_to_hsl(self.r, self.g, self.b);
        colorutils_rs::Hsl::from_components(h, s / 100.0, l / 100.0)
    }

    /// LAB
    pub fn from_lab(l: f32, a: f32, b: f32) -> Self {
        let rgb = colorutils_rs::Lab::new(l, a, b).to_rgb8();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_lab(&self) -> colorutils_rs::Lab {
        let (r, g, b, _a) = self.as_tuple();
        colorutils_rs::Rgb::<u8>::new(r, g, b).to_lab()
    }

    /// XYB
    pub fn from_xyb(x: f32, y: f32, b: f32) -> Self {
        let rgb = colorutils_rs::Xyb::new(x, y, b).to_rgb(colorutils_rs::TransferFunction::Srgb);
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_xyb(&self) -> colorutils_rs::Xyb {
        let (r, g, b, _a) = self.as_tuple();
        let rgb = colorutils_rs::Rgb::<u8>::new(r, g, b);
        colorutils_rs::Xyb::from_rgb(rgb, colorutils_rs::TransferFunction::Srgb)
    }

    /// XYZ
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        let rgb = colorutils_rs::Xyz::new(x, y, z).to_srgb();
        Self::from_rgb(rgb.r, rgb.g, rgb.b)
    }

    pub fn to_xyz(&self) -> colorutils_rs::Xyz {
        let (r, g, b, _a) = self.as_tuple();
        let rgb = colorutils_rs::Rgb::<u8>::new(r, g, b);
        colorutils_rs::Xyz::from_srgb(rgb)
    }
}

impl LowerHex for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}",
            self.r, self.g, self.b, self.a
        )
    }
}

impl UpperHex for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:02X}{:02X}{:02X}{:02X}",
            self.r, self.g, self.b, self.a
        )
    }
}

/// convert
impl From<u32> for Color {
    /// `0xRRGGBBAA` packing (what [`Color::hex`] emits).
    fn from(x: u32) -> Self {
        Self::from_rgba((x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, x as u8)
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
        color.as_tuple()
    }
}

impl From<Color> for [u8; 4] {
    fn from(color: Color) -> Self {
        color.as_array()
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

/// Lossless RGB conversion from a colorous gradient/palette output
/// (`TURBO.eval_rational(..).into()`); alpha is set to fully opaque.
impl From<colorous::Color> for Color {
    fn from(c: colorous::Color) -> Self {
        Self::from_rgb(c.r, c.g, c.b)
    }
}

/// Converts back into colorous (e.g. to feed other colorous APIs); the alpha
/// channel is dropped, since colorous colors carry no alpha.
impl From<Color> for colorous::Color {
    fn from(c: Color) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
        }
    }
}

impl FromStr for Color {
    type Err = &'static str;

    /// Parses `#RRGGBBAA` (the leading `#` is optional; a 6-digit value is
    /// treated as fully opaque).
    fn from_str(x: &str) -> std::result::Result<Self, Self::Err> {
        let hex = x.trim_start_matches('#');
        let hex = match hex.len() {
            6 => format!("{hex}ff"),
            8 => hex.to_string(),
            _ => return Err("Failed to convert `Color` from str: invalid length"),
        };

        u32::from_str_radix(&hex, 16)
            .map(Self::from)
            .map_err(|_| "Failed to convert `Color` from str: invalid hex")
    }
}

impl TryFrom<&str> for Color {
    type Error = &'static str;

    /// Same grammar as the [`FromStr`] impl, which this delegates to so that
    /// `str::parse` and `TryFrom` cannot drift apart.
    fn try_from(x: &str) -> std::result::Result<Self, Self::Error> {
        x.parse()
    }
}

/// Declares a named [`Color`] constant from a hex literal, under FFmpeg's own
/// colour name and with the exact value `ffmpeg -colors` prints for it.
///
/// ```text
/// color!(AliceBlue, "#f0f8ff");
/// ```
///
/// The literal follows the [`FromStr`] grammar (`#RRGGBB`, or `#RRGGBBAA` when
/// the alpha channel matters) and is decoded through the `const`
/// [`Color::from_hex_str`], so a mistyped value is a compile error rather than a
/// wrong colour at runtime.
///
/// FFmpeg spells these names itself — `Darkorange`, `GoldenRod`, `HoneyDew`,
/// `SeaShell`, `LightGrey` — and the constants keep that spelling, hence the
/// `non_upper_case_globals` allowance. Copying a whole `ffmpeg -colors` row into
/// `color!("<name>", "#<hex>")` therefore produces the correct constant.
macro_rules! color {
    ($name:ident, $hex:literal) => {
        #[allow(non_upper_case_globals)]
        pub const $name: Color = Color::from_hex_str($hex);
    };
}

// The 140 colour names `av_parse_color()` accepts, in `ffmpeg -colors` order and
// with FFmpeg's own spelling and hex values — one row of that listing per line,
// so the two can be diffed side by side. `AliceBlue` is `#f0f8ff`, `Darkorange`
// keeps its lowercase `o`, `GoldenRod` its capital `R`; the values are X11's,
// which is why `MediumPurple` is `#9370d8` and not CSS's `#9370db`.
color!(AliceBlue, "#f0f8ff");
color!(AntiqueWhite, "#faebd7");
color!(Aqua, "#00ffff");
color!(Aquamarine, "#7fffd4");
color!(Azure, "#f0ffff");
color!(Beige, "#f5f5dc");
color!(Bisque, "#ffe4c4");
color!(Black, "#000000");
color!(BlanchedAlmond, "#ffebcd");
color!(Blue, "#0000ff");
color!(BlueViolet, "#8a2be2");
color!(Brown, "#a52a2a");
color!(BurlyWood, "#deb887");
color!(CadetBlue, "#5f9ea0");
color!(Chartreuse, "#7fff00");
color!(Chocolate, "#d2691e");
color!(Coral, "#ff7f50");
color!(CornflowerBlue, "#6495ed");
color!(Cornsilk, "#fff8dc");
color!(Crimson, "#dc143c");
color!(Cyan, "#00ffff");
color!(DarkBlue, "#00008b");
color!(DarkCyan, "#008b8b");
color!(DarkGoldenRod, "#b8860b");
color!(DarkGray, "#a9a9a9");
color!(DarkGreen, "#006400");
color!(DarkKhaki, "#bdb76b");
color!(DarkMagenta, "#8b008b");
color!(DarkOliveGreen, "#556b2f");
color!(Darkorange, "#ff8c00");
color!(DarkOrchid, "#9932cc");
color!(DarkRed, "#8b0000");
color!(DarkSalmon, "#e9967a");
color!(DarkSeaGreen, "#8fbc8f");
color!(DarkSlateBlue, "#483d8b");
color!(DarkSlateGray, "#2f4f4f");
color!(DarkTurquoise, "#00ced1");
color!(DarkViolet, "#9400d3");
color!(DeepPink, "#ff1493");
color!(DeepSkyBlue, "#00bfff");
color!(DimGray, "#696969");
color!(DodgerBlue, "#1e90ff");
color!(FireBrick, "#b22222");
color!(FloralWhite, "#fffaf0");
color!(ForestGreen, "#228b22");
color!(Fuchsia, "#ff00ff");
color!(Gainsboro, "#dcdcdc");
color!(GhostWhite, "#f8f8ff");
color!(Gold, "#ffd700");
color!(GoldenRod, "#daa520");
color!(Gray, "#808080");
color!(Green, "#008000");
color!(GreenYellow, "#adff2f");
color!(HoneyDew, "#f0fff0");
color!(HotPink, "#ff69b4");
color!(IndianRed, "#cd5c5c");
color!(Indigo, "#4b0082");
color!(Ivory, "#fffff0");
color!(Khaki, "#f0e68c");
color!(Lavender, "#e6e6fa");
color!(LavenderBlush, "#fff0f5");
color!(LawnGreen, "#7cfc00");
color!(LemonChiffon, "#fffacd");
color!(LightBlue, "#add8e6");
color!(LightCoral, "#f08080");
color!(LightCyan, "#e0ffff");
color!(LightGoldenRodYellow, "#fafad2");
color!(LightGreen, "#90ee90");
color!(LightGrey, "#d3d3d3");
color!(LightPink, "#ffb6c1");
color!(LightSalmon, "#ffa07a");
color!(LightSeaGreen, "#20b2aa");
color!(LightSkyBlue, "#87cefa");
color!(LightSlateGray, "#778899");
color!(LightSteelBlue, "#b0c4de");
color!(LightYellow, "#ffffe0");
color!(Lime, "#00ff00");
color!(LimeGreen, "#32cd32");
color!(Linen, "#faf0e6");
color!(Magenta, "#ff00ff");
color!(Maroon, "#800000");
color!(MediumAquaMarine, "#66cdaa");
color!(MediumBlue, "#0000cd");
color!(MediumOrchid, "#ba55d3");
color!(MediumPurple, "#9370d8");
color!(MediumSeaGreen, "#3cb371");
color!(MediumSlateBlue, "#7b68ee");
color!(MediumSpringGreen, "#00fa9a");
color!(MediumTurquoise, "#48d1cc");
color!(MediumVioletRed, "#c71585");
color!(MidnightBlue, "#191970");
color!(MintCream, "#f5fffa");
color!(MistyRose, "#ffe4e1");
color!(Moccasin, "#ffe4b5");
color!(NavajoWhite, "#ffdead");
color!(Navy, "#000080");
color!(OldLace, "#fdf5e6");
color!(Olive, "#808000");
color!(OliveDrab, "#6b8e23");
color!(Orange, "#ffa500");
color!(OrangeRed, "#ff4500");
color!(Orchid, "#da70d6");
color!(PaleGoldenRod, "#eee8aa");
color!(PaleGreen, "#98fb98");
color!(PaleTurquoise, "#afeeee");
color!(PaleVioletRed, "#d87093");
color!(PapayaWhip, "#ffefd5");
color!(PeachPuff, "#ffdab9");
color!(Peru, "#cd853f");
color!(Pink, "#ffc0cb");
color!(Plum, "#dda0dd");
color!(PowderBlue, "#b0e0e6");
color!(Purple, "#800080");
color!(Red, "#ff0000");
color!(RosyBrown, "#bc8f8f");
color!(RoyalBlue, "#4169e1");
color!(SaddleBrown, "#8b4513");
color!(Salmon, "#fa8072");
color!(SandyBrown, "#f4a460");
color!(SeaGreen, "#2e8b57");
color!(SeaShell, "#fff5ee");
color!(Sienna, "#a0522d");
color!(Silver, "#c0c0c0");
color!(SkyBlue, "#87ceeb");
color!(SlateBlue, "#6a5acd");
color!(SlateGray, "#708090");
color!(Snow, "#fffafa");
color!(SpringGreen, "#00ff7f");
color!(SteelBlue, "#4682b4");
color!(Tan, "#d2b48c");
color!(Teal, "#008080");
color!(Thistle, "#d8bfd8");
color!(Tomato, "#ff6347");
color!(Turquoise, "#40e0d0");
color!(Violet, "#ee82ee");
color!(Wheat, "#f5deb3");
color!(White, "#ffffff");
color!(WhiteSmoke, "#f5f5f5");
color!(Yellow, "#ffff00");
color!(YellowGreen, "#9acd32");

/// Brings a hue angle into the standard `[0, 360)` domain.
///
/// A hue names an angle, so a full turn is the same direction as none at all:
/// `-90.0`, `270.0` and `630.0` are one and the same colour. `colorutils_rs`
/// does not read it that way — it answers **black** for a negative hue, and for
/// a hue at or past a full turn it leaves the branch ladder that decodes the
/// angle, landing on a sector that is simply wrong — while `palette` wraps on
/// its own. Wrapping here, once, is what makes every conversion in this module
/// agree; `rem_euclid` rather than `%` because the latter keeps the sign.
fn normalize_hue(hue: f32) -> f32 {
    hue.rem_euclid(360.0)
}

/// Brings a saturation / value / lightness percentage into the standard
/// `[0, 100]` domain.
///
/// Both libraries feed an out-of-range percentage straight into their
/// arithmetic, which does not clamp it but transforms it into something else
/// entirely: a saturation of `150` comes back out of [`hsv_to_hsl`] as `300`,
/// and `-50` as `-200`. A percentage of a whole is defined on `[0, 100]`, so
/// that is what callers are held to and what they get back.
fn clamp_percent(percent: f32) -> f32 {
    percent.clamp(0.0, 100.0)
}

/// Convert RGB to HSV color space.
///
/// Returns `[h, s, v]` where `h` is an angle in **degrees** and `s` / `v` are
/// **percentages** — *not* the 0-1 fractions some HSV APIs use. Both channels
/// come back inside their standard domain (`h` in `[0, 360)`, `s` / `v` in
/// `[0, 100]`), so a saturation that the underlying arithmetic lands a hair over
/// `100` is reported as exactly `100`. It is the exact counterpart of
/// [`hsv_to_rgb`]: the two round-trip without loss, so every 8-bit colour comes
/// back from `hsv_to_rgb(rgb_to_hsv(..))` unchanged.
///
/// ```
/// # use rsmedia::colors::{hsv_to_rgb, rgb_to_hsv};
/// assert_eq!(rgb_to_hsv(255, 0, 0), [0.0, 100.0, 100.0]);
/// assert_eq!(hsv_to_rgb(0.0, 100.0, 100.0), [255, 0, 0]);
/// ```
///
/// For the sibling space see [`rgb_to_hsl`] / [`hsv_to_hsl`].
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> [f32; 3] {
    let hsv = colorutils_rs::Rgb::<u8>::new(r, g, b).to_hsv();
    [
        normalize_hue(hsv.h),
        clamp_percent(hsv.s * 100.0),
        clamp_percent(hsv.v * 100.0),
    ]
}

/// Convert HSV to RGB color space.
///
/// The inverse of [`rgb_to_hsv`], with the same convention: `h` is an angle in
/// **degrees** and `s` / `v` are **percentages**. Both arguments are taken
/// through `normalize_hue` and `clamp_percent` first, so an out-of-domain
/// value names a colour rather than producing a broken one — `colorutils_rs`
/// alone returns black for any negative hue.
///
/// Unlike the other conversions, this leg is `palette`'s rather than
/// `colorutils_rs`'s, and deliberately so. `colorutils_rs`' `Hsv::to_rgb8`
/// narrows the channels to `u8` with a **truncating** cast where its own
/// `Hsl::to_rgb8` *rounds*, and that one difference drags better than half of
/// all colours a unit darker: measured over the full 2^24 8-bit colours,
/// `rgb_to_hsv` → `hsv_to_rgb` came back wrong for 9,594,803 of them — always
/// one unit low, never one high — while the HSL pair was exact. `palette`
/// rounds, so the pair is now exact too: no 8-bit colour changes value on the
/// round trip.
///
/// ```
/// # use rsmedia::colors::{hsv_to_rgb, rgb_to_hsv};
/// // Exact, not merely close — including the dark colours that used to lose a unit.
/// for r in [0u8, 1, 90, 128, 254, 255] {
///     let [h, s, v] = rgb_to_hsv(r, 0, 45);
///     assert_eq!(hsv_to_rgb(h, s, v), [r, 0, 45]);
/// }
/// // A hue is an angle: -90° and 270° are the same direction, so the same colour.
/// assert_eq!(hsv_to_rgb(-90.0, 100.0, 100.0), hsv_to_rgb(270.0, 100.0, 100.0));
/// // A percentage is clamped to its domain rather than left to misbehave.
/// assert_eq!(hsv_to_rgb(0.0, 150.0, 100.0), hsv_to_rgb(0.0, 100.0, 100.0));
/// ```
///
/// For the sibling space see [`hsl_to_rgb`] / [`hsv_to_hsl`].
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    use palette::convert::IntoColorUnclamped;
    use palette::{Hsv, Srgb};

    // `Hsv` stores s/v as fractions (0..=1), while this API takes percentages —
    // hence the `/ 100.0` (matching `rgb_to_hsv`'s `* 100.0`). Coming back is
    // `Srgb::from_format`, which rounds to nearest rather than truncating.
    let srgb: Srgb<f32> = Hsv::new_srgb(
        normalize_hue(h),
        clamp_percent(s) / 100.0,
        clamp_percent(v) / 100.0,
    )
    .into_color_unclamped();
    let rgb: Srgb<u8> = Srgb::from_format(srgb);
    [rgb.red, rgb.green, rgb.blue]
}

/// Convert RGB to HSL color space.
///
/// Returns `[h, s, l]` under the same convention as [`rgb_to_hsv`]: `h` is an
/// angle in **degrees** and `s` / `l` are **percentages** — *not* the 0-1
/// fractions some HSL APIs use. Both come back inside their standard domain
/// (`h` in `[0, 360)`, `s` / `l` in `[0, 100]`). It is the exact counterpart of
/// [`hsl_to_rgb`], so the two round-trip:
///
/// ```
/// # use rsmedia::colors::{hsl_to_rgb, rgb_to_hsl};
/// assert_eq!(rgb_to_hsl(255, 0, 0), [0.0, 100.0, 50.0]);
/// assert_eq!(hsl_to_rgb(0.0, 100.0, 50.0), [255, 0, 0]);
/// ```
///
/// Implemented by `colorutils_rs` — the crate behind [`rgb_to_hsv`] too.
///
/// # Upstream quirk
///
/// The conversion is [`colorutils_rs::Hsl::from_rgb`]'s (`rgb2hsl` in that
/// crate's `hsl.rs`), which works saturation out as `delta / (1 - |2l - 1|)`.
/// That denominator is `0` for a fully black or a fully white input, so pure
/// `#000000` and `#ffffff` come back with a **NaN** saturation — unlike
/// `colorutils_rs`' own `rgb_to_hsv`, which guards the same division. Both are
/// greys and have no saturation to report, so they are given `0.0` here. Every
/// other input passes through untouched, bit for bit.
///
/// ```
/// # use rsmedia::colors::rgb_to_hsl;
/// assert_eq!(rgb_to_hsl(0, 0, 0), [0.0, 0.0, 0.0]);          // not NaN
/// assert_eq!(rgb_to_hsl(255, 255, 255), [0.0, 0.0, 100.0]);  // not NaN
/// ```
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> [f32; 3] {
    let hsl = colorutils_rs::Rgb::<u8>::new(r, g, b).to_hsl();
    // `Hsl::from_rgb` divides by `1 - |2l - 1|`, which is 0 for pure black and
    // pure white, making saturation `0 / 0` — a NaN. Both are greys, so 0 is the
    // honest answer; everything else is upstream's value unchanged.
    let s = if hsl.s.is_finite() { hsl.s } else { 0.0 };
    [
        normalize_hue(hsl.h),
        clamp_percent(s * 100.0),
        clamp_percent(hsl.l * 100.0),
    ]
}

/// Convert HSL to RGB color space.
///
/// The inverse of [`rgb_to_hsl`] — `h` an angle in **degrees**, `s` / `l` as
/// **percentages**. Both arguments go through `normalize_hue` and
/// `clamp_percent` first, exactly as in [`hsv_to_rgb`]. As there, the channels
/// keep their fractional part instead of being rounded to integers first, and
/// the narrowing back to `u8` rounds rather than truncates, so [`rgb_to_hsl`] →
/// `hsl_to_rgb` is exact — every 8-bit colour returns unchanged:
///
/// ```
/// # use rsmedia::colors::{hsl_to_rgb, rgb_to_hsl};
/// assert_eq!(hsl_to_rgb(120.0, 100.0, 25.0), [0, 128, 0]);
/// let [h, s, l] = rgb_to_hsl(128, 64, 200);
/// assert_eq!(hsl_to_rgb(h, s, l), [128, 64, 200]);
/// ```
///
/// Wrapping the hue is not merely cosmetic here. `colorutils_rs` decodes the
/// angle with a `[0, 360)` branch ladder whose catch-all arm reflects the angle
/// about the wrong axis, so a hue *past* a full turn lands on an arbitrary
/// sector: left alone, `hsl_to_rgb(400.0, 100.0, 50.0)` answers `[255, 0, 170]`
/// where the correct colour, `40°`, is `[255, 170, 0]`. Reduced first, both
/// spellings agree:
///
/// ```
/// # use rsmedia::colors::hsl_to_rgb;
/// assert_eq!(hsl_to_rgb(400.0, 100.0, 50.0), [255, 170, 0]);
/// assert_eq!(hsl_to_rgb(400.0, 100.0, 50.0), hsl_to_rgb(40.0, 100.0, 50.0));
/// ```
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [u8; 3] {
    // `Hsl` stores s/l as fractions (0..=1), while this API takes percentages —
    // hence the `/ 100.0` (matching `rgb_to_hsl`'s `* 100.0`).
    let hsl = colorutils_rs::Hsl::from_components(
        normalize_hue(h),
        clamp_percent(s) / 100.0,
        clamp_percent(l) / 100.0,
    );
    let rgb = hsl.to_rgb8();
    [rgb.r, rgb.g, rgb.b]
}

/// Convert HSV to HSL color space.
///
/// Both spaces carry the same hue codomain, so only the other two channels need
/// recomputing. As everywhere else in this module `h` is an angle in **degrees**
/// and `s` / `v` / `l` are **percentages**, and as everywhere else the inputs go
/// through `normalize_hue` and `clamp_percent` and the outputs stay inside
/// the same domain.
///
/// `colorutils_rs` has no HSV ↔ HSL bridge, so this one comes from `palette`,
/// which solves the pair directly — no 8-bit RGB leg — and guards black, white
/// and grey on its own. [`hsv_to_hsl`] → [`hsl_to_hsv`] therefore returns its
/// input up to f32 rounding:
///
/// ```
/// # use rsmedia::colors::{hsl_to_hsv, hsv_to_hsl};
/// assert_eq!(hsv_to_hsl(0.0, 100.0, 100.0), [0.0, 100.0, 50.0]);
/// assert_eq!(hsv_to_hsl(120.0, 100.0, 50.0), [120.0, 100.0, 25.0]);
/// assert_eq!(hsl_to_hsv(120.0, 100.0, 25.0), [120.0, 100.0, 50.0]);
/// assert_eq!(hsv_to_hsl(-90.0, 100.0, 100.0), [270.0, 100.0, 50.0]);
/// ```
pub fn hsv_to_hsl(h: f32, s: f32, v: f32) -> [f32; 3] {
    use palette::convert::IntoColorUnclamped;
    use palette::{Hsl, Hsv};

    let hsl: Hsl = Hsv::new_srgb(
        normalize_hue(h),
        clamp_percent(s) / 100.0,
        clamp_percent(v) / 100.0,
    )
    .into_color_unclamped();
    [
        normalize_hue(hsl.hue.into_positive_degrees()),
        clamp_percent(hsl.saturation * 100.0),
        clamp_percent(hsl.lightness * 100.0),
    ]
}

/// Convert HSL to HSV color space.
///
/// The inverse of [`hsv_to_hsl`] — and, like it, `palette`'s conversion rather
/// than any formula of ours. See [`hsv_to_hsl`] for the units and the domain
/// contract; going back and forth keeps the channels to within f32 rounding.
pub fn hsl_to_hsv(h: f32, s: f32, l: f32) -> [f32; 3] {
    use palette::convert::IntoColorUnclamped;
    use palette::{Hsl, Hsv};

    let hsv: Hsv = Hsl::new_srgb(
        normalize_hue(h),
        clamp_percent(s) / 100.0,
        clamp_percent(l) / 100.0,
    )
    .into_color_unclamped();
    [
        normalize_hue(hsv.hue.into_positive_degrees()),
        clamp_percent(hsv.saturation * 100.0),
        clamp_percent(hsv.value * 100.0),
    ]
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

    /// `FromStr` is the canonical parser; `TryFrom<&str>` delegates to it, so the
    /// two must agree (including on the error cases).
    #[test]
    fn test_from_str_matches_try_from() {
        for text in ["#FF0000", "00FF00", "#0000FF80", "010203"] {
            let parsed: Color = text.parse().unwrap();
            assert_eq!(parsed, Color::try_from(text).unwrap());
        }

        // Both reject the same inputs, and neither accepts them silently.
        for text in ["", "#FF00", "#FF00000000", "nothex!!"] {
            assert!(text.parse::<Color>().is_err(), "{text} should not parse");
            assert!(Color::try_from(text).is_err(), "{text} should not convert");
        }
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Red, Color::from_rgb(255, 0, 0));
        assert_eq!(Green, Color::from_rgb(0, 128, 0));
        assert_eq!(Blue, Color::from_rgb(0, 0, 255));
    }

    #[test]
    fn test_color_macro_hex_literals() {
        // FFmpeg name plus FFmpeg hex: 6 digits means fully opaque...
        assert_eq!(Crimson, Color::from_rgba(220, 20, 60, 0xFF));
        // ...and 8 digits carry their own alpha.
        color!(Ghost, "#f8f8ff80");
        assert_eq!(Ghost, Color::from_rgba(248, 248, 255, 128));
    }

    /// [`Color::from_hex_str`] is the `const` twin of [`FromStr`]: the same
    /// grammar, and the same colour for every spelling both accept.
    #[test]
    fn test_from_hex_str_matches_from_str() {
        for literal in [
            "#f0f8ff",
            "f0f8ff",
            "#F0F8FF",
            "F0f8Ff",
            "#f0f8ff80",
            "f0f8ff80",
            "#00000000",
        ] {
            assert_eq!(
                Color::from_hex_str(literal),
                literal.parse::<Color>().unwrap(),
                "{literal}"
            );
        }

        // 6 digits is opaque; 8 digits is taken as written.
        assert_eq!(Color::from_hex_str("f0f8ff").a, 0xFF);
        assert_eq!(Color::from_hex_str("#00ff0080").a, 0x80);
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
                (h - eh).abs() < 0.5,
                "Hue mismatch for RGB({r},{g},{b}): {h} vs {eh}"
            );
            assert!(
                (s - es).abs() <= 0.5,
                "S mismatch for RGB({r},{g},{b}): {s} vs {es}"
            );
            assert!(
                (v - ev).abs() <= 0.5,
                "V mismatch for RGB({r},{g},{b}): {v} vs {ev}"
            );

            // 往返转换只允许 ±1 的浮点舍入（s/v 不再被截断成整数百分比）
            let [r2, g2, b2] = hsv_to_rgb(h, s, v);
            assert!(
                (r as i32 - r2 as i32).abs() <= 1
                    && (g as i32 - g2 as i32).abs() <= 1
                    && (b as i32 - b2 as i32).abs() <= 1,
                "roundtrip mismatch for RGB({r},{g},{b}): [{r2},{g2},{b2}]"
            );
        }
    }

    /// Every hue-taking conversion wraps its angle into `[0, 360)`, so any
    /// spelling of one direction names one colour. This is the fix for a real
    /// defect: left to `colorutils_rs`, a negative hue came back as black out of
    /// [`hsv_to_rgb`] and as a wrong sector out of [`hsl_to_rgb`].
    #[test]
    fn test_conversions_normalize_hue() {
        let cases = [
            (0.0, 0.0),
            (360.0, 0.0),
            (720.0, 0.0),
            (-90.0, 270.0),
            (-1.0, 359.0),
            (370.0, 10.0),
            (450.0, 90.0),
        ];

        for (input, expected) in cases {
            assert_eq!(
                hsv_to_hsl(input, 100.0, 100.0)[0],
                expected,
                "hsv_to_hsl({input})"
            );
            assert_eq!(
                hsl_to_hsv(input, 100.0, 50.0)[0],
                expected,
                "hsl_to_hsv({input})"
            );

            // The two RGB-facing directions take the hue as an argument too, and
            // are compared against their own reduced form rather than a literal.
            assert_eq!(
                hsv_to_rgb(input, 100.0, 100.0),
                hsv_to_rgb(expected, 100.0, 100.0),
                "hsv_to_rgb({input})"
            );
            assert_eq!(
                hsl_to_rgb(input, 100.0, 50.0),
                hsl_to_rgb(expected, 100.0, 50.0),
                "hsl_to_rgb({input})"
            );
        }

        // The defect itself, pinned: a negative hue is a direction, not black.
        assert_eq!(
            hsv_to_rgb(-90.0, 100.0, 100.0),
            hsv_to_rgb(270.0, 100.0, 100.0)
        );
        assert_ne!(hsv_to_rgb(-90.0, 100.0, 100.0), [0, 0, 0]);

        // `400°` used to decode through the wrong arm of upstream's branch ladder
        // and come back as `[255, 0, 170]`; `40°` is orange.
        assert_eq!(hsl_to_rgb(400.0, 100.0, 50.0), [255, 170, 0]);
    }

    /// Percentages outside `[0, 100]` are clamped into it, in both directions:
    /// as arguments, where the libraries would otherwise turn `150` into a
    /// reported saturation of `300`, and as results.
    #[test]
    fn test_conversions_clamp_percentages() {
        // Same answer whether the caller clamps or we do.
        assert_eq!(hsv_to_hsl(0.0, 150.0, 100.0), hsv_to_hsl(0.0, 100.0, 100.0));
        assert_eq!(hsv_to_hsl(0.0, -50.0, 100.0), hsv_to_hsl(0.0, 0.0, 100.0));
        assert_eq!(hsl_to_hsv(0.0, -50.0, 50.0), hsl_to_hsv(0.0, 0.0, 50.0));
        assert_eq!(hsv_to_rgb(0.0, 150.0, 100.0), hsv_to_rgb(0.0, 100.0, 100.0));
        assert_eq!(hsl_to_rgb(0.0, -50.0, 50.0), hsl_to_rgb(0.0, 0.0, 50.0));

        // A lightness past the top is simply white, not a wrapped-around colour.
        assert_eq!(hsl_to_rgb(0.0, 100.0, 200.0), [255, 255, 255]);
        // …and the reported channels never leave the domain.
        assert_eq!(hsl_to_hsv(0.0, 150.0, 50.0)[1], 100.0);

        for r in [0u8, 1, 128, 254, 255] {
            for g in [0u8, 1, 128, 254, 255] {
                for b in [0u8, 1, 128, 254, 255] {
                    let [h, s, v] = rgb_to_hsv(r, g, b);
                    let [h2, s2, l] = rgb_to_hsl(r, g, b);
                    let in_domain = |hue: f32, a: f32, b: f32| {
                        (0.0..360.0).contains(&hue)
                            && (0.0..=100.0).contains(&a)
                            && (0.0..=100.0).contains(&b)
                    };
                    assert!(in_domain(h, s, v), "rgb_to_hsv({r},{g},{b}) = {h},{s},{v}");
                    assert!(
                        in_domain(h2, s2, l),
                        "rgb_to_hsl({r},{g},{b}) = {h2},{s2},{l}"
                    );
                }
            }
        }
    }

    /// The `u16` builders share the array-based functions' contract, so a hue of
    /// a full turn is no turn and an oversized percentage is clamped rather than
    /// wrapped into a different colour.
    #[test]
    fn test_from_hsv_from_hsl_follow_the_domain() {
        for h in [0u16, 90, 359, 360, 450, 720] {
            for c in [0u16, 50, 100] {
                let expect_hsv = {
                    let [r, g, b] = hsv_to_rgb(f32::from(h), f32::from(c), 100.0);
                    Color::from_rgb(r, g, b)
                };
                assert_eq!(Color::from_hsv(h, c, 100), expect_hsv, "from_hsv({h},{c})");

                let expect_hsl = {
                    let [r, g, b] = hsl_to_rgb(f32::from(h), f32::from(c), 50.0);
                    Color::from_rgb(r, g, b)
                };
                assert_eq!(Color::from_hsl(h, c, 50), expect_hsl, "from_hsl({h},{c})");
            }
        }

        assert_eq!(Color::from_hsv(360, 100, 100), Color::from_hsv(0, 100, 100));
        assert_eq!(Color::from_hsl(360, 100, 50), Color::from_hsl(0, 100, 50));
        assert_eq!(Color::from_hsv(0, 200, 100), Color::from_hsv(0, 100, 100));
        assert_eq!(Color::from_hsl(0, 65535, 50), Color::from_hsl(0, 100, 50));
    }

    /// Known RGB → HSL values (hue in degrees, saturation / lightness in percent).
    #[test]
    fn test_rgb_hsl_known_values() {
        let cases: &[(u8, u8, u8, f32, f32, f32)] = &[
            (255, 0, 0, 0.0, 100.0, 50.0),         // 纯红
            (0, 255, 0, 120.0, 100.0, 50.0),       // 纯绿
            (0, 0, 255, 240.0, 100.0, 50.0),       // 纯蓝
            (255, 255, 0, 60.0, 100.0, 50.0),      // 黄
            (255, 165, 0, 38.823_53, 100.0, 50.0), // 橙
            (128, 64, 200, 268.235_3, 55.284_6, 51.764_7),
            (1, 2, 3, 210.0, 50.0, 0.784_314), // 极暗
            (0, 0, 0, 0.0, 0.0, 0.0),          // 纯黑：无色调、无饱和度
            (255, 255, 255, 0.0, 0.0, 100.0),  // 纯白
            (128, 128, 128, 0.0, 0.0, 50.196), // 中灰
        ];

        for &(r, g, b, eh, es, el) in cases {
            let [h, s, l] = rgb_to_hsl(r, g, b);
            assert!(
                (h - eh).abs() < 0.5,
                "HSL hue mismatch for RGB({r},{g},{b}): {h} vs {eh}"
            );
            assert!(
                (s - es).abs() <= 0.5,
                "HSL saturation mismatch for RGB({r},{g},{b}): {s} vs {es}"
            );
            assert!(
                (l - el).abs() <= 0.5,
                "HSL lightness mismatch for RGB({r},{g},{b}): {l} vs {el}"
            );

            // 往返只允许 ±1（与 HSV 一致）
            let [r2, g2, b2] = hsl_to_rgb(h, s, l);
            assert!(
                (r as i32 - r2 as i32).abs() <= 1
                    && (g as i32 - g2 as i32).abs() <= 1
                    && (b as i32 - b2 as i32).abs() <= 1,
                "HSL roundtrip mismatch for RGB({r},{g},{b}): [{r2},{g2},{b2}]"
            );
        }
    }

    /// HSV ↔ HSL must land where the RGB detour through the other space lands.
    /// Both directions are checked on a grid, since the two paths round
    /// independently (`Hsv::to_rgb8` truncates, `Hsl::to_rgb8` rounds).
    #[test]
    fn test_hsv_hsl_agree_with_rgb_detour() {
        let near = |a: [u8; 3], b: [u8; 3]| (0..3).all(|i| (a[i] as i32 - b[i] as i32).abs() <= 1);

        for h in [0.0, 45.0, 120.0, 210.0, 300.0, 359.0] {
            for s in [0.0, 25.0, 50.0, 100.0] {
                for v in [0.0, 10.0, 50.0, 100.0] {
                    let direct = hsv_to_rgb(h, s, v);
                    let [sh, ss, sl] = hsv_to_hsl(h, s, v);
                    let via_hsl = hsl_to_rgb(sh, ss, sl);
                    assert!(
                        near(direct, via_hsl),
                        "HSV({h},{s},{v}): {direct:?} vs HSL detour {via_hsl:?}"
                    );
                }
            }
        }

        for h in [0.0, 45.0, 120.0, 210.0, 300.0, 359.0] {
            for s in [0.0, 25.0, 50.0, 100.0] {
                for l in [0.0, 10.0, 50.0, 100.0] {
                    let direct = hsl_to_rgb(h, s, l);
                    let [vh, vs, vv] = hsl_to_hsv(h, s, l);
                    let via_hsv = hsv_to_rgb(vh, vs, vv);
                    assert!(
                        near(direct, via_hsv),
                        "HSL({h},{s},{l}): {direct:?} vs HSV detour {via_hsv:?}"
                    );
                }
            }
        }
    }

    /// HSV → HSL → HSV returns its input (no RGB in between, so no quantisation).
    #[test]
    fn test_hsv_hsl_round_trip() {
        for h in [0.0, 77.0, 180.0, 359.0] {
            for s in [0.0, 33.0, 100.0] {
                for v in [1.0, 42.0, 100.0] {
                    let [hh, hs, hl] = hsv_to_hsl(h, s, v);
                    let [h2, s2, v2] = hsl_to_hsv(hh, hs, hl);
                    assert!(
                        (h - h2).abs() < 1e-4 && (s - s2).abs() < 0.01 && (v - v2).abs() < 0.01,
                        "HSV({h},{s},{v}) -> [{h2},{s2},{v2}]"
                    );
                }
            }
        }
    }

    /// Neither conversion may leak the `0 / 0` NaN that `colorutils_rs`' own
    /// `rgb2hsl` produces for pure black and pure white.
    #[test]
    fn test_to_hsl_and_to_hsv_stay_finite() {
        for (color, edge) in [(Black, 0.0), (White, 1.0)] {
            let hsl = color.to_hsl();
            assert_eq!(
                (hsl.h, hsl.s, hsl.l),
                (0.0, 0.0, edge),
                "HSL of {color:?} must be finite"
            );

            let hsv = color.to_hsv();
            assert_eq!(
                (hsv.h, hsv.s, hsv.v),
                (0.0, 0.0, edge),
                "HSV of {color:?} must be finite"
            );
        }
    }

    /// The HSV round trip is exact, not ±1. It used to lose a unit on better
    /// than half of all colours — `colorutils_rs`' `Hsv::to_rgb8` truncates
    /// where its `Hsl::to_rgb8` rounds — which is why [`hsv_to_rgb`] is
    /// `palette`'s conversion now. Sweeping every value of one channel at a time
    /// covers the whole 0..=255 range without an exhaustive 2^24 run.
    #[test]
    fn test_round_trip() {
        let mut cases: Vec<[u8; 3]> = vec![
            [123, 45, 67],  // 随机颜色
            [255, 128, 0],  // 橙色
            [75, 200, 220], // 青色系
            [30, 150, 80],  // 绿色系
            [0, 0, 45],     // the colour that used to come back as (0, 0, 44)
        ];
        for v in 0..=255u8 {
            cases.push([v, 0, 45]);
            cases.push([45, v, 0]);
            cases.push([0, 45, v]);
            cases.push([v, v, 255 - v]);
        }

        for [r, g, b] in cases {
            let [h, s, v] = rgb_to_hsv(r, g, b);
            assert_eq!(
                hsv_to_rgb(h, s, v),
                [r, g, b],
                "HSV round trip of RGB({r}, {g}, {b})"
            );
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
