//! FFI wrapper macros.
//!
//! Three macros cover everything the crate needs from FFmpeg's constants. Pick between them by
//! asking "can two values be meaningfully combined?", not by which FFmpeg version the constants
//! come from. Full capability matrix:
//!
//! | capability | `ffi_enum_wrap_from!` | `ffi_enum_wrap!` | `ffi_enum!` |
//! |------------|-----------------------|------------------|-------------|
//! | intended for | mutually exclusive IDs | bit sets | bit sets |
//! | declares the named FFI type | yes | yes, plus an explicit `size_of` check | no — constants may be bare integers *or* named types, normalised by `as` |
//! | `as_raw()` | no | yes | yes |
//! | conversion variant to raw value | yes (`From<Enum> for ffi`) | yes (`From<Enum> for ffi`) | yes (`From<Enum> for repr`, i.e. `Into<repr>`) |
//! | conversion raw value to variant | yes (`From<ffi> for Enum`) | no | no |
//! | combine flags into a raw mask | no | no | yes (`BitOr`, either operand order) |
//! | test a bit inside a raw mask | no | no | yes (`BitAnd`: `mask & Enum::A`) |
//! | fallback for an unlisted value | panics; every table fails fast (the expression form is still supported but unused) | n/a | n/a |
//!
//! In practice the split is mostly by kind: the `ffi_enum!` call sites are bit sets
//! (`AVCodecFlag`, `AVCodecFlag2`, `AVFormatFlag`, `AVPixFmtFlag`, `AVSeekFlag`) and the
//! `ffi_enum_wrap_from!` call sites are IDs (`PixelFormat`, `SampleFormat`, `MediaType`,
//! `HWDeviceType`, plus the swscale value sets in `scale.rs`: `SwsDither`, `AlphaBlend`,
//! `ScalerFlags`, `Intent`, `Backend`).
//!
//! One table is an ID **and** sits with bit sets: `ScaleAlgorithm` (`scale.rs`) is a mutually
//! exclusive choice ("only one may be active at a time" per FFmpeg's header) whose members are
//! `SWS_*` bits — it is therefore an `ffi_enum!` bit-set table even though callers pick one
//! value, and combining two of its bits is a caller error FFmpeg rejects rather than something
//! the type prevents.
//!
//! `ffi_enum_wrap!` currently has no user — see the note on the macro itself for why the need
//! for it disappeared.
//!
//! Points worth remembering, because they are easy to get wrong:
//!
//! - A fieldless enum cannot hold an unnamed discriminant, so the bit-set operators yield the
//!   raw integer rather than the enum, and `ffi_enum!` can never convert a raw value back into
//!   a variant. `BitOr` is implemented for both operand orders (`A | B`, `A | raw`, `raw | A`),
//!   so a chain such as `A | B | C` reads as written; the read side is `raw & A` (`BitAnd`,
//!   mask on the left). If a combination must be stored, keep it as the raw value.
//! - `ffi_enum!` does not declare an FFI type, but that is a statement about the macro's shape,
//!   not a restriction on its constants: `SWS_*` is a bare constant on FFmpeg 6/7 and the
//!   `ffi::SwsFlags` alias on 8+, and one table covers both.
//! - Every generated enum is `#[non_exhaustive]`. FFmpeg gains and drops these values across
//!   releases (several tables already carry `#[cfg(feature = ...)]` variants), so a downstream
//!   `match` must keep a wildcard arm — which is exactly what the attribute now enforces.
//! - Every generated enum converts **variant to raw** with `Into` (`Into<repr>` for bit sets,
//!   `Into<ffi>` for IDs), and that is the conversion to reach for by default. `as_raw()` exists
//!   only on the bit-set macros, where the same value is also needed for bit-level masking; treat
//!   it as the explicit counterpart of the operators rather than a second generic conversion.

/// Wraps a mutually exclusive FFI enum: from a single `variant => constant` table it generates
/// the Rust enum plus the `From` impls in both directions.
///
/// This removes the hand-written "enum + two-way `match`" boilerplate that used to be
/// duplicated across modules (pixel / fmt / stream / hwaccel). One table is the single source
/// of truth and expands into:
///
/// - `pub enum`: `#[repr(..)]` comes from the `repr =` argument (`i32`/`u32`, declaring the
///   enum's signedness); every discriminant is normalised via `$const as $repr`, so
///   `variant as _` equals the corresponding FFmpeg value.
/// - `impl From<$ffi> for $enum`: `value as $repr` is normalised, then matched against the
///   constants. Values not listed take the `fallback`. Deprecated same-value aliases (such as
///   `AV_PIX_FMT_Y400A` ≡ `AV_PIX_FMT_YA8`) need not be listed: they share the main variant's
///   value, so they hit its arm automatically.
/// - `impl From<$enum> for $ffi`: matches the variant back to the constant, normalised via
///   `$const as $ffi`. Exhaustive over the Rust variants, so a forgotten variant fails to
///   compile.
///
/// # Type normalisation
///
/// The type bindgen assigns to an enum constant depends on the C enum's signedness and on the
/// generation environment (regenerated locally vs. pre-generated per platform): it may be
/// `c_int`, `c_uint`, a bare `u32`, and so on, and is not guaranteed to match `repr`. The macro
/// therefore normalises with an explicit `as` at all three seams (discriminant, `From<$ffi>`
/// match, `From<$enum>` return), which means `repr` only has to declare the signedness implied
/// by the enum's semantics (`i32` if any value is negative, otherwise `u32`) and does not have
/// to match the constants' type alias textually. A wrong signedness that makes two
/// discriminants collide is reported at compile time.
///
/// # Panicking fallback
///
/// ```ignore
/// ffi_enum_wrap_from!(
///     /// Pixel format definitions in bindings.
///     #[allow(non_camel_case_types)]
///     PixelFormat => ffi::AVPixelFormat,
///     repr = i32,
///     fallback = panic {
///         NONE => ffi::AV_PIX_FMT_NONE;
///         YUV420P => ffi::AV_PIX_FMT_YUV420P;
///         YUYV422 => ffi::AV_PIX_FMT_YUYV422;
///     }
/// );
/// ```
///
/// # Custom fallback (falls back to `Self::NONE`)
///
/// ```ignore
/// ffi_enum_wrap_from!(
///     /// Pixel format definitions in bindings.
///     #[allow(non_camel_case_types)]
///     PixelFormat => ffi::AVPixelFormat,
///     repr = i32,
///     fallback = Self::NONE {
///         NONE => ffi::AV_PIX_FMT_NONE;
///         YUV420P => ffi::AV_PIX_FMT_YUV420P;
///         YUYV422 => ffi::AV_PIX_FMT_YUYV422;
///     }
/// );
/// ```
///
/// # Notes
///
/// - Every table in this crate uses `fallback = panic`. An FFmpeg value the table does not model
///   is a fast failure, not a silent degradation: mapping it onto some variant would produce a
///   subtly wrong result that is much harder to diagnose than a panic. The `fallback =
///   <expression>` form remains supported for tables where degrading is genuinely the right
///   answer, but nothing uses it today.
/// - A variant's doc comment is forwarded to both the enum variant and the `match` arm. Doc on
///   a match arm is an `unused_doc_comments` warning, so the macro inserts
///   `#[allow(unused_doc_comments)]` on every arm.
/// - When the FFI type is a plain integer alias — `ffi::AVPixelFormat` and `ffi::AVSampleFormat`
///   are both `c_int` — the generated `From` impls **are** the integer conversions, so
///   `i32::from(PixelFormat::RGB24)` and `PixelFormat::from(raw_i32)` work without any extra
///   impl (and a separate `impl From<i32>` would be rejected as a duplicate impl of the same
///   type). The reverse one runs through the `fallback`, so an unlisted integer panics: reach
///   for `from_ffi_checked` when the value arrives from FFmpeg.
/// - `fallback = panic` and `fallback = <expression>` are two **entry** rules that differ only
///   in the fallback they hand to the internal `@expand` rule, which holds the single copy of
///   the expansion. Each hands over a **closure** — `|value| panic!(...)`, `|_| <expr>` — that
///   `@expand` calls with the raw value: macro hygiene would otherwise stop an identifier
///   written in one rule from resolving to the `value` binding of the `fn from` generated by
///   another. The expression rule keeps taking a `$fallback:path` rather than `$fb:expr`,
///   because `fallback = Self::NONE { ... }` would then be parsed as a struct literal instead
///   of a path followed by the variant list.
///
/// `@expand` is an implementation detail: it matches only the two entry rules above and is not
/// part of the macro's interface.
macro_rules! ffi_enum_wrap_from {
    // panic 版：fallback = panic { 变体列表 }
    (
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident,
        fallback = panic {
            $( $(#[$m:meta])* $variant:ident => $const:path; )*
        }
    ) => {
        ffi_enum_wrap_from!(
            @expand
            $(#[$em])*
            $enum => $ffi,
            repr = $repr,
            @fallback |value| panic!("Invalid {} value: {}", stringify!($enum), value),
            {
                $( $(#[$m])* $variant => $const; )*
            }
        );
    };

    // 自定义回退版：fallback = 表达式 { 变体列表 }
    (
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident,
        fallback = $fallback:path {
            $( $(#[$m:meta])* $variant:ident => $const:path; )*
        }
    ) => {
        ffi_enum_wrap_from!(
            @expand
            $(#[$em])*
            $enum => $ffi,
            repr = $repr,
            @fallback |_| $fallback,
            {
                $( $(#[$m])* $variant => $const; )*
            }
        );
    };

    // 展开体：两条入口规则把 fallback 归一成 `@fallback <闭包>` 后走这里，代码只有这一份。
    // 闭包是宏卫生的绕行：入口规则里的标识符引用不到本规则 `fn from` 的 `value` 绑定，
    // 因此由 `value` 所属的作用域来**调用**它，而不是让调用方直接写 `value`。
    (
        @expand
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident,
        @fallback $fb:expr,
        {
            $( $(#[$m:meta])* $variant:ident => $const:path; )*
        }
    ) => {
        $(#[$em])*
        #[repr($repr)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum $enum {
            $(
                $(#[$m])*
                $variant = $const as $repr,
            )*
        }

        #[allow(non_upper_case_globals)]
        impl From<$ffi> for $enum {
            fn from(value: $ffi) -> Self {
                $(
                    $(#[$m])*
                    const $variant: $repr = $const as $repr;
                )*
                match value as $repr {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $variant => $enum::$variant,
                    )*
                    _ => ($fb)(value as $repr),
                }
            }
        }

        /// The variant for `value`, or [`None`] when the table does not list it.
        ///
        /// [`From`] fails fast on an unlisted value because that usually signals a
        /// programming error. This is the counterpart to use when the value comes
        /// from *outside* the program — the format of an arbitrary media file, say —
        /// where "unsupported" has to be reported instead of aborting the process.
        #[allow(non_upper_case_globals)]
        impl $enum {
            pub fn from_ffi_checked(value: $ffi) -> Option<Self> {
                $(
                    $(#[$m])*
                    #[allow(unused_doc_comments)]
                    const $variant: $repr = $const as $repr;
                )*
                match value as $repr {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $variant => Some($enum::$variant),
                    )*
                    _ => None,
                }
            }
        }

        impl From<$enum> for $ffi {
            fn from(value: $enum) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $enum::$variant => $const as $ffi,
                    )*
                }
            }
        }
    };
}

/// Wraps constants that belong to a **named FFI type**: declares that type and generates the
/// **forward** conversion (variant → raw), but no reverse conversion.
///
/// Where it sits in the macro family:
/// - [`ffi_enum!`]: bit sets. The constants may be bare integers or a named FFI type in the
///   bindings; the macro normalises them and adds `Into<repr>` plus `BitOr`/`BitAnd`.
/// - `ffi_enum_wrap!` (this macro): the constants are a **named FFI type alias** in the bindings
///   (e.g. `ffi::SwsFlags`). Declares the type, converts **only** variant → raw.
/// - [`ffi_enum_wrap_from!`]: named FFI type **plus** the reverse direction as well (`From<ffi>`,
///   which needs mutually exclusive values, and `from_ffi_checked`).
///
/// # Which of the two directions is possible
///
/// **Variant → raw is a total function, so it is generated** (`impl From<$enum> for $ffi`, the
/// same shape [`ffi_enum_wrap_from!`] produces): a variant is one constant, and it maps to
/// exactly one FFI value no matter how combinable the table's members are.
///
/// **Raw → variant is not**, which is why it is absent: this shape is for **combinable bit
/// flags**, where a combination such as `FAST_BILINEAR | ACCURATE_RND` is not any single
/// variant, and an arbitrary FFI value need not be listed in the table at all. Combinations and
/// unknown values are handled bitwise through `as_raw()` instead — see [`ffi_enum!`], which is
/// the macro the crate actually uses for such tables.
///
/// The `$ffi` declaration is not a mere comment: the expansion contains `size_of::<$ffi>()` as a
/// compile-time existence check.
///
/// Discriminants are normalised via `$const as $repr`: the underlying integer of an FFI type
/// alias drifts across platforms (e.g. `SwsFlags` is `c_int` on MSVC and `c_uint` on Unix), so
/// `repr` only has to declare the signedness implied by the semantics; between 32-bit integers
/// an `as` cast is lossless at the bit-pattern level.
///
/// # Example
///
/// ```ignore
/// ffi_enum_wrap!(
///     /// Sws scale filter flags (SWS_*)
///     #[allow(non_camel_case_types)]
///     SwsFlags => ffi::SwsFlags,
///     repr = u32 {
///         /// fast bilinear filtering
///         FAST_BILINEAR => ffi::SWS_FAST_BILINEAR;
///         /// 2-tap cubic B-spline
///         BICUBIC => ffi::SWS_BICUBIC;
///     }
/// );
///
/// let raw: ffi::SwsFlags = SwsFlags::BICUBIC.into(); // variant -> raw
/// ```
///
/// # Note
///
/// Currently unused: `ScaleAlgorithm` is defined with [`ffi_enum!`] instead. That macro's `as`
/// normalisation accepts both the bare `SWS_*` constants of FFmpeg 6/7 and the `ffi::SwsFlags`
/// alias of FFmpeg 8+, so a single table covers every supported version — whereas the
/// `size_of::<$ffi>()` check here requires the alias to exist in all of them.
#[allow(unused_macros)]
macro_rules! ffi_enum_wrap {
    (
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident {
            $( $(#[$m:meta])* $var:ident => $const:path; )*
        }
    ) => {
        $(#[$em])*
        #[repr($repr)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum $enum {
            $(
                $(#[$m])*
                $var = $const as $repr,
            )*
        }

        // 编译期校验 ffi 类型声明的存在性
        const _: usize = ::core::mem::size_of::<$ffi>();

        impl $enum {
            /// 获取底层FFI原始整数
            pub fn as_raw(&self) -> $repr {
                *self as $repr
            }
        }

        // 变体 → 原始 FFI 值的正向转换：每个变体唯一对应一个常量，与成员是否可组合无关，
        // 所以它是全函数，可以生成（反向 raw → 变体做不到，见宏文档）。
        impl From<$enum> for $ffi {
            fn from(value: $enum) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $enum::$var => $const as $ffi,
                    )*
                }
            }
        }
    };
}

/// Defines a **bit set**: an `#[repr]` enum of FFmpeg flag constants, plus the conversions
/// needed to assemble a mask and hand it to FFI.
///
/// The constants may be bare integers (`u32`/`i32`) or a named FFI type alias in the bindings;
/// both are normalised by `$ffi_val as $repr_ty`. That normalisation is what lets a single
/// table cover FFmpeg versions that type the same flags differently — for instance `SWS_*` is
/// a bare constant on FFmpeg 6/7 and the `ffi::SwsFlags` alias on 8+.
///
/// Expands to:
/// - `pub enum` whose discriminants equal the FFmpeg values, plus `as_raw()`.
/// - `impl From<$enum> for $repr_ty` (that is, `Into<repr>`), so a flag can be passed straight
///   to an API taking `impl Into<repr>`.
/// - `BitOr<Self>` and `BitOr<repr>`, both with `Output = $repr_ty`, so `A | B`, `A | raw` and
///   chained combinations all produce the raw mask.
/// - `BitAnd<Self>` for `$repr_ty` (`raw & A`), so a mask can be queried with a named flag:
///   `flags & AVSeekFlag::BYTE != 0`.
///
/// # Why the operators yield the raw integer
///
/// A fieldless Rust enum cannot represent an unnamed discriminant, and most useful
/// combinations (`BACKWARD | ANY`, `GLOBAL_HEADER | NOTIMESTAMPS`) have no variant of their
/// own. The operators therefore return `$repr_ty`: they exist to assemble a mask **at the FFI
/// boundary**, not to model a first-class set type. All three `|` operand orders are accepted
/// (`A | B`, `A | raw`, `raw | A`), so a longer chain keeps reading naturally. When a
/// combination has to be stored or queried, keep it in the raw integer — `mask & Enum::X`
/// (or `Enum::X.as_raw()`) then tests an individual bit.
///
/// `raw` means `$repr_ty` — no other integer type is accepted, because a table only
/// implements the operators for its own `repr` and for itself. An FFI constant whose type
/// differs from the `repr` therefore needs an explicit cast, which in practice means the
/// `AVSEEK_FLAG_*` style case: rsmpeg types those constants `u32` while the consumer
/// (`av_seek_frame`) takes `int`, so the table is declared `repr = i32` and mixing becomes
/// `AVSeekFlag::FRAME | (ffi::AVSEEK_FLAG_BYTE as i32)`. Tables declared `repr = u32` need no
/// cast for the same constants.
///
/// # Parameters
///
/// `ffi_enum!(EnumName, ReprType { Variant => ffi::CONSTANT; ... });`
///
/// `ReprType` is `i32` or `u32`. Declare the signedness the FFI call expects, and note that a
/// flag set whose constants are unsigned but whose consumer wants `i32` (such as
/// `AVSEEK_FLAG_*`) must be declared `i32` so that `Into<i32>` is generated.
///
/// # Example
///
/// ```ignore
/// // u32 flag bits
/// ffi_enum!(AvPixFmtFlag, u32 {
///     BE => ffi::AV_PIX_FMT_FLAG_BE;
///     #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
///     XYZ => ffi::AV_PIX_FMT_FLAG_XYZ;
/// });
///
/// // i32 flag bits, combinable and convertible
/// ffi_enum!(AVSeekFlag, i32 {
///     BACKWARD => ffi::AVSEEK_FLAG_BACKWARD;
///     BYTE     => ffi::AVSEEK_FLAG_BYTE;
///     ANY      => ffi::AVSEEK_FLAG_ANY;
///     FRAME    => ffi::AVSEEK_FLAG_FRAME;
/// });
///
/// let mask = AVSeekFlag::BACKWARD | AVSeekFlag::ANY; // i32
/// let raw: i32 = AVSeekFlag::FRAME.into();
/// let decided_by_keyframe_only = mask & AVSeekFlag::ANY != 0; // read a bit back
/// ```
macro_rules! ffi_enum {
    (
        $(#[$enum_doc:meta])*
        $enum_ident:ident, $repr_ty:ty {
            $(
                $(#[$var_meta:meta])*
                $var:ident => $ffi_val:expr;
            )*
        }
    ) => {
        $(#[$enum_doc])*
        #[repr($repr_ty)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum $enum_ident {
            $(
                $(#[$var_meta])*
                $var = $ffi_val as $repr_ty,
            )*
        }

        impl $enum_ident {
            /// 获取底层FFI原始整数
            pub fn as_raw(&self) -> $repr_ty {
                *self as $repr_ty
            }
        }

        impl From<$enum_ident> for $repr_ty {
            fn from(value: $enum_ident) -> Self {
                value.as_raw()
            }
        }

        /// `Enum::A | Enum::B`: combines flag bits, yielding the raw integer.
        impl ::std::ops::BitOr for $enum_ident {
            type Output = $repr_ty;

            fn bitor(self, rhs: Self) -> Self::Output {
                self.as_raw() | rhs.as_raw()
            }
        }

        /// `Enum::A | raw`: mixes a named flag with a raw mask, yielding the raw integer.
        impl ::std::ops::BitOr<$repr_ty> for $enum_ident {
            type Output = $repr_ty;

            fn bitor(self, rhs: $repr_ty) -> Self::Output {
                self.as_raw() | rhs
            }
        }

        /// `raw | Enum::A`: continues a chain that has already produced the raw integer,
        /// so `A | B | C` compiles as written (the first `|` yields `$repr_ty`).
        impl ::std::ops::BitOr<$enum_ident> for $repr_ty {
            type Output = $repr_ty;

            fn bitor(self, rhs: $enum_ident) -> Self::Output {
                self | rhs.as_raw()
            }
        }

        /// `raw & Enum::A`: reads a flag bit back out of a mask that is already raw, so a
        /// query such as `flags & AVSeekFlag::BYTE != 0` needs no `as_raw()` on the flag.
        /// Note the operand order: the mask is the `repr` side.
        impl ::std::ops::BitAnd<$enum_ident> for $repr_ty {
            type Output = $repr_ty;

            fn bitand(self, rhs: $enum_ident) -> Self::Output {
                self & rhs.as_raw()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    //! One test per macro, pinning the capabilities in the module-level matrix and using the
    //! crate's real types so these double as regression tests for them.
    //!
    //! Negative cells cannot be asserted by a normal test (that needs `trybuild` or
    //! `static_assertions`), so each test uses **only** the capabilities marked `yes` — adding
    //! one of the `no` capabilities at that call site would fail to compile. Those `no` cells
    //! were confirmed once with a throwaway probe: `as_raw()`, `BitOr` and `BitAnd` on
    //! `PixelFormat`, `BitOr`/`BitAnd` on `ProbeWrapFlags`, the reverse `From<i32>` for
    //! `AVSeekFlag` and `ProbeWrapFlags`, `Into<i32>` for `ProbeWrapFlags`, and a second
    //! `From<i32>` for `PixelFormat` (E0119 — the generated `From<ffi::AVPixelFormat>` already *is*
    //! that impl, because the alias is `c_int`) are all rejected by the compiler.

    use rsmpeg::ffi;

    /// `ffi_enum!` is the **bit-set** macro. It is the only one that can combine flags, and the
    /// combination is the raw integer, because a fieldless enum cannot hold an unnamed
    /// combination such as `BACKWARD | ANY`.
    #[test]
    #[allow(clippy::unnecessary_cast)] // `SWS_*` is a bare integer before FFmpeg 8.
    fn ffi_enum_provides_bit_combination() {
        use crate::io::AVSeekFlag;

        // Discriminants are the FFmpeg values.
        assert_eq!(
            AVSeekFlag::BACKWARD.as_raw(),
            ffi::AVSEEK_FLAG_BACKWARD as i32
        );

        // `Into<repr>`: a single flag can be handed to an `impl Into<i32>` parameter — the
        // crate has exactly one (`Seekable::seek_to_frame`), and it is also what lets the
        // raw mask below be passed to it.
        let single: i32 = AVSeekFlag::FRAME.into();
        assert_eq!(single, ffi::AVSEEK_FLAG_FRAME as i32);

        // `BitOr<Self>`: two named flags combine into the raw mask.
        let two = AVSeekFlag::BACKWARD | AVSeekFlag::ANY;
        assert_eq!(
            two,
            (ffi::AVSEEK_FLAG_BACKWARD | ffi::AVSEEK_FLAG_ANY) as i32
        );

        // `BitOr<repr>`: a named flag can be mixed with an already-raw mask.
        let mixed = AVSeekFlag::FRAME | (ffi::AVSEEK_FLAG_BYTE as i32);
        assert_eq!(
            mixed,
            (ffi::AVSEEK_FLAG_FRAME | ffi::AVSEEK_FLAG_BYTE) as i32
        );

        // The result is the raw integer, so further combination stays in the raw domain —
        // and `repr | Variant` is provided precisely so that a chain can be written as
        // `A | B | C` and still produce the raw mask. The reverse direction (raw -> variant)
        // does not exist here at all; that is `ffi_enum_wrap_from!`'s job.
        let three = AVSeekFlag::BACKWARD | AVSeekFlag::ANY | AVSeekFlag::FRAME;
        assert_eq!(three, 1 | 4 | 8);

        // `BitAnd`: a mask that is already raw is queried with a named flag — the read side of
        // the operators, where `as_raw()` would otherwise be needed on the flag.
        assert_ne!(two & AVSeekFlag::BACKWARD, 0);
        assert_eq!(two & AVSeekFlag::BYTE, 0);

        // `repr` picks the conversion target, which is how one table covers every FFmpeg
        // version: `SWS_*` is a bare constant on 6/7 and a named alias on 8+, and both
        // normalise through `as`. Here `repr = u32`, so `Into<u32>` is generated.
        let sws: u32 = crate::scale::ScaleAlgorithm::BICUBIC.into();
        assert_eq!(sws, ffi::SWS_BICUBIC as u32);
    }

    /// `ffi_enum_wrap_from!` is for **mutually exclusive IDs**. It is the only macro that
    /// converts in the reverse direction (raw value -> variant), and it has no bit operators
    /// because an ID is not a bit set.
    ///
    /// It is also the only one of the three that exposes no `as_raw()`: an ID is turned into a
    /// raw value with `Into`, so an `as_raw()` method would just be a second spelling of the
    /// same thing.
    #[test]
    fn ffi_enum_wrap_from_provides_two_way_conversion() {
        use crate::pixel::PixelFormat;

        // Variant -> FFI value (`Into`, not `as_raw`).
        let raw: ffi::AVPixelFormat = PixelFormat::YUV420P.into();
        assert_eq!(raw, ffi::AV_PIX_FMT_YUV420P);

        // FFI value -> variant: the direction no other macro offers.
        assert_eq!(PixelFormat::from(ffi::AV_PIX_FMT_RGB24), PixelFormat::RGB24);
    }

    /// The ID tables' `From` impls **are** the `i32` conversions: `ffi::AVPixelFormat` and
    /// `ffi::AVSampleFormat` are `c_int` aliases, so `Into<i32>` / `From<i32>` come for free in
    /// both directions. Spelled with `i32` on purpose — that is the contract downstream code
    /// relies on when handing these values to other FFmpeg-facing crates.
    #[test]
    fn id_enums_convert_to_and_from_i32() {
        use crate::fmt::SampleFormat;
        use crate::pixel::PixelFormat;

        let raw: i32 = PixelFormat::RGB24.into();
        assert_eq!(raw, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(PixelFormat::from(raw), PixelFormat::RGB24);

        let raw: i32 = SampleFormat::FLTP.into();
        assert_eq!(raw, ffi::AV_SAMPLE_FMT_FLTP);
        assert_eq!(SampleFormat::from(raw), SampleFormat::FLTP);
    }

    /// `fallback = panic` on `PixelFormat`: an unlisted value is a hard error.
    #[test]
    #[should_panic(expected = "Invalid PixelFormat value")]
    fn pixel_format_panics_on_unknown_value() {
        use crate::pixel::PixelFormat;
        // `AV_PIX_FMT_NONE` (-1) *is* in the table, so use a value the table does not list.
        let _ = PixelFormat::from(i32::MIN);
    }

    /// `fallback = panic` on `SampleFormat`: same policy. Degrading an unknown format to `NONE`
    /// would risk encoding into the wrong format, which is far harder to diagnose than a panic.
    #[test]
    #[should_panic(expected = "Invalid SampleFormat value")]
    fn sample_format_panics_on_unknown_value() {
        use crate::fmt::SampleFormat;
        // `AV_SAMPLE_FMT_NONE` (-1) *is* in the table, so use a value it does not list.
        let _ = SampleFormat::from(i32::MIN);
    }

    /// `fallback = panic` on `HWDeviceType`: same policy. An unknown device type means the
    /// caller's assumption about the source is wrong, so it should not fall back to "no device".
    #[test]
    #[should_panic(expected = "Invalid HWDeviceType value")]
    fn hw_device_type_panics_on_unknown_value() {
        use crate::hwaccel::HWDeviceType;
        // `HWDeviceType::from` takes `ffi::AVHWDeviceType`, whose underlying integer drifts by
        // platform: `c_uint` on Unix, but `c_int` on Windows because MSVC gives an all
        // non-negative C enum a signed `int`. Deriving the probe value from an FFI constant keeps
        // the argument exactly that type — an explicitly typed `u32::MAX` compiles on Unix and
        // fails on Windows.
        let unknown = ffi::AV_HWDEVICE_TYPE_NONE + 9999;
        let _ = HWDeviceType::from(unknown);
    }

    /// The macro's second rule, `fallback = <expression>`, is still supported — this probe keeps
    /// it exercised so it cannot rot — but **no table in the crate uses it any more**: every ID
    /// table fails fast, because degrading an unrecognised FFmpeg value silently is how you end
    /// up producing a subtly wrong file instead of an error.
    #[test]
    fn expression_fallback_rule_is_still_supported() {
        ffi_enum_wrap_from!(
            #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
            ProbeId => ffi::AVPixelFormat,
            repr = i32,
            fallback = Self::NONE {
                NONE => ffi::AV_PIX_FMT_NONE;
                RGB24 => ffi::AV_PIX_FMT_RGB24;
            }
        );

        // Listed values round-trip in both directions as usual.
        assert_eq!(ProbeId::from(ffi::AV_PIX_FMT_RGB24), ProbeId::RGB24);
        assert_eq!(
            ffi::AVPixelFormat::from(ProbeId::RGB24),
            ffi::AV_PIX_FMT_RGB24
        );
        // ...while an unlisted value takes the expression fallback instead of panicking.
        assert_eq!(ProbeId::from(i32::MIN), ProbeId::NONE);
    }

    // `ffi_enum_wrap!` declares the named FFI type (checked at compile time via `size_of`) and
    // converts **variant → raw** only: no reverse conversion and no bit operators, so combining
    // two of its flags requires manual `as_raw()` arithmetic.
    //
    // It has no user in the crate today, and this probe shows why the need disappeared: the
    // only constants with a genuinely named FFI type (`SWS_*` on FFmpeg 8+) also have to be
    // normalised with `as` to stay compatible with 6/7, which `ffi_enum!` already does.
    // The probe is declared over `ffi::AVPixelFormat` only because that alias exists in every
    // supported version — `ffi::SwsFlags` would not compile on 6/7.
    ffi_enum_wrap!(
        /// Test-only probe: a flag table declared over a named FFI type.
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        ProbeWrapFlags => ffi::AVPixelFormat,
        repr = i32 {
            BE => ffi::AV_PIX_FMT_FLAG_BE;
            PLANAR => ffi::AV_PIX_FMT_FLAG_PLANAR;
        }
    );

    #[test]
    fn ffi_enum_wrap_provides_as_raw_and_a_forward_conversion() {
        // Same `as_raw()` as its siblings.
        assert_eq!(ProbeWrapFlags::BE.as_raw(), ffi::AV_PIX_FMT_FLAG_BE as i32);

        // The forward conversion is total, so it exists — `Into<$ffi>`, exactly like the one
        // `ffi_enum_wrap_from!` generates (here `$ffi` is `ffi::AVPixelFormat` = `c_int`).
        let raw: ffi::AVPixelFormat = ProbeWrapFlags::BE.into();
        assert_eq!(raw, ffi::AV_PIX_FMT_FLAG_BE as i32);

        // The reverse direction and the bit operators stay absent: combining therefore has to be
        // spelled out by hand — `ProbeWrapFlags::BE | ProbeWrapFlags::PLANAR` would not compile,
        // whereas the same expression on an `ffi_enum!` bit set does.
        let both = ProbeWrapFlags::BE.as_raw() | ProbeWrapFlags::PLANAR.as_raw();
        assert_eq!(
            both,
            (ffi::AV_PIX_FMT_FLAG_BE | ffi::AV_PIX_FMT_FLAG_PLANAR) as i32
        );
    }
}
