//! The crate's macros: three FFI wrappers plus one builder generator.
//!
//! Three macros cover everything the crate needs from FFmpeg's constants. Pick between them by
//! asking "can two values be meaningfully combined?", not by which FFmpeg version the constants
//! come from. Full capability matrix:
//!
//! | capability | `ffi_enum_from!` | `ffi_enum_typed!` | `ffi_enum!` |
//! |------------|------------------|-------------------|-------------|
//! | intended for | mutually exclusive IDs | bit sets | bit sets |
//! | declares the named FFI type | yes | yes, plus an explicit `size_of` check | no — constants may be bare integers *or* named types, normalised by `as` |
//! | `as_raw()` | no | yes | yes |
//! | conversion variant to raw value | yes (`From<Enum> for ffi`) | yes (`From<Enum> for ffi`) | yes (`From<Enum> for repr`, i.e. `Into<repr>`) |
//! | conversion raw value to variant | yes (`From<ffi> for Enum`) | no | no |
//! | conversion variant to flag set | no | no | yes (`From<Enum> for FlagSet<Enum>`) |
//! | combine flags | no | no | yes (`BitOr`, either operand order, yielding [`FlagSet<E>`](crate::FlagSet)) |
//! | test one bit in a set | no | no | yes ([`FlagSet::contains`](crate::FlagSet::contains), or `set & Enum::A`) |
//! | fallback for an unlisted value | panics; every table fails fast (the expression form is still supported but unused) | n/a | n/a |
//!
//! The suffixes are the whole naming scheme, and each one names the capability it adds on top of
//! `ffi_enum!`:
//!
//! - `ffi_enum!` — the base case: an enum over FFmpeg flag constants, its conversions and the bit
//!   operators. It declares no FFI type, because its constants may be bare integers on some
//!   FFmpeg versions and a named alias on others.
//! - `ffi_enum_typed!` — the same, **plus** the named FFI type alias is declared and checked at
//!   compile time (`size_of::<$ffi>()`). Forward conversion only.
//! - `ffi_enum_from!` — the same, **plus** the reverse conversion (`From<ffi> for Enum`, and
//!   `from_ffi_checked`). That is why it is the ID-table macro: a combination such as
//!   `BACKWARD | ANY` is not any single variant, so there would be nothing to convert back to.
//!
//! In practice the split is mostly by kind.
//!
//! The `ffi_enum!` call sites are bit sets — `AVCodecFlag` / `AVCodecFlag2` / `ThreadType`
//! (`codec.rs`), `AVFormatFlag` (`fmt.rs`), `AVPixFmtFlag` (`pixel.rs`), `AVSeekFlag`
//! (`io.rs`), `ErrRecognition` (`decode.rs`), `ScaleQuality` (`scale.rs`), `AVLogFlag`
//! (`init.rs`) — plus two tables that are really IDs: `ScaleAlgorithm` (`scale.rs`, a
//! mutually exclusive choice — "only one may be active at a time" per FFmpeg's header —
//! whose members are `SWS_*` bits) and `AVLogLevel` (`init.rs`, an ordered level, not a
//! mask). Neither of those two can reject a combination at the type level, so combining
//! their values is a caller error FFmpeg rejects rather than something the type prevents.
//!
//! The `ffi_enum_from!` call sites are IDs: `PixelFormat` (`pixel.rs`), `SampleFormat`
//! (`fmt.rs`), `MediaType` (`stream.rs`), `HWDeviceType` (`hwaccel.rs`), `SkipFrame`
//! (`decode.rs`), plus the swscale value sets in `scale.rs`: `SwsDither`, `SwsAlphaBlend`,
//! `SwsScaler`, `SwsIntent`, `SwsBackend`.
//!
//! The two tables above that are IDs yet cannot use `ffi_enum_from!` — `ScaleAlgorithm` and
//! `AVLogLevel` — are the reason the split is "declared shape" and not "semantics": that macro
//! needs a named FFI type alias to put after `=>`, and their constants are bare integers on the
//! older FFmpeg versions. They are declared with `ffi_enum!` and carry the caveat in their own
//! documentation instead.
//!
//! `ffi_enum_typed!` currently has no user — see the note on the macro itself for why the need
//! for it disappeared.
//!
//! The fourth macro, [`impl_codec_builder_setters!`], is not an FFI wrapper: it expands into the
//! `impl` block that
//! [`EncoderBuilder`](crate::encode::EncoderBuilder) and
//! [`DecoderBuilder`](crate::decode::DecoderBuilder) share, so a codec option common to both
//! sides has one definition and one copy of its documentation. It lives here because it is a
//! macro like the others, and because `#[macro_use] mod macros;` (in `lib.rs`) is what puts every
//! macro in this file in scope crate-wide — which is why no call site imports any of them.
//!
//! Points worth remembering, because they are easy to get wrong:
//!
//! - A fieldless enum cannot hold an unnamed discriminant, so the bit-set operators cannot yield
//!   the enum back: `LOW_DELAY | CLOSED_GOP` is not a variant of anything. They yield
//!   [`FlagSet<E>`](crate::FlagSet) — the crate's type for "a mask of `Enum`'s bits" — which is also what
//!   makes `ffi_enum!` unable to ever convert a raw value back into a variant. `BitOr` is
//!   implemented for both operand orders as well as for a set on either side (`A | B`,
//!   `A | raw`, `raw | A`, `set | A`, `A | set`), so a chain such as `A | B | C` reads as
//!   written and stays a `FlagSet` throughout; the read side is
//!   [`FlagSet::contains`](crate::FlagSet::contains), or
//!   `set & A` when the common bits are what matter. A mask that cannot be described as a
//!   combination of named flags goes through
//!   [`FlagSet::from_bits`](crate::FlagSet::from_bits).
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
//!   A `FlagSet` converts to the same raw `repr` (it is what makes a combination acceptable
//!   anywhere a single flag is), and [`FlagSet::bits`](crate::FlagSet::bits) is the explicit
//!   spelling of that read.

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
/// ffi_enum_from!(
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
/// ffi_enum_from!(
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
macro_rules! ffi_enum_from {
    // panic 版：fallback = panic { 变体列表 }
    (
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident,
        fallback = panic {
            $( $(#[$m:meta])* $variant:ident => $const:path; )*
        }
    ) => {
        ffi_enum_from!(
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
        ffi_enum_from!(
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
/// - `ffi_enum_typed!` (this macro): the constants are a **named FFI type alias** in the bindings
///   (e.g. `ffi::SwsFlags`). Declares the type, converts **only** variant → raw.
/// - [`ffi_enum_from!`]: named FFI type **plus** the reverse direction as well (`From<ffi>`,
///   which needs mutually exclusive values, and `from_ffi_checked`).
///
/// # Which of the two directions is possible
///
/// **Variant → raw is a total function, so it is generated** (`impl From<$enum> for $ffi`, the
/// same shape [`ffi_enum_from!`] produces): a variant is one constant, and it maps to
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
/// ffi_enum_typed!(
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
macro_rules! ffi_enum_typed {
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
/// - `impl From<$enum> for `[`FlagSet`](crate::FlagSet)`, plus the set's `Into<$repr_ty>`, its
///   `contains`, and the `|` / `&` operators whose result is the set — so one flag and any
///   combination of flags are accepted by the same `impl Into<FlagSet<$enum>>` parameter.
/// - `BitAnd<Self>` for `$repr_ty` (`raw & A`), so a mask that is already raw can be queried
///   with a named flag: `flags & AVSeekFlag::BYTE != 0`.
///
/// # Why the operators yield a `FlagSet`, not the enum
///
/// A fieldless Rust enum cannot represent an unnamed discriminant, and most useful
/// combinations (`BACKWARD | ANY`, `GLOBAL_HEADER | NOTIMESTAMPS`) have no variant of their
/// own. The operators therefore cannot return `$enum`; they return
/// [`FlagSet`](crate::FlagSet) instead. That keeps the combination **typed**: a
/// `FlagSet<AVCodecFlag>` cannot be handed to an API expecting `FlagSet<ScaleQuality>` even
/// though both are 32-bit masks, while `A | B | C` still reads as written because the set
/// implements `|` against itself and against the enum. Five `|` operand pairs are accepted —
/// `A | B`, `A | raw`, `raw | A`, `set | A`, `A | set` — so no operand order has to be looked
/// up. Reading goes through [`FlagSet::contains`](crate::FlagSet::contains) (one bit) or
/// `set & A` (the common bits).
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
/// let set = AVSeekFlag::BACKWARD | AVSeekFlag::ANY; // FlagSet<AVSeekFlag>
/// let raw: i32 = AVSeekFlag::FRAME.into();          // a single flag, straight to FFI
/// let also_raw: i32 = set.into();                   // ...and so is a combination
/// let decided_by_keyframe_only = set.contains(AVSeekFlag::ANY); // read a bit back
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

        /// A single flag is a one-bit set.
        impl From<$enum_ident> for $crate::flags::FlagSet<$enum_ident> {
            fn from(value: $enum_ident) -> Self {
                $crate::flags::FlagSet::from_bits(value.as_raw() as u32)
            }
        }

        /// The set hands its mask back to FFmpeg, as the type the FFI field wants.
        impl From<$crate::flags::FlagSet<$enum_ident>> for $repr_ty {
            fn from(value: $crate::flags::FlagSet<$enum_ident>) -> Self {
                value.bits() as $repr_ty
            }
        }

        impl $crate::flags::FlagSet<$enum_ident> {
            /// Whether `flag`'s bit is set in this set.
            ///
            /// ```
            /// # use rsmedia::{AVCodecFlag, FlagSet};
            /// let set = AVCodecFlag::LOW_DELAY | AVCodecFlag::CLOSED_GOP;
            /// assert!(set.contains(AVCodecFlag::LOW_DELAY));
            /// assert!(!set.contains(AVCodecFlag::BITEXACT));
            /// ```
            pub fn contains(self, flag: $enum_ident) -> bool {
                self.bits() & (flag.as_raw() as u32) != 0
            }
        }

        /// `Enum::A | Enum::B`: combines flag bits into a [`FlagSet`](crate::FlagSet).
        ///
        /// The result is the *set*, not a bare integer, so a chain keeps its type:
        /// `A | B | C` is a `FlagSet<Self>` throughout, and the per-table conversions
        /// (`From<Self>`, `Into<repr>`) apply to it exactly as they do to a single flag.
        impl ::std::ops::BitOr for $enum_ident {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitor(self, rhs: Self) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self.as_raw() as u32 | rhs.as_raw() as u32)
            }
        }

        /// `Enum::A | raw`: mixes a named flag with an untyped mask, still yielding the set.
        ///
        /// `raw` means `$repr_ty`, as for the rest of this macro; a constant of another width
        /// needs an explicit cast. Prefer [`FlagSet::from_bits`](crate::FlagSet::from_bits) when
        /// the whole mask is raw — this form exists so a *mostly named* expression reads as one.
        impl ::std::ops::BitOr<$repr_ty> for $enum_ident {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitor(self, rhs: $repr_ty) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self.as_raw() as u32 | rhs as u32)
            }
        }

        /// `raw | Enum::A`: the mirror of the arm above, so either operand order reads naturally.
        impl ::std::ops::BitOr<$enum_ident> for $repr_ty {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitor(self, rhs: $enum_ident) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self as u32 | rhs.as_raw() as u32)
            }
        }

        /// `set | Enum::A`: continues a chain that has already produced the set, so an
        /// expression such as `(A | B) | C` compiles with the same result as `A | B | C`.
        impl ::std::ops::BitOr<$enum_ident> for $crate::flags::FlagSet<$enum_ident> {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitor(self, rhs: $enum_ident) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self.bits() | rhs.as_raw() as u32)
            }
        }

        /// `Enum::A | set`: the mirror of the arm above.
        impl ::std::ops::BitOr<$crate::flags::FlagSet<$enum_ident>> for $enum_ident {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitor(self, rhs: $crate::flags::FlagSet<$enum_ident>) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self.as_raw() as u32 | rhs.bits())
            }
        }

        /// `set & Enum::A`: keeps only `flag`'s bit, so `(set & F).is_empty()` asks whether `F`
        /// is clear — the read-side counterpart of
        /// [`FlagSet::contains`](crate::FlagSet::contains).
        impl ::std::ops::BitAnd<$enum_ident> for $crate::flags::FlagSet<$enum_ident> {
            type Output = $crate::flags::FlagSet<$enum_ident>;

            fn bitand(self, rhs: $enum_ident) -> Self::Output {
                $crate::flags::FlagSet::from_bits(self.bits() & rhs.as_raw() as u32)
            }
        }

        /// `set |= Enum::A`: adds one named flag in place. The assign form of `set | A`, so
        /// accumulating a mask bit by bit needs no `into()` at each step.
        impl ::std::ops::BitOrAssign<$enum_ident> for $crate::flags::FlagSet<$enum_ident> {
            fn bitor_assign(&mut self, rhs: $enum_ident) {
                *self = $crate::flags::FlagSet::from_bits(self.bits() | rhs.as_raw() as u32);
            }
        }

        /// `set &= Enum::A`: keeps only one named flag in place. The assign form of `set & A`.
        impl ::std::ops::BitAndAssign<$enum_ident> for $crate::flags::FlagSet<$enum_ident> {
            fn bitand_assign(&mut self, rhs: $enum_ident) {
                *self = $crate::flags::FlagSet::from_bits(self.bits() & rhs.as_raw() as u32);
            }
        }

        /// `raw & Enum::A`: reads a flag bit back out of a mask that is *already* raw, keeping
        /// the raw integer. Note the operand order: the mask is the `repr` side.
        impl ::std::ops::BitAnd<$enum_ident> for $repr_ty {
            type Output = $repr_ty;

            fn bitand(self, rhs: $enum_ident) -> Self::Output {
                self & rhs.as_raw()
            }
        }
    };
}

/// Generates the setters that [`EncoderBuilder`](crate::encode::EncoderBuilder) and
/// [`DecoderBuilder`](crate::decode::DecoderBuilder) **share**.
///
/// The options common to both sides — codec name and private options, filters, hardware
/// acceleration and its frame pool, threads and flags, scaling policy — have the **same field
/// name** in both builders, so one definition expands verbatim into each `impl` block: the
/// semantics and the documentation of a shared option exist once, one edit reaches both sides,
/// and a drift such as "the encoder grew `with_flags2`, the decoder did not" cannot happen.
/// Options that belong to one side only (the encoder's bit rate / quality / profile, the
/// decoder's output format / discard granularity, …) stay in their own file.
///
/// To add a shared option: add the method and its documentation here, then add a field of the
/// same name to both builders and to their `Default`. If the setter's AVOption key can also
/// arrive through `with_options`, list that key in `with_options`'s documentation — a duplicate
/// key is resolved in the dictionary's favour, as documented there.
macro_rules! impl_codec_builder_setters {
    () => {
        /// Set the codec name — the encoder or decoder to use (`"libx264"`,
        /// `"aac"`, `"mov_text"`, `"h264_nvenc"`, `"h264_cuvid"`).
        ///
        /// Takes anything that converts into a `String`, so a string literal, a
        /// `&str` and a `String` are all passed directly: no `.to_string()`, no
        /// `Some(...)` wrapper.
        ///
        /// * decoder — follow the codec the input stream declares (chosen by the
        ///   container);
        /// * encoder — take the default for the media type (`libx264` / `aac` /
        ///   `subrip`).
        ///
        /// # Example
        ///
        /// ```ignore
        /// let builder = EncoderBuilder::new_video(640, 480).with_codec_name("libx264");
        /// ```
        pub fn with_codec_name(mut self, codec_name: impl Into<String>) -> Self {
            self.codec_name = Some(codec_name.into());
            self
        }

        /// Set the thread count.
        ///
        /// 与 FFmpeg 的 `AVCodecContext.thread_count` 同为 `i32`，直接写该字段。
        ///
        /// 传进来的值**原样**交给 FFmpeg，不做任何取值判断：`0` 是 FFmpeg 的
        /// "自行推导"语义（解码器会在 `avcodec_open2` 里把它改成实际线程数），
        /// 其它值（含负数）也照写，是否合法由 FFmpeg 决定。
        ///
        /// **不调用本方法**时 rsmedia 完全不碰该字段，线程数由 FFmpeg 自己定
        /// （解码器上下文默认是 `1`；想要自动推导请显式传 `0`）。同名 AVOption
        /// 若经 `with_options` 透传，以透传值为准（见 [`Self::with_options`]）。
        pub fn with_thread_count(mut self, thread_count: i32) -> Self {
            self.thread_count = Some(thread_count);
            self
        }

        /// Set `AVCodecContext.flags` (`AV_CODEC_FLAG_*`).
        ///
        /// Takes a [`FlagSet<AVCodecFlag>`](crate::FlagSet): a single flag
        /// (`AVCodecFlag::LOW_DELAY`) or any `|` combination of them
        /// (`AVCodecFlag::CLOSED_GOP | AVCodecFlag::LOW_DELAY`) — both are the same parameter,
        /// so no `Some(...)` wrapper and no raw integer is involved. A mask that arrives from
        /// elsewhere as a bare `u32` goes through
        /// [`FlagSet::from_bits`](crate::FlagSet::from_bits), which keeps the conversion
        /// visible.
        ///
        /// 解码器未设置时取 `AVCodecFlag::LOW_DELAY`（rsmedia 的解码默认值）。
        /// 编码器侧则是在上下文既有 flags 上按位合并：FFmpeg 的默认位（如
        /// `CLOSED_GOP`）与 `with_global_header` 的 `GLOBAL_HEADER` 都保留，
        /// 因此同时设置不会互相覆盖。
        ///
        /// # Example
        ///
        /// ```ignore
        /// use rsmedia::codec::AVCodecFlag;
        /// // Closed GOP + low latency, e.g. for a low-latency stream.
        /// let builder = EncoderBuilder::new_video(640, 480)
        ///     .with_flags(AVCodecFlag::CLOSED_GOP | AVCodecFlag::LOW_DELAY);
        /// ```
        pub fn with_flags(mut self, flags: impl Into<crate::flags::FlagSet<AVCodecFlag>>) -> Self {
            self.flags = Some(flags.into());
            self
        }

        /// Set `AVCodecContext.flags2` (`AV_CODEC_FLAG2_*`).
        ///
        /// Same shape as [`Self::with_flags`]: a single flag or a `|` combination, as a
        /// [`FlagSet<AVCodecFlag2>`](crate::FlagSet).
        pub fn with_flags2(
            mut self,
            flags2: impl Into<crate::flags::FlagSet<AVCodecFlag2>>,
        ) -> Self {
            self.flags2 = Some(flags2.into());
            self
        }

        /// Set `AVCodecContext.thread_type` (`FF_THREAD_*`).
        ///
        /// Chooses the multithreading granularity: [`ThreadType::FRAME`](crate::ThreadType::FRAME)
        /// (frame-level, best compression, more latency) or
        /// [`ThreadType::SLICE`](crate::ThreadType::SLICE) (slice-level, lower
        /// latency, needs codec support). Same shape as [`Self::with_flags`], so both
        /// granularities can be requested: `ThreadType::FRAME | ThreadType::SLICE`.
        /// Left unset, FFmpeg picks its default.
        pub fn with_thread_type(
            mut self,
            thread_type: impl Into<crate::flags::FlagSet<ThreadType>>,
        ) -> Self {
            self.thread_type = Some(thread_type.into());
            self
        }

        /// Codec (private) options used for this stream.
        ///
        /// 只用于 builder 未建模的**编解码器私有参数**（编码器如 `preset`、`tune`、
        /// `x264-params`；解码器如 `threads` 之外的各种解码开关）。
        ///
        /// 同一个 AVOption 若同时由 typed setter 与这里给出，**以这里为准**：typed
        /// setter 写的是 `AVCodecContext` 字段，而 `avcodec_open2` 在处理完字段之后
        /// 才应用本字典，同名的键因此覆盖 setter（写进同一个字典的 `crf`/`profile`/
        /// `level` 也是本字典后合并）。想让 setter 生效，就不要把同名键放进这里。
        ///
        /// 与 typed setter 同名的键：`threads`/`flags`/`flags2`/`thread_type`（对应
        /// [`Self::with_thread_count`]/[`Self::with_flags`]/[`Self::with_flags2`]/
        /// [`Self::with_thread_type`]）；编码器另有 `b`/`maxrate`/`bufsize`/`crf`/
        /// `profile`/`level`/`g`/`bf`，解码器另有 `skip_frame`/`err_detect`。
        pub fn with_options(mut self, options: impl Into<Option<Options>>) -> Self {
            self.codec_opts = options.into();
            self
        }

        /// Set the filters applied to frames on their way to/from this codec.
        ///
        /// 解码器：作用于**解码后**的帧（缩放/叠加/去噪…），滤镜输出即
        /// [`decode`](crate::Decoder::decode) 交付的帧。编码器：作用于**编码前**的帧。
        pub fn with_filters(mut self, filters: impl Into<Option<Vec<Filter>>>) -> Self {
            self.filters = filters.into();
            self
        }

        /// Enable hardware acceleration with the specified device type.
        ///
        /// * `device_config` - Device to use for hardware acceleration. Accepts a
        ///   [`HWDeviceConfig`] directly, or `None` to decode/encode on the CPU —
        ///   the same `impl Into<Option<_>>` shape [`Self::with_options`] uses, so
        ///   a device chosen at runtime needs no `Some(...)` wrapper.
        pub fn with_hardware_device(
            mut self,
            device_config: impl Into<Option<HWDeviceConfig>>,
        ) -> Self {
            self.hw_device_config = device_config.into();
            self
        }

        /// 设置硬件帧池的预分配表面数（`AVHWFramesContext::initial_pool_size`）。
        ///
        /// 只在启用了硬件加速时有效。默认 `DEFAULT_HW_POOL_SIZE`（[`crate::hwaccel`]
        /// 里的常量，20 张）对 1080p 是够用的启发值，但表面数是**预分配**的、直接
        /// 决定显存占用：4K 一张 NV12 面约 12MB，20 张就是约 240MB。高分辨率、
        /// 多路并发或显存紧张时按需调小；传 `0` 表示交给后端按需分配（FFmpeg 默认行为）。
        ///
        /// 解码器侧池子还要容纳 DPB（参考帧窗口），调到 1~2 张会限制参考帧复用、
        /// 影响压缩效率，建议至少留够 `refs + 2`。
        ///
        /// ```no_run
        /// # use rsmedia::encode::EncoderBuilder;
        /// # use rsmedia::hwaccel::HWDeviceConfig;
        /// # fn main() -> rsmedia::Result<()> {
        /// let config = HWDeviceConfig::auto_platform()?;
        /// let encoder = EncoderBuilder::new_video(3840, 2160)
        ///     .with_hardware_device(Some(config))
        ///     .with_hw_pool_size(4) // 4K 下只预占约 48MB 显存
        ///     .build()?;
        /// # drop(encoder);
        /// # Ok(())
        /// # }
        /// ```
        pub fn with_hw_pool_size(mut self, pool_size: i32) -> Self {
            self.hw_pool_size = Some(pool_size);
            self
        }

        /// Set the scaling algorithm used when converting frames to the target
        /// pixel format (encoder: input frames -> the encoder's format; decoder:
        /// decoded frames -> the output format, e.g. NV12 -> RGBA).
        ///
        /// The algorithm picks the scaling kernel and is **mutually exclusive** —
        /// FFmpeg's header states *"Scaler selection options. Only one may be active
        /// at a time."* Defaults to [`crate::scale::ScaleAlgorithm::BICUBIC`]; the
        /// quality/behaviour bits are set separately with [`Self::with_scale_quality`].
        pub fn with_scale_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
            self.scale_algorithm = algorithm;
            self
        }

        /// Set the scaling quality/behaviour bits used when converting frames to
        /// the target pixel format.
        ///
        /// Unlike the algorithm (exactly one bit), the quality flags are a set, given as a
        /// [`FlagSet<ScaleQuality>`](crate::FlagSet) like [`Self::with_flags`] — one bit
        /// (`ScaleQuality::BITEXACT`), several combined with `|`
        /// (`ScaleQuality::FULL_CHR_H_INT | ScaleQuality::ACCURATE_RND`), or
        /// [`FlagSet::EMPTY`](crate::FlagSet::EMPTY) for none. Defaults to
        /// [`ScaleQuality::default_mask`](crate::scale::ScaleQuality::default_mask).
        pub fn with_scale_quality(
            mut self,
            quality: impl Into<crate::flags::FlagSet<ScaleQuality>>,
        ) -> Self {
            self.scale_quality = quality.into();
            self
        }

        /// Enable (`true`) or disable (`false`) pooled allocation of the scaler's
        /// destination frames (see [`Scaler::with_buffer_pool`]).
        ///
        /// Off by default. With it on, frames this codec scales are allocated from an
        /// internal `AVBufferPool` instead of being freshly allocated per frame, so a
        /// steady stream of same-geometry conversions stops allocating after a couple
        /// of frames. Note that the pool only zeroes the bytes swscale never writes
        /// (alignment offset, stride slack, plane gaps, tail padding) — it does **not**
        /// zero the visible pixels, and `AVFrame::alloc_buffer` does not zero anything
        /// either (it goes through `av_frame_get_buffer` → `av_buffer_alloc` →
        /// `av_malloc`).
        pub fn with_buffer_pool(mut self, enabled: bool) -> Self {
            self.scale_pool = enabled;
            self
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
    /// combination is a [`FlagSet`](crate::FlagSet) — not the enum (a fieldless enum cannot hold
    /// an unnamed combination such as `BACKWARD | ANY`) and not a bare integer (which would let
    /// one flag type's mask slip into another's parameter).
    #[test]
    #[allow(clippy::unnecessary_cast)] // `SWS_*` is a bare integer before FFmpeg 8.
    fn ffi_enum_provides_bit_combination() {
        use crate::FlagSet;
        use crate::io::AVSeekFlag;

        // Discriminants are the FFmpeg values.
        assert_eq!(
            AVSeekFlag::BACKWARD.as_raw(),
            ffi::AVSEEK_FLAG_BACKWARD as i32
        );

        // `Into<repr>`: a single flag can be handed to an `impl Into<i32>` parameter — the
        // crate has exactly one (`Seekable::seek_to_frame`), and the same conversion on the
        // *set* is what lets a combination be passed to it too.
        let single: i32 = AVSeekFlag::FRAME.into();
        assert_eq!(single, ffi::AVSEEK_FLAG_FRAME as i32);

        // `BitOr<Self>`: two named flags combine into a set of that flag type.
        let two: FlagSet<AVSeekFlag> = AVSeekFlag::BACKWARD | AVSeekFlag::ANY;
        assert_eq!(
            two.bits(),
            (ffi::AVSEEK_FLAG_BACKWARD | ffi::AVSEEK_FLAG_ANY) as u32
        );
        // ...and the set converts to the raw `repr` just like a single flag does.
        let two_raw: i32 = two.into();
        assert_eq!(
            two_raw,
            (ffi::AVSEEK_FLAG_BACKWARD | ffi::AVSEEK_FLAG_ANY) as i32
        );

        // `BitOr<repr>`: a named flag can be mixed with an already-raw mask, and stays typed.
        let mixed = AVSeekFlag::FRAME | (ffi::AVSEEK_FLAG_BYTE as i32);
        assert_eq!(
            mixed.bits(),
            (ffi::AVSEEK_FLAG_FRAME | ffi::AVSEEK_FLAG_BYTE) as u32
        );

        // The first `|` yields the set, and `set | Enum` / `Enum | set` both continue it, so
        // `A | B | C` reads as written and stays a `FlagSet`. The reverse direction
        // (raw -> variant) does not exist here at all; that is `ffi_enum_from!`'s job.
        let three = AVSeekFlag::BACKWARD | AVSeekFlag::ANY | AVSeekFlag::FRAME;
        assert_eq!(three.bits(), 1 | 4 | 8);
        assert_eq!(three, two | AVSeekFlag::FRAME);
        assert_eq!(three, AVSeekFlag::FRAME | two);

        // The read side: `contains` for a single bit, `set & Enum` for the common bits.
        assert!(three.contains(AVSeekFlag::FRAME));
        assert!(two.contains(AVSeekFlag::BACKWARD));
        assert!(!two.contains(AVSeekFlag::BYTE));
        assert_eq!(two & AVSeekFlag::BACKWARD, AVSeekFlag::BACKWARD.into());
        assert!((two & AVSeekFlag::BYTE).is_empty());

        // A mask that is *already* raw is still queried by the raw `BitAnd`, which keeps the
        // integer: `raw & Enum::A`.
        let raw_two: i32 = two.into();
        assert_ne!(raw_two & AVSeekFlag::BACKWARD, 0);
        assert_eq!(raw_two & AVSeekFlag::BYTE, 0);

        // `repr` picks the conversion target, which is how one table covers every FFmpeg
        // version: `SWS_*` is a bare constant on 6/7 and a named alias on 8+, and both
        // normalise through `as`. Here `repr = u32`, so `Into<u32>` is generated.
        let sws: u32 = crate::scale::ScaleAlgorithm::BICUBIC.into();
        assert_eq!(sws, ffi::SWS_BICUBIC as u32);
    }

    /// `ffi_enum_from!` is for **mutually exclusive IDs**. It is the only macro that
    /// converts in the reverse direction (raw value -> variant), and it has no bit operators
    /// because an ID is not a bit set.
    ///
    /// It is also the only one of the three that exposes no `as_raw()`: an ID is turned into a
    /// raw value with `Into`, so an `as_raw()` method would just be a second spelling of the
    /// same thing.
    #[test]
    fn ffi_enum_from_provides_two_way_conversion() {
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
        ffi_enum_from!(
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

    // `ffi_enum_typed!` declares the named FFI type (checked at compile time via `size_of`) and
    // converts **variant → raw** only: no reverse conversion and no bit operators, so combining
    // two of its flags requires manual `as_raw()` arithmetic.
    //
    // It has no user in the crate today, and this probe shows why the need disappeared: the
    // only constants with a genuinely named FFI type (`SWS_*` on FFmpeg 8+) also have to be
    // normalised with `as` to stay compatible with 6/7, which `ffi_enum!` already does.
    // The probe is declared over `ffi::AVPixelFormat` only because that alias exists in every
    // supported version — `ffi::SwsFlags` would not compile on 6/7.
    ffi_enum_typed!(
        /// Test-only probe: a flag table declared over a named FFI type.
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        ProbeWrapFlags => ffi::AVPixelFormat,
        repr = i32 {
            BE => ffi::AV_PIX_FMT_FLAG_BE;
            PLANAR => ffi::AV_PIX_FMT_FLAG_PLANAR;
        }
    );

    #[test]
    fn ffi_enum_typed_provides_as_raw_and_a_forward_conversion() {
        // Same `as_raw()` as its siblings.
        assert_eq!(ProbeWrapFlags::BE.as_raw(), ffi::AV_PIX_FMT_FLAG_BE as i32);

        // The forward conversion is total, so it exists — `Into<$ffi>`, exactly like the one
        // `ffi_enum_from!` generates (here `$ffi` is `ffi::AVPixelFormat` = `c_int`).
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
