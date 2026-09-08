/// FFI 枚举包装宏：从「枚举变体 <-> FFmpeg 常量」的单一配对表，生成枚举定义与两端的 `From` 实现。
///
/// 消除各模块中"手写枚举 + 手写双向 match"的多处定义（如 pixel / fmt / stream / hwaccel）。
/// 输入一张「变体 => 常量」的表作为单一事实来源，自动展开：
///
/// - `pub enum`：`#[repr(..)]` 由 `repr =` 参数指定（i32/u32，需与 FFmpeg 常量类型一致），
///   判别值即 FFmpeg 常量值，因此 `variant as _` 直接等于对应 FFmpeg 值。
/// - `impl From<$ffi> for $enum`：`match` 常量 → 变体，未列出的值走 `fallback`。
///   弃用的同值别名常量（如 `AV_PIX_FMT_Y400A` ≡ `AV_PIX_FMT_YA8`）无需列出，
///   与主变体同值，match 时自动命中主变体分支。
/// - `impl From<$enum> for $ffi`：`match` 变体 → 常量（对 Rust 变体穷尽，漏写变体将编译报错）。
///
/// panic 语法：
/// ```rust,ignore
/// ffi_enum!(
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
/// 自定义 fallback 示例（回退到 Self::NONE）
/// ```rust,ignore
/// ffi_enum!(
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
/// 注意：
/// - 变体行的 doc 注释会同时转发到枚举变体与 match 分支，match 分支上的 doc
///   是 `unused_doc_comments` 警告，宏在每个分支上自动插入
///   `#[allow(unused_doc_comments)]` 压制。
/// - `fallback = panic` 与 `fallback = <表达式>` 是两条独立规则：panic 文本由宏
///   内部生成才能引用 `fn from` 的 `value` 绑定（卫生性，调用方表达式无法引用）。
macro_rules! ffi_enum {
    // panic 版：fallback = panic { 变体列表 }
    (
        $(#[$em:meta])*
        $enum:ident => $ffi:ty,
        repr = $repr:ident,
        fallback = panic {
            $( $(#[$m:meta])* $variant:ident => $const:path; )*
        }
    ) => {
        $(#[$em])*
        #[repr($repr)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $enum {
            $(
                $(#[$m])*
                $variant = $const,
            )*
        }

        impl From<$ffi> for $enum {
            fn from(value: $ffi) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $const => $enum::$variant,
                    )*
                    _ => panic!("Invalid {} value: {}", stringify!($enum), value),
                }
            }
        }

        impl From<$enum> for $ffi {
            fn from(value: $enum) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $enum::$variant => $const,
                    )*
                }
            }
        }
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
        $(#[$em])*
        #[repr($repr)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $enum {
            $(
                $(#[$m])*
                $variant = $const,
            )*
        }

        impl From<$ffi> for $enum {
            fn from(value: $ffi) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $const => $enum::$variant,
                    )*
                    _ => $fallback,
                }
            }
        }

        impl From<$enum> for $ffi {
            fn from(value: $enum) -> Self {
                match value {
                    $(
                        $(#[$m])*
                        #[allow(unused_doc_comments)]
                        $enum::$variant => $const,
                    )*
                }
            }
        }
    };
}

/// FFI 旗标常量聚合宏：把一组 `*_FLAG_*` 常量定义为 Rust 枚举，判别值即常量值。
///
/// 与 [`ffi_enum!`] 的区别：`ffi_const!` 只做单向定义（常量 → 枚举判别值），
/// 不生成与 FFI 类型的双向 `From`，仅提供 `as_raw()` 取原始值。
///
/// # 参数
/// ffi_const!(EnumName, ReprType { ... });
/// ReprType: i32 / u32
///
/// # Example
/// ```rust,ignore
/// // u32 标志位
/// ffi_const!(AvPixFmtFlag, u32 {
///     BE => ffi::AV_PIX_FMT_FLAG_BE;
///     #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
///     XYZ => ffi::AV_PIX_FMT_FLAG_XYZ;
/// });
///
/// // i32 示例：像素格式ID
/// ffi_const!(AvPixelFormat, i32 {
///     YUV420P => ffi::AV_PIX_FMT_YUV420P;
///     RGB24 => ffi::AV_PIX_FMT_RGB24;
/// });
/// ```
macro_rules! ffi_const {
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
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $enum_ident {
            $(
                $(#[$var_meta])*
                $var = $ffi_val,
            )*
        }

        impl $enum_ident {
            /// 获取底层FFI原始整数
            pub fn as_raw(&self) -> $repr_ty {
                *self as $repr_ty
            }
        }
    };
}
