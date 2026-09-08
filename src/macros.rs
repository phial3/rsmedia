/// FFI 枚举包装宏：从「枚举变体 <-> FFmpeg 常量」的单一配对表，生成枚举定义与两端的 `From` 实现。
///
/// 消除各模块中"手写枚举 + 手写双向 match"的多处定义（如 pixel / fmt / stream / hwaccel）。
/// 输入一张「变体 => 常量」的表作为单一事实来源，自动展开：
///
/// - `pub enum`：`#[repr(..)]` 由 `repr =` 参数指定（i32/u32，声明该枚举的符号性），
///   判别值经 `$const as $repr` 归一化，因此 `variant as _` 直接等于对应 FFmpeg 值。
/// - `impl From<$ffi> for $enum`：`value as $repr` 归一化后 `match` 常量 → 变体，
///   未列出的值走 `fallback`。弃用的同值别名常量（如 `AV_PIX_FMT_Y400A` ≡ `AV_PIX_FMT_YA8`）
///   无需列出，与主变体同值，match 时自动命中主变体分支。
/// - `impl From<$enum> for $ffi`：`match` 变体 → 常量，经 `$const as $ffi` 归一化
///   （对 Rust 变体穷尽，漏写变体将编译报错）。
///
/// # 类型归一化
/// bindgen 对枚举常量的类型取决于 C 枚举符号性与生成环境（本地重新生成 / 平台预生成），
/// 可能是 `c_int`/`c_uint`/裸 `u32` 等，不保证与 `repr` 一致。宏在全部三处接缝
/// （判别值、`From<$ffi>` 匹配、`From<$enum>` 返回）都做 `as` 显式归一化，
/// 因此 `repr` 只需按枚举语义声明符号性（有负值写 `i32`，无负值写 `u32`），
/// 不必与常量的实际类型别名逐字匹配。若符号性声明错误导致判别值冲突，编译期报错。
///
/// panic 语法：
/// ```rust,ignore
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
/// 自定义 fallback 示例（回退到 Self::NONE）
/// ```rust,ignore
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
/// 注意：
/// - 变体行的 doc 注释会同时转发到枚举变体与 match 分支，match 分支上的 doc
///   是 `unused_doc_comments` 警告，宏在每个分支上自动插入
///   `#[allow(unused_doc_comments)]` 压制。
/// - `fallback = panic` 与 `fallback = <表达式>` 是两条独立规则：panic 文本由宏
///   内部生成才能引用 `fn from` 的 `value` 绑定（卫生性，调用方表达式无法引用）。
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
        $(#[$em])*
        #[repr($repr)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
                        $enum::$variant => $const as $ffi,
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
                        $enum::$variant => $const as $ffi,
                    )*
                }
            }
        }
    };
}

/// FFI 类型包装枚举宏（无 `From`）：把一组属于某 ffi 自定义类型别名的常量定义为 Rust 枚举，
/// 并声明常量所属的 ffi 类型。
///
/// 三级宏体系中的定位：
/// - [`ffi_enum!`]：旗标常量是绑定中的裸基本类型（`u32`/`i32`），不声明 ffi 类型，无 `From`。
/// - `ffi_enum_wrap!`（本宏）：常量在绑定中是 **ffi 自定义类型别名**（如 `ffi::SwsFlags`），
///   宏内声明该类型，但不生成 `From`。
/// - [`ffi_enum_wrap_from!`]：ffi 自定义类型 + 双向 `From`（互斥枚举值场景）。
///
/// 为什么声明了 ffi 类型却不实现 `From`：适用于**可组合位标志**（如 `SWS_*` 旗标），
/// 组合值（`FAST_BILINEAR | ACCURATE_RND`）不属于任何变体，
/// `From<$ffi> for enum` 无法全函数实现，互转通过 `as_raw()` 按位使用完成。
/// `$ffi` 声明并非纯注释：展开时以 `size_of::<$ffi>()` 做编译期存在性校验。
///
/// 判别值经 `$const as $repr` 归一化：ffi 类型别名的底层整型随平台漂移
/// （如 `SwsFlags` 在 MSVC 上为 `c_int`、Unix 上为 `c_uint`），
/// `repr` 只需按语义声明符号性，32 位整型间 `as` 转换位模式无损。
///
/// # Example
/// ```rust,ignore
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
/// ```
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
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
    };
}

/// FFI 旗标常量聚合宏：把一组 `*_FLAG_*` 常量定义为 Rust 枚举，判别值即常量值。
///
/// 与 [`ffi_enum_wrap_from!`] 的区别：`ffi_enum!` 只做单向定义（常量 → 枚举判别值），
/// 不生成与 FFI 类型的双向 `From`，仅提供 `as_raw()` 取原始值。
///
/// 判别值经 `$ffi_val as $repr` 归一化：bindgen 对旗标常量的类型可能是 `u32`
/// 或 `c_int`/`c_uint`（取决于 FFmpeg 头的 `#define` 形式与生成环境），
/// `repr` 不必与其逐字匹配，32 位整型间 `as` 转换位模式无损。
///
/// # 参数
/// ffi_enum!(EnumName, ReprType { ... });
/// ReprType: i32 / u32
///
/// # Example
/// ```rust,ignore
/// // u32 标志位
/// ffi_enum!(AvPixFmtFlag, u32 {
///     BE => ffi::AV_PIX_FMT_FLAG_BE;
///     #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
///     XYZ => ffi::AV_PIX_FMT_FLAG_XYZ;
/// });
///
/// // i32 示例：像素格式ID
/// ffi_enum!(AvPixelFormat, i32 {
///     YUV420P => ffi::AV_PIX_FMT_YUV420P;
///     RGB24 => ffi::AV_PIX_FMT_RGB24;
/// });
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
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
    };
}
