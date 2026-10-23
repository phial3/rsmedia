//! The classification every stream lookup is keyed by.
//!
//! [`MediaType`] is the crate's leaf type for "what kind of stream this is":
//! it is a plain wrapper over FFmpeg's `AVMEDIA_TYPE_*` and depends on nothing
//! else here, which is what lets the IO layer, the codec layer and the stream
//! layer all key off it without any of them depending on each other.
//!
//! It sits here rather than in [`crate::stream`] to keep the two layers
//! independent: `io` needs `MediaType` for
//! [`crate::io::Reader::find_best_stream`], while `stream` needs `io`'s traits
//! for `StreamInfo::from_reader` / `from_writer`. Owning the type in either of
//! them would make the other depend back on it.

use crate::strutils;

use rsmpeg::{avutil, ffi};

use std::fmt::{Display, Formatter};

ffi_enum_from!(
    /// Media type (FFmpeg `AVMEDIA_TYPE_*`): the classification of a stream.
    ///
    /// Generated from one `variant => constant` table with a two-way `From`. A value the table
    /// does not list panics rather than degrading to `UNKNOWN`, so a stream type this crate does
    /// not model is reported immediately instead of being silently treated as unknown — the
    /// listed `AVMEDIA_TYPE_UNKNOWN` still converts to `UNKNOWN` as usual.
    MediaType => ffi::AVMediaType,
    repr = i32,
    fallback = panic {
        UNKNOWN => ffi::AVMEDIA_TYPE_UNKNOWN;
        VIDEO => ffi::AVMEDIA_TYPE_VIDEO;
        AUDIO => ffi::AVMEDIA_TYPE_AUDIO;
        DATA => ffi::AVMEDIA_TYPE_DATA;
        SUBTITLE => ffi::AVMEDIA_TYPE_SUBTITLE;
        ATTACHMENT => ffi::AVMEDIA_TYPE_ATTACHMENT;
    }
);

impl MediaType {
    pub fn get_media_name(&self) -> String {
        // 同 `SampleFormat::get_sample_fmt_name`：FFmpeg 给的名字非 UTF-8 时降级显示。
        avutil::get_media_type_string(*self as _)
            .map_or("Unknown".to_string(), strutils::cstr_to_string_lossy)
    }
}

impl Display for MediaType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.get_media_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个变体都要能取到 FFmpeg 自己的名字。
    ///
    /// `UNKNOWN` 是唯一的例外，而且是**实测**的：`av_get_media_type_string` 对
    /// `AVMEDIA_TYPE_UNKNOWN` 返回 NULL，于是 `get_media_name` 走降级分支给出
    /// 大写开头的 `"Unknown"`。其余变体拿到的是 FFmpeg 的小写名字；某个变体
    /// 退化成 `"Unknown"` 就意味着 `*self as _` 转出的值 FFmpeg 不认（枚举表写错）。
    #[test]
    fn test_get_media_name_covers_every_variant() {
        for (variant, expected) in [
            (MediaType::VIDEO, "video"),
            (MediaType::AUDIO, "audio"),
            (MediaType::DATA, "data"),
            (MediaType::SUBTITLE, "subtitle"),
            (MediaType::ATTACHMENT, "attachment"),
        ] {
            assert_eq!(variant.get_media_name(), expected, "{variant:?}");
            // `Display` 走的是同一个出口。
            assert_eq!(variant.to_string(), expected, "{variant:?}");
        }
        assert_eq!(MediaType::UNKNOWN.get_media_name(), "Unknown");
    }
}
