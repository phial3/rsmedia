//! 输入/输出设备枚举（libavdevice）。
//!
//! FFmpeg 的设备采集入口（摄像头、麦克风、屏幕）由 libavdevice 提供：
//! 直接用 [`input_video_devices`] 等函数枚举本机构建的设备 demuxer 名
//! （rsmedia 支持的全部 FFmpeg 版本均自动注册设备，无需手动初始化），
//! 最后以该名字作为 `format` 打开 demuxer（如 `avfoundation` /
//! `dshow` / `pulse`）。rsmedia 目前提供**枚举**；采集循环可用
//! [`ReaderBuilder::with_format`](crate::io::StreamReaderBuilder::with_format)
//! 指定设备 demuxer，采集参数（帧率/分辨率）走该 demuxer 的 options。
//!
//! # 用法
//!
//! ```no_run
//! use rsmedia::device;
//!
//! for name in device::input_video_devices() {
//!     println!("video capture device: {name}");
//! }
//! ```

use rsmpeg::ffi;
use std::ffi::CStr;

/// 本机可用的**视频采集**设备 demuxer 名（如 `avfoundation`、`dshow`）。
///
/// 设备是否存在取决于 FFmpeg 构建配置与操作系统，列表可能为空。
pub fn input_video_devices() -> Vec<String> {
    let mut fmt: *const ffi::AVInputFormat = std::ptr::null_mut();
    unsafe {
        collect(|| {
            fmt = ffi::av_input_video_device_next(fmt);
            fmt_name(fmt)
        })
    }
}

/// 本机可用的**音频采集**设备 demuxer 名（如 `pulse`、`avfoundation`）。
pub fn input_audio_devices() -> Vec<String> {
    let mut fmt: *const ffi::AVInputFormat = std::ptr::null_mut();
    unsafe {
        collect(|| {
            fmt = ffi::av_input_audio_device_next(fmt);
            fmt_name(fmt)
        })
    }
}

/// 本机可用的**视频输出**设备 demuxer 名（如 `sdl2`）。
pub fn output_video_devices() -> Vec<String> {
    let mut fmt: *const ffi::AVOutputFormat = std::ptr::null_mut();
    unsafe {
        collect(|| {
            fmt = ffi::av_output_video_device_next(fmt);
            fmt_name_out(fmt)
        })
    }
}

/// 本机可用的**音频输出**设备 demuxer 名。
pub fn output_audio_devices() -> Vec<String> {
    let mut fmt: *const ffi::AVOutputFormat = std::ptr::null_mut();
    unsafe {
        collect(|| {
            fmt = ffi::av_output_audio_device_next(fmt);
            fmt_name_out(fmt)
        })
    }
}

/// 从 `AVInputFormat` 取设备名（空指针安全）。
unsafe fn fmt_name(fmt: *const ffi::AVInputFormat) -> Option<String> {
    unsafe {
        if fmt.is_null() {
            return None;
        }
        let name = (*fmt).name;
        if name.is_null() {
            return None;
        }
        Some(CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

/// 从 `AVOutputFormat` 取设备名（空指针安全）。
unsafe fn fmt_name_out(fmt: *const ffi::AVOutputFormat) -> Option<String> {
    unsafe {
        if fmt.is_null() {
            return None;
        }
        let name = (*fmt).name;
        if name.is_null() {
            return None;
        }
        Some(CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

/// `av_*_device_next` 族函数的通用枚举循环：闭包推进到下一个设备并取出
/// 名字（`None` 表示枚举结束）。
unsafe fn collect<F>(mut next_name: F) -> Vec<String>
where
    F: FnMut() -> Option<String>,
{
    let mut out = Vec::new();
    while let Some(name) = next_name() {
        out.push(name);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 枚举不应 panic；名字非空。CI 容器通常无设备，列表可能为空。
    #[test]
    fn test_device_enumeration_does_not_panic() {
        for name in input_video_devices() {
            assert!(!name.is_empty());
        }
        for name in input_audio_devices() {
            assert!(!name.is_empty());
        }
        for name in output_video_devices() {
            assert!(!name.is_empty());
        }
        for name in output_audio_devices() {
            assert!(!name.is_empty());
        }
    }
}
