use crate::error::{Context, Result, RsmediaError};
use crate::{PixelFormat, SampleFormat, imgutils, time};

use rsmpeg::avutil::{AVFrame, AVSamples};
use rsmpeg::ffi;
use rsmpeg::swresample::SwrContext;
use rsmpeg::swscale::SwsContext;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Video Scaler SwsContext ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// FFmpeg `SWS_*` 定义参考（对应 swscale 头的开关位，见
// https://ffmpeg.org/doxygen/trunk/swscale_8h_source.html ）：
//   SWS_STRICT         1 << 11   Return an error on underspecified conversions.
//   SWS_PRINT_INFO     1 << 12   Emit verbose log of scaling parameters.
//   SWS_FULL_CHR_H_INT 1 << 13   Perform full chroma upsampling when upscaling to RGB.
//   SWS_FULL_CHR_H_INP 1 << 14   Perform full chroma interpolation when downscaling RGB.
//   SWS_ACCURATE_RND   1 << 18   Force bit-exact output rounding.
//   SWS_BITEXACT       1 << 19   Disable platform-specific optimizations for bit-exactness.
//   SWS_UNSTABLE       1 << 20   Prefer experimental code paths.
//   SWS_DIRECT_BGR     1 << 15   Deprecated: no effect.
//   SWS_ERROR_DIFFUSION 1 << 23   Deprecated: set `SwsContext.dither` instead.
//   SWS_FAST_BILINEAR  1 <<  0   fast bilinear filtering
//   SWS_BILINEAR       1 <<  1   bilinear filtering
//   SWS_BICUBIC        1 <<  2   2-tap cubic B-spline
//   SWS_X              1 <<  3   experimental
//   SWS_POINT          1 <<  4   nearest neighbor
//   SWS_AREA           1 <<  5   area averaging
//   SWS_BICUBLIN       1 <<  6   bicubic luma, bilinear chroma
//   SWS_GAUSS          1 <<  7   gaussian approximation
//   SWS_SINC           1 <<  8   unwindowed sinc
//   SWS_LANCZOS        1 <<  9   3-tap sinc/sinc
//   SWS_SPLINE         1 << 10   unwindowed natural cubic spline
//
// 版本差异（详见 https://github.com/FFmpeg/FFmpeg/blob/n8.1.2/doc/APIchanges）：
// - FFmpeg 6/7：`SWS_*` 是裸整型常量，`ffi::SwsFlags` 类型别名不存在；
//   libswscale 只有 legacy 路径（`sws_getContext()` 初始化 + `sws_scale_frame()`）。
// - FFmpeg 8+：`SWS_*` 常量类型化为 `ffi::SwsFlags`（MSVC 上底层为 `c_int`，Unix 为
//   `c_uint`），`sws_init_context()` 被废弃，官方推荐 `sws_alloc_context()` → 设置字段
//   → `sws_scale_frame()` 的全动态模式（FFmpeg 9 起拒绝 legacy/modern 混用）。
// `ffi_enum!` 判别值的 `as u32` 归一化使 6/7 的裸常量与 8+ 的 `ffi::SwsFlags` 别名
// 常量都能编译，故这里统一用 `ffi_enum!` 单表定义，无需按版本复制变体表。
ffi_enum!(
    /// Sws scale filter flags (SWS_*)
    #[allow(non_camel_case_types)]
    SwsFlags, u32 {
        /// fast bilinear filtering
        FAST_BILINEAR => ffi::SWS_FAST_BILINEAR;
        /// bilinear filtering
        BILINEAR => ffi::SWS_BILINEAR;
        /// 2-tap cubic B-spline
        BICUBIC => ffi::SWS_BICUBIC;
        /// experimental
        X => ffi::SWS_X;
        /// nearest neighbor
        POINT => ffi::SWS_POINT;
        /// area averaging
        AREA => ffi::SWS_AREA;
        /// bicubic luma, bilinear chroma
        BICUBLIN => ffi::SWS_BICUBLIN;
        /// gaussian approximation
        GAUSS => ffi::SWS_GAUSS;
        /// unwindowed sinc
        SINC => ffi::SWS_SINC;
        /// 3‑tap sinc/sinc
        LANCZOS => ffi::SWS_LANCZOS;
        /// unwindowed natural cubic spline
        SPLINE => ffi::SWS_SPLINE;
    }
);

impl SwsFlags {
    /// 返回该算法对应的完整 swscale flags（算法位 + 质量 flag）。
    /// （`SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND | SWS_BITEXACT`）恒被附加。
    #[allow(clippy::unnecessary_cast)]
    pub fn complete(self) -> u32 {
        let mut flag = self.as_raw() as u32;
        flag |= ffi::SWS_FULL_CHR_H_INT as u32;
        flag |= ffi::SWS_ACCURATE_RND as u32;
        flag |= ffi::SWS_BITEXACT as u32;
        flag
    }
}

#[allow(clippy::derivable_impls)]
impl Default for SwsFlags {
    fn default() -> Self {
        Self::BICUBIC
    }
}

/// 创建软件缩放上下文（按 FFmpeg 版本走新旧 API 路径）：
/// - FFmpeg 6/7：legacy 路径，`sws_getContext()` 一次性传入源/目标参数完成初始化；
/// - FFmpeg 8+：modern 全动态路径，`sws_alloc_context()` 分配后仅设置 flags 字段，
///   尺寸/格式等参数由 `sws_scale_frame()` 从帧属性推导（`sws_init_context()` 自
///   FFmpeg 8.0 起废弃，FFmpeg 9 起拒绝 legacy/modern API 混用）。
#[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
fn setup_scaler(
    src_width: i32,
    src_height: i32,
    src_pix_fmt: ffi::AVPixelFormat,
    dst_width: i32,
    dst_height: i32,
    dst_pix_fmt: ffi::AVPixelFormat,
    flags: u32,
) -> Result<SwsContext> {
    SwsContext::get_context(
        src_width,
        src_height,
        src_pix_fmt,
        dst_width,
        dst_height,
        dst_pix_fmt,
        flags,
        None,
        None,
        None,
    )
    .context("Failed to create a swscale context.")
}

/// FFmpeg 8+ 的 modern 全动态路径：参数签名与 6/7 分支保持一致以便调用方无感切换，
/// 除 flags 外的参数（尺寸/格式）由 [`SwsContext::scale_full_frame`] 从帧属性推导，
/// 此处忽略。
#[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
#[allow(unused_variables)]
fn setup_scaler(
    src_width: i32,
    src_height: i32,
    src_pix_fmt: ffi::AVPixelFormat,
    dst_width: i32,
    dst_height: i32,
    dst_pix_fmt: ffi::AVPixelFormat,
    flags: u32,
) -> Result<SwsContext> {
    let mut sws_ctx = SwsContext::alloc().context("Failed to allocate a swscale context.")?;
    sws_ctx.set_flags(flags);
    Ok(sws_ctx)
}

/// # Safety
///
/// ffi::sws_scale_frame
pub fn scale_frame(
    src_frame: &AVFrame,
    dst_width: i32,
    dst_height: i32,
    dst_pix_fmt: PixelFormat,
) -> Result<AVFrame> {
    scale_with_flags(
        src_frame,
        dst_width,
        dst_height,
        dst_pix_fmt,
        SwsFlags::default(),
    )
}

/// # Safety
///
/// ffi::sws_scale_frame
pub fn scale_with_flags(
    src_frame: &AVFrame,
    dst_width: i32,
    dst_height: i32,
    dst_pix_fmt: PixelFormat,
    scaler_algo: SwsFlags,
) -> Result<AVFrame> {
    if !src_frame.hw_frames_ctx.is_null() {
        return Err(RsmediaError::unsupported(
            "Hardware frames are not supported in this software scalar",
        ));
    }

    let mut dst_frame = AVFrame::new();
    dst_frame.set_width(dst_width);
    dst_frame.set_height(dst_height);
    dst_frame.set_format(dst_pix_fmt.into());
    dst_frame
        .alloc_buffer()
        .context("Failed to allocate destination frame buffer")?;
    imgutils::copy_frame_metadata(src_frame, &mut dst_frame, false)?;
    let mut sws_ctx = setup_scaler(
        src_frame.width,
        src_frame.height,
        src_frame.format,
        dst_width,
        dst_height,
        dst_pix_fmt.into(),
        scaler_algo.complete(),
    )
    .context("Failed to create swscale context.")?;

    // FFmpeg 6/7：legacy 初始化的上下文直调 `sws_scale_frame`（对已初始化上下文属
    // 向后兼容用法）；FFmpeg 8+：全动态上下文必须走 modern 封装
    // [`SwsContext::scale_full_frame`]，FFmpeg 9 起对未初始化的上下文直调底层
    // `sws_scale_frame` 会因新旧 API 混用而拒绝（AVERROR EINVAL）。
    #[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
    {
        let ret = unsafe {
            ffi::sws_scale_frame(
                sws_ctx.as_mut_ptr(),
                dst_frame.as_mut_ptr(),
                src_frame.as_ptr(),
            )
        };
        if ret < 0 {
            return Err(RsmediaError::custom(format!(
                "Failed to call sws_scale_frame, ret: {ret}"
            )));
        }
    }

    #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
    sws_ctx
        .scale_full_frame(&mut dst_frame, src_frame)
        .context("Failed to scale frame.")?;

    log::debug!(
        "Sws scale from src:[{}x{}, {:?}] to dst:[{}x{}, {:?}]",
        src_frame.width,
        src_frame.height,
        PixelFormat::from(src_frame.format),
        dst_width,
        dst_height,
        dst_pix_fmt
    );

    Ok(dst_frame)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Audio Resampler SwrContext /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

fn setup_resampler(
    in_ch_layout: ffi::AVChannelLayout,
    in_sample_fmt: ffi::AVSampleFormat,
    in_sample_rate: i32,
    out_ch_layout: ffi::AVChannelLayout,
    out_sample_fmt: ffi::AVSampleFormat,
    out_sample_rate: i32,
) -> Result<SwrContext> {
    let mut swr_ctx = SwrContext::new(
        &out_ch_layout,
        out_sample_fmt,
        out_sample_rate,
        &in_ch_layout,
        in_sample_fmt,
        in_sample_rate,
    )
    .context("Could not allocate resample context")?;

    swr_ctx.init().context("Could not open resample context")?;

    Ok(swr_ctx)
}

/// 校验重采样输入帧：不支持硬件帧，且采样率/样本数必须有效。
fn check_resampler_input(src_frame: &AVFrame) -> Result<()> {
    if !src_frame.hw_frames_ctx.is_null() {
        return Err(RsmediaError::unsupported(
            "Hardware frames are not supported in this software re-sampler",
        ));
    }

    if src_frame.sample_rate < 1 || src_frame.nb_samples < 1 {
        return Err(RsmediaError::custom("Invalid input frame."));
    }
    Ok(())
}

/// Audio resampling frame
pub fn convert(
    src_frame: &AVFrame,
    out_ch_layout: ffi::AVChannelLayout,
    out_sample_fmt: ffi::AVSampleFormat,
    out_sample_rate: i32,
) -> Result<AVSamples> {
    check_resampler_input(src_frame)?;

    let mut resampler = Resampler::new(
        src_frame.ch_layout,
        src_frame.format,
        src_frame.sample_rate,
        out_ch_layout,
        out_sample_fmt,
        out_sample_rate,
    )
    .context("Failed to create resample context.")?;

    let samples = resampler.convert(src_frame, out_ch_layout, out_sample_fmt)?;

    log::debug!(
        "Swr convert from src:[{}, {:?}, {}] to dst:[{}, {:?}, {}]",
        src_frame.ch_layout.nb_channels,
        SampleFormat::from(src_frame.format),
        src_frame.sample_rate,
        out_ch_layout.nb_channels,
        SampleFormat::from(out_sample_fmt),
        out_sample_rate
    );

    Ok(samples)
}

/// Audio resampling frame
///
/// # Arguments
///
///
pub fn convert_frame(
    src_frame: &AVFrame,
    out_ch_layout: ffi::AVChannelLayout,
    out_sample_fmt: ffi::AVSampleFormat,
    out_sample_rate: i32,
) -> Result<AVFrame> {
    check_resampler_input(src_frame)?;

    let mut resampler = Resampler::new(
        src_frame.ch_layout,
        src_frame.format,
        src_frame.sample_rate,
        out_ch_layout,
        out_sample_fmt,
        out_sample_rate,
    )?;

    let mut dst_frame = AVFrame::new();
    // copy props
    imgutils::copy_frame_metadata(src_frame, &mut dst_frame, false)?;
    dst_frame.set_format(out_sample_fmt);
    dst_frame.set_ch_layout(out_ch_layout);
    // 输出帧的缓冲必须按重采样后的输出样本数分配，而不是简单地使用输入样本数。
    // 当输入/输出采样率不同时，swr_convert_frame() 会写入比输入样本数更多的输出样本，
    // 但 FFmpeg 不会自动扩大已分配的输出缓冲（只把放不下的部分存入内部 FIFO），
    // 若这里 mb_samples 设得过小，将导致 swr_convert 越界写。
    // 用 swr_get_out_samples() 得到所需输出样本数的上界来分配缓冲。
    let out_samples = resampler.get_out_samples(src_frame.nb_samples).max(1);
    dst_frame.set_nb_samples(out_samples);
    dst_frame.set_sample_rate(out_sample_rate);
    dst_frame.set_time_base(time::new_rational(1, out_sample_rate));
    dst_frame
        .alloc_buffer()
        .context("Failed to allocate destination frame buffer")?;

    // 转换输入 AVFrame 中的样本并将其写入输出 AVFrame。
    // 输入和输出 AVFrame 必须设置通道布局、采样率和格式。
    // 如果输出 AVFrame 没有分配数据指针，则将在调用 av_frame_get_buffer() 分配帧时设置 nb_samples 字段。
    // 输出的 AVFrame 可以是 NULL，或者分配的样本少于所需的数量。在这种情况下，未写入输出的剩余样本将被添加到内部 FIFO 缓冲区，在下次调用此函数或 swr_convert() 时返回。
    // 如果转换采样率，内部重采样延迟缓冲区中可能会有剩余数据。要以输出方式获取这些数据，请调用此函数或 swr_convert()，并输入 NULL。
    resampler
        .convert_frame(src_frame, &mut dst_frame)
        .context("Failed to convert frame.")?;

    log::debug!(
        "Swr convert_frame from src:[{}, {:?}, {}] to dst:[{}, {:?}, {}]",
        src_frame.ch_layout.nb_channels,
        SampleFormat::from(src_frame.format),
        src_frame.sample_rate,
        out_ch_layout.nb_channels,
        SampleFormat::from(out_sample_fmt),
        out_sample_rate
    );

    Ok(dst_frame)
}

/// Persistent streaming resampler.
///
/// Unlike [`convert_frame`] (which creates a temporary context on each call),
/// `Resampler` holds a reusable `SwrContext` for continuous streaming input:
/// when resampling (different input/output sample rates), the internal filter
/// delay buffers samples between calls and outputs them with subsequent data;
/// at the end, [`Resampler::flush`] must be called to drain the tail, otherwise
/// the last few milliseconds of samples will be lost.
pub struct Resampler {
    swr: SwrContext,
}

impl Resampler {
    pub fn new(
        in_ch_layout: ffi::AVChannelLayout,
        in_sample_fmt: ffi::AVSampleFormat,
        in_sample_rate: i32,
        out_ch_layout: ffi::AVChannelLayout,
        out_sample_fmt: ffi::AVSampleFormat,
        out_sample_rate: i32,
    ) -> Result<Self> {
        Ok(Self {
            swr: setup_resampler(
                in_ch_layout,
                in_sample_fmt,
                in_sample_rate,
                out_ch_layout,
                out_sample_fmt,
                out_sample_rate,
            )?,
        })
    }

    /// Upper bound estimate of the number of output samples for the given number of input samples.
    pub fn get_out_samples(&self, in_samples: i32) -> i32 {
        self.swr.get_out_samples(in_samples).max(1)
    }

    /// Convert an input frame into an output frame with allocated buffer
    /// (persistent context: samples that cannot be written due to insufficient
    /// output capacity remain internally buffered and are returned with subsequent calls).
    ///
    /// The caller must set format/layout/sample_rate/nb_samples on `dst` and
    /// call `alloc_buffer`; after conversion, `dst.nb_samples` is the actual
    /// number of output samples.
    pub fn convert_frame(&mut self, src: &AVFrame, dst: &mut AVFrame) -> Result<()> {
        self.swr
            .convert_frame(Some(src), dst)
            .context("Failed to convert frame with streaming resampler")
    }

    /// Raw sample conversion (direct `swr_convert` wrapper), for caller-managed
    /// sample buffers such as [`AVSamples`].
    ///
    /// Returns the number of samples output per channel; a negative value has
    /// been mapped to an error. Returns 0 when the input sample count is > 0,
    /// indicating that the conversion result is temporarily buffered inside swr
    /// (possible during resampling) and will be output along with subsequent inputs.
    ///
    /// # Safety
    ///
    /// The buffers pointed to by `out`/`in_` and their sample counts must satisfy
    /// the validity requirements of `swr_convert`.
    pub fn convert(
        &mut self,
        src_frame: &AVFrame,
        out_ch_layout: ffi::AVChannelLayout,
        out_sample_fmt: ffi::AVSampleFormat,
    ) -> Result<AVSamples> {
        // 容量按输出样本数的上界分配，避免上采样（in < out）时尾部样本被丢弃。
        let capacity = self.get_out_samples(src_frame.nb_samples);
        let mut out_samples =
            AVSamples::new(out_ch_layout.nb_channels, capacity, out_sample_fmt, 0)
                .context("Create samples buffer failed.")?;

        let ret = unsafe {
            self.swr
                .convert(
                    out_samples.audio_data.as_mut_ptr(),
                    out_samples.nb_samples,
                    src_frame.extended_data as *const _,
                    src_frame.nb_samples,
                )
                .context("Could not convert input samples")?
        };

        if ret < 0 {
            return Err(RsmediaError::custom(format!(
                "Failed to convert input samples, ret: {ret}"
            )));
        }

        Ok(out_samples)
    }

    /// Drain the remaining samples from the resampler (EOF flush).
    ///
    /// `dst` must already have an allocated buffer; after conversion,
    /// `dst.nb_samples` is the actual number of samples obtained (possibly 0).
    /// Repeat the call until 0 is returned to fully drain.
    pub fn flush(&mut self, dst: &mut AVFrame) -> Result<()> {
        self.swr
            .convert_frame(None, dst)
            .context("Failed to flush streaming resampler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Context, Result};
    use crate::{SampleFormat, time};
    use rsmpeg::avutil::AVChannelLayout;
    use rsmpeg::ffi;

    /// 音频格式特征描述
    #[warn(dead_code)]
    struct AudioFormatDesc {
        format: ffi::AVSampleFormat,
        name: &'static str,
        bytes_per_sample: usize,
    }

    impl std::fmt::Debug for AudioFormatDesc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    /// 定义所有支持的音频格式
    const AUDIO_FORMATS: &[AudioFormatDesc] = &[
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_U8,
            name: "U8",
            bytes_per_sample: 1,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_U8P,
            name: "U8P",
            bytes_per_sample: 1,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S16,
            name: "S16",
            bytes_per_sample: 2,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S16P,
            name: "S16P",
            bytes_per_sample: 2,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S32,
            name: "S32",
            bytes_per_sample: 4,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S32P,
            name: "S32P",
            bytes_per_sample: 4,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_FLT,
            name: "FLT",
            bytes_per_sample: 4,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_FLTP,
            name: "FLTP",
            bytes_per_sample: 4,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_DBL,
            name: "DBL",
            bytes_per_sample: 8,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_DBLP,
            name: "DBLP",
            bytes_per_sample: 8,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S64,
            name: "S64",
            bytes_per_sample: 8,
        },
        AudioFormatDesc {
            format: ffi::AV_SAMPLE_FMT_S64P,
            name: "S64P",
            bytes_per_sample: 8,
        },
    ];

    /// 安全地填充测试数据
    unsafe fn fill_test_data(frame: &mut AVFrame, format_desc: &AudioFormatDesc) -> Result<()> {
        let nb_samples = frame.nb_samples as usize;
        let nb_channels = frame.ch_layout.nb_channels as usize;
        let is_planar = SampleFormat::from(format_desc.format).is_planar();

        macro_rules! fill_samples {
            ($type:ty, $max_val:expr) => {
                if is_planar {
                    for ch in 0..nb_channels {
                        let data = unsafe {
                            std::slice::from_raw_parts_mut(frame.data[ch] as *mut $type, nb_samples)
                        };
                        for (i, sample) in data.iter_mut().enumerate() {
                            *sample = ((i * nb_channels + ch) as f64
                                / (nb_samples * nb_channels) as f64
                                * $max_val as f64) as $type;
                        }
                    }
                } else {
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            frame.data[0] as *mut $type,
                            nb_samples * nb_channels,
                        )
                    };
                    for i in 0..(nb_samples * nb_channels) {
                        data[i] = (i as f64 / (nb_samples * nb_channels) as f64 * $max_val as f64)
                            as $type;
                    }
                }
            };
        }

        match format_desc.format {
            ffi::AV_SAMPLE_FMT_U8 | ffi::AV_SAMPLE_FMT_U8P => {
                fill_samples!(u8, u8::MAX)
            }
            ffi::AV_SAMPLE_FMT_S16 | ffi::AV_SAMPLE_FMT_S16P => {
                fill_samples!(i16, i16::MAX)
            }
            ffi::AV_SAMPLE_FMT_S32 | ffi::AV_SAMPLE_FMT_S32P => {
                fill_samples!(i32, i32::MAX)
            }
            ffi::AV_SAMPLE_FMT_FLT | ffi::AV_SAMPLE_FMT_FLTP => {
                fill_samples!(f32, 1.0)
            }
            ffi::AV_SAMPLE_FMT_DBL | ffi::AV_SAMPLE_FMT_DBLP => {
                fill_samples!(f64, 1.0)
            }
            ffi::AV_SAMPLE_FMT_S64 | ffi::AV_SAMPLE_FMT_S64P => {
                fill_samples!(i64, i64::MAX)
            }
            _ => return Err(RsmediaError::custom("Unsupported sample format")),
        }
        Ok(())
    }

    fn create_test_frame(
        format_desc: &AudioFormatDesc,
        sample_rate: i32,
        nb_channels: i32,
        nb_samples: i32,
    ) -> Result<AVFrame> {
        let mut frame = AVFrame::new();

        frame.set_format(format_desc.format);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(nb_channels).into_inner());
        frame.set_nb_samples(nb_samples);
        frame.set_sample_rate(sample_rate);
        frame.set_time_base(time::new_rational(1, sample_rate));

        frame
            .alloc_buffer()
            .context("Failed to allocate frame buffer")?;

        unsafe {
            fill_test_data(&mut frame, format_desc).context("Failed to fill test data")?;
        }

        Ok(frame)
    }

    #[test]
    fn test_format_conversion() -> Result<()> {
        let sample_rate = 44100;
        let nb_samples = 1024;
        let nb_channels = 2;

        // 测试所有格式组合
        for in_fmt in AUDIO_FORMATS {
            println!("\nTesting input format: {:?}", in_fmt);

            let src_frame = create_test_frame(in_fmt, sample_rate, nb_channels, nb_samples)
                .with_context(|| format!("Failed to create source frame for {:?}", in_fmt))?;

            for out_fmt in AUDIO_FORMATS {
                let ch_layout = AVChannelLayout::from_nb_channels(nb_channels).into_inner();

                let result = convert_frame(&src_frame, ch_layout, out_fmt.format, sample_rate)
                    .with_context(|| {
                        format!("Failed to convert from {:?} to {:?}", in_fmt, out_fmt)
                    })?;

                // 验证转换结果
                assert_eq!(
                    result.format, out_fmt.format,
                    "Format mismatch converting from {:?} to {:?}",
                    in_fmt, out_fmt
                );
                assert_eq!(
                    result.nb_samples, src_frame.nb_samples,
                    "Sample count mismatch converting from {:?} to {:?}",
                    in_fmt, out_fmt
                );
                assert_eq!(
                    result.ch_layout.nb_channels, src_frame.ch_layout.nb_channels,
                    "Channel count mismatch converting from {:?} to {:?}",
                    in_fmt, out_fmt
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_format_conversion_with_different_rates() -> Result<()> {
        let sample_rates = &[44100, 48000, 96000];
        let nb_samples = 1024;
        let nb_channels = 2;

        for in_fmt in AUDIO_FORMATS {
            for &in_rate in sample_rates {
                let src_frame = create_test_frame(in_fmt, in_rate, nb_channels, nb_samples)?;

                for out_fmt in AUDIO_FORMATS {
                    for &out_rate in sample_rates {
                        if in_rate == out_rate {
                            continue;
                        }

                        println!(
                            "Converting {:?} @{}Hz to {:?} @{}Hz",
                            in_fmt, in_rate, out_fmt, out_rate
                        );

                        let ch_layout = AVChannelLayout::from_nb_channels(nb_channels).into_inner();

                        let result =
                            convert_frame(&src_frame, ch_layout, out_fmt.format, out_rate)?;

                        // 验证转换结果
                        assert_eq!(result.format, out_fmt.format);
                        assert_eq!(result.sample_rate, out_rate);
                        assert_eq!(result.ch_layout.nb_channels, nb_channels);
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_channel_conversion() -> Result<()> {
        let nb_samples = 1024;
        let channel_layouts = &[1, 2];
        let sample_rates = &[44100, 48000, 96000];

        for in_fmt in AUDIO_FORMATS {
            for &in_rate in sample_rates {
                for &in_channels in channel_layouts {
                    println!(
                        "\nSource: format={:?}, rate={}Hz, channels={}",
                        in_fmt, in_rate, in_channels
                    );

                    let src_frame = create_test_frame(in_fmt, in_rate, in_channels, nb_samples)?;

                    assert_eq!(src_frame.ch_layout.nb_channels, in_channels);

                    for out_fmt in AUDIO_FORMATS {
                        for &out_rate in sample_rates {
                            for &out_channels in channel_layouts {
                                // 跳过相同的配置
                                if in_fmt.format == out_fmt.format
                                    && in_rate == out_rate
                                    && in_channels == out_channels
                                {
                                    continue;
                                }

                                println!(
                                    "Converting to: format={:?}, rate={}Hz, channels={}",
                                    out_fmt, out_rate, out_channels
                                );

                                let result = convert_frame(
                                    &src_frame,
                                    AVChannelLayout::from_nb_channels(out_channels).into_inner(),
                                    out_fmt.format,
                                    out_rate,
                                )?;

                                // 验证基本参数
                                assert_eq!(result.format, out_fmt.format);
                                assert_eq!(result.sample_rate, out_rate);
                                assert_eq!(result.ch_layout.nb_channels, out_channels);

                                // 验证数据有效性
                                unsafe {
                                    if SampleFormat::from(out_fmt.format).is_planar() {
                                        for ch in 0..out_channels as usize {
                                            assert!(
                                                !result.data[ch].is_null(),
                                                "Channel {} data pointer is null",
                                                ch
                                            );

                                            let data = std::slice::from_raw_parts(
                                                result.data[ch],
                                                result.nb_samples as usize
                                                    * out_fmt.bytes_per_sample,
                                            );

                                            assert!(
                                                data.iter().any(|&x| x != 0),
                                                "Channel {} contains all zeros (total size: {})",
                                                ch,
                                                data.len()
                                            );
                                        }
                                    } else {
                                        assert!(!result.data[0].is_null(), "Data pointer is null");
                                        let data = std::slice::from_raw_parts(
                                            result.data[0],
                                            result.nb_samples as usize
                                                * out_channels as usize
                                                * out_fmt.bytes_per_sample,
                                        );

                                        assert!(
                                            data.iter().any(|&x| x != 0),
                                            "Output buffer contains all zeros (total size: {})",
                                            data.len()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// 用确定性数据填充 YUV420P 帧的全部平面（保证转换产物非零可断言）。
    unsafe fn fill_yuv420p(frame: &mut AVFrame) {
        for plane in 0..3usize {
            let height = if plane == 0 {
                frame.height
            } else {
                frame.height / 2
            };
            let data = unsafe {
                std::slice::from_raw_parts_mut(
                    frame.data[plane],
                    frame.linesize[plane] as usize * height as usize,
                )
            };
            for (i, b) in data.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        }
    }

    /// 视频缩放：64x64 YUV420P -> 32x32 RGB24。
    ///
    /// 覆盖 [`scale_with_flags`] 的完整执行路径——FFmpeg 6/7 走 legacy
    /// `sws_scale_frame`，FFmpeg 8+ 走 modern `scale_full_frame`。
    #[test]
    fn test_scale_frame_video() -> Result<()> {
        let mut src = AVFrame::new();
        src.set_width(64);
        src.set_height(64);
        src.set_format(PixelFormat::YUV420P.into());
        src.alloc_buffer().context("alloc src buffer")?;
        unsafe { fill_yuv420p(&mut src) };

        let dst = scale_with_flags(&src, 32, 32, PixelFormat::RGB24, SwsFlags::LANCZOS)
            .context("scale failed")?;

        assert_eq!(dst.width, 32);
        assert_eq!(dst.height, 32);
        assert_eq!(dst.format, PixelFormat::RGB24.into());

        // 输出缓冲非零
        unsafe {
            let data = std::slice::from_raw_parts(dst.data[0], dst.linesize[0] as usize * 32);
            assert!(!data.iter().all(|&b| b == 0), "scaled output is empty");
        }
        Ok(())
    }

    /// FFmpeg 8+ modern 全动态参数的完整用法验证（6/7 无这些类型化字段）：
    ///
    /// - `flags`（`u32` 位标志）：算法位（`SWS_*` 滤波器选择）+ 质量位
    ///   （`SWS_ACCURATE_RND`/`SWS_BITEXACT` 等），rsmedia 公开路径经
    ///   [`SwsFlags::complete`] 组装；
    /// - `threads`：并行线程数，0 = 自动；
    /// - `dither`（`SwsDither`）：抖动算法，作用于色深降低/Bayer 输出，
    ///   默认 AUTO；
    /// - `alpha_blend`（`SwsAlphaBlend`）：目标带 alpha 通道时的逐像素混合
    ///   方式，默认 NONE（直接覆盖）；
    /// - `scaler` / `backends`（仅 FFmpeg 9）：显式选择 scaler 类型与实现
    ///   后端，`scaler` 非默认值时覆盖 flags 的算法位。
    #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
    #[test]
    fn test_scale_modern_options() -> Result<()> {
        use rsmpeg::swscale::SwsContext;

        let mut src = AVFrame::new();
        src.set_width(64);
        src.set_height(64);
        src.set_format(PixelFormat::YUV420P.into());
        src.alloc_buffer().context("alloc src buffer")?;
        unsafe { fill_yuv420p(&mut src) };

        let mut ctx = SwsContext::alloc().context("allocate sws context")?;
        ctx.set_flags(SwsFlags::LANCZOS.complete());
        ctx.set_threads(0);
        ctx.set_dither(ffi::SWS_DITHER_AUTO);
        ctx.set_alpha_blend(ffi::SWS_ALPHA_BLEND_NONE);
        #[cfg(feature = "ffmpeg9")]
        {
            // 显式指定 scaler 类型（覆盖 flags 算法位）与允许的实现后端。
            ctx.set_scaler(ffi::SWS_SCALE_BICUBIC);
            ctx.set_backends(ffi::SWS_BACKEND_ALL);
        }

        let mut dst = AVFrame::new();
        dst.set_width(32);
        dst.set_height(32);
        dst.set_format(PixelFormat::RGB24.into());
        dst.alloc_buffer().context("alloc dst buffer")?;

        ctx.scale_full_frame(&mut dst, &src)
            .context("scale with modern options failed")?;

        assert_eq!(dst.width, 32);
        assert_eq!(dst.height, 32);
        assert_eq!(dst.format, PixelFormat::RGB24.into());
        unsafe {
            let data = std::slice::from_raw_parts(dst.data[0], dst.linesize[0] as usize * 32);
            assert!(!data.iter().all(|&b| b == 0), "scaled output is empty");
        }
        Ok(())
    }
}
