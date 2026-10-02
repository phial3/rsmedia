use crate::error::{Context, Result, RsmediaError};
use crate::time::Rational;
use crate::{SampleFormat, imgutils};

use rsmpeg::avutil::{AVChannelLayout, AVFrame, AVSamples};
use rsmpeg::ffi;
use rsmpeg::swresample::SwrContext;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Audio Resampler SwrContext /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// 一组音频格式参数：声道布局 + 采样格式 + 采样率。
///
/// 这三样在重采样里总是一起出现（输入侧一份、输出侧一份），打包传递有两个好处：
/// 像 [`SwrContext::new`] 那样 6 个位置参数的调用不会再被写反，各处也不必反复罗列
/// "布局/格式/采样率"三个字段。
///
/// `pub(crate)`：对外的 [`Resampler::new`] 与 [`convert_frame`] 保留 FFI 镜像签名
/// （裸 `ffi::AVChannelLayout` / `ffi::AVSampleFormat` / `i32`），在模块边界上转换 ——
/// 与 crate 里其它 FFI 镜像层（`VideoParams`、`imgutils::fill_linesizes`…）一致。
#[derive(Debug, Clone, Copy)]
pub(crate) struct AudioSpec {
    ch_layout: ffi::AVChannelLayout,
    sample_fmt: ffi::AVSampleFormat,
    sample_rate: i32,
}

impl AudioSpec {
    pub(crate) fn new(
        ch_layout: ffi::AVChannelLayout,
        sample_fmt: ffi::AVSampleFormat,
        sample_rate: i32,
    ) -> Self {
        Self {
            ch_layout,
            sample_fmt,
            sample_rate,
        }
    }

    /// `frame` 自己的音频格式。
    pub(crate) fn from_frame(frame: &AVFrame) -> Self {
        Self::new(frame.ch_layout, frame.format, frame.sample_rate)
    }

    /// 只换采样格式，声道布局与采样率不变。
    ///
    /// 「只换格式」是解码输出与「进滤镜图前」两处的全部需求：它们的目标布局/采样率就是
    /// 帧自己的，只有采样格式由配置决定。
    pub(crate) fn with_sample_fmt(self, sample_fmt: ffi::AVSampleFormat) -> Self {
        Self { sample_fmt, ..self }
    }

    /// 未指定（UNSPEC）的声道布局按声道数取默认布局 —— 见 [`default_layout_for`]。
    /// 上下文按归一化后的布局建立，`swr` 才不会每帧都报 `AVERROR_OUTPUT_CHANGED`。
    fn normalized(self) -> Self {
        Self {
            ch_layout: default_layout_for(self.ch_layout).unwrap_or(self.ch_layout),
            ..self
        }
    }

    fn nb_channels(self) -> i32 {
        self.ch_layout.nb_channels
    }

    /// `frame` 是否已经是这个格式。
    fn matches(self, frame: &AVFrame) -> bool {
        self == Self::from_frame(frame)
    }
}

/// 声道布局按 FFmpeg 自己的规则比较：`AVChannelLayout` 的掩码在 union 里，手写比较会漏掉
/// "同为 2 声道但 FL|FR ≠ FL|FC"这种差别 —— 而布局不对正是重采样该修的东西。
impl PartialEq for AudioSpec {
    fn eq(&self, other: &Self) -> bool {
        self.sample_fmt == other.sample_fmt
            && self.sample_rate == other.sample_rate
            // SAFETY: 两个引用都指向已初始化的 `AVChannelLayout`；比较函数只读它们。
            && unsafe { ffi::av_channel_layout_compare(&self.ch_layout, &other.ch_layout) == 0 }
    }
}

fn setup_resampler(in_spec: AudioSpec, out_spec: AudioSpec) -> Result<SwrContext> {
    let mut swr_ctx = SwrContext::new(
        &out_spec.ch_layout,
        out_spec.sample_fmt,
        out_spec.sample_rate,
        &in_spec.ch_layout,
        in_spec.sample_fmt,
        in_spec.sample_rate,
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
        return Err(RsmediaError::msg("Invalid input frame."));
    }
    Ok(())
}

/// The default channel layout for `layout`, or [`None`] when it is already
/// specified (or names no channels at all).
///
/// Frames decoded from containers that carry no channel mask (a plain WAV, say)
/// report `AV_CHANNEL_ORDER_UNSPEC`. `swr_alloc_set_opts2` replaces such an order
/// with the default layout for the channel count when it builds the context, and
/// `swr_convert` then rejects every frame whose order still says UNSPEC with
/// `AVERROR_INPUT_CHANGED` / `AVERROR_OUTPUT_CHANGED`. Interpreting the
/// unspecified layout as the default one is what FFmpeg itself does; this is the
/// one place that decision is made, for input frames and output layouts alike.
fn default_layout_for(layout: ffi::AVChannelLayout) -> Option<ffi::AVChannelLayout> {
    (layout.order == ffi::AV_CHANNEL_ORDER_UNSPEC && layout.nb_channels > 0)
        .then(|| AVChannelLayout::from_nb_channels(layout.nb_channels).into_inner())
}

/// 采样格式的可读名称；未收录的取值退化为原始整数（来自外部的值不 panic）。
fn sample_fmt_name(sample_fmt: ffi::AVSampleFormat) -> String {
    SampleFormat::from_ffi_checked(sample_fmt)
        .map_or_else(|| sample_fmt.to_string(), |fmt| fmt.get_sample_fmt_name())
}

/// Runs `convert` with `frame`'s unspecified channel layout filled in.
///
/// The clone only happens in that one case, and it is `av_frame_clone`: the buffers are
/// **shared by reference count** (sample data is not copied), so the cost is a refcount
/// bump. A frame whose layout is specified is passed through untouched.
fn with_normalized_layout<R>(
    frame: &AVFrame,
    convert: impl FnOnce(&AVFrame) -> Result<R>,
) -> Result<R> {
    let normalized;
    let frame = match default_layout_for(frame.ch_layout) {
        Some(layout) => {
            let mut copy = frame.clone();
            copy.set_ch_layout(layout);
            normalized = copy;
            &normalized
        }
        None => frame,
    };
    convert(frame)
}

/// Resamples one frame into a new frame, in one call.
///
/// The context is created fresh for this call, so this is for isolated
/// conversions; a stream should use [`Resampler`] instead, whose delay buffers
/// carry samples across calls (and whose [`flush`](Resampler::flush) drains the
/// tail).
///
/// The output is allocated at the *upper bound* of what the conversion can
/// produce, so `dst.nb_samples` after the call is the real count. Metadata is
/// copied from the source frame (pts included), and the result's time base is
/// `1 / out_sample_rate`.
///
/// # Arguments
///
/// * `src_frame` - Decoded frame to convert; must be a software frame with a
///   valid sample rate and at least one sample.
/// * `out_ch_layout` - Channel layout of the output.
/// * `out_sample_fmt` - Sample format of the output.
/// * `out_sample_rate` - Sample rate of the output.
///
/// This is an FFI-level entry point — it takes a bare `AVFrame` plus
/// `ffi::AVChannelLayout` / `ffi::AVSampleFormat` and `i32` rates, mirroring
/// `SwrContext`'s own signatures. The high-level audio path uses
/// `MediaFrame`'s `u32` sample rate and converts at this boundary.
pub fn convert_frame(
    src_frame: &AVFrame,
    out_ch_layout: ffi::AVChannelLayout,
    out_sample_fmt: ffi::AVSampleFormat,
    out_sample_rate: i32,
) -> Result<AVFrame> {
    check_resampler_input(src_frame)?;

    let normalized = with_normalized_layout(src_frame, |frame| Ok(frame.clone()))?;
    let src_frame = &normalized;

    // 输出布局同样要归一化：上下文按它构建，目标帧每次调用又把它交回 swr，
    // 未指定的 order 会触发 `AVERROR_OUTPUT_CHANGED`。
    let out_spec = AudioSpec::new(out_ch_layout, out_sample_fmt, out_sample_rate).normalized();

    let mut resampler = Resampler::build(AudioSpec::from_frame(src_frame), out_spec)?;
    convert_with(&mut resampler, src_frame, out_spec)
}

/// 用给定的重采样上下文把一帧转成新分配的输出帧（不改动上下文，可跨帧复用）。
fn convert_with(
    resampler: &mut Resampler,
    src_frame: &AVFrame,
    out_spec: AudioSpec,
) -> Result<AVFrame> {
    let mut dst_frame = AVFrame::new();
    // copy props
    imgutils::copy_frame_metadata(src_frame, &mut dst_frame, false)?;
    dst_frame.set_format(out_spec.sample_fmt);
    dst_frame.set_ch_layout(out_spec.ch_layout);
    // 输出帧的缓冲必须按重采样后的输出样本数分配，而不是简单地使用输入样本数。
    // 当输入/输出采样率不同时，swr_convert() 会写入比输入样本数更多的输出样本，
    // 但 FFmpeg 不会自动扩大已分配的输出缓冲（只把放不下的部分存入内部 FIFO），
    // 若这里 nb_samples 设得过小，将导致 swr_convert 越界写。
    // 用 swr_get_out_samples() 得到所需输出样本数的上界来分配缓冲。
    let out_samples = resampler.get_out_samples(src_frame.nb_samples).max(1);
    dst_frame.set_nb_samples(out_samples);
    dst_frame.set_sample_rate(out_spec.sample_rate);
    dst_frame.set_time_base(
        Rational::new(1, out_spec.sample_rate)
            .unwrap_or(Rational::ZERO)
            .into(),
    );
    dst_frame
        .alloc_buffer()
        .context("Failed to allocate destination frame buffer")?;

    // 转换输入 AVFrame 中的样本并将其写入输出 AVFrame。
    // 输入和输出 AVFrame 必须设置通道布局、采样率和格式。
    // 如果输出 AVFrame 没有分配数据指针，则将在调用 av_frame_get_buffer() 分配帧时设置 nb_samples 字段。
    // 输出的 AVFrame 可以是 NULL，或者分配的样本少于所需的数量。在这种情况下，未写入输出的剩余样本将被添加到内部 FIFO 缓冲区，在下次调用此函数时返回。
    // 如果转换采样率，内部重采样延迟缓冲区中可能会有剩余数据。要以输出方式获取这些数据，请调用此函数，并输入 NULL。
    resampler
        .convert_frame(src_frame, &mut dst_frame)
        .context("Failed to convert frame.")?;

    tracing::debug!(
        "Swr convert_frame from src:[{}, {:?}, {}] to dst:[{}, {:?}, {}]",
        src_frame.ch_layout.nb_channels,
        sample_fmt_name(src_frame.format),
        src_frame.sample_rate,
        out_spec.nb_channels(),
        sample_fmt_name(out_spec.sample_fmt),
        out_spec.sample_rate
    );

    Ok(dst_frame)
}

/// 错误是否表示「帧规格与重采样上下文不一致」——FFmpeg 要求此时重建上下文。
///
/// `swr_convert_frame` 用 `AVERROR_INPUT_CHANGED` / `AVERROR_OUTPUT_CHANGED` 报告
/// 该情况，两者也可能按位或在一起，故用位测试而不是相等比较。`Context` 包装过的
/// 错误要走到 [`RsmediaError::root`] 才能看到原始形态。
fn is_spec_change_error(err: &RsmediaError) -> bool {
    let changed = ffi::AVERROR_INPUT_CHANGED | ffi::AVERROR_OUTPUT_CHANGED;
    matches!(
        err.root(),
        RsmediaError::FFmpeg(rsmpeg::error::RsmpegError::AVError(code)) if code & changed != 0
    )
}

/// 把一帧重采样到目标格式；**已经是目标格式、或者是空帧就原样返回**。
///
/// 解码与编码三处共用这一条路径，是音频侧对应 [`Scaler::scale_if_needed`] 的入口。
/// 其中「目标 == 现状」是最常见的：解码输出与「进滤镜图前」的目标布局/采样率就是帧自己的，
/// 只有采样格式是配置决定的目标值，因此目标与现状是否一致只在这里判一次，调用方不必各写
/// 一遍。空帧（`nb_samples < 1`）一并放行：`swr` 无从转换它，[`check_resampler_input`]
/// 也会拒绝，而调用方要的显然是"把这一帧接着往下送"。
///
/// 需要转换时才建立（或复用）上下文 —— **输入格式要到第一帧才知道**，调用方手上只有目标
/// 格式，而建上下文输入输出两侧都得给全，所以 `slot` 是 [`Option`]`<Resampler>`：`None`
/// 表示"还没有帧决定过输入格式"。已经有一个、但输出格式与本次不同时按本次的格式重建
/// （解码与「进滤镜图前」的输出布局/采样率取自帧本身，帧一变输出侧就跟着变）；输入格式的
/// 变化不在这里比较，交给 FFmpeg 报告（`AVERROR_INPUT_CHANGED` /
/// `AVERROR_OUTPUT_CHANGED`）—— 它能看出我们比不出来的情况，例如 UNSPEC 布局按声道数
/// 补出的默认布局与显式布局之间的差别。
///
/// # Errors
///
/// 见 [`Resampler::convert_frame_owned`]：硬件帧与空帧由 [`check_resampler_input`] 拒掉
/// （空帧在这里已提前返回），其余按重采样本身的错误报出。
pub(crate) fn resample_if_needed(
    slot: &mut Option<Resampler>,
    src_frame: AVFrame,
    out_spec: AudioSpec,
) -> Result<AVFrame> {
    if out_spec.matches(&src_frame) || src_frame.nb_samples < 1 {
        return Ok(src_frame);
    }
    let out_spec = out_spec.normalized();
    let stale = slot
        .as_ref()
        .is_some_and(|resampler| resampler.out_spec != out_spec);
    if slot.is_none() || stale {
        *slot = Some(Resampler::build(
            AudioSpec::from_frame(&src_frame),
            out_spec,
        )?);
    }
    slot.as_mut()
        .expect("filled just above")
        .convert_frame_owned(&src_frame)
}

/// Persistent streaming resampler.
///
/// Unlike [`convert_frame`] (which creates a temporary context on each call),
/// `Resampler` holds a reusable `SwrContext` for continuous streaming input:
/// when resampling (different input/output sample rates), the internal filter
/// delay buffers samples between calls and outputs them with subsequent data;
/// at the end, [`Resampler::flush_frames`] must be called to drain the tail,
/// otherwise the last few milliseconds of samples will be lost. A context whose
/// output sample rate equals its input rate does no resampling and keeps no
/// delay, so draining one that only converts the sample format is a no-op.
///
/// Three ways to drive it, in increasing order of how much the caller has to arrange:
///
/// * [`Resampler::convert_frame_owned`] — allocates the destination frame at the upper bound
///   [`Resampler::get_out_samples`] reports, for the output spec this resampler was built
///   with. This is the streaming entry point: the decode and encode pipelines reach it
///   through `resample_if_needed`, which also skips the work entirely when the frame already
///   has the target format.
/// * [`Resampler::convert_frame`] — the caller pre-allocates `dst` and sets its format,
///   layout, sample rate and `nb_samples` on it.
/// * [`Resampler::convert`] — a raw `swr_convert` into an [`AVSamples`] buffer, for
///   caller-managed sample buffers; the returned count says how many samples per channel the
///   buffer actually holds.
///
/// A frame that no longer matches the context is absorbed by rebuilding it — see
/// [`Resampler::convert_frame_owned`].
pub struct Resampler {
    swr: SwrContext,
    /// 上下文构建时用的**输出格式**（声道布局已归一化）。
    ///
    /// 留着是因为重建上下文时输出侧要原样恢复 —— `setup_resampler` 输入输出都得给全，
    /// 而重建发生在只拿得到新输入帧的地方（见 [`Self::convert_frame_owned`]）。也用于
    /// [`Self::convert`]：输出样本缓冲必须按上下文真正使用的声道数/采样格式分配，取这里
    /// 的规格就不存在形参不一致的可能。
    out_spec: AudioSpec,
}

impl Resampler {
    /// Build a resampler from the input format to the output format.
    ///
    /// The six values are two `AudioSpec` triples laid out flat — six positional
    /// arguments of which three describe each side — the shape `SwrContext` itself takes, and
    /// the one [`convert_frame`] mirrors.
    ///
    /// # Errors
    ///
    /// [`RsmediaError::Other`] if the context cannot be allocated or opened, e.g. for a
    /// sample format or layout FFmpeg has no converter for.
    pub fn new(
        in_ch_layout: ffi::AVChannelLayout,
        in_sample_fmt: ffi::AVSampleFormat,
        in_sample_rate: i32,
        out_ch_layout: ffi::AVChannelLayout,
        out_sample_fmt: ffi::AVSampleFormat,
        out_sample_rate: i32,
    ) -> Result<Self> {
        // 与 `convert`/`convert_frame` 用同一套归一化，保证留下来的输出格式就是上下文
        // 真正使用的那个。
        let out_spec = AudioSpec::new(out_ch_layout, out_sample_fmt, out_sample_rate).normalized();
        Self::build(
            AudioSpec::new(in_ch_layout, in_sample_fmt, in_sample_rate),
            out_spec,
        )
    }

    /// 按输入/输出两侧的格式建立上下文。
    fn build(in_spec: AudioSpec, out_spec: AudioSpec) -> Result<Self> {
        Ok(Self {
            swr: setup_resampler(in_spec, out_spec)?,
            out_spec,
        })
    }

    /// 按 `src_frame` 的输入格式重建上下文，输出格式不变。
    ///
    /// 上下文里的延迟缓冲随旧上下文一起丢弃，但那些样本属于刚刚结束的那套格式，
    /// 本就无从转换。
    fn rebuild_for(&mut self, src_frame: &AVFrame) -> Result<()> {
        let out_spec = self.out_spec;
        *self = Self::build(AudioSpec::from_frame(src_frame), out_spec)?;
        Ok(())
    }

    /// Upper bound estimate of the number of output samples for the given number of input samples.
    pub fn get_out_samples(&self, in_samples: i32) -> i32 {
        self.swr.get_out_samples(in_samples).max(1)
    }

    /// Resample one frame into a **newly allocated** output frame.
    ///
    /// Differs from [`Self::convert_frame`] only in who allocates: the destination is built
    /// here, at the upper bound from [`Self::get_out_samples`] — upsampling produces more
    /// samples than went in — so `nb_samples` on the result is the real count. Metadata is
    /// copied from the source frame (pts included) and the result's time base is
    /// `1 / out_sample_rate`. The output spec is the one this resampler was built with
    /// (see [`Self::new`]).
    ///
    /// The context is reused across frames, which is the point: when the sample rate
    /// changes, swr keeps a few trailing samples in its internal delay buffer and emits them
    /// with the following input, so a context created per frame would drop that tail every
    /// time. Call [`Self::flush`] after the last frame to drain what is left.
    ///
    /// # Errors
    ///
    /// [`RsmediaError::Unsupported`] for a hardware frame and [`RsmediaError::Other`] for a
    /// frame with no sample rate or no samples, both from the same checks
    /// [`convert_frame`] performs; otherwise whatever the conversion reports.
    ///
    /// When FFmpeg reports that the frame no longer matches the context
    /// (`AVERROR_INPUT_CHANGED` / `AVERROR_OUTPUT_CHANGED`), the context is rebuilt for the
    /// new *input* spec — the output spec stays what this resampler was built for — and the
    /// conversion is retried once. The old delay buffer is dropped with the old context, but
    /// those samples belong to the spec that has just ended.
    pub fn convert_frame_owned(&mut self, src_frame: &AVFrame) -> Result<AVFrame> {
        check_resampler_input(src_frame)?;

        let out_spec = self.out_spec;
        match convert_with(self, src_frame, out_spec) {
            Err(err) if is_spec_change_error(&err) => {
                tracing::debug!(
                    "Resampler spec changed ({err}); rebuilding the context for the new frame"
                );
                self.rebuild_for(src_frame)?;
                convert_with(self, src_frame, out_spec)
            }
            result => result,
        }
    }

    /// Convert an input frame into an output frame with allocated buffer
    /// (persistent context: samples that cannot be written with `swr_convert` due to
    /// insufficient output capacity remain internally buffered and are returned with subsequent calls).
    ///
    /// The caller must set format/layout/sample_rate/nb_samples on `dst` and
    /// call `alloc_buffer`; after conversion, `dst.nb_samples` is the actual
    /// number of output samples.
    pub fn convert_frame(&mut self, src: &AVFrame, dst: &mut AVFrame) -> Result<()> {
        with_normalized_layout(src, |src| {
            self.swr
                .convert_frame(Some(src), dst)
                .context("Failed to convert frame with streaming resampler")
        })
    }

    /// Raw sample conversion (a direct `swr_convert`), for caller-managed sample
    /// buffers such as [`AVSamples`].
    ///
    /// Returns the buffer and **the number of samples per channel it holds**. The
    /// buffer is allocated at the upper bound from
    /// [`get_out_samples`](Self::get_out_samples) — upsampling produces more
    /// samples than went in — so the count, not the capacity, says how much of it
    /// is valid:
    ///
    /// * `> 0` — that many samples were written for each channel;
    /// * `0` — the input had samples but nothing came out yet: swr holds them in
    ///   its internal delay buffer and emits them with the following input
    ///   (possible whenever the sample rate changes).
    ///
    /// A negative `swr_convert` result becomes an error, never a count.
    ///
    /// The output buffer is allocated for the **output spec this resampler was
    /// built with** (see [`Self::new`]) — there is no longer a way to pass a
    /// mismatched layout/fmt, which previously was a runtime error here and a
    /// potential heap overrun if it slipped through.
    pub fn convert(&mut self, src_frame: &AVFrame) -> Result<(AVSamples, i32)> {
        // 输出样本缓冲按**上下文**的输出声道数/采样格式分配：`swr_convert`
        // 始终按上下文（而非调用方传入的形参）写数据。输出规格即 `new` 里归一化后
        // 的 `out_spec`，分配因此总是与上下文一致，也消除了形参不一致的越界写隐患。
        let ch_layout = self.out_spec.ch_layout;

        with_normalized_layout(src_frame, |src_frame| {
            // 容量按输出样本数的上界分配，避免上采样（in < out）时尾部样本被丢弃。
            let capacity = self.get_out_samples(src_frame.nb_samples);
            let mut out_samples =
                AVSamples::new(ch_layout.nb_channels, capacity, self.out_spec.sample_fmt, 0)
                    .context("Create samples buffer failed.")?;

            let converted = unsafe {
                self.swr
                    .convert(
                        out_samples.audio_data.as_mut_ptr(),
                        capacity,
                        src_frame.extended_data as *const _,
                        src_frame.nb_samples,
                    )
                    .context("Could not convert input samples")?
            };

            // `AVSamples::nb_samples` 是**容量**（rsmpeg 的约定），所以实际样本数
            // 单独返回，调用方无需猜测缓冲区里有多少是有效的。
            Ok((out_samples, converted))
        })
    }

    /// 分配一个符合本重采样器**输出规格**的帧，容量 `nb_samples` 样本/声道。
    ///
    /// 这是 [`Self::convert_frame`] / [`Self::flush`] 里那段"手动搭目标帧"
    /// 的标准写法——按 `new` 时归一化后的输出规格设格式/布局/采样率/时基并分配缓冲。
    /// 集中的意义：整个 crate 只有一个地方知道"输出帧长什么样"。
    pub fn alloc_out_frame(&self, nb_samples: i32) -> Result<AVFrame> {
        let mut dst = AVFrame::new();
        dst.set_format(self.out_spec.sample_fmt);
        dst.set_ch_layout(self.out_spec.ch_layout);
        dst.set_sample_rate(self.out_spec.sample_rate);
        dst.set_nb_samples(nb_samples.max(1));
        dst.set_time_base(
            Rational::new(1, self.out_spec.sample_rate)
                .unwrap_or(Rational::ZERO)
                .into(),
        );
        dst.alloc_buffer()
            .context("Failed to allocate a frame for the resampler output")?;
        Ok(dst)
    }

    /// Drain the remaining samples from the resampler (EOF flush).
    ///
    /// `dst` must already have an allocated buffer; after conversion,
    /// `dst.nb_samples` is the actual number of samples (possibly 0).
    /// Repeat until 0 samples are returned to fully drain.
    pub fn flush(&mut self, dst: &mut AVFrame) -> Result<()> {
        self.swr
            .convert_frame(None, dst)
            .context("Failed to flush streaming resampler")
    }

    /// Drain the whole delay line into newly allocated frames.
    ///
    /// The end-of-stream counterpart of [`Self::convert_frame_owned`]: that one takes a
    /// frame and returns the resampled one, this one takes no input and returns what the
    /// delay line still holds — frames in this resampler's output spec, ready to be sent
    /// on like any other. An empty `Vec` means nothing was left, which is the usual
    /// answer: a context built for a 1:1 sample rate does no resampling and therefore
    /// keeps no delay at all.
    ///
    /// `swr` need not hand everything over in a single call, so this loops until a call
    /// comes back empty. The loop is bounded by `MAX_DRAIN_ITERATIONS` — the same cap
    /// every drain loop in this crate uses — because an unbounded one would let "finish
    /// this audio stream" hang forever.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::flush`] reports, plus [`RsmediaError::Other`] if the delay line
    /// does not empty within the iteration bound.
    pub fn flush_frames(&mut self) -> Result<Vec<AVFrame>> {
        let capacity = self.out_spec.sample_rate.max(1);
        let mut frames = Vec::new();
        for _ in 0..crate::MAX_DRAIN_ITERATIONS {
            let mut dst = self.alloc_out_frame(capacity)?;
            self.flush(&mut dst)?;
            if dst.nb_samples <= 0 {
                return Ok(frames);
            }
            frames.push(dst);
        }
        Err(RsmediaError::msg(format!(
            "Resampler keeps producing samples while draining ({} iterations); giving up",
            crate::MAX_DRAIN_ITERATIONS
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Context, Result};
    use crate::{SampleFormat, time::Rational};
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
            _ => return Err(RsmediaError::msg("Unsupported sample format")),
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
        frame.set_time_base(
            Rational::new(1, sample_rate)
                .unwrap_or(Rational::ZERO)
                .into(),
        );

        frame
            .alloc_buffer()
            .context("Failed to allocate frame buffer")?;

        unsafe {
            fill_test_data(&mut frame, format_desc).context("Failed to fill test data")?;
        }

        Ok(frame)
    }

    /// 无声道掩码的容器（如不带 `dwChannelMask` 的 WAV）解码出的帧布局是
    /// `AV_CHANNEL_ORDER_UNSPEC`；重采样器必须接受它——此前 `swr_convert`
    /// 会以 `AVERROR_INPUT_CHANGED`/`OUTPUT_CHANGED` 拒绝每一帧。
    #[test]
    fn test_convert_frame_with_unspec_channel_layout() -> Result<()> {
        let mut frame = create_test_frame(&AUDIO_FORMATS[2], 44100, 2, 1024)?;
        let mut unspec = AVChannelLayout::from_nb_channels(2).into_inner();
        unspec.order = ffi::AV_CHANNEL_ORDER_UNSPEC;
        unspec.u.mask = 0;
        frame.set_ch_layout(unspec);

        // 输入与输出布局都按 UNSPEC 传入：两侧都要被归一化。
        let out = convert_frame(
            &frame,
            AVChannelLayout::from_nb_channels(2).into_inner(),
            ffi::AV_SAMPLE_FMT_FLTP,
            44100,
        )?;
        assert_eq!(out.format, ffi::AV_SAMPLE_FMT_FLTP);
        assert_eq!(out.ch_layout.order, ffi::AV_CHANNEL_ORDER_NATIVE);
        assert_eq!(out.ch_layout.nb_channels, 2);
        assert_eq!(out.nb_samples, frame.nb_samples);
        Ok(())
    }

    #[test]
    fn test_format_conversion() -> Result<()> {
        let sample_rate = 44100;
        let nb_samples = 1024;
        let nb_channels = 2;

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

    /// 流式重采样器会在内部保留采样率换算的余数，跨调用携带；`flush` 必须把尾巴
    /// 排空。否则每个 chunk 都会丢掉不足一个输出样本的余量，长音频累计起来就是
    /// 可听的时长缺失——一次性 `convert_frame` 每次新建上下文，暴露不出这个问题。
    #[test]
    fn test_streaming_resampler_carries_delay_and_flushes() -> Result<()> {
        let (in_rate, out_rate) = (48_000, 44_100);
        let (channels, in_samples, chunks) = (2, 1024, 5);
        let layout = || AVChannelLayout::from_nb_channels(channels).into_inner();

        let mut resampler = Resampler::new(
            layout(),
            ffi::AV_SAMPLE_FMT_FLTP,
            in_rate,
            layout(),
            ffi::AV_SAMPLE_FMT_FLTP,
            out_rate,
        )?;

        let mut produced = 0i64;
        for chunk in 0..chunks {
            let src = create_test_frame(&AUDIO_FORMATS[7], in_rate, channels, in_samples)?;
            let (_, converted) = resampler.convert(&src)?;
            assert!(
                converted >= 0,
                "chunk {chunk}: swr_convert returned {converted}"
            );
            produced += i64::from(converted);
        }

        // 排空延迟缓冲：容量按输出上界分配，`nb_samples` 回填实际数量。
        let mut tail = AVFrame::new();
        tail.set_format(ffi::AV_SAMPLE_FMT_FLTP);
        tail.set_ch_layout(layout());
        tail.set_sample_rate(out_rate);
        tail.set_nb_samples(resampler.get_out_samples(in_samples));
        tail.alloc_buffer()
            .context("Failed to allocate flush frame")?;
        resampler.flush(&mut tail)?;
        assert!(
            tail.nb_samples > 0,
            "flush produced nothing: the delay line was never drained"
        );
        produced += i64::from(tail.nb_samples);

        let expected = f64::from(in_samples * chunks) * f64::from(out_rate) / f64::from(in_rate);
        assert!(
            (produced as f64 - expected).abs() <= 2.0,
            "expected ~{expected:.0} samples across {chunks} chunks + flush, got {produced}"
        );
        Ok(())
    }

    /// [`Resampler::flush_frames`] 排出的正是上面那条测试里"少了的那一段"。
    #[test]
    fn test_flush_frames_returns_the_delay_line() -> Result<()> {
        let (in_rate, out_rate) = (48_000, 44_100);
        let (channels, in_samples, chunks) = (2, 1024, 5);
        let layout = || AVChannelLayout::from_nb_channels(channels).into_inner();

        let mut resampler = Resampler::new(
            layout(),
            ffi::AV_SAMPLE_FMT_FLTP,
            in_rate,
            layout(),
            ffi::AV_SAMPLE_FMT_FLTP,
            out_rate,
        )?;
        let mut produced = 0i64;
        for _ in 0..chunks {
            let src = create_test_frame(&AUDIO_FORMATS[7], in_rate, channels, in_samples)?;
            produced += i64::from(resampler.convert_frame_owned(&src)?.nb_samples);
        }

        let tail: i64 = resampler
            .flush_frames()?
            .iter()
            .map(|frame| i64::from(frame.nb_samples))
            .sum();
        assert!(
            tail > 0,
            "flush_frames produced nothing: the delay line was never drained"
        );
        // 排空后再排一次必须为空 —— 循环按"取到空帧"终止，不会无限吐下去。
        assert!(resampler.flush_frames()?.is_empty());

        let expected = f64::from(in_samples * chunks) * f64::from(out_rate) / f64::from(in_rate);
        assert!(
            ((produced + tail) as f64 - expected).abs() <= 2.0,
            "expected ~{expected:.0} samples across {chunks} chunks + flush_frames, got {}",
            produced + tail
        );
        Ok(())
    }

    /// 只换采样格式（采样率不变）的上下文不做重采样 ⇒ **没有延迟线可排**。
    ///
    /// 这正是 [`resample_if_needed`] 在「进滤镜图前」与解码输出两处的用法：目标布局与
    /// 采样率都取自帧本身，只有采样格式由配置决定。所以那两个 [`Resampler`] 不需要
    /// 排空 —— 只有输出规格取自**编码器**（采样率可能与输入不同）的那个才需要。
    #[test]
    fn test_a_format_only_resampler_keeps_no_delay_line() -> Result<()> {
        let (rate, channels, samples) = (48_000, 2, 1024);
        let layout = AVChannelLayout::from_nb_channels(channels).into_inner();
        let mut resampler = Resampler::new(
            layout,
            ffi::AV_SAMPLE_FMT_S16,
            rate,
            layout,
            ffi::AV_SAMPLE_FMT_FLTP,
            rate,
        )?;

        let src = create_test_frame(&AUDIO_FORMATS[2], rate, channels, samples)?;
        let out = resampler.convert_frame_owned(&src)?;
        assert_eq!(
            out.nb_samples, samples,
            "same rate in and out: one sample out per sample in"
        );
        assert!(
            resampler.flush_frames()?.is_empty(),
            "a context that does no resampling must have nothing to drain"
        );
        Ok(())
    }

    /// [`Resampler`] 跨帧复用同一个上下文：采样率换算的余数留在上下文里、由后续帧
    /// 带出，因此总样本数逼近理论值（只差上下文内尚未排出的那一份延迟）；而逐帧
    /// [`convert_frame`] 每次新建上下文，**每帧**都丢掉这份延迟，缺口随帧数线性放大。
    ///
    /// 这正是解码/编码路径持有一个 `Resampler`、而不是每帧调 `convert_frame` 的原因，
    /// 也是本测试要钉住的差别。
    #[test]
    fn test_resampler_reuses_context_across_frames() -> Result<()> {
        let (in_rate, out_rate) = (44_100, 48_000);
        let (channels, nb_samples, frames) = (2, 1024, 50);
        let layout = || AVChannelLayout::from_nb_channels(channels).into_inner();

        let src = create_test_frame(&AUDIO_FORMATS[2], in_rate, channels, nb_samples)?;

        let mut resampler = Resampler::new(
            layout(),
            ffi::AV_SAMPLE_FMT_S16,
            in_rate,
            layout(),
            ffi::AV_SAMPLE_FMT_FLTP,
            out_rate,
        )?;
        let mut streamed = 0i64;
        for _ in 0..frames {
            streamed += i64::from(resampler.convert_frame_owned(&src)?.nb_samples);
        }

        let mut rebuilt_each_call = 0i64;
        for _ in 0..frames {
            let out = convert_frame(&src, layout(), ffi::AV_SAMPLE_FMT_FLTP, out_rate)?;
            rebuilt_each_call += i64::from(out.nb_samples);
        }

        let expected =
            frames as i64 * i64::from(nb_samples) * i64::from(out_rate) / i64::from(in_rate);
        assert!(
            expected - streamed < i64::from(nb_samples),
            "reusing the context may only lose one flush-less delay (< 1 frame), \
             expected ~{expected}, got {streamed}"
        );
        assert!(
            rebuilt_each_call < streamed,
            "a context per frame must lose the delay every time: \
             per-call {rebuilt_each_call} vs streamed {streamed}"
        );
        Ok(())
    }

    /// 帧规格中途变化（采样率换了）时，`swr` 以 `AVERROR_INPUT_CHANGED` 拒绝复用旧
    /// 上下文；[`Resampler::convert_frame_owned`] 必须按新规格重建并完成这一帧，而不是
    /// 把错误抛给调用方。取值走的是调用方那条路（`resample_if_needed`），
    /// 所以输入规格的比较也一并覆盖：输出规格没变，不该因此重建。
    #[test]
    fn test_resampler_rebuilds_on_spec_change() -> Result<()> {
        let channels = 2;
        let nb_samples = 1024;
        let layout = || AVChannelLayout::from_nb_channels(channels).into_inner();

        let mut slot: Option<Resampler> = None;
        let out_spec = || AudioSpec::new(layout(), ffi::AV_SAMPLE_FMT_FLTP, 48_000);
        let first = create_test_frame(&AUDIO_FORMATS[2], 44_100, channels, nb_samples)?;
        let first_out = resample_if_needed(&mut slot, first, out_spec())?;
        // 首帧的输出样本数由采样率比决定（约 1024 * 48/44.1），并扣掉留在上下文里的
        // 那一份延迟；这里只要它非空且规格正确即可，具体数值随 FFmpeg 版本浮动。
        assert!(first_out.nb_samples > 0);

        // 换成 32kHz 的帧：与上下文（44.1kHz）不符，须重建。
        let second = create_test_frame(&AUDIO_FORMATS[2], 32_000, channels, nb_samples)?;
        let out = resample_if_needed(&mut slot, second, out_spec())?;
        // 重建后的首帧按 32kHz → 48kHz 等比输出约 1024 * 1.5 = 1536 个样本，但重采样
        // 滤波器自身的启动延迟会扣掉几十个样本（它们留在新上下文里、由后续帧带出），
        // 所以这里只校验量级，不钉死具体数值。
        let expected = nb_samples as f64 * 48_000.0 / 32_000.0;
        assert!(
            (f64::from(out.nb_samples) - expected).abs() < 64.0,
            "expected ~{expected:.0} samples after the rebuild, got {}",
            out.nb_samples
        );
        assert_eq!(out.format, ffi::AV_SAMPLE_FMT_FLTP);
        assert_eq!(out.sample_rate, 48_000);
        Ok(())
    }
}
