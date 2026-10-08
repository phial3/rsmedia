//! Filter module
//! This module provides a set of filters that can be used to process media data.
//! The filters are implemented using the ffmpeg library.
//!
//! See: <https://ffmpeg.org/ffmpeg-filters.html>
use crate::error::{Context, Result, RsmediaError};
use crate::fmt::{FrameFormat, SampleFormat};
use crate::pixel::PixelFormat;
use crate::state::ProcessState;
use crate::stream::MediaType;
use crate::strutils;
use crate::time::Rational;

use rsmpeg::avfilter::{AVFilter, AVFilterContextMut, AVFilterGraph, AVFilterInOut, AVFilterRef};
use rsmpeg::avutil::{AVChannelLayout, AVFrame};
use rsmpeg::ffi;

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Filter {
    name: &'static str,
    media_type: MediaType,
    spec: String,
    /// 滤镜图要求的**输入**格式声明（如 GIF 调色板链要求 RGB 输入）。
    ///
    /// `Some(fmt)` 时：编码器/解码器侧把输入帧先转换到 `fmt` 再进图，且
    /// buffer/abuffer 源按 `fmt` 配置（sink 仍按协商格式）。`None` 时沿用
    /// 协商格式（默认，src/sink 同格式，零行为变化）。
    input_format: Option<FrameFormat>,
}

impl Filter {
    pub fn new(name: &'static str, media_type: MediaType, spec: String) -> Self {
        Self {
            name,
            media_type,
            spec,
            input_format: None,
        }
    }

    /// 声明滤镜图要求的输入格式（视频=像素格式 / 音频=采样格式，覆盖默认的
    /// "协商格式"）。
    ///
    /// `PixelFormat`/`SampleFormat` 均可隐式转入 [`FrameFormat`]，调用形如
    /// `.with_input_format(PixelFormat::RGB24)` 或
    /// `.with_input_format(SampleFormat::FLT)`。
    pub fn with_input_format(mut self, format: impl Into<FrameFormat>) -> Self {
        self.input_format = Some(format.into());
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn media_type(&self) -> MediaType {
        self.media_type
    }

    pub fn spec(&self) -> String {
        self.spec.clone()
    }

    pub fn input_format(&self) -> Option<FrameFormat> {
        self.input_format
    }
}

/// 滤镜图中的一个**节点**：一个 [`Filter`] 加上它在图里的接线。
///
/// [`Filter`] 只描述「滤镜是什么」，接线（这一路输入来自哪个上游、输出叫什么名字）
/// 属于图的结构，放在这里。两者分开的好处是：单输入滤镜可以线性链式拼接（`Vec<Filter>`），
/// 而多输入滤镜（`overlay` / `amix` / `hstack` / `vstack` / `concat`）只能作为节点
/// 显式接线，**无法**被误塞进线性链——那正是过去 `filter::video::overlay()` 公开却
/// 建不出图的根源。
///
/// 输出侧同理：单输出滤镜用 [`with_label`](Self::with_label)，多输出 pad 的滤镜
/// （`split` / `asplit`，见 [`video::split`] / [`audio::asplit`]）用
/// [`with_outputs`](Self::with_outputs) 逐个 pad 标名，每个名字各接一条下游链路
/// ——这是图中唯一合法的 fan-out 方式（同一个标签被消费两次仍会被拒）。
///
/// ```
/// use rsmedia::{filter, MediaType};
///
/// # fn main() -> rsmedia::Result<()> {
/// // 线性节点：不写接线，自动接在链尾
/// let node = filter::FilterNode::new(filter::video::scale(1280, 720, None)?);
///
/// // 多输入节点：显式指定两路来源（图输入标签或前序节点的输出标签）
/// let node = filter::FilterNode::new(filter::video::overlay("10", "10")?)
///     .with_inputs(["base", "logo"])
///     .with_label("composed");
///
/// // 多输出节点：一路输入复制成两路（fan-out 必须先 split，再分别接下游）
/// let node = filter::FilterNode::new(filter::video::split(2))
///     .with_inputs(["in0"])
///     .with_outputs(["copy_a", "copy_b"]);
/// # let _ = (node, MediaType::VIDEO);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FilterNode {
    filter: Filter,
    /// 各路输入分别接到哪个上游标签，顺序即滤镜的输入 pad 顺序。
    /// 空 = 自动接线（接在链尾；链首接图的首个输入）。
    inputs: Vec<String>,
    /// 各输出 pad 的标签，顺序即滤镜的输出 pad 顺序；空 = 由构建器自动分配一个
    /// `n{序号}`（自动分配**只给一个**标签：单输出 pad 的滤镜正好够用；多输出滤镜
    /// 不标注的话，静态 pad 会在建图时因个数不符报 `InvalidConfig`，而动态 pad 的
    /// `split`/`asplit` 会安静地只生成 1 路输出）。
    outputs: Vec<String>,
}

impl FilterNode {
    /// 用单个滤镜建一个节点（不写接线，自动接在链尾）。
    pub fn new(filter: impl Into<Filter>) -> Self {
        Self {
            filter: filter.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// 绑定本节点的输入到指定的上游标签，顺序即滤镜的输入 pad 顺序。
    ///
    /// 上游可以是图输入（[`FilterGraphBuilder::add_input_with`] 声明的标签）或前序
    /// 节点的输出（[`FilterNode::with_label`]）。**静态**输入 pad 的滤镜必须逐个指全：
    /// 只给部分标签会在建图时报 [`RsmediaError::InvalidConfig`]，不会静默接错。
    /// 动态输入 pad 的滤镜（`hstack`/`amix`/`concat`…）不校验个数，路数是否与滤镜
    /// 自己的 `inputs=` / `n=` 选项一致由 FFmpeg 在 `config()` 阶段核对。
    pub fn with_inputs<I, S>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs = inputs.into_iter().map(Into::into).collect();
        self
    }

    /// 给本节点的输出打标签，供下游节点或图输出引用。
    ///
    /// 标签是滤镜图内的接线标识，只能包含 ASCII 字母、数字与下划线。不设置时
    /// [`FilterGraphBuilder::build`] 会按节点序号自动分配 `n0`、`n1`…，所以即便
    /// 不写标签也能被 [`FilterGraphBuilder::add_output`] 引用（写 `n0` 即可），只是
    /// 这种标签会随节点增删而移位；要稳定引用就显式写一个。
    ///
    /// 这是单输出滤镜的简写；多输出 pad 的滤镜（`split` / `asplit`）请改用
    /// [`with_outputs`](Self::with_outputs) 逐个 pad 标注，否则未被标注的 pad
    /// 会以「有 pad 没有下游」的协商错误暴露出来。
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.outputs = vec![label.into()];
        self
    }

    /// 逐个输出 pad 绑定标签，顺序即滤镜的输出 pad 顺序。
    ///
    /// 用于多输出 pad 的滤镜（`split` / `asplit`）把一路输入复制成多路：每个标签
    /// 各接一条下游链路或声明为一个图输出。标签个数必须与滤镜的输出 pad 数一致，
    /// 否则 [`FilterGraphBuilder::build`] 报 [`RsmediaError::InvalidConfig`]；
    /// **动态**输出 pad 的滤镜（`split`/`asplit`，即带 `AVFILTER_FLAG_DYNAMIC_OUTPUTS`
    /// 的那些）不参与这个校验——它们的 pad 数正是由这里的标签个数决定的。
    ///
    /// ```no_run
    /// # use rsmedia::filter::{self, FilterGraphBuilder, FilterNode, VideoEndpoint};
    /// # use rsmedia::Rational;
    /// # use rsmedia::PixelFormat;
    /// # fn main() -> rsmedia::Result<()> {
    /// # let endpoint = VideoEndpoint::new(320, 240, PixelFormat::YUV420P)
    /// #     .with_time_base(Rational::new(1, 25).unwrap())
    /// #     .with_frame_rate(Rational::new(25, 1).unwrap());
    /// let mut builder = FilterGraphBuilder::new();
    /// builder.add_input_with("src", endpoint);
    /// // 一路输入复制成两路：一路原样输出、一路水平翻转后输出。
    /// builder.add_node(
    ///     FilterNode::new(filter::video::split(2))
    ///         .with_inputs(["src"])
    ///         .with_outputs(["copy_a", "copy_b"]),
    /// );
    /// builder.add_node(
    ///     FilterNode::new(filter::video::hflip())
    ///         .with_inputs(["copy_b"])
    ///         .with_label("mirrored"),
    /// );
    /// builder.add_output("copy_a", endpoint);
    /// builder.add_output("mirrored", endpoint);
    /// let graph = builder.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_outputs<I, S>(mut self, outputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outputs = outputs.into_iter().map(Into::into).collect();
        self
    }

    /// 本节点承载的滤镜。
    pub fn filter(&self) -> &Filter {
        &self.filter
    }

    /// 本节点绑定的输入标签（自动接线时为空切片）。
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// 本节点的**第一个**输出标签（未设置时为 `None`）。
    ///
    /// 多输出节点请用 [`outputs`](Self::outputs) 取全部标签。
    pub fn label(&self) -> Option<&str> {
        self.outputs.first().map(String::as_str)
    }

    /// 本节点各输出 pad 的标签（自动接线时为空切片）。
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }
}

impl From<Filter> for FilterNode {
    fn from(filter: Filter) -> Self {
        Self::new(filter)
    }
}

/// Whether the named FFmpeg filter exists in this build
/// （如 `drawtext` 依赖 libfreetype、`subtitles` 依赖 libass）
/// 用于**前置**跳过不可用滤镜，避免依赖 FFmpeg 运行时错误字符串来判断
pub fn get_by_name(name: &str) -> Result<Option<AVFilterRef<'static>>> {
    let filter_name = strutils::str_to_cstring(name)?;
    Ok(AVFilter::get_by_name(&filter_name))
}

/// 第一级（**选项级**）要转义的字符：选项分隔符 `:` `=`、选项值的括号 `{` `}`，以及
/// 全部图级分隔符。
///
/// 之所以把图级分隔符也算进来，是因为第一级还要单独用于**引号内**的值（见
/// [`escape_option_level`]）：那种位置没有第二级可选，多转义几个字符是无害的
/// （第二遍 unescape 会把它们还原），漏转义则会让值被拆开。
const OPTION_SPECIALS: &str = "\\':,[]={};";

/// 第二级（**图级**）要转义的字符：图级分隔符与链路括号。
const GRAPH_SPECIALS: &str = "\\'[],;";

/// 走 FFmpeg 自己的 `av_escape`（BACKSLASH 模式，不带 `AV_ESCAPE_FLAG_STRICT`）。
///
/// `av_escape` 只有两种失败，两种都如实报出，**没有任何降级**：
///
/// * 入参含内部 NUL —— 它无法成为 C 字符串，也不可能是滤镜描述的一部分；
///   [`NulError`](std::ffi::NulError) 经 `From` 变成 [`RsmediaError::InvalidConfig`]，
///   错误消息会指出是 NUL 的问题。
/// * 分配失败（`av_escape` 返回负值）—— 带上 `AVERROR` 变体与上下文报出。
///
/// 以前这里失败时会 `input.replace('\0', "")` 然后原样放行：滤镜描述会悄悄少几个字符
/// 却照常建图成功，调用方完全看不出自己给的值没被用上。这就是"掩盖问题"。
fn escape_backslash(input: &str, specials: &str, all_whitespace: bool) -> Result<String> {
    let c_input = CString::new(input)?;
    // `specials` 是 crate 里的常量，NUL 是编译期就能排除的；这里不是降级，是不变量。
    let c_specials = CString::new(specials).expect("a crate constant cannot contain a NUL byte");
    let flags = if all_whitespace {
        ffi::AV_ESCAPE_FLAG_WHITESPACE as i32
    } else {
        0
    };

    let mut escaped_ptr = std::ptr::null_mut();
    // SAFETY: 两个入参都是本函数内构造、在调用期间一直有效的 NUL 结尾 C 字符串；`escaped_ptr`
    // 是可写的本地变量，成功后由 FFmpeg 填入一块 `av_malloc` 的缓冲（本函数随后释放）。
    let ret = unsafe {
        ffi::av_escape(
            &mut escaped_ptr,
            c_input.as_ptr(),
            c_specials.as_ptr(),
            ffi::AV_ESCAPE_MODE_BACKSLASH,
            flags,
        )
    };
    if ret < 0 {
        return Err(RsmediaError::av_error(ret).with_context("Failed to escape a filter value"));
    }
    if escaped_ptr.is_null() {
        // 成功却拿不到缓冲说明不变量被破坏（`av_escape` 成功时必然写回一块 `av_malloc`
        // 的内存）—— 如实报出，而不是当成"未转义"继续往下走。
        return Err(RsmediaError::msg(
            "av_escape reported success but returned no buffer",
        ));
    }

    // SAFETY: 成功路径上 `escaped_ptr` 是 `av_escape` 写入的 NUL 结尾 C 字符串。
    let escaped = unsafe { CStr::from_ptr(escaped_ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `escaped_ptr` 由 `av_escape` 用 `av_malloc` 分配，必须用 `av_free` 释放。
    unsafe { ffi::av_free(escaped_ptr as *mut _) };

    Ok(escaped)
}

/// 第一级：把一个值写进**选项值**位置。
///
/// 两种调用方式：
/// * 作为 [`escape_filter_value`] 的第一步（未经引号的值）；
/// * **单独**用于外层已经有 `'` 引号的值（`drawtext` 的 `text='…'` / `fontfile='…'`）：
///   `av_get_token` 在引号内原样拷贝、不处理反斜杠，所以那种位置只需要一级 —— 恰好
///   一个 `\` 会被第二遍 unescape 吃掉。这也是唯一"只转一级"合法的场合。
///
/// # Errors
///
/// 见 [`escape_backslash`]：值含 NUL 字节时报 [`RsmediaError::InvalidConfig`]，
/// `av_escape` 自身失败时报对应的 [`RsmediaError`]。
fn escape_option_level(input: &str) -> Result<String> {
    escape_backslash(input, OPTION_SPECIALS, true)
}

/// 第二级：把第一级的输出再护一层，供**未经引号**直接写进滤镜描述的值使用。
///
/// 它的作用不是"再转义一批新字符"（图级字符在第一级里已经转过了），而是把第一级留下的
/// 反斜杠**翻倍**，让它们在 FFmpeg 的第一遍 unescape 之后仍然存在。
///
/// # Errors
///
/// 见 [`escape_backslash`]。
fn escape_graph_level(input: &str) -> Result<String> {
    escape_backslash(input, GRAPH_SPECIALS, false)
}

/// 把一个调用方提供的值安全地写进滤镜描述（值外层**不加引号**时使用）：依次做
/// **选项级**（[`escape_option_level`]）与**图级**（[`escape_graph_level`]）转义。
///
/// # 为什么是两级
///
/// FFmpeg 对滤镜描述做两次 unescape，顺序与转义相反：
///
/// 1. 图级 —— `graphparser.c` 的 `av_get_token(filter, "[],;")` 拆滤镜与链路；
/// 2. 选项级 —— `avfilter.c::ff_filter_opt_parse` → `av_opt_get_key_value(&args, "=", ":")`
///    → `av_get_token(opts, ":")` 拆选项。
///
/// 所以判据不是"哪些字符被转义了"，而是**每个字符最后留下几个反斜杠**（`movie=<path>`
/// 的报错会回显解析后的路径，用它实测过）：
///
/// | 输入字符 | 第一级后 | 最终 | 依据 |
/// |---|---|---|---|
/// | `,` `;` `[` `]` | 1 个 | **3 个** | 第 1 遍后必须落到"第 2 遍不当作分隔符"的状态；实测 2 个时 `,` 会在第 1 遍变成裸分隔符，图被拆开（`No such filter: 'b.mp4'`） |
/// | `:` `=` `{` `}` | 1 个 | **2 个** | 第 1 遍后必须**仍是**转义态，否则第 2 遍把 `:` 当分隔符、值被截断（实测 1 个时 `/tmp/a:b.mp4` 变成 `/tmp/a`） |
/// | `'` | 1 个 | **3 个** | 同图级：第 1 遍后必须还是 `\'`，否则 `av_get_token` 会进入引号模式把后面整段吞掉 |
/// | `\` | 2 个 | **4 个** | 每级都把自己翻倍 |
/// | 空白 | 1 个 | **2 个** | 第 1 遍后必须仍是 `\ ` |
///
/// # Errors
///
/// 见 [`escape_backslash`]。值里的 NUL 报 [`RsmediaError::InvalidConfig`] —— 滤镜描述
/// 本来就是 C 字符串，NUL 无法表示，也没有"丢掉几个字符继续"的余地。
fn escape_filter_value(input: &str) -> Result<String> {
    escape_graph_level(&escape_option_level(input)?)
}

/// A filter option that FFmpeg **evaluates**, i.e. one declared `<string>` in
/// `ffmpeg -h filter=<name>` rather than `<int>`, `<float>` or `<double>`.
///
/// Several filters take options that are not plain numbers: `eq`'s `brightness`
/// and `contrast`, `volume`'s `volume`, `fps`'s `fps` are all `<string>`
/// *because* FFmpeg evaluates them. They accept expression syntax
/// (`"sin(t)"`, `"iw/2"`, `"min(cw/2,ch/2)"`), alternative units
/// (`volume` takes `"-6dB"`), exact rationals (`fps` takes `"30000/1001"`
/// where a float cannot represent the rate), and — with `eval=frame` — a
/// different value on every frame.
///
/// Typing such an option as `f32` silently restricts the wrapper to constants,
/// which is a loss of capability rather than a style choice. [`Expr`] keeps both
/// spellings reachable through a single parameter type:
///
/// ```
/// use rsmedia::filter::{audio, video};
///
/// // A constant still reads like a constant.
/// let _ = video::eq(0.1, 1.2);
///
/// // ...and the expression form is reachable without dropping to raw strings.
/// let _ = video::eq("sin(t)", "1 + 0.2*sin(t)");
///
/// // `volume` additionally accepts dB, which no float can spell.
/// let _ = audio::volume("-6dB");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Expr<'a> {
    /// A constant value, written into the filter description as a decimal.
    Const(f64),
    /// An FFmpeg expression, passed through verbatim apart from the usual filter
    /// escaping (so `"if(lt(val,100),val,val+20)"` can be written as-is).
    Expression(&'a str),
}

impl Expr<'_> {
    /// The option value as it must appear in a filter description.
    ///
    /// # Errors
    ///
    /// [`Expr::Const`] never fails. [`Expr::Expression`] goes through
    /// the filter-value escaping path, so it reports a [`RsmediaError::InvalidConfig`]
    /// when the expression contains a NUL byte — a filter description is a C string and
    /// cannot carry one.
    ///
    /// This replaces the `Display` impl this type used to have: rendering an option value
    /// is fallible now, and `Display` has no way to report that (its only failure exit is
    /// `fmt::Error`, which makes `format!` panic). Constructors call this method and
    /// propagate the error instead.
    pub fn to_filter_value(self) -> Result<String> {
        match self {
            Self::Const(value) => Ok(format!("{value}")),
            Self::Expression(expr) => escape_filter_value(expr),
        }
    }
}

impl From<f64> for Expr<'_> {
    fn from(value: f64) -> Self {
        Self::Const(value)
    }
}

impl From<i32> for Expr<'_> {
    fn from(value: i32) -> Self {
        Self::Const(f64::from(value))
    }
}

impl<'a> From<&'a str> for Expr<'a> {
    fn from(expr: &'a str) -> Self {
        Self::Expression(expr)
    }
}

/// The literal FFmpeg's `<duration>` options take — e.g. `afade`'s `st`/`d`,
/// `trim`'s `start`/`end`, `fade`'s `start_time`.
///
/// A bare number in a `<duration>` option means **seconds**: `atrim=start=0.5`
/// trims half a second, not half a microsecond (verified against FFmpeg 9.0).
/// The exact seconds value is therefore the most faithful spelling, and
/// `f64`'s shortest-round-trip formatting keeps it readable (`1.5`, not
/// `1.5000000000000002`).
fn duration_literal(duration: Duration) -> String {
    format!("{}", duration.as_secs_f64())
}

/// Render a filter's numeric option value.
///
/// Filter options are plain decimals by the time they reach FFmpeg, so the
/// question is only which Rust type to accept. These parameters are typed
/// `impl Into<f64>` for two reasons, in this order:
///
/// * **No boilerplate.** Every spelling a caller already has keeps working
///   unchanged — an integer literal (`highpass(80)`), a float literal
///   (`nlmeans(1.0)`), an `f32` value from elsewhere in a pipeline, and an
///   `f64` computation. A plain `f64` parameter would force `.0` onto every
///   integer call site, which is exactly the "make the caller write
///   boilerplate" cost this crate avoids elsewhere.
/// * **No narrowing.** FFmpeg declares most of these `<double>` (and the rest
///   `<float>`/`<int>`); an `f32` or `i32` parameter would reject values FFmpeg
///   accepts — fractional dB, fractional gain, fractional frequency (31.5 Hz is
///   a standard 1/3-octave centre). `f64` is the narrowest type that contains all
///   three declarations.
///
/// Options FFmpeg declares `<float>` are still `f32` inside FFmpeg; a wider
/// value is rounded there, exactly as it would be from the `ffmpeg` CLI.
fn number(value: impl Into<f64>) -> String {
    format!("{}", value.into())
}

/// 转义文本，但保留 FFmpeg 的 `%{...}` 展开块（如 `%{localtime}`、`%{pts:hms}`）。
///
/// 用于 `drawtext` 等需要显示动态时间/帧号的场景，避免 `{` `}` 被转义后无法展开。
/// 转义文本，但保留 FFmpeg 的 `%{...}` 展开块（如 `%{localtime}`、`%{pts:hms}`）。
///
/// 用于 `drawtext` 等需要显示动态时间/帧号的场景，避免 `{` `}` 被转义后无法展开。
///
/// # Errors
///
/// 见 [`escape_backslash`]。
fn escape_filter_expr(input: &str) -> Result<String> {
    let mut result = String::new();
    let mut rest = input;
    while let Some(pos) = rest.find("%{") {
        // 转义 `%{` 之前的部分
        result.push_str(&escape_option_level(&rest[..pos])?);
        // 找到匹配的 `}`，整体保留
        if let Some(end_rel) = rest[pos..].find('}') {
            result.push_str(&rest[pos..pos + end_rel + 1]);
            rest = &rest[pos + end_rel + 1..];
        } else {
            result.push_str(&escape_option_level(&rest[pos..])?);
            rest = "";
        }
    }
    result.push_str(&escape_option_level(rest)?);
    Ok(result)
}

/// 校验一个**闭集**滤镜选项：值必须是 `names` 之一，或 `0..=max` 里的整数。
///
/// FFmpeg 这类选项在绑定里是 `<int>` 加一组具名常量 —— `ffmpeg -h filter=yadif`
/// 里 `mode` 声明 `from 0 to 3`，同时给四个值起了 `send_frame`、`send_field`…
/// 的名字。**名字和数字都合法**，只按名字收会把 FFmpeg 认的值挡在门外，只按整数
/// 收同理，所以两边都要收。
///
/// 不做校验的话，拼错的名字会被原样写进滤镜串，直到建图才变成一句 FFmpeg 的解析
/// 错误（"Unable to parse \"mode\" option value \"…\""），既没有本 crate 的上下文、
/// 也定位不到是哪个参数；在这里前置拦下，是 `InvalidConfig` 且点名了候选值。
///
/// `max` 是 FFmpeg 声明的上界（含），取自 `ffmpeg -h filter=<name>` 的范围。
fn check_closed_set(
    filter: &str,
    option: &str,
    value: &str,
    names: &[&str],
    max: i32,
) -> Result<()> {
    let in_numeric_range = value.parse::<i32>().is_ok_and(|n| (0..=max).contains(&n));
    if in_numeric_range || names.contains(&value) {
        return Ok(());
    }
    let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
    Err(RsmediaError::invalid_config(format!(
        "{filter} {option} must be one of {} (or an integer in 0..={max}), got '{value}'",
        quoted.join(", ")
    )))
}

/// `amix` 的 `duration`（如何判定流结束）只认三个档位。
///
/// 单独抽出来是因为它有两个入口 —— [`audio::amix`] 与 [`FilterGraphBuilder::amix`]
/// —— 之前只有后者校验，于是同一个概念在一个入口拼错立刻报错、在另一个入口静默写进
/// 滤镜串直到建图才炸。
fn check_amix_duration(duration: &str) -> Result<()> {
    check_closed_set(
        "amix",
        "duration",
        duration,
        &["longest", "shortest", "first"],
        2,
    )
}

pub mod video {
    use super::*;

    /// Scales video dimensions.
    ///
    /// # Arguments
    ///
    /// * `width`: Target width.
    /// * `height`: Target height.
    /// * `flags`: Optional `SWS_FLAG_*` string, specifying the scaling algorithm and other options. Default is "bicubic".
    ///   Possible values for scaling algorithm flags:
    ///     - `fast_bilinear`: Select fast bilinear scaling algorithm.
    ///     - `bilinear`: Select bilinear scaling algorithm.
    ///     - `bicubic`: Select bicubic scaling algorithm (default).
    ///     - `experimental`: Select experimental scaling algorithm.
    ///     - `neighbor`: Select nearest neighbor rescaling algorithm.
    ///     - `area`: Select averaging area rescaling algorithm.
    ///     - `bicublin`: Select bicubic scaling algorithm for the luma component, bilinear for chroma components.
    ///     - `gauss`: Select Gaussian rescaling algorithm.
    ///     - `sinc`: Select sinc rescaling algorithm.
    ///     - `lanczos`: Select Lanczos rescaling algorithm. The default width (alpha) is 3 and can be changed by setting param0.
    ///     - `spline`: Select natural bicubic spline rescaling algorithm.
    ///       Other possible flags:
    ///     - `print_info`: Enable printing/debug logging.
    ///     - `accurate_rnd`: Enable accurate rounding.
    ///     - `full_chroma_int`: Enable full chroma interpolation.
    ///     - `full_chroma_inp`: Select full chroma input.
    ///     - `bitexact`: Enable bitexact output.
    ///
    /// See: <https://ffmpeg.org/ffmpeg-scaler.html#Scaler-Options>
    /// # Errors
    ///
    /// [`RsmediaError::InvalidConfig`] when `flags` contains a NUL byte, or whatever
    /// Anything `av_escape` fails with is propagated as well.
    pub fn scale<'a>(width: u32, height: u32, flags: impl Into<Option<&'a str>>) -> Result<Filter> {
        let flags: Option<&str> = flags.into();
        // 默认与 FFmpeg `scale` 滤镜一致，也与本 crate 的 `Scaler::default()`
        // 一致（BICUBIC）；早先这里是 `fast_bilinear`，与上方文档矛盾。
        let flags_str = escape_filter_value(flags.unwrap_or("bicubic"))?;

        Ok(Filter::new(
            "scale",
            MediaType::VIDEO,
            format!("scale=w={width}:h={height}:flags={flags_str}"),
        ))
    }

    /// Converts video pixel format.
    /// `format`: <https://ffmpeg.org/ffmpeg-filters.html#format>
    /// `aformat`: <https://ffmpeg.org/ffmpeg-filters.html#aformat-1>
    pub fn format(format: PixelFormat) -> Filter {
        Filter::new(
            "format",
            MediaType::VIDEO,
            format!("format=pix_fmts={}", format.get_pix_fmt_name()),
        )
    }

    /// Crops video to a specified rectangle.
    /// `x` and `y` can be negative but runtime validation against input frame is better.
    /// `w` and `h` must be positive.
    pub fn crop(x: i32, y: i32, w: u32, h: u32) -> Filter {
        Filter::new(
            "crop",
            MediaType::VIDEO,
            format!("crop=x={x}:y={y}:w={w}:h={h}"),
        )
    }

    /// 在视频上绘制文字的 Builder，对应 FFmpeg `drawtext` 滤镜。
    ///
    /// `fontfile` 可选；不指定时用**相对路径** `fonts/Arial.ttf`（避免依赖 system
    /// fontconfig，例如 Windows 等没有 fontconfig 配置的平台会崩溃）。相对路径
    /// 按**进程当前工作目录**解析：只有工作目录恰好在仓库根目录时才找得到，
    /// 因此生产代码请总是用 [`DrawText::fontfile`] 传绝对路径（如用
    /// `env!("CARGO_MANIFEST_DIR")` 拼出字体路径）。路径不存在时滤镜图初始化会失败
    /// （`drawtext` 报找不到字体文件），不会 panic。
    /// 也支持给文字加描边盒子（`boxed`）。
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::filter::video::DrawText;
    /// let f = DrawText::new("Hello", 10, 10, 24, "white")
    ///     .fontfile("/path/to/your-font.ttf")
    ///     .boxed("black@0.5")
    ///     .build();
    /// ```
    pub struct DrawText {
        text: String,
        x: i32,
        y: i32,
        fontfile: Option<String>,
        fontsize: u32,
        fontcolor: String,
        box_enabled: bool,
        box_color: String,
        box_border_w: u32,
        /// `text` holds an FFmpeg expression (`%{localtime}`, `%{frame_num}`, …)
        /// rather than a literal string; `%{...}` must survive escaping intact.
        text_is_expression: bool,
    }

    impl DrawText {
        /// 创建文字水印。
        ///
        /// * `text` - 要绘制的文本。
        /// * `x` / `y` - 文字左上角坐标。
        /// * `fontsize` - 字号。
        /// * `fontcolor` - 文字颜色（如 `"white"`、`"white@0.5"`）。
        pub fn new(text: &str, x: i32, y: i32, fontsize: u32, fontcolor: &str) -> Self {
            Self {
                text: text.to_string(),
                x,
                y,
                fontfile: None,
                fontsize,
                fontcolor: fontcolor.to_string(),
                box_enabled: false,
                box_color: "black@0.5".to_string(),
                box_border_w: 0,
                text_is_expression: false,
            }
        }

        /// 指定字体文件路径。
        pub fn fontfile(mut self, path: &str) -> Self {
            self.fontfile = Some(path.to_string());
            self
        }

        /// 开启文字背景盒子（描边效果）。
        pub fn boxed(mut self, color: &str) -> Self {
            self.box_enabled = true;
            self.box_color = color.to_string();
            self
        }

        /// 使用 FFmpeg 文本展开表达式显示动态内容（如当前时间、帧号）。
        ///
        /// 常用表达式：`%{localtime}`（本地时间）、`%{pts:hms}`（时间戳时分秒）、
        /// `%{frame_num}`（帧号）。表达式中的 `%{...}` 不会被转义。
        ///
        /// # Examples
        ///
        /// 在右上角显示当前时间：
        /// ```
        /// use rsmedia::filter::video::DrawText;
        /// let f = DrawText::new("", 0, 0, 24, "white")
        ///     .time_text("%{localtime}")
        ///     .build();
        /// ```
        pub fn time_text(mut self, fmt: &str) -> Self {
            self.text = fmt.to_string();
            self.text_is_expression = true;
            self
        }

        /// 生成最终的 [`Filter`]。
        /// # Errors
        ///
        /// A NUL byte in any of the values is [`RsmediaError::InvalidConfig`]: a filter
        /// description is a C string and cannot carry one.
        pub fn build(self) -> Result<Filter> {
            let text_spec = if self.text_is_expression {
                escape_filter_expr(&self.text)?
            } else {
                escape_option_level(&self.text)?
            };
            let mut spec = format!(
                "drawtext=text='{}':x={}:y={}:fontsize={}:fontcolor={}",
                text_spec,
                self.x,
                self.y,
                self.fontsize,
                escape_filter_value(&self.fontcolor)?
            );
            // 缺省使用项目自带字体，避免依赖 system fontconfig（Windows 等平台没有
            // fontconfig 配置会在查字体时崩溃）；用户显式指定字体时优先用用户的。
            // 注意 `fonts/Arial.ttf` 是相对路径，按进程当前工作目录解析（见结构体文档）。
            let fontfile = self
                .fontfile
                .unwrap_or_else(|| "fonts/Arial.ttf".to_string());
            spec.push_str(&format!(":fontfile='{}'", escape_option_level(&fontfile)?));
            if self.box_enabled {
                spec.push_str(&format!(
                    ":box=1:boxcolor={}:boxborderw={}",
                    escape_filter_value(&self.box_color)?,
                    self.box_border_w
                ));
            }
            Ok(Filter::new("drawtext", MediaType::VIDEO, spec))
        }
    }

    /// 画矩形框
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn drawbox(x: i32, y: i32, w: u32, h: u32, color: &str, thickness: i32) -> Result<Filter> {
        if thickness < 0 {
            // FFmpeg 't=fill' is also possible
            tracing::warn!("Box thickness is negative ({thickness}), using absolute value.",);
        }
        let color = escape_filter_value(color)?;
        Ok(Filter::new(
            "drawbox",
            MediaType::VIDEO,
            format!(
                "drawbox=x={}:y={}:w={}:h={}:color={}:t={}",
                x,
                y,
                w,
                h,
                color,
                thickness.abs()
            ),
        ))
    }

    /// 去除水印
    ///
    /// # Arguments
    ///
    /// `x` and `y` are the top-left corner of the logo.
    /// `w` and `h` are the width and height of the logo.
    /// See: <https://ffmpeg.org/ffmpeg-filters.html#delogo>
    pub fn delogo(x: i32, y: i32, w: u32, h: u32) -> Filter {
        Filter::new(
            "delogo",
            MediaType::VIDEO,
            format!("delogo=x={x}:y={y}:w={w}:h={h}"),
        )
    }

    /// 去除水印（delogo）的 Builder：支持**多个区域**串联及可选参数。
    ///
    /// FFmpeg 的 `delogo` 通过周围像素插值填补指定矩形区域，适合去除固定位置的
    /// 简单文字/logo 水印。对半透明、异形或复杂背景的水印效果有限。
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rsmedia::filter::video::Delogo;
    /// // 去除两处水印并开启调试可视化
    /// let filters = Delogo::new()
    ///     .add_region(10, 10, 120, 30)
    ///     .add_region(640, 10, 120, 30)
    ///     .show()
    ///     .build();
    /// ```
    pub struct Delogo {
        regions: Vec<(i32, i32, u32, u32)>,
        band: Option<i32>,
        show: bool,
    }

    impl Default for Delogo {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Delogo {
        /// 创建一个空的水印去除器（需通过 [`add_region`](Delogo::add_region) 添加区域）。
        pub fn new() -> Self {
            Self {
                regions: Vec::new(),
                band: None,
                show: false,
            }
        }

        /// 添加一个待去除的水印区域（左上角坐标 + 宽高）。
        pub fn add_region(mut self, x: i32, y: i32, w: u32, h: u32) -> Self {
            self.regions.push((x, y, w, h));
            self
        }

        /// 扫描带宽度（插值取样宽度）。
        ///
        /// **版本差异**：`band` 选项仅 FFmpeg ≤ 7 提供（旧名 `t`），FFmpeg 8+
        /// 已从 `delogo` 移除。因此默认**不输出**该选项（保证在新版可用），
        /// 只有显式调用本方法时才会写入 —— 在 FFmpeg 8+ 上调用会得到
        /// FFmpeg 自己的 "Option not found" 错误。
        pub fn band(mut self, band: i32) -> Self {
            self.band = Some(band);
            self
        }

        /// 显示去除区域（调试用），将待去除区域标记出来再输出。
        pub fn show(mut self) -> Self {
            self.show = true;
            self
        }

        /// 生成滤镜列表，每个水印区域对应一个 `delogo` 滤镜（按顺序串联）。
        pub fn build(self) -> Vec<Filter> {
            self.regions
                .into_iter()
                .map(|(x, y, w, h)| {
                    let mut params = format!("delogo=x={x}:y={y}:w={w}:h={h}");
                    if let Some(band) = self.band {
                        params.push_str(&format!(":band={band}"));
                    }
                    if self.show {
                        params.push_str(":show=1");
                    }
                    Filter::new("delogo", MediaType::VIDEO, params)
                })
                .collect()
        }
    }

    /// zoompan - 平移和缩放效果
    ///
    /// `zoom`/`x`/`y` 均为 FFmpeg 表达式（如 `"1.5"`、`"iw/2-(iw/zoom/2)"`），
    /// 内部会做选项级 + 图级转义。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn zoompan(
        zoom: &str,
        x: &str,
        y: &str,
        duration: impl Into<Option<i32>>,
    ) -> Result<Filter> {
        let duration: Option<i32> = duration.into();
        let (zoom, x, y) = (
            escape_filter_value(zoom)?,
            escape_filter_value(x)?,
            escape_filter_value(y)?,
        );
        let mut params = format!("zoompan=z={zoom}:x={x}:y={y}");
        if let Some(d) = duration {
            params.push_str(&format!(":d={d}"));
        }
        Ok(Filter::new("zoompan", MediaType::VIDEO, params))
    }

    /// Rotates by a multiple of 90°, optionally with a vertical flip.
    ///
    /// No interpolation is involved, so this is much cheaper than [`rotate`] —
    /// but it can only express the four quarter turns.
    ///
    /// | `mode` | direction                | effect                                     |
    /// |--------|--------------------------|--------------------------------------------|
    /// | `0`    | `cclock_flip`            | 90° counter-clockwise, then flip vertically |
    /// | `1`    | `clock`                  | 90° clockwise                              |
    /// | `2`    | `cclock`                 | 90° counter-clockwise                      |
    /// | `3`    | `clock_flip`             | 90° clockwise, then flip vertically        |
    ///
    /// # Why 4..=7 is rejected
    ///
    /// FFmpeg's `dir` option declares the range `0..=7`, so it *accepts* 4..=7
    /// without complaint — and then passes every frame through untouched.
    /// `transpose=5` is byte-for-byte identical to applying no filter at all
    /// (verified with `ffmpeg -vf transpose=N` on FFmpeg 9.0). Handing that back
    /// as a working filter would turn "the video silently did not rotate" into a
    /// value this crate appears to vouch for, so those modes are rejected here.
    ///
    /// See: <https://ffmpeg.org/ffmpeg-filters.html#transpose-1>
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `mode` is outside `0..=3`.
    pub fn transpose(mode: i32) -> Result<Filter> {
        if !(0..=3).contains(&mode) {
            return Err(RsmediaError::invalid_config(format!(
                "transpose mode must be in 0..=3 (0=cclock_flip, 1=clock, 2=cclock, 3=clock_flip), \
                 got {mode}: FFmpeg accepts 4..=7 but passes frames through unchanged"
            )));
        }
        Ok(Filter::new(
            "transpose",
            MediaType::VIDEO,
            format!("transpose={mode}"),
        ))
    }

    /// rotate - 任意角度旋转滤镜（支持动画表达式）
    ///
    /// `angle` 为**角度**（度），内部转成 FFmpeg `rotate` 需要的弧度表达式
    /// (`{angle}*PI/180`)；`rotate` 的选项本身是表达式，因此也可以直接
    /// 用 `Filter::new("rotate", ...)` 写 `PI/4` 之类的弧度表达式。
    /// 注意：性能较低，可能有插值模糊；用于精准旋转或动态旋转场景。
    pub fn rotate(angle: i32) -> Filter {
        // FFmpeg 的 rotate 角度以弧度为单位，这里把调用者给的度数换算过去。
        Filter::new("rotate", MediaType::VIDEO, format!("rotate={angle}*PI/180"))
    }

    /// Flips video horizontally.
    pub fn hflip() -> Filter {
        Filter::new("hflip", MediaType::VIDEO, "hflip".to_string())
    }
    /// Flips video vertically.
    pub fn vflip() -> Filter {
        Filter::new("vflip", MediaType::VIDEO, "vflip".to_string())
    }

    /// 视频淡入淡出
    /// Fades video in from the start.
    /// `duration_frames`: Fade duration in number of frames.
    pub fn fade_in(duration_frames: u32) -> Filter {
        Filter::new(
            "fade",
            MediaType::VIDEO,
            format!("fade=t=in:start_frame=0:nb_frames={duration_frames}"),
        )
    }

    /// Fades video out.
    /// `start_frame`: Frame number to start the fade out.
    /// `duration_frames`: Fade duration in number of frames.
    pub fn fade_out(start_frame: u32, duration_frames: u32) -> Filter {
        Filter::new(
            "fade",
            MediaType::VIDEO,
            format!("fade=t=out:start_frame={start_frame}:nb_frames={duration_frames}"),
        )
    }

    /// 视频锐化
    /// Applies unsharp mask filter (default settings).
    pub fn unsharp() -> Filter {
        // Add parameters if needed: lx, ly, la, cx, cy, ca
        Filter::new("unsharp", MediaType::VIDEO, "unsharp".to_string())
    }

    /// 视频模糊
    /// Applies box blur filter.
    /// `radius`: Radius of the luma blur — a constant, or an expression such as
    /// `"min(cw/2,ch/2)"` (`boxblur`'s `luma_radius` is declared `<string>`
    /// because FFmpeg evaluates it).
    /// # Errors
    ///
    /// [`Expr::to_filter_value`] is fallible, so this constructor is too.
    pub fn blur<'a>(radius: impl Into<Expr<'a>>) -> Result<Filter> {
        // Consider adding other boxblur params: luma_power, chroma_radius, chroma_power, alpha_radius, alpha_power
        let radius = radius.into().to_filter_value()?;
        Ok(Filter::new(
            "boxblur",
            MediaType::VIDEO,
            format!("boxblur=luma_radius={radius}"),
        ))
    }

    /// 亮度/对比度调节
    ///
    /// `brightness` / `contrast` are FFmpeg `<string>` options: constants work
    /// (`eq(0.1, 1.2)`), and so do expressions (`eq("sin(t)", "1.2")`), which is
    /// what `eval=frame` needs. See [`Expr`].
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn eq<'a>(
        brightness: impl Into<Expr<'a>>,
        contrast: impl Into<Expr<'a>>,
    ) -> Result<Filter> {
        let brightness = brightness.into().to_filter_value()?;
        let contrast = contrast.into().to_filter_value()?;
        Ok(Filter::new(
            "eq",
            MediaType::VIDEO,
            format!("eq=brightness={brightness}:contrast={contrast}"),
        ))
    }

    /// 帧率控制
    ///
    /// `fps` is a `<string>` option, so besides a constant it accepts the exact
    /// rationals a float cannot represent (`"30000/1001"`) and `"source"`.
    /// A rate that has to be exact should be given as a rational expression —
    /// the same caveat as [`crate::EncoderBuilder::with_fps`].
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn fps<'a>(fps: impl Into<Expr<'a>>) -> Result<Filter> {
        let fps = fps.into().to_filter_value()?;
        Ok(Filter::new("fps", MediaType::VIDEO, format!("fps={fps}")))
    }

    /// 去交错（Deinterlace），将隔行扫描转为逐行扫描。
    /// `mode`: `send_frame`(默认), `send_field`, `send_frame_nospatial`, `send_field_nospatial`.
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `mode` is not one of the four
    /// directions FFmpeg names (`send_frame`, `send_field`, `send_frame_nospatial`,
    /// `send_field_nospatial`) or an integer in `0..=3`, or when a value contains a NUL
    /// byte — a filter description is a C string and cannot carry one. Anything
    /// `av_escape` fails with is propagated too.
    pub fn yadif(mode: &str) -> Result<Filter> {
        check_closed_set(
            "yadif",
            "mode",
            mode,
            &[
                "send_frame",
                "send_field",
                "send_frame_nospatial",
                "send_field_nospatial",
            ],
            3,
        )?;
        let mode = escape_filter_value(mode)?;
        Ok(Filter::new(
            "yadif",
            MediaType::VIDEO,
            format!("yadif=mode={mode}"),
        ))
    }

    /// 补边（Pad），在视频周围添加指定颜色的边。
    ///
    /// 参数顺序 `(x, y, w, h)` 与 [`crop`] / [`delogo`] / [`Delogo::add_region`] 一致 ——
    /// 四者都是"先位置、后尺寸"，顺序反着写（`w, h, x, y`）编译得过但画出来是错的。
    ///
    /// * `x` / `y` - 原视频在输出画布上的偏移。
    /// * `w` / `h` - 输出尺寸（不包含负值表达式）。
    /// * `color` - 填充颜色，如 `"black"`。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn pad(x: i32, y: i32, w: u32, h: u32, color: &str) -> Result<Filter> {
        let color = escape_filter_value(color)?;
        Ok(Filter::new(
            "pad",
            MediaType::VIDEO,
            format!("pad=w={w}:h={h}:x={x}:y={y}:color={color}"),
        ))
    }

    /// 烧录字幕（Subtitles）。
    /// `path`: 字幕文件路径（`srt`/`ass` 等）；路径中的转义字符（如 `,`/`;`/`[]`）
    /// 会被自动转义，调用者传原始路径即可。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn subtitles(path: &str) -> Result<Filter> {
        let escaped = escape_filter_value(path)?;
        Ok(Filter::new(
            "subtitles",
            MediaType::VIDEO,
            format!("subtitles={escaped}"),
        ))
    }

    /// 设置显示宽高比（DAR）。
    /// `ratio` 为宽高比，如 `16`/`9`。
    pub fn setdar(num: i32, den: i32) -> Filter {
        Filter::new("setdar", MediaType::VIDEO, format!("setdar={num}/{den}"))
    }

    /// 设置采样宽高比（SAR）。
    pub fn setsar(num: i32, den: i32) -> Filter {
        Filter::new("setsar", MediaType::VIDEO, format!("setsar={num}/{den}"))
    }

    /// 色相/饱和度/亮度调节。
    /// `hue` 为色相偏移（度，-180 ~ 180）。
    pub fn hue(hue: i32) -> Filter {
        Filter::new("hue", MediaType::VIDEO, format!("hue=h={hue}"))
    }

    /// 反相（负片效果）。
    pub fn negate() -> Filter {
        Filter::new("negate", MediaType::VIDEO, "negate".to_string())
    }

    /// 添加噪点。
    /// `amount` 为噪点强度（0-100）。
    pub fn noise(amount: u32) -> Filter {
        Filter::new("noise", MediaType::VIDEO, format!("noise=alls={amount}"))
    }

    /// 视频降噪（hqdn3d），减少亮度/色度噪声。
    ///
    /// * `luma` - 亮度空间降噪强度（`luma_spatial`，FFmpeg: 0~DBL_MAX，默认 0；
    ///   常用 4）。
    /// * `chroma` - 色度空间降噪强度（`chroma_spatial`，同上，常用 3）。
    ///
    /// `hqdn3d.luma_spatial` / `chroma_spatial` 在 FFmpeg 里是 `<double>`，因此这里
    /// 收 `f64`：包一层 `f32` 会比 FFmpeg 实际接受的域更窄。
    pub fn hqdn3d(luma: impl Into<f64>, chroma: impl Into<f64>) -> Filter {
        Filter::new(
            "hqdn3d",
            MediaType::VIDEO,
            format!(
                "hqdn3d=luma_spatial={}:chroma_spatial={}",
                number(luma),
                number(chroma)
            ),
        )
    }

    /// 视频降噪（nlmeans），非局部均值降噪，降噪效果更好但更耗时。
    /// `strength` 为降噪强度（建议 0-20，默认 1.0）。
    ///
    /// 对应 `nlmeans.s`，FFmpeg 声明为 `<double>`。
    pub fn nlmeans(strength: impl Into<f64>) -> Filter {
        Filter::new(
            "nlmeans",
            MediaType::VIDEO,
            format!("nlmeans=s={}", number(strength)),
        )
    }

    /// 智能模糊 / 磨皮（smartblur）
    ///
    /// 与整体模糊不同，smartblur 只平滑**平坦区域**、保留边缘，因此是证件照
    /// "磨皮"的常用滤镜：`strength` 取小正值（如 `0.05~0.15`）即可抹平细纹
    /// 而不糊掉五官轮廓。
    ///
    /// * `luma_strength` - 亮度平滑强度（-1~1）。**正值 = 平滑/磨皮**，
    ///   负值 = 锐化；证件照建议 `0.05~0.2`。
    /// * `luma_radius` - 平滑半径（0.1~5，默认 1），越大越柔和，证件照建议 `3` 左右。
    ///
    /// 色度/alpha 无需单独设置：FFmpeg 的 `smartblur` **没有** `chroma_mode` 这类选项，
    /// 而是在 `init` 里把低于合法下限的 `chroma_radius` / `chroma_strength` /
    /// `chroma_threshold`（默认值 `-0.9` / `-2` / `-31`，即各自下限减一）直接替换成
    /// 对应的 luma 值——所以不显式指定时色度就是跟随亮度。若要让色度与亮度不同，
    /// 请用 [`Filter::new`] 逃生舱传完整 spec。
    pub fn smartblur(luma_strength: impl Into<f64>, luma_radius: impl Into<f64>) -> Filter {
        Filter::new(
            "smartblur",
            MediaType::VIDEO,
            format!(
                "smartblur=luma_radius={}:luma_strength={}",
                number(luma_radius),
                number(luma_strength)
            ),
        )
    }

    /// Gamma 校正（画质增强）。
    ///
    /// `gamma` 为 gamma 值（通常 0.5-2.0，1.0 表示不变），也可以是表达式。
    ///
    /// **版本差异**：FFmpeg 8+ 移除了独立的 `gamma` 滤镜，该功能并入 `eq`
    /// （`eq=gamma=…`）。为在新版本上可用，这里直接生成 `eq` 滤镜，
    /// 语义与旧 `gamma` 滤镜一致。
    /// # Errors
    ///
    /// See [`Expr::to_filter_value`].
    pub fn gamma<'a>(gamma: impl Into<Expr<'a>>) -> Result<Filter> {
        let gamma = gamma.into().to_filter_value()?;
        Ok(Filter::new(
            "eq",
            MediaType::VIDEO,
            format!("eq=gamma={gamma}"),
        ))
    }

    /// 饱和度调节（画质增强）。
    /// `saturation` 为饱和度倍数（1.0 表示不变，0 为黑白），也可以是表达式。
    /// # Errors
    ///
    /// See [`Expr::to_filter_value`].
    pub fn saturation<'a>(saturation: impl Into<Expr<'a>>) -> Result<Filter> {
        let saturation = saturation.into().to_filter_value()?;
        Ok(Filter::new(
            "eq",
            MediaType::VIDEO,
            format!("eq=saturation={saturation}"),
        ))
    }

    /// 鲜艳度调节（画质增强）。
    /// `vibrance` 为鲜艳度（FFmpeg: -2.0 ~ 2.0，默认 0 表示不变），对应
    /// `vibrance=intensity`。
    pub fn vibrance(vibrance: impl Into<f64>) -> Filter {
        Filter::new(
            "vibrance",
            MediaType::VIDEO,
            format!("vibrance=intensity={}", number(vibrance)),
        )
    }

    /// 去块效应（画质增强），减轻压缩产生的马赛克/块状伪影。
    pub fn deblock() -> Filter {
        Filter::new("deblock", MediaType::VIDEO, "deblock".to_string())
    }

    /// 高斯模糊。
    /// `sigma`: 高斯标准差（越大越模糊，默认 0.5）。
    pub fn gblur(sigma: impl Into<f64>) -> Filter {
        Filter::new(
            "gblur",
            MediaType::VIDEO,
            format!("gblur=sigma={}", number(sigma)),
        )
    }

    /// 平均值模糊（boxblur，参数化版本）。
    /// * `luma_radius` - 亮度模糊半径（像素），可以是表达式，如 `"2"` 或
    ///   `"min(cw/2,ch/2)"`（传**未转义**的表达式，内部会做两层转义）。
    /// * `luma_power` - 亮度模糊强度（1 表示完全平均，2 表示两遍）。
    ///
    /// 注意：`blur(radius)` 是 convenience 版，只设 `luma_radius`；
    /// 这里保留 boxblur 完整参数供精细控制。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn boxblur(luma_radius: &str, luma_power: u32) -> Result<Filter> {
        let luma_radius = escape_filter_value(luma_radius)?;
        Ok(Filter::new(
            "boxblur",
            MediaType::VIDEO,
            format!("boxblur=luma_radius={luma_radius}:luma_power={luma_power}"),
        ))
    }

    /// 叠加（overlay），将一个视频流（overlay）叠加到主视频流上。
    ///
    /// 这是**多输入**滤镜：用 [`FilterGraphBuilder::overlay`] 直接建整图，或
    /// `FilterNode::new(filter::video::overlay(...)).with_inputs([底层, 叠加层])`
    /// 接进自定义的多输入图——线性滤镜链（`DecoderBuilder::with_filters` 等）
    /// 装不下它，会以「输入 pad 数不匹配」报错。
    /// * `x` / `y` - 叠加层在基底上的偏移（支持表达式，如 `"main_w-overlay_w-10"`）。
    ///   表达式可含 `,`（如 `"if(eq(t,0),0,W-w)"`），会按滤镜语法转义，
    ///   直接照写即可，无需自己加反斜杠。
    ///
    /// ⚠️ 这里**没有**不透明度参数：`overlay` 的 `alpha` 是"alpha 格式"枚举
    /// （`auto` / `straight` / `premultiplied`，取值 0~2），不是不透明度 ——
    /// 把 0~1 的透明度写进 `alpha=` 会被 FFmpeg 静默取整成一个格式档位。
    /// 要给叠加层做半透明，先用 [`FilterGraphBuilder`] 在该路上接一个
    /// `colorchannelmixer=aa=<0~1>`（或 `format=rgba` + `colorchannelmixer`）。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn overlay(x: &str, y: &str) -> Result<Filter> {
        // x/y 是表达式，`,` 等字符会被滤镜语法解析吃掉（整条描述先按 `,` 拆分
        // 滤镜），报出来的错与真实原因无关，因此与其它 `&str` 参数一致地转义。
        let (x, y) = (escape_filter_value(x)?, escape_filter_value(y)?);
        Ok(Filter::new(
            "overlay",
            MediaType::VIDEO,
            format!("overlay=x={x}:y={y}"),
        ))
    }

    /// 横向并排（hstack）：把多路视频并成一行。
    ///
    /// 多输入滤镜，接线方式见 [`overlay`]；要求各路**高度一致**，
    /// 输出宽度为各路宽度之和（[`FilterGraphBuilder::hstack`] 会自动收口到声明的
    /// 输出尺寸）。
    pub fn hstack(inputs: u32) -> Filter {
        Filter::new(
            "hstack",
            MediaType::VIDEO,
            format!("hstack=inputs={inputs}"),
        )
    }

    /// 纵向堆叠（vstack）：把多路视频叠成一列。
    ///
    /// 多输入滤镜，要求各路**宽度一致**，输出高度为各路高度之和。
    pub fn vstack(inputs: u32) -> Filter {
        Filter::new(
            "vstack",
            MediaType::VIDEO,
            format!("vstack=inputs={inputs}"),
        )
    }

    /// 首尾拼接（concat，仅视频段）：把多路视频按顺序接成一路。
    ///
    /// 多输入滤镜，要求各路尺寸、像素格式、时间基一致。
    /// `segments`: 段数（对应 `concat=n=N:v=1:a=0`）。
    pub fn concat(segments: u32) -> Filter {
        Filter::new(
            "concat",
            MediaType::VIDEO,
            format!("concat=n={segments}:v=1:a=0"),
        )
    }

    /// 把一路视频复制成 `outputs` 路相同内容（`split`），是视频 fan-out 的显式手段。
    ///
    /// **多输出**滤镜：用 [`FilterNode::with_outputs`] 给每个输出 pad 标名，每个名字
    /// 各接一条下游链路——同一个标签被两处消费会被建图期拒绝，正是提示在这里插一个
    /// `split`。复制的是同一份像素（后续各链路互不影响）。
    ///
    /// `outputs` 取值范围是 FFmpeg 的 `split.outputs`：**1~INT_MAX**，默认 2。
    /// 传 `1` 是合法的（等价于直连，只是多一层拷贝）。本函数**不校验**该值，
    /// 传 `0` 或超过 `INT_MAX` 的数会原样写进 spec，由建图时的 FFmpeg 报错。
    pub fn split(outputs: u32) -> Filter {
        Filter::new("split", MediaType::VIDEO, format!("split={outputs}"))
    }

    /// 色度键抠像（chromakey），将指定颜色转为透明。
    /// * `color` - 要抠掉的颜色，如 `"green@0.5"`。
    /// * `similarity` - 颜色相似度阈值（FFmpeg: 1e-05~1，默认 0.01，越大越宽松）。
    /// * `blend` - 混合比例（0~1）。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn chromakey(
        color: &str,
        similarity: impl Into<f64>,
        blend: impl Into<f64>,
    ) -> Result<Filter> {
        let color = escape_filter_value(color)?;
        Ok(Filter::new(
            "chromakey",
            MediaType::VIDEO,
            format!(
                "chromakey=color={color}:similarity={}:blend={}",
                number(similarity),
                number(blend)
            ),
        ))
    }

    /// RGB 色键（colorkey），将指定 RGB 颜色转为透明。
    /// `color` - 如 `"black"` 或 `"0x000000"`。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn colorkey(
        color: &str,
        similarity: impl Into<f64>,
        blend: impl Into<f64>,
    ) -> Result<Filter> {
        let color = escape_filter_value(color)?;
        Ok(Filter::new(
            "colorkey",
            MediaType::VIDEO,
            format!(
                "colorkey=color={color}:similarity={}:blend={}",
                number(similarity),
                number(blend)
            ),
        ))
    }

    /// 曲线调节（curves），通过控制点微调 R/G/B 通道色调。
    /// `preset`/`points` 二选一；`points` 形如 `"0/0 0.5/0.5 1/1"`（无需自行转义）。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn curves<'a>(
        preset: impl Into<Option<&'a str>>,
        points: impl Into<Option<&'a str>>,
    ) -> Result<Filter> {
        let preset: Option<&str> = preset.into();
        let points: Option<&str> = points.into();
        let spec = match (preset, points) {
            (Some(p), _) => format!("curves=preset={}", escape_filter_value(p)?),
            (None, Some(pt)) => format!("curves=all={}", escape_filter_value(pt)?),
            _ => "curves".to_string(),
        };
        Ok(Filter::new("curves", MediaType::VIDEO, spec))
    }

    /// 逐行/隔行转换（bwdif）去隔行，现代去隔行替代方案。
    /// `mode`: `send_frame` / `send_field`(默认)。FFmpeg 的 `bwdif` 只有这两档
    /// （取值 0~1），没有 `send_frame_nospatial`。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `mode` is not one of the two
    /// directions FFmpeg names (`send_frame`, `send_field`) or an integer in `0..=1`, or
    /// when a value contains a NUL byte — a filter description is a C string and cannot
    /// carry one. Anything `av_escape` fails with is propagated too.
    ///
    /// Note the range is narrower than [`yadif`]'s: `bwdif` has no
    /// `*_nospatial` variants.
    pub fn bwdif(mode: &str) -> Result<Filter> {
        check_closed_set("bwdif", "mode", mode, &["send_frame", "send_field"], 1)?;
        let mode = escape_filter_value(mode)?;
        Ok(Filter::new(
            "bwdif",
            MediaType::VIDEO,
            format!("bwdif=mode={mode}"),
        ))
    }

    /// GIF 单遍调色板滤镜链（palettegen/paletteuse），输出 pal8 帧供 `gif`
    /// 编码器直接编码。
    ///
    /// 构建的滤镜图：
    /// `fps=<fps>,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse[输出]`
    ///
    /// palettegen 统计**全部**输入帧的最优 256 色调色板（EOF 时才输出），
    /// paletteuse 用它把帧量化为 pal8 —— 即 FFmpeg 官方推荐的单遍 GIF 调色板
    /// 管线；palettegen/paletteuse 之间的帧缓存在图内完成，无需两遍编码。
    ///
    /// * `fps` - 输出帧率（GIF 体积敏感，通常 10~15）。
    /// * `dither` - 抖动算法（`"bayer"`/`"floyd_steinberg"`/`"none"` 等），
    ///   `None` 使用 FFmpeg 默认（sierra2_4a）。
    ///
    /// 输入为 RGB 帧：滤镜声明了 RGB24 输入格式，编码器侧自动把输入帧转到
    /// RGB24 再进图；输出 pal8 与 `gif` 编码器原生格式一致。
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rsmedia::{EncoderBuilder, filter::video};
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// let encoder = EncoderBuilder::new_video(320, 240)
    ///     .with_codec_name("gif")
    ///     .with_fps(10.0)
    ///     .with_filters(vec![video::gif_palette(10.0, None)?])
    ///     .build()?;
    /// # drop(encoder);
    /// # Ok(())
    /// # }
    /// ```
    pub fn gif_palette<'a>(fps: f32, dither: impl Into<Option<&'a str>>) -> Result<Filter> {
        let dither: Option<&str> = dither.into();
        // 不能用 `map(...)`：闭包里没法用 `?`，而转义现在是可能失败的。
        let dither_part = match dither {
            Some(d) => format!(":dither={}", escape_filter_value(d)?),
            None => String::new(),
        };
        Ok(Filter::new(
            "paletteuse",
            MediaType::VIDEO,
            format!("fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse{dither_part}"),
        )
        .with_input_format(PixelFormat::RGB24))
    }

    /// LUT 调色（lutyuv）：按亮度/色度查找表逐通道映射，证件照"美白"常用
    /// 亮度表把中间调整体上提而不压高光。
    ///
    /// * `y` / `u` / `v` - 各通道的 LUT 表达式（FFmpeg eval 语法，如
    ///   `"if(lt(val,100),val,val+20)"`），传 `None` 表示该通道不变。
    ///   表达式中的逗号会被自动转义。
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::filter::video;
    ///
    /// # fn main() -> rsmedia::Result<()> {
    /// // 亮度整体 +10（简单提亮美白），色度不动
    /// let _f = video::lutyuv(Some("val+10"), None, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn lutyuv<'a>(
        y: impl Into<Option<&'a str>>,
        u: impl Into<Option<&'a str>>,
        v: impl Into<Option<&'a str>>,
    ) -> Result<Filter> {
        let y: Option<&str> = y.into();
        let u: Option<&str> = u.into();
        let v: Option<&str> = v.into();
        let mut parts = Vec::new();
        if let Some(y_expr) = y {
            parts.push(format!("y={}", escape_filter_value(y_expr)?));
        }
        if let Some(u) = u {
            parts.push(format!("u={}", escape_filter_value(u)?));
        }
        if let Some(v) = v {
            parts.push(format!("v={}", escape_filter_value(v)?));
        }
        let spec = if parts.is_empty() {
            "lutyuv".to_string()
        } else {
            format!("lutyuv={}", parts.join(":"))
        };
        Ok(Filter::new("lutyuv", MediaType::VIDEO, spec))
    }

    /// 拼版（tile）：把多帧按 `cols x rows` 网格排成一张图，证件照"一张 6 寸
    /// 相纸排 8 张一寸"即此滤镜。
    ///
    /// * `cols` / `rows` - 网格行列数（总格数 = cols*rows，输入帧数不足时
    ///   未填满的格子用 `color` 填充）。
    /// * `padding` - **内边框厚度**（每格四周各加这么多像素），FFmpeg 取值
    ///   0~1024（默认 0）；它不是"格与格之间的间距"。整图外边框另有 `margin`，
    ///   本函数不暴露。
    /// * `color` - 未使用区域的颜色（"set the color of the unused area"，
    ///   默认 `black`）：既填未填满的格子，也填 `padding` 留出的内边框，如 `"white"`。
    ///
    /// 注意：tile 是**攒帧**滤镜——每 cols*rows 帧吐 1 帧，EOF 时输出残余格。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn tile(cols: u32, rows: u32, padding: u32, color: &str) -> Result<Filter> {
        let color = escape_filter_value(color)?;
        Ok(Filter::new(
            "tile",
            MediaType::VIDEO,
            format!("tile={cols}x{rows}:padding={padding}:color={color}"),
        ))
    }
}

pub mod audio {
    use super::*;

    /// 创建音频重采样过滤器
    pub fn resample(nb_channels: i32, sample_rate: i32, format: SampleFormat) -> Filter {
        // 统一走解析函数以便复用 channel_desc 处理，避免 describe().unwrap() panic
        let channel_desc = audio_channel_desc(nb_channels);

        // async=1 可能更适合实时场景，避免缓冲问题。
        let spec_str = format!(
            "aresample=osr={}:osf={}:ochl={}:async=1",
            sample_rate,
            format.get_sample_fmt_name(),
            channel_desc,
        );

        // 注意：滤镜本体是 aresample，name 必须与之保持一致，避免与真实滤镜名混淆
        Filter::new("aresample", MediaType::AUDIO, spec_str)
    }

    /// Converts audio sample format.
    /// `format`: <https://ffmpeg.org/ffmpeg-filters.html#format>
    /// `aformat`: <https://ffmpeg.org/ffmpeg-filters.html#aformat-1>
    pub fn format(nb_channels: i32, sample_rate: i32, format: SampleFormat) -> Filter {
        let channel_desc = audio_channel_desc(nb_channels);

        Filter::new(
            "aformat",
            MediaType::AUDIO,
            format!(
                "aformat=sample_fmts={}:sample_rates={}:channel_layouts={}",
                format.get_sample_fmt_name(),
                sample_rate,
                channel_desc
            ),
        )
    }

    /// 把通道数解析为 FFmpeg 通道布局描述；失败时回退到数字通道数，避免 panic。
    /// 音频端点建图（[`FilterGraph::create_audio_source`]）也复用同一逻辑。
    pub(super) fn audio_channel_desc(nb_channels: i32) -> String {
        AVChannelLayout::from_nb_channels(nb_channels)
            .describe()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_else(|_| format!("{nb_channels}"))
    }

    /// 音量调整
    /// Adjusts audio volume.
    ///
    /// `volume` is an FFmpeg `<string>` option, so all three of its spellings are
    /// reachable: a linear multiplier (`volume(0.5)`), a **dB** value — which no
    /// numeric type could carry — (`volume("-6dB")`), and an expression
    /// (`volume("if(gt(t,10),0,1)")`). See [`Expr`].
    /// # Errors
    ///
    /// See [`Expr::to_filter_value`].
    pub fn volume<'a>(volume: impl Into<Expr<'a>>) -> Result<Filter> {
        let volume = volume.into().to_filter_value()?;
        Ok(Filter::new(
            "volume",
            MediaType::AUDIO,
            format!("volume={volume}"),
        ))
    }

    /// loudnorm - EBU R128音量标准化
    ///
    /// `loudnorm.I`（整合响度，LUFS）在 FFmpeg 里是 `<double>`。
    pub fn loudnorm(integrated_loudness: impl Into<f64>) -> Filter {
        Filter::new(
            "loudnorm",
            MediaType::AUDIO,
            format!("loudnorm=I={}:TP=-1.5:LRA=11", number(integrated_loudness)),
        )
    }

    /// 单频段均衡器
    /// Applies a single-band peaking equalizer.
    ///
    /// * `frequency` - Center frequency in Hz.
    /// * `gain` - Gain in dB.
    /// * `width` - Bandwidth in Hz (`width_type=h`).
    ///
    /// 三个参数在 FFmpeg 里都是 `<double>`，所以统一收 `f64`：中心频率与带宽
    /// 都不是"整数才合理"的量。
    pub fn equalizer(
        frequency: impl Into<f64>,
        gain: impl Into<f64>,
        width: impl Into<f64>,
    ) -> Filter {
        Filter::new(
            "equalizer",
            MediaType::AUDIO,
            format!(
                "equalizer=f={}:width_type=h:width={}:g={}", // width_type=h (Hz)
                number(frequency),
                number(width),
                number(gain),
            ),
        )
    }

    /// 多频段均衡器 (bass, mid, treble)
    /// Applies a simple 3-band equalizer using firequalizer.
    /// See: <https://ffmpeg.org/ffmpeg-filters.html#firequalizer>
    pub fn three_band_equalizer(
        bass_gain: impl Into<f64>,
        mid_gain: impl Into<f64>,
        treble_gain: impl Into<f64>,
    ) -> Filter {
        Filter::new(
            "firequalizer",
            MediaType::AUDIO,
            format!(
                "firequalizer=gain='if(lt(f,200),{},if(gt(f,5000),{},{}))':scale=linlog",
                number(bass_gain),
                number(treble_gain),
                number(mid_gain),
            ),
        )
    }

    /// 压缩器
    /// Applies dynamic range compression.
    ///
    /// * `ratio`: Compression ratio (1 - 20).
    /// * `attack`: Attack time in ms (optional, default 20).
    /// * `release`: Release time in ms (optional, default 250).
    ///
    /// `acompressor` 的 `ratio`/`attack`/`release` 在 FFmpeg 里都是 `<double>`。
    ///
    /// See: <https://ffmpeg.org/ffmpeg-filters.html#acompressor>
    pub fn compressor(
        ratio: impl Into<f64>,
        attack: impl Into<Option<f64>>,
        release: impl Into<Option<f64>>,
    ) -> Result<Filter> {
        let attack: Option<f64> = attack.into();
        let release: Option<f64> = release.into();
        let ratio: f64 = ratio.into();
        if ratio < 1.0 {
            return Err(RsmediaError::invalid_config(format!(
                "Compressor ratio must be >= 1.0: {ratio}"
            )));
        }
        let mut spec = format!("acompressor=ratio={}", number(ratio));
        if let Some(a) = attack {
            spec.push_str(&format!(":attack={a}"));
        }
        if let Some(r) = release {
            spec.push_str(&format!(":release={r}"));
        }
        // Add other params: makeup, knee, link, detection, mix...
        Ok(Filter::new("acompressor", MediaType::AUDIO, spec))
    }

    /// 高通滤波
    ///
    /// `highpass.frequency` 在 FFmpeg 里是 `<double>`。
    pub fn highpass(freq: impl Into<f64>) -> Filter {
        Filter::new(
            "highpass",
            MediaType::AUDIO,
            format!("highpass=f={}", number(freq)),
        )
    }

    /// 低通滤波
    ///
    /// `lowpass.frequency` 在 FFmpeg 里是 `<double>`。
    pub fn lowpass(freq: impl Into<f64>) -> Filter {
        Filter::new(
            "lowpass",
            MediaType::AUDIO,
            format!("lowpass=f={}", number(freq)),
        )
    }

    /// 音频变速
    /// Changes audio tempo without changing pitch.
    /// * `rate`: Speed multiplier (0.5 to 100.0).
    ///
    /// `atempo.tempo` 在 FFmpeg 里是 `<double>`。
    pub fn atempo(rate: impl Into<f64>) -> Filter {
        Filter::new(
            "atempo",
            MediaType::AUDIO,
            format!("atempo={}", number(rate)),
        )
    }

    /// 延时（ms）
    ///
    /// 这里固定用 `all=1` 把同一个延时应用到所有通道；逐通道写法
    /// (`delays=100|100`) 必须按实际通道数逐个列出，通道数不匹配时会被
    /// 静默忽略，因此不在此暴露。（滤镜链的分隔符是 `,`，`|` 只是部分滤镜
    /// 选项值内部的分隔符。）
    pub fn adelay(delay_ms: i32) -> Filter {
        Filter::new(
            "adelay",
            MediaType::AUDIO,
            format!("adelay=delays={delay_ms}:all=1"),
        )
    }

    /// 创建FFT降噪过滤器
    /// Applies FFT noise reduction (simple).
    /// `noise_reduction`: Noise reduction factor in dB (e.g., 12.5).
    /// `noise_floor`: Noise floor in dB (e.g., -50.5).
    ///
    /// `afftdn.nr` / `nf` 在 FFmpeg 里是 `<float>`：收 `i32` 会把分数 dB 挡在门外。
    pub fn fft_denoise(noise_reduction: impl Into<f64>, noise_floor: impl Into<f64>) -> Filter {
        Filter::new(
            "afftdn",
            MediaType::AUDIO,
            format!(
                "afftdn=nr={}:nf={}:nt=w",
                number(noise_reduction),
                number(noise_floor)
            ),
        )
    }

    /// 创建高级FFT降噪过滤器
    /// Applies FFT noise reduction (advanced).
    ///
    /// * `noise_reduction`: Noise reduction in dB（`afftdn.nr`，`<float>`，
    ///   FFmpeg 取值 0.01~97，默认 12）。
    /// * `noise_floor`: Noise floor in dB（`afftdn.nf`，`<float>`，
    ///   FFmpeg 取值 -80~-20，默认 -50）。
    /// * `noise_type`: `afftdn.nt`，`'w'`（white，默认）/ `'v'`（vinyl）/ `'s'`（shellac）/
    ///   `'c'`（custom）。FFmpeg 只认这四个枚举值（0~3），本函数**不做校验**：
    ///   传别的值（例如 `'p'`）会原样写进 spec，最终在建图时报错。
    ///   `'c'` 还要配合 `band_noise` 才有意义，本函数不暴露该选项。
    /// * `track_residual`: `afftdn.tr`，FFmpeg 声明为 **`<boolean>`**：跟踪残余噪声
    ///   （`track_residual`），不是"时间平滑系数"。
    ///
    /// 早先这个参数叫 `time_smoothing`、类型是 `Option<f32>`，会被格式化成
    /// `tr=0.5` 这样的值——而 `tr` 是布尔选项，FFmpeg 直接拒绝整条滤镜链：
    /// `Unable to parse "tr" option value "0.5" as boolean`。也就是说**只要传入
    /// 任何非整数，`advanced_fft_denoise` 都会让 `FilterGraph` 建不起来**；
    /// 名字与类型都指向"一个浮点系数"，掩盖了真实的选项语义。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn advanced_fft_denoise<'a>(
        noise_reduction: impl Into<f64>,
        noise_floor: impl Into<f64>,
        noise_type: impl Into<Option<&'a str>>,
        track_residual: bool,
    ) -> Result<Filter> {
        let noise_type: Option<&str> = noise_type.into();
        let nt = escape_filter_value(noise_type.unwrap_or("w"))?;
        let tr = u8::from(track_residual);
        Ok(Filter::new(
            "afftdn",
            MediaType::AUDIO,
            format!(
                "afftdn=nr={}:nf={}:nt={nt}:tr={tr}",
                number(noise_reduction),
                number(noise_floor)
            ),
        ))
    }

    /// 创建自适应非局部均值降噪过滤器
    /// Applies Non-Local Means de-noising (anlmdn).
    /// `strength`: Denoising strength (0 to inf, default 1e-05).
    /// `patch_size`: Patch **duration** (FFmpeg's `p`, `<duration>`, default 0.002 = 2 ms).
    /// `search_range`: Research **duration** (FFmpeg's `r`, `<duration>`, default 0.006 = 6 ms).
    ///
    /// `p` / `r` are `<duration>` options, not sample counts — FFmpeg rejects anything outside
    /// `[0.001, 0.1]` / `[0.002, 0.3]`, so they are typed [`Duration`] here (like `afade` and
    /// `trim`) rather than `i32`.
    ///
    /// # Known upstream issue (FFmpeg <= 9.0)
    ///
    /// FFmpeg 的 `anlmdn` 滤镜在流结束（EOF）冲刷不满一窗的尾巴帧时存在**堆越界写**：
    /// 输出缓冲按尾巴帧的样本数分配，而 `filter_channel` 固定写入完整窗口（默认
    /// 44.1kHz 下 H=177 样本）。当输入总样本数不是 H 的整数倍时会越界写堆内存，
    /// 可能导致进程随机崩溃。上游尚未修复；若使用本滤镜，建议保证输入总样本数为
    /// 窗口尺寸（`H = 2*round(pd*sample_rate/1e6)+1`，默认参数 44.1kHz 下为 177）
    /// 的整数倍，或改用 `Filter::fft_denoise` / `Filter::denoise`。
    pub fn anlm_denoise(
        strength: impl Into<Option<f64>>,
        patch_size: impl Into<Option<Duration>>,
        search_range: impl Into<Option<Duration>>,
    ) -> Filter {
        let strength: Option<f64> = strength.into();
        let patch_size: Option<Duration> = patch_size.into();
        let search_range: Option<Duration> = search_range.into();
        let mut params = Vec::new();
        if let Some(s) = strength {
            params.push(format!("s={s}"));
        }
        if let Some(p) = patch_size {
            params.push(format!("p={}", duration_literal(p)));
        }
        if let Some(r) = search_range {
            params.push(format!("r={}", duration_literal(r)));
        }
        let spec = if params.is_empty() {
            "anlmdn".to_string()
        } else {
            format!("anlmdn={}", params.join(":"))
        };
        Filter::new("anlmdn", MediaType::AUDIO, spec)
    }

    /// 音频降噪（便捷方法），使用 FFT 降噪并自动估计噪声特征。
    /// `strength` 为降噪强度（dB，建议 10-30）。
    pub fn denoise(strength: impl Into<f64>) -> Filter {
        Filter::new(
            "afftdn",
            MediaType::AUDIO,
            format!("afftdn=nr={}:nt=w", number(strength)),
        )
    }

    /// 音频淡入淡出（afade）。
    ///
    /// * `fade_type` - `in` 或 `out`。
    /// * `start` - 起点。
    /// * `duration` - 淡变时长。
    ///
    /// `afade.st` / `d` 在 FFmpeg 里是 **`<duration>`**，不是裸浮点数：它们有自己的
    /// 量纲（时间），也接受 `"1.5s"`、`"00:00:01.5"` 这样的时长字面量。
    /// 用 [`Duration`] 表达"这是一段时间"，比 `f32` 秒更贴近语义，也免掉了
    /// `f32` 在 10⁴ 秒量级上约 1 ms 的 ULP 误差。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `fade_type` is neither `in` nor `out`
    /// (nor the integers `0`/`1` FFmpeg maps them to), or when a value contains a NUL
    /// byte — a filter description is a C string and cannot carry one. Anything
    /// `av_escape` fails with is propagated too.
    pub fn afade(fade_type: &str, start: Duration, duration: Duration) -> Result<Filter> {
        check_closed_set("afade", "type", fade_type, &["in", "out"], 1)?;
        let fade_type = escape_filter_value(fade_type)?;
        Ok(Filter::new(
            "afade",
            MediaType::AUDIO,
            format!(
                "afade=t={fade_type}:st={}:d={}",
                duration_literal(start),
                duration_literal(duration)
            ),
        ))
    }

    /// 回声（aecho）。
    /// * `in_gain` / `out_gain` - 输入/输出增益。
    /// * `delays` - 延迟序列（ms，如 `"60|30"`）。
    /// * `decays` - 衰减系数（如 `"0.4|0.3"`）。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
    /// description is a C string and cannot carry one. Anything `av_escape` fails with is
    /// propagated too.
    pub fn aecho(
        in_gain: impl Into<f64>,
        out_gain: impl Into<f64>,
        delays: &str,
        decays: &str,
    ) -> Result<Filter> {
        let (delays, decays) = (escape_filter_value(delays)?, escape_filter_value(decays)?);
        Ok(Filter::new(
            "aecho",
            MediaType::AUDIO,
            format!(
                "aecho=in_gain={}:out_gain={}:delays={delays}:decays={decays}",
                number(in_gain),
                number(out_gain)
            ),
        ))
    }

    /// 混音（amix），将多路输入混成一路。
    ///
    /// 这是**多输入**滤镜：用 [`FilterGraphBuilder::amix`] 直接建整图，或
    /// `FilterNode::new(filter::audio::amix(n, "longest")).with_inputs([...])`
    /// 接进自定义的多输入图。各路采样率 / 采样格式 / 通道布局不同时，FFmpeg 会在
    /// 链路协商阶段自动插入 `aresample`。
    /// `inputs`: 输入路数；`duration`: `longest`/`shortest`/`first`。
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `duration` is not one of `longest`,
    /// `shortest`, `first` (nor the integers `0`/`1`/`2` FFmpeg maps them to), or when a
    /// value contains a NUL byte — a filter description is a C string and cannot carry
    /// one. Anything `av_escape` fails with is propagated too.
    pub fn amix(inputs: u32, duration: &str) -> Result<Filter> {
        check_amix_duration(duration)?;
        let duration = escape_filter_value(duration)?;
        Ok(Filter::new(
            "amix",
            MediaType::AUDIO,
            format!("amix=inputs={inputs}:duration={duration}"),
        ))
    }

    /// 把一路音频复制成 `outputs` 路相同内容（`asplit`），是音频 fan-out 的显式手段。
    ///
    /// **多输出**滤镜：用 [`FilterNode::with_outputs`] 给每个输出 pad 标名，每个名字
    /// 各接一条下游链路（同一个标签接两处会被建图期拒绝）。
    ///
    /// `outputs` 取值范围是 FFmpeg 的 `(a)split.outputs`：**1~INT_MAX**，默认 2。
    /// 传 `1` 是合法的（等价于直连）。本函数**不校验**该值，`0` 或超过 `INT_MAX`
    /// 的数会原样写进 spec，由建图时的 FFmpeg 报错。
    pub fn asplit(outputs: u32) -> Filter {
        Filter::new("asplit", MediaType::AUDIO, format!("asplit={outputs}"))
    }

    /// 反转音频（areverse）。
    pub fn areverse() -> Filter {
        Filter::new("areverse", MediaType::AUDIO, "areverse".to_string())
    }

    /// 低频增益（bass）。
    /// `freq` - 中心频率，`gain` - 增益（dB）。
    ///
    /// `bass.frequency` / `gain` 在 FFmpeg 里都是 `<double>`。
    pub fn bass(freq: impl Into<f64>, gain: impl Into<f64>) -> Filter {
        Filter::new(
            "bass",
            MediaType::AUDIO,
            format!("bass=f={}:g={}", number(freq), number(gain)),
        )
    }

    /// 高频增益（treble）。
    /// `freq` - 中心频率，`gain` - 增益（dB）。
    ///
    /// `treble.frequency` / `gain` 在 FFmpeg 里都是 `<double>`。
    pub fn treble(freq: impl Into<f64>, gain: impl Into<f64>) -> Filter {
        Filter::new(
            "treble",
            MediaType::AUDIO,
            format!("treble=f={}:g={}", number(freq), number(gain)),
        )
    }

    /// 低频搁架滤波器（lowshelf）。
    /// `freq` - 转折频率，`gain` - 增益（dB）。
    ///
    /// `lowshelf.frequency` / `gain` 在 FFmpeg 里都是 `<double>`。
    pub fn lowshelf(freq: impl Into<f64>, gain: impl Into<f64>) -> Filter {
        Filter::new(
            "lowshelf",
            MediaType::AUDIO,
            format!("lowshelf=f={}:g={}", number(freq), number(gain)),
        )
    }

    /// 高频搁架滤波器（highshelf）。
    /// `freq` - 转折频率，`gain` - 增益（dB）。
    ///
    /// `highshelf.frequency` / `gain` 在 FFmpeg 里都是 `<double>`。
    pub fn highshelf(freq: impl Into<f64>, gain: impl Into<f64>) -> Filter {
        Filter::new(
            "highshelf",
            MediaType::AUDIO,
            format!("highshelf=f={}:g={}", number(freq), number(gain)),
        )
    }
}

/// 修改时间戳表达式（加速、减速、对齐等）。
/// 典型值：`"0.5*PTS"`（2倍速）、`"1.5*PTS"`（慢放）、`"PTS-STARTPTS"`。
/// `expr`: FFmpeg expression (e.g., "0.5*PTS", "PTS-STARTPTS").
/// # Errors
///
/// Returns [`RsmediaError::InvalidConfig`] when a value contains a NUL byte: a filter
/// description is a C string and cannot carry one. Anything `av_escape` fails with is
/// propagated too.
pub fn setpts(media_type: MediaType, expr: &str) -> Result<Filter> {
    // 音频滤镜在视频滤镜名前加 `a` 前缀（如 `asetpts`/`setpts`）。
    let name = if media_type == MediaType::AUDIO {
        "asetpts"
    } else {
        "setpts"
    };
    let escaped_expr = escape_filter_value(expr)?;
    Ok(Filter::new(
        name,
        media_type,
        format!("{name}={escaped_expr}"),
    ))
}

/// 将视频/音频裁剪到指定的时间范围。
///
/// `trim.start` / `end`（音频侧 `atrim`）在 FFmpeg 里都是 **`<duration>`**，
/// 所以两端都收 [`Duration`]：裁剪点是一段时间，不是一个裸浮点数。
///
/// 选项按名字写（`start=` / `end=`）而不是按位置：`trim` 的选项表里
/// `start`/`end` 之间还夹着 `starti`/`endi`，位置写法是否稳定取决于 FFmpeg
/// 版本，而名字写法在任何版本上都只有一个意思。
pub fn trim(media_type: MediaType, start: Duration, end: Duration) -> Filter {
    // 同上：`atrim`/`trim`。
    let name = if media_type == MediaType::AUDIO {
        "atrim"
    } else {
        "trim"
    };
    Filter::new(
        name,
        media_type,
        format!(
            "{name}=start={}:end={}",
            duration_literal(start),
            duration_literal(end)
        ),
    )
}

/// 过滤器参数配置
#[derive(Debug, Clone)]
pub enum FilterParams {
    Video(VideoParams),
    Audio(AudioParams),
}

impl FilterParams {
    pub fn media_type(&self) -> MediaType {
        match self {
            FilterParams::Video(_) => MediaType::VIDEO,
            FilterParams::Audio(_) => MediaType::AUDIO,
        }
    }

    /// 这一路的**输入**端点：按帧实际推入时的格式（`src_format`）声明。
    pub(crate) fn input_endpoint(&self) -> Endpoint {
        match self {
            FilterParams::Video(p) => Endpoint::Video(p.clone().into()),
            FilterParams::Audio(p) => Endpoint::Audio(p.clone().into()),
        }
    }

    /// 这一路的**输出**端点：sink 按协商格式（`format`）约束，`src_format` →
    /// `format` 的转换由图内滤镜完成（如 GIF 调色板链 RGB24 → pal8）。
    pub(crate) fn output_endpoint(&self) -> Endpoint {
        match self {
            FilterParams::Video(p) => {
                let endpoint: VideoEndpoint = p.clone().into();
                Endpoint::Video(VideoEndpoint {
                    format: p.format,
                    ..endpoint
                })
            }
            FilterParams::Audio(p) => {
                let endpoint: AudioEndpoint = p.clone().into();
                Endpoint::Audio(AudioEndpoint {
                    format: p.format,
                    ..endpoint
                })
            }
        }
    }
}

/// Video filter parameters.
///
/// The sizes mirror the shape of the corresponding `AVFrame` / `AVCodecContext`
/// field, which is why they keep FFmpeg's `int` width (`i32`) instead of the
/// `u32` used by the high-level API ([`EncoderBuilder`](crate::EncoderBuilder),
/// [`MediaFrame`](crate::MediaFrame)). This mirror layer is the **only** place
/// in the crate that deliberately keeps `i32` sizes. The rational parameters,
/// by contrast, are plain [`Rational`] values — FFmpeg's `AVRational` never
/// appears outside [`crate::time`].
#[derive(Debug, Clone)]
pub struct VideoParams {
    /// Frame width in pixels, mirroring FFmpeg's `int`.
    pub width: i32,
    /// Frame height in pixels, mirroring FFmpeg's `int`.
    pub height: i32,
    /// 滤镜图**输出**（sink）像素格式：编码器协商格式。
    pub format: PixelFormat,
    /// 滤镜图**输入**（buffer 源）像素格式：默认与 `format` 相同；当滤镜链
    /// 声明了不同的输入格式（如 GIF 调色板链要求 RGB 输入、输出 pal8）时，
    /// 编码/解码两条流水线都把它设为声明的格式，并在进图前把帧转成同一格式；
    /// src→sink 的格式转换由滤镜图内完成。
    pub src_format: PixelFormat,
    /// 输入时间基。
    pub time_base: Rational,
    /// 帧率。
    pub frame_rate: Rational,
    /// 像素宽高比；[`Rational::ZERO`] 表示未知。
    pub pixel_aspect: Rational,
}

/// 音频过滤器参数
#[derive(Debug, Clone)]
pub struct AudioParams {
    pub nb_channels: i32,
    pub sample_rate: i32,
    /// 滤镜图**输出**（abuffersink 约束）采样格式：编码器/解码器协商格式。
    pub format: SampleFormat,
    /// 滤镜图**输入**（abuffer）采样格式：默认与 `format` 相同；当滤镜链
    /// 声明了不同的输入格式（[`Filter::with_input_format`]）时，由编码器/
    /// 解码器侧设置为声明的格式，src→sink 的格式转换由图内滤镜完成。
    pub src_format: SampleFormat,
    /// 输入时间基（通常为 `1 / sample_rate`）。
    pub time_base: Rational,
}

/// 滤镜图端点（`buffer`/`abuffer` 源，或 `buffersink`/`abuffersink` 汇）的格式声明。
///
/// 端点描述「这一路帧进来（或出去）时是什么格式、什么时间基」，构建器据此建
/// `buffer` 源与 `buffersink` 汇。**各路端点不必格式一致**：FFmpeg 在链路协商
/// 阶段会自动插入 `scale` / `aresample` 完成像素格式、尺寸与采样格式的转换，
/// 所以叠加一个不同尺寸的 logo 只需如实声明它的尺寸。
#[derive(Debug, Clone, Copy)]
pub enum Endpoint {
    /// 视频端点。
    Video(VideoEndpoint),
    /// 音频端点。
    Audio(AudioEndpoint),
}

impl Endpoint {
    /// 端点的媒体类型。
    pub fn media_type(&self) -> MediaType {
        match self {
            Endpoint::Video(_) => MediaType::VIDEO,
            Endpoint::Audio(_) => MediaType::AUDIO,
        }
    }
}

impl From<VideoEndpoint> for Endpoint {
    fn from(value: VideoEndpoint) -> Self {
        Endpoint::Video(value)
    }
}

impl From<AudioEndpoint> for Endpoint {
    fn from(value: AudioEndpoint) -> Self {
        Endpoint::Audio(value)
    }
}

/// Format declaration for a video endpoint.
///
/// Like [`VideoParams`], the sizes stay `i32` — FFmpeg's `int` — and so does the
/// rest of this crate's size API: a value has to reach `AVFrame.width` and
/// `AVCodecContext.width` unchanged, so `u32` would only add a narrowing step in
/// which an out-of-range value wraps silently. The rational fields are plain
/// [`Rational`].
///
/// The three rationals share a type and sit next to each other, so both a struct
/// literal and a positional constructor can swap them without the compiler
/// noticing. Build the value with [`VideoEndpoint::new`] plus the `with_*`
/// setters, where each one is named by what it sets.
#[derive(Debug, Clone, Copy)]
pub struct VideoEndpoint {
    /// Width in pixels
    pub width: i32,
    /// Height in pixels
    pub height: i32,
    /// 像素格式。
    pub format: PixelFormat,
    /// 时间基。同一张图内各路输入应当一致，否则画面会错位。
    pub time_base: Rational,
    /// 帧率。
    pub frame_rate: Rational,
    /// 像素宽高比。
    pub pixel_aspect: Rational,
}

impl VideoEndpoint {
    /// 用尺寸与像素格式建一个视频端点。
    ///
    /// 三个有理数**不**走位置参数：它们类型相同、含义又相近（时间基与帧率还互为
    /// 倒数），写反了编译得过、要到出片才看得出错。用下面的 `with_*` 逐个按名字设置。
    ///
    /// 默认值：时间基与帧率 [`Rational::ZERO`]（`buffer` 源会直接拒绝 `time_base=0/1`，
    /// 不会静默建出一个错的图）、像素宽高比 [`Rational::ONE`]（1:1，最常见的情形）。
    pub fn new(width: i32, height: i32, format: PixelFormat) -> Self {
        Self {
            width,
            height,
            format,
            time_base: Rational::ZERO,
            frame_rate: Rational::ZERO,
            pixel_aspect: Rational::ONE,
        }
    }

    /// 设置时间基。同一张图内各路输入应当一致，否则画面会错位。
    pub fn with_time_base(mut self, time_base: Rational) -> Self {
        self.time_base = time_base;
        self
    }

    /// 设置帧率。
    pub fn with_frame_rate(mut self, frame_rate: Rational) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// 设置像素宽高比。
    pub fn with_pixel_aspect(mut self, pixel_aspect: Rational) -> Self {
        self.pixel_aspect = pixel_aspect;
        self
    }
}

/// 音频端点的格式声明。
#[derive(Debug, Clone, Copy)]
pub struct AudioEndpoint {
    /// 通道数。
    pub nb_channels: i32,
    /// 采样率。
    pub sample_rate: i32,
    /// 采样格式。
    pub format: SampleFormat,
    /// 时间基。
    pub time_base: Rational,
}

impl AudioEndpoint {
    /// 用通道数、采样率、采样格式与时间基建一个音频端点。
    pub fn new(
        nb_channels: i32,
        sample_rate: i32,
        format: SampleFormat,
        time_base: Rational,
    ) -> Self {
        Self {
            nb_channels,
            sample_rate,
            format,
            time_base,
        }
    }
}

impl From<VideoParams> for VideoEndpoint {
    /// `VideoParams.src_format` 是这一路帧推入滤镜图时的实际格式，故取它。
    fn from(params: VideoParams) -> Self {
        Self {
            width: params.width,
            height: params.height,
            format: params.src_format,
            time_base: params.time_base,
            frame_rate: params.frame_rate,
            pixel_aspect: params.pixel_aspect,
        }
    }
}

impl From<AudioParams> for AudioEndpoint {
    fn from(params: AudioParams) -> Self {
        Self {
            nb_channels: params.nb_channels,
            sample_rate: params.sample_rate,
            format: params.src_format,
            time_base: params.time_base,
        }
    }
}

impl From<VideoParams> for Endpoint {
    fn from(params: VideoParams) -> Self {
        Endpoint::Video(params.into())
    }
}

impl From<AudioParams> for Endpoint {
    fn from(params: AudioParams) -> Self {
        Endpoint::Audio(params.into())
    }
}

/// 标签是否可安全地写进滤镜图描述：非空，且只含 ASCII 字母、数字与下划线。
fn is_valid_label(label: &str) -> bool {
    !label.is_empty()
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// 滤镜实例的静态输入 pad 数，以及它是否允许比静态列表更多的输入
/// （`AVFILTER_FLAG_DYNAMIC_INPUTS`）。
///
/// `hstack` / `vstack` / `concat` / `amix` 的实际路数由滤镜自己的 `inputs=` / `n=`
/// 选项决定，`AVFilter.inputs` 只列出静态 pad，所以这两类必须分开处理：静态滤镜按
/// pad 数精确校验接线，动态滤镜的 pad 由标签数量生成、交由 FFmpeg 在 `config`
/// 阶段核对选项。
fn filter_input_pads(name: &str) -> Result<(usize, bool)> {
    let filter = get_by_name(name)?;
    if filter.is_none() {
        return Err(RsmediaError::unsupported(format!(
            "filter '{name}' is not available in this FFmpeg build"
        )));
    }
    let filter = filter.unwrap();
    // SAFETY: `filter` 指向 FFmpeg 的静态滤镜定义，`avfilter_filter_pad_count` 只读
    // 其中以 NULL 结尾的 pad 数组（`is_output = 0` 取输入侧）。
    let pads = unsafe { ffi::avfilter_filter_pad_count(filter.as_ptr(), 0) } as usize;
    Ok((
        pads,
        filter.flags & ffi::AVFILTER_FLAG_DYNAMIC_INPUTS as i32 != 0,
    ))
}

/// 滤镜实例的静态输出 pad 数，以及它的输出 pad 是否可以在协商时由选项/标签改写
/// （`AVFILTER_FLAG_DYNAMIC_OUTPUTS`）。
///
/// `split` / `asplit` 的 `AVFilter.outputs` 为空数组（pad 数由 `outputs=` 选项决定），
/// 因此它们的实际输出路数只能相信调用者给出的标签个数，其余滤镜按静态 pad 数精确校验
/// ——多标一个标签会生成 `filter[a][b]` 这种 pad 数对不上的描述，FFmpeg 只会给出一句
/// 难懂的解析错误，所以在这里提前拒掉。
fn filter_output_pads(name: &str) -> Result<(usize, bool)> {
    let filter = get_by_name(name)?;
    if filter.is_none() {
        return Err(RsmediaError::unsupported(format!(
            "filter '{name}' is not available in this FFmpeg build"
        )));
    }
    let filter = filter.unwrap();
    // SAFETY: 同 `filter_input_pads`，`is_output = 1` 取输出侧。
    let pads = unsafe { ffi::avfilter_filter_pad_count(filter.as_ptr(), 1) } as usize;
    Ok((
        pads,
        filter.flags & ffi::AVFILTER_FLAG_DYNAMIC_OUTPUTS as i32 != 0,
    ))
}

/// 单输入线性链的图输入 / 图输出标签。
///
/// 线性链的 spec 不带标签（由 FFmpeg 按 `,` 自动接线），而 FFmpeg 对不带标签的
/// 开放端点约定用 `in` / `out` 命名（参见 `avfilter_graph_parse` 里 "first input
/// can be omitted if it is [in]" 的处理），所以这两个名字既是逻辑标签、也是端点名。
const SINGLE_INPUT_LABEL: &str = "in";
/// 见 [`SINGLE_INPUT_LABEL`]。
const SINGLE_OUTPUT_LABEL: &str = "out";

/// 把若干 [`AVFilterInOut`] 手工串成链表，返回链表头（空则返回 `None`）。
///
/// rsmpeg 只提供单节点构造，而 `avfilter_graph_parse_ptr` 要的是链表，节点顺序
/// 无关紧要（配对按名字，见 [`FilterGraph::setup_endpoints`]），名字才是关键。
///
/// 一条链上只能有**一个** Rust 侧所有者：rsmpeg 的 `Drop` 调的是
/// `avfilter_inout_free`，它会顺着 `next` 释放**整条链**。所以先把所有节点
/// `into_raw()` 交出所有权（不析构），链接完成后只把头节点包回 RAII —— 其余节点
/// 从此归头节点那一份所有者管
fn chain_inouts(nodes: Vec<AVFilterInOut>) -> Option<AVFilterInOut> {
    let ptrs: Vec<NonNull<ffi::AVFilterInOut>> =
        nodes.into_iter().map(AVFilterInOut::into_raw).collect();
    let head = *ptrs.first()?;
    // SAFETY: 指针全部来自上面刚交出所有权的节点，此刻没有任何包装体持有它们，
    // 且 `from_raw` 之前不会再有第二个所有者产生。`next` 是普通字段，写入独占。
    unsafe {
        for window in ptrs.windows(2) {
            (*window[0].as_ptr()).next = window[1].as_ptr();
        }
        Some(AVFilterInOut::from_raw(head))
    }
}

const DEFAULT_ORDERING: Ordering = Ordering::SeqCst;

/// 过滤器图表 —— 支持 **n 路输入 / m 路输出**。
///
/// 单输入线性链（`DecoderBuilder::with_filters` / `EncoderBuilder::with_filters`）
/// 与多输入合成图（[`FilterGraphBuilder`]）走同一套实现：前者就是「1 进 1 出」的
/// 退化情形。下标 0 是**主输入 / 主输出**，[`Self::process_frame`]、
/// [`Self::is_flushed`] 这些便捷方法都以它为准。
pub struct FilterGraph {
    graph: AVFilterGraph,
    /// 每路输出的状态；下标 0 为主输出。
    states: Vec<ProcessState>,
    initialized: AtomicBool,
    /// 每路输入的 EOF 是否已推过（`av_buffersrc_add_frame(src, NULL)`）。EOF 只能
    /// 推一次，重复推送会拿到 `AVERROR_EOF`，所以第二遍起直接返回 `Ok(())`。
    eof_sent: Vec<bool>,
    /// 图输入标签，下标是逻辑输入序号。
    input_labels: Vec<String>,
    /// 图输出标签，下标是逻辑输出序号。
    output_labels: Vec<String>,
    /// `buffer` / `abuffer` 源实例名，按逻辑输入序号排列。
    ///
    /// 实例名同时用作建图时交给 FFmpeg 的端点名，因此**必须**等于该路输入在 spec
    /// 里的标签：`avfilter_graph_parse_ptr` 按名字把调用方给的开放端点与图里的开放
    /// 端点配对，名字对不上就配不上（见 [`Self::setup_endpoints`]）。
    src_names: Vec<CString>,
    /// `buffersink` / `abuffersink` 汇实例名，按逻辑输出序号排列；命名规则同
    /// [`Self::src_names`]（名字 = 该路输出在 spec 里的标签）。
    sink_names: Vec<CString>,
    /// 各图输入端点的人类可读描述（`base=64x48/yuv420p …`），**只用于错误信息**：
    /// 取帧失败时最常见的原因是某路输入帧的像素/采样格式或尺寸与声明的端点不符，
    /// 而 FFmpeg 只会回一句 `EINVAL`，没有这份描述用户无从下手。
    input_specs: Vec<String>,
}

/// 端点的人类可读描述（错误信息用）。
fn describe_endpoint(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Video(v) => format!(
            "{}x{} {:?} {}fps",
            v.width,
            v.height,
            v.format,
            v.frame_rate.num()
        ),
        Endpoint::Audio(a) => format!("{}ch {:?} {}Hz", a.nb_channels, a.format, a.sample_rate),
    }
}

impl FilterGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: AVFilterGraph::new(),
            states: Vec::new(),
            initialized: AtomicBool::new(false),
            eof_sent: Vec::new(),
            input_labels: Vec::new(),
            output_labels: Vec::new(),
            src_names: Vec::new(),
            sink_names: Vec::new(),
            input_specs: Vec::new(),
        }
    }

    /// 第 `output` 路输出的推进阶段；下标越界（图尚未初始化）按 `Normal` 处理：
    /// 未建图 = 还没推进过任何阶段，因此 `is_drained_at`/`is_flushed_at` 都为假。
    fn state_at(&self, output: usize) -> ProcessState {
        self.states
            .get(output)
            .copied()
            .unwrap_or(ProcessState::Normal)
    }

    /// Rebuilds the graph from scratch, discarding everything the old one held.
    ///
    /// A filter graph has no "rewind": frames that went in cannot be taken back
    /// (`av_buffersrc_add_frame` offers no such operation), and once the sink has
    /// seen EOF it stays at EOF forever — every later `av_buffersrc_add_frame`
    /// fails with `AVERROR_EOF`. So the only correct way to restart a filtered
    /// pipeline (after a seek, or to reuse a drained decoder) is to throw the
    /// graph away and build a new one, which is what this does. The old
    /// `AVFilterGraph` is dropped, freeing its filters and every frame still
    /// buffered inside them.
    ///
    /// `params`/`filters` are the same values [`Self::init`] was given; the caller
    /// has to keep them for exactly this reason.
    pub(crate) fn rebuild(&mut self, params: &FilterParams, filters: &[Filter]) -> Result<()> {
        self.graph = AVFilterGraph::new();
        self.states.clear();
        self.initialized.store(false, DEFAULT_ORDERING);
        self.eof_sent.clear();
        self.input_labels.clear();
        self.output_labels.clear();
        self.src_names.clear();
        self.sink_names.clear();
        // 输入描述随 `input_labels` 一起重建：留着旧值会让错误信息指向上一张图。
        self.input_specs.clear();
        self.init(params, filters)
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(DEFAULT_ORDERING)
    }

    /// 建一张已初始化的滤镜图（`new` + [`init`](Self::init) 的合并入口）。
    ///
    /// 解码与编码两条流水线都用它建图，媒体类型校验、滤镜可用性校验
    /// （本构建没编入该滤镜 → [`Unsupported`](crate::RsmediaError::Unsupported)，
    /// 媒体类型不符 → [`InvalidConfig`](crate::RsmediaError::InvalidConfig)）
    /// 因此只有 [`init`](Self::init) 一处实现——调用方不需要在门外再抄一遍这些
    /// 检查，两份检查只会随 FFmpeg 版本漂移。
    ///
    /// **转义不在这一层**：滤镜描述里的特殊字符在构造 `Filter` 时（各便捷构造函数
    /// 内部走 `escape_filter_value`）就已经转义好了，`init` 只负责把 spec 用 `,`
    /// 拼成线性链。
    pub(crate) fn build(params: &FilterParams, filters: &[Filter]) -> Result<FilterGraph> {
        let mut graph = Self::new();
        graph
            .init(params, filters)
            .context("Failed to initialize filter graph")?;
        Ok(graph)
    }

    /// 主输出（下标 0）是否正在排空（EOF 已送出，图里还有缓冲帧要出）。
    ///
    /// 判据是"EOF 已送到每一路输入"（最后一路 EOF 推送成功的那一刻置位），
    /// **不是** `av_buffersink_get_frame` 返回过 `EAGAIN`：流中段的 `EAGAIN`
    /// （还在等其它输入、滤镜缓冲未攒够）不改变状态，否则它在流中段就永久为真
    /// （契约见 `state::ProcessState`）。
    pub fn is_drained(&self) -> bool {
        self.is_drained_at(0)
    }

    /// 第 `output` 路输出是否已 drain。
    pub fn is_drained_at(&self, output: usize) -> bool {
        self.state_at(output).is_drained()
    }

    /// 主输出（下标 0）是否已到流末尾（`av_buffersink_get_frame` 返回 `EOF`）。
    pub fn is_flushed(&self) -> bool {
        self.is_flushed_at(0)
    }

    /// 第 `output` 路输出是否已到流末尾。
    pub fn is_flushed_at(&self, output: usize) -> bool {
        self.state_at(output).is_flushed()
    }

    /// 初始化过滤器图表（单输入线性链：1 路进、1 路出）。
    ///
    /// 滤镜之间靠 `,` 顺序串联，由 FFmpeg 自动接线，因此这一路只接受单输入 /
    /// 单输出滤镜；需要 `overlay` / `amix` 这类多输入滤镜时改用
    /// [`FilterGraphBuilder`]。
    ///
    /// `filters` 至少要有**一个**元素：空切片会拼出空描述串，走到 FFmpeg 那里
    /// 只会得到一句无法定位的解析错误。想要"什么都不做"请显式传一个 `null` 滤镜。
    pub fn init(&mut self, params: &FilterParams, filters: &[Filter]) -> Result<()> {
        if self.is_initialized() {
            return Err(RsmediaError::invalid_config(
                "Filter graph already initialized",
            ));
        }
        if filters.is_empty() {
            // 空描述串不是"直通"，而是一串解析失败；调用方改传一个滤镜即可 ⇒
            // 调用方配置问题。
            return Err(RsmediaError::invalid_config(
                "filter graph has no filter: init needs at least one, \
                 pass a `null` filter for a passthrough",
            ));
        }
        for filter in filters {
            Self::check_filter(filter, params.media_type())?;
        }

        // 线性链描述：没有标签，靠 `,` 顺序串联。
        let filter_spec = filters
            .iter()
            .map(|f| f.spec())
            .collect::<Vec<_>>()
            .join(",");

        let (input, output) = (params.input_endpoint(), params.output_endpoint());

        // 线性链的 spec 不带标签，FFmpeg 把这种开放端点约定为 `in` / `out`
        // （见 `SINGLE_INPUT_LABEL` 的说明），端点名与实例名都用这两个名字。
        self.input_labels = vec![SINGLE_INPUT_LABEL.to_string()];
        self.output_labels = vec![SINGLE_OUTPUT_LABEL.to_string()];
        // 与 `FilterGraphBuilder::build` 一样登记输入描述：`init` 这条路径
        // （解码/编码流水线用的单输入线性链）同样会在取帧失败时打印它，漏了就只剩
        // 一句没有上下文的 `EINVAL`。
        self.input_specs = vec![format!(
            "{SINGLE_INPUT_LABEL}={}",
            describe_endpoint(&input)
        )];
        self.eof_sent = vec![false];
        self.states = vec![ProcessState::Normal];

        self.setup_endpoints(
            &[(SINGLE_INPUT_LABEL.to_string(), input)],
            &[(SINGLE_OUTPUT_LABEL.to_string(), output)],
            &filter_spec,
        )?;

        self.graph.config()?;
        self.initialized.store(true, DEFAULT_ORDERING);

        Ok(())
    }

    /// 校验单个滤镜：是否存在于本次构建、媒体类型是否与图一致。
    fn check_filter(filter: &Filter, media_type: MediaType) -> Result<()> {
        // 名字必须是本 FFmpeg 构建里真实存在的滤镜：缺失时在此前置报错，
        // 而不是等到 parse 阶段返回一句难以定位的字符串错误。"这个构建没编入它"
        // （如 `drawtext` 需要 libfreetype、`subtitles` 需要 libass）是**本构建
        // 缺能力**而不是调用方配置错 ⇒ `Unsupported`，调用方据此跳过或降级。
        if get_by_name(filter.name())?.is_none() {
            return Err(RsmediaError::unsupported(format!(
                "filter '{}' is not available in this FFmpeg build",
                filter.name()
            )));
        }
        if filter.media_type() != media_type {
            return Err(RsmediaError::invalid_config(format!(
                "Filter '{}' media type mismatch: expected {:?}, got {:?}",
                filter.name(),
                media_type,
                filter.media_type()
            )));
        }
        Ok(())
    }

    /// 建一路视频输入端点（`buffer` 源），按端点声明的尺寸、像素格式、时间基与帧率配置。
    ///
    /// `buffer`: <https://ffmpeg.org/ffmpeg-filters.html#buffer>
    fn create_video_source(
        &self,
        name: &CStr,
        endpoint: &VideoEndpoint,
    ) -> Result<AVFilterContextMut<'_>> {
        // buffer 源按"这一路实际推入的帧格式"配置；sink 仍按目标格式约束。二者
        // 不同时（如 GIF 调色板链 RGB→pal8），格式转换由图内的滤镜完成。
        let args = CString::new(format!(
            "width={}:height={}:pix_fmt={}:time_base={}/{}:frame_rate={}/{}:pixel_aspect={}/{}",
            endpoint.width,
            endpoint.height,
            endpoint.format.get_pix_fmt_name(),
            endpoint.time_base.num(),
            endpoint.time_base.den(),
            endpoint.frame_rate.num(),
            endpoint.frame_rate.den(),
            endpoint.pixel_aspect.num(),
            endpoint.pixel_aspect.den(),
        ))?;

        let buffersrc = get_by_name("buffer")?.ok_or_else(|| {
            RsmediaError::unsupported("filter 'buffer' is not available in this FFmpeg build")
        })?;
        self.graph
            .create_filter_context(&buffersrc, name, Some(&args))
            .context("Failed to create video buffer source")
    }

    /// 建一路视频输出端点（`buffersink` 汇），把图内输出约束到 `format`。
    ///
    /// `buffersink`: <https://ffmpeg.org/ffmpeg-filters.html#buffersink>
    fn create_video_sink(
        &self,
        name: &CStr,
        format: PixelFormat,
    ) -> Result<AVFilterContextMut<'_>> {
        let buffersink = get_by_name("buffersink")?.ok_or_else(|| {
            RsmediaError::unsupported("filter 'buffersink' is not available in this FFmpeg build")
        })?;

        let mut sink_ctx = self
            .graph
            .alloc_filter_context(&buffersink, name)
            .context("Failed to allocate video buffer sink")?;

        // 先分配再设置选项、最后初始化。FFmpeg 8 起 `pix_fmts`(binary) 废弃为数组选项
        // `pixel_formats`，两者均为非运行时选项，须在 init 之前设置。
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        sink_ctx
            .opt_set_array(
                c"pixel_formats",
                0,
                Some(&[ffi::AVPixelFormat::from(format)]),
                ffi::AV_OPT_TYPE_PIXEL_FMT,
            )
            .context("Failed to set video sink filter context pixel format")?;
        #[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
        sink_ctx
            .opt_set_bin(c"pix_fmts", &(ffi::AVPixelFormat::from(format)))
            .context("Failed to set video sink filter context pixel format")?;
        sink_ctx
            .init_str(None)
            .context("Failed to init video buffer sink")?;
        Ok(sink_ctx)
    }

    /// 建一路音频输入端点（`abuffer` 源），按端点声明的采样格式、采样率与通道数配置。
    ///
    /// `abuffer`: <https://ffmpeg.org/ffmpeg-filters.html#abuffer>
    fn create_audio_source(
        &self,
        name: &CStr,
        endpoint: &AudioEndpoint,
    ) -> Result<AVFilterContextMut<'_>> {
        // 与 `audio::audio_channel_desc` 共用同一套「通道数 → 布局描述」逻辑
        // （失败时回退到数字通道数，不会 panic）。
        let channel_desc = audio::audio_channel_desc(endpoint.nb_channels);

        let args = CString::new(format!(
            "time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout={}",
            endpoint.time_base.num(),
            endpoint.time_base.den(),
            endpoint.sample_rate,
            endpoint.format.get_sample_fmt_name(),
            channel_desc,
        ))?;

        let buffersrc = get_by_name("abuffer")?.ok_or_else(|| {
            RsmediaError::unsupported("filter 'abuffer' is not available in this FFmpeg build")
        })?;
        self.graph
            .create_filter_context(&buffersrc, name, Some(&args))
            .context("Failed to create audio buffer source")
    }

    /// 建一路音频输出端点（`abuffersink` 汇），把图内输出约束到指定的采样格式 /
    /// 采样率 / 通道布局。
    ///
    /// `abuffersink`: <https://ffmpeg.org/ffmpeg-filters.html#abuffersink>
    fn create_audio_sink(
        &self,
        name: &CStr,
        endpoint: &AudioEndpoint,
    ) -> Result<AVFilterContextMut<'_>> {
        let buffersink = get_by_name("abuffersink")?.ok_or_else(|| {
            RsmediaError::unsupported("filter 'abuffersink' is not available in this FFmpeg build")
        })?;

        let mut sink_ctx = self
            .graph
            .alloc_filter_context(&buffersink, name)
            .context("Failed to allocate audio buffer sink")?;

        // 先分配再设置选项、最后初始化，兼容 FFmpeg 8 中 sink 选项为非运行时选项的限制。
        // FFmpeg8 将如下参数废弃, 且新旧选项不能混用:
        // - buffersink ：新数组选项 pixel_formats （旧 pix_fmts 已废弃）
        // - abuffersink ：新数组选项 `sample_formats`/`samplerates`/`channel_layouts`
        // （旧 `sample_fmts`/`sample_rates`/`ch_layouts`(binary/string) 已废弃）
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        sink_ctx
            .opt_set_array(
                c"sample_formats",
                0,
                Some(&[endpoint.format as i32]),
                ffi::AV_OPT_TYPE_SAMPLE_FMT,
            )
            .context("Failed to set audio sink sample format")?;
        #[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
        sink_ctx.opt_set_bin(c"sample_fmts", &(endpoint.format as i32))?;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        sink_ctx
            .opt_set_array(
                c"samplerates",
                0,
                Some(&[endpoint.sample_rate]),
                ffi::AV_OPT_TYPE_INT,
            )
            .context("Failed to set audio sink sample rate")?;
        #[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
        sink_ctx.opt_set_bin(c"sample_rates", &endpoint.sample_rate)?;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        {
            let layout = AVChannelLayout::from_nb_channels(endpoint.nb_channels).into_inner();
            sink_ctx
                .opt_set_array(
                    c"channel_layouts",
                    0,
                    Some(&[layout]),
                    ffi::AV_OPT_TYPE_CHLAYOUT,
                )
                .context("Failed to set audio sink channel layout")?;
        }
        #[cfg(any(feature = "ffmpeg6", feature = "ffmpeg7"))]
        sink_ctx.opt_set(
            c"ch_layouts",
            &AVChannelLayout::from_nb_channels(endpoint.nb_channels).describe()?,
        )?;
        sink_ctx
            .init_str(None)
            .context("Failed to init audio buffer sink")?;
        Ok(sink_ctx)
    }

    /// 建图的开放端点：为每一路输入建 `buffer`/`abuffer` 源、为每一路输出建
    /// `buffersink`/`abuffersink` 汇，再交给 `avfilter_graph_parse_ptr` 与 spec 的
    /// 开放端点配对。
    ///
    /// `inputs` / `outputs` 是 `(端点名, 端点声明)`：**端点名必须等于该端点在 spec
    /// 里的标签**（不带标签的线性链用 `in` / `out`，见 [`SINGLE_INPUT_LABEL`]）——
    /// `avfilter_graph_parse_ptr` 是按名字把调用方给的开放端点与图里的开放端点配对的
    /// （实测：名字对不上时不会报错，端点会静默地不接线，直到 `config()` 才以
    /// "pad not connected" 失败）。顺序无关紧要，名字同时用作滤镜实例名，方便
    /// `get_filter` 查找与错误定位。
    fn setup_endpoints(
        &mut self,
        inputs: &[(String, Endpoint)],
        outputs: &[(String, Endpoint)],
        spec: &str,
    ) -> Result<()> {
        let mut src_names = Vec::with_capacity(inputs.len());
        let mut sink_names = Vec::with_capacity(outputs.len());
        let mut src_list = Vec::with_capacity(inputs.len());
        let mut sink_list = Vec::with_capacity(outputs.len());

        for (name, endpoint) in inputs {
            let name = strutils::str_to_cstring(name)?;
            let mut ctx = match endpoint {
                Endpoint::Video(v) => self.create_video_source(&name, v)?,
                Endpoint::Audio(a) => self.create_audio_source(&name, a)?,
            };
            src_list.push(AVFilterInOut::new(&name, &mut ctx, 0));
            src_names.push(name);
        }

        for (name, endpoint) in outputs {
            let name = strutils::str_to_cstring(name)?;
            let mut ctx = match endpoint {
                Endpoint::Video(v) => self.create_video_sink(&name, v.format)?,
                Endpoint::Audio(a) => self.create_audio_sink(&name, a)?,
            };
            sink_list.push(AVFilterInOut::new(&name, &mut ctx, 0));
            sink_names.push(name);
        }

        let spec_cstr = CString::new(spec)?;

        // FFmpeg 的参数命名与直觉相反：`inputs` 参数收的是**图的输出端**（汇的链表），
        // `outputs` 参数收的是**图的输入端**（源的链表）。
        let (rest_inputs, rest_outputs) = self
            .graph
            .parse_ptr(&spec_cstr, chain_inouts(sink_list), chain_inouts(src_list))
            .with_context(|| {
                format!(
                    "Failed to parse filter graph: {}",
                    spec_cstr.to_string_lossy()
                )
            })?;

        // 返回值是**配对后剩余**的节点链表，不是入参指针的原样回显：
        // `avfilter_graph_parse_ptr` 会把按名字配上的那些节点自己 `av_free` 掉
        // （节点由 `avfilter_inout_alloc` 分配，由 FFmpeg 释放才是对称的），
        // 没配上名字的节点才留在这对返回值里。全部配对成功时两个返回值都是 `None`；
        // 非空则意味着有端点没接上（名字写错正是这种情形），后续 `config()` 会以
        // "pad not connected" 失败。
        //
        // rsmpeg 侧在成功路径上对传入的链表做了 `into_raw()`（不析构），仅失败路径才
        // 析构，所以这里只释放这对返回值即可，不会重复释放。
        drop(rest_inputs);
        drop(rest_outputs);

        self.src_names = src_names;
        self.sink_names = sink_names;
        Ok(())
    }

    /// 处理主输入（下标 0）的单帧：推入一帧（`Some`）或 EOF（`None`），再取一帧。
    ///
    /// 单输入线性链的便捷入口；多输入图请用 [`Self::push_frame_to`] /
    /// [`Self::receive_frame_from`] 指明是哪一路。
    pub fn process_frame(&mut self, frame: Option<AVFrame>) -> Result<Option<AVFrame>> {
        self.push_frame_to(0, frame)?;
        self.receive_frame_from(0)
    }

    /// 把一帧（`Some`）或 EOF（`None`）推入第 `input` 路输入。
    ///
    /// EOF 只会推一次：重复 `av_buffersrc_add_frame(src, NULL)` 会返回
    /// `AVERROR_EOF`，所以第二次起直接返回 `Ok(())`。
    pub fn push_frame_to(&mut self, input: usize, frame: Option<AVFrame>) -> Result<()> {
        if !self.is_initialized() {
            return Err(RsmediaError::invalid_config("Filter graph not initialized"));
        }
        if input >= self.eof_sent.len() {
            return Err(RsmediaError::invalid_config(format!(
                "input index {input} out of range: graph has {} inputs",
                self.eof_sent.len()
            )));
        }
        let is_eof = frame.is_none();
        if is_eof && self.eof_sent[input] {
            return Ok(());
        }

        // src_ctx 在本块结束时释放借用（它借的是 `self.graph`，下面的状态记账
        // 因此必须等它释放之后再做）
        {
            let mut src_ctx = self.get_src_context(input)?;
            src_ctx
                .buffersrc_add_frame(frame, None)
                .context("Error submitting the frame to the filter graph.")?;
        }

        // EOF 只在**推送成功之后**才记账：失败的推送并没有让图进入 EOF，若先置位，
        // 这一路输入就被永久"毒化"——后面再推（含重试）都会被上面的短路直接跳过，
        // 帧静默丢失而调用方看到的是一路 `Ok(())`。
        if is_eof {
            self.eof_sent[input] = true;
            // `Drained` 的判据是"EOF 已送到每一路输入"，与 `state::ProcessState`
            // 的定义一致，因此在这里推进，而不是等取帧拿到 `EAGAIN`：后者会让
            // "刚推完 EOF、一次 `EAGAIN` 都没遇到过"的图仍停在 `Normal`。
            // 多输入图里只喂了一路时不会触发（还有输入没送 EOF）。
            if self.eof_sent.iter().all(|sent| *sent) {
                for state in &mut self.states {
                    if state.is_normal() {
                        *state = ProcessState::Drained;
                    }
                }
            }
        }
        Ok(())
    }

    /// 从第 `output` 路输出取一帧：`Ok(Some)` 是产出帧，`Ok(None)` 表示这一刻没有帧
    /// ——要么图还需要更多输入（`EAGAIN`，状态保持在 `Normal`），要么正在排空
    /// （EOF 已送出，状态置为 `Drained`）或已到流末尾（状态置为 `Flushed`）。
    pub fn receive_frame_from(&mut self, output: usize) -> Result<Option<AVFrame>> {
        if output >= self.states.len() {
            return Err(RsmediaError::invalid_config(format!(
                "output index {output} out of range: graph has {} outputs",
                self.states.len()
            )));
        }

        let filter_result = {
            // 借用在块结束时释放，避免与后面的状态更新冲突
            let mut sink_ctx = self.get_sink_context(output)?;
            sink_ctx.buffersink_get_frame(None)
        };

        // 获取处理后的帧
        match filter_result {
            Ok(frame) => Ok(Some(frame)),
            Err(rsmpeg::error::RsmpegError::BufferSinkDrainError) => {
                // `EAGAIN` 只说明**这一刻**没有帧：流中段同样会出现（帧同步类滤镜
                // 还在等其它输入、滤镜自身的缓冲未攒够），并不等于已经在排空。
                // 推进到 `Drained` 由 `push_frame_to` 在"EOF 已送到每一路输入"时完成
                // （契约见 `state::ProcessState`）；这里只记日志。
                tracing::debug!(
                    "filter graph: output {output} has no frame available (EAGAIN, eof sent: {})",
                    self.eof_sent.iter().all(|sent| *sent)
                );
                Ok(None)
            }
            Err(rsmpeg::error::RsmpegError::BufferSinkEofError) => {
                tracing::debug!("filter graph: buffer sink eof error");
                self.states[output] = ProcessState::Flushed;
                Ok(None)
            }
            Err(e) => Err(RsmediaError::from(e).with_context(format!(
                "filter graph output {output} produced no frame; a frequent cause is an input \
                 frame whose pixel/sample format or size differs from the one declared for that \
                 graph input, which FFmpeg reports only as EINVAL. Declared inputs: [{}]. \
                 Convert the frame to the declared format (e.g. MediaFrame::convert_to) or \
                 declare the format you actually feed",
                self.input_specs.join(", ")
            ))),
        }
    }

    /// 刷新过滤器链：给**所有**输入推一次 EOF，再把主输出（下标 0）缓存的帧全部取出。
    ///
    /// 多输出图请对每一路调用 [`Self::drain_output`]——本方法只收主输出，其余输出上
    /// 的帧仍留在图里，不会被丢弃。
    pub fn flush(&mut self) -> Result<Vec<AVFrame>> {
        self.drain_output(0)
    }

    /// 给所有输入推 EOF，然后把第 `output` 路输出里缓存的帧全部取出。
    pub fn drain_output(&mut self, output: usize) -> Result<Vec<AVFrame>> {
        if !self.is_initialized() {
            return Err(RsmediaError::invalid_config("Filter graph not initialized"));
        }
        if output >= self.states.len() {
            return Err(RsmediaError::invalid_config(format!(
                "output index {output} out of range: graph has {} outputs",
                self.states.len()
            )));
        }
        if self.state_at(output).is_flushed() {
            tracing::debug!("Filter graph already flushed.");
            return Ok(Vec::new());
        }

        // 只有推过 EOF，图才会吐出缓冲帧（每一路都推，否则多输入图仍会等数据）。
        for input in 0..self.eof_sent.len() {
            self.push_frame_to(input, None)?;
        }

        let mut frames = Vec::new();
        let mut drained_iterations = 0usize;

        loop {
            match self.receive_frame_from(output) {
                Ok(Some(frame)) => {
                    drained_iterations = 0;
                    frames.push(frame);
                }
                Ok(None) => {
                    if self.state_at(output).is_flushed() {
                        break;
                    }
                    // EAGAIN：图里仍有缓冲帧要出，继续拉取；但个别滤镜可能一直回
                    // EAGAIN 而不进入 Flushed，故设上限收尾（与解码/编码排空一致）。
                    // 触顶时不能静默返回半截结果：残留帧会被丢掉，必须让调用者知道。
                    if drained_iterations >= crate::MAX_DRAIN_ITERATIONS {
                        tracing::error!(
                            "Filter graph keeps returning EAGAIN while flushing; \
                             giving up after {} iterations",
                            crate::MAX_DRAIN_ITERATIONS
                        );
                        return Err(RsmediaError::msg(format!(
                            "Filter graph stalled while flushing: still no output after {} \
                             iterations; {} already-dequeued frames are discarded",
                            crate::MAX_DRAIN_ITERATIONS,
                            frames.len()
                        )));
                    }
                    drained_iterations += 1;
                    tracing::trace!("Filter graph draining during flush...");
                }
                Err(e) => {
                    tracing::error!("Error encountered during filter graph flush: {e}");
                    return Err(e);
                }
            }
        }

        Ok(frames)
    }

    /// 第 `input` 路源（`buffer`/`abuffer`）的实例名。
    fn src_name(&self, input: usize) -> Result<&CString> {
        self.src_names.get(input).ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "input index {input} out of range: graph has {} inputs",
                self.src_names.len()
            ))
        })
    }

    /// 第 `output` 路汇（`buffersink`/`abuffersink`）的实例名。
    fn sink_name(&self, output: usize) -> Result<&CString> {
        self.sink_names.get(output).ok_or_else(|| {
            RsmediaError::invalid_config(format!(
                "output index {output} out of range: graph has {} outputs",
                self.sink_names.len()
            ))
        })
    }

    /// 动态获取第 `input` 路源过滤器上下文
    fn get_src_context(&mut self, input: usize) -> Result<AVFilterContextMut<'_>> {
        let name = self.src_name(input)?;
        self.graph
            .get_filter(name.as_c_str())
            .context("Source filter context not found")
    }

    /// 动态获取第 `output` 路目标过滤器上下文
    fn get_sink_context(&mut self, output: usize) -> Result<AVFilterContextMut<'_>> {
        let name = self.sink_name(output)?;
        self.graph
            .get_filter(name.as_c_str())
            .context("Sink filter context not found")
    }

    /// 图输入路数。
    pub fn input_count(&self) -> usize {
        self.src_names.len()
    }

    /// 图的输出路数。
    pub fn output_count(&self) -> usize {
        self.sink_names.len()
    }

    /// Frame rate of the primary (index `0`) output link, as reported by
    /// `av_buffersink_get_frame_rate`.
    ///
    /// A filter such as `fps` or `framerate` rewrites the frame rate, and an
    /// encoder fed from this graph has to follow it: a time base derived from
    /// the *input* parameters no longer matches the pts the graph emits.
    ///
    /// When no frame rate has been negotiated the sink reports `0/0`, which
    /// folds to [`Rational::ZERO`] — the spelling every consumer in this crate
    /// already reads as "unset". That is a *value*, not an error.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when the graph declares no
    /// output, or when the sink is missing from the graph.
    pub fn output_frame_rate(&mut self) -> Result<Rational> {
        self.output_frame_rate_at(0)
    }

    /// Frame rate of the `output`-th output link.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `output` is out of range, or
    /// when the sink is missing from the graph. An out-of-range index is a
    /// programming error and is deliberately *not* folded into "no frame rate":
    /// callers must be able to tell the two apart.
    pub fn output_frame_rate_at(&mut self, output: usize) -> Result<Rational> {
        Ok(self.get_sink_context(output)?.get_frame_rate().into())
    }

    /// Time base of the primary (index `0`) output link, as reported by
    /// `av_buffersink_get_time_base`.
    ///
    /// A frame's pts comes out of the graph in *this* time base, not in the one
    /// the input endpoint declared, so anything consuming filter output has to
    /// rescale against it.
    ///
    /// An unnegotiated sink reports `0/0`, which folds to [`Rational::ZERO`].
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when the graph declares no
    /// output, or when the sink is missing from the graph.
    pub fn output_time_base(&mut self) -> Result<Rational> {
        self.output_time_base_at(0)
    }

    /// Time base of the `output`-th output link.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `output` is out of range, or
    /// when the sink is missing from the graph.
    pub fn output_time_base_at(&mut self, output: usize) -> Result<Rational> {
        Ok(self.get_sink_context(output)?.get_time_base().into())
    }

    /// Size `(width, height)` of the primary (index `0`) output link, as
    /// reported by `av_buffersink_get_w` / `av_buffersink_get_h`.
    ///
    /// `scale`, `crop`, `pad`, `rotate` and `transpose` all change the size, and
    /// an encoder fed from this graph has to be opened with the *filter's*
    /// size — sending a frame of a different size fails.
    ///
    /// A sink that has not negotiated a size reports `(0, 0)`; an audio sink has
    /// no size at all and reports the same. Both are values, not errors.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when the graph declares no
    /// output, or when the sink is missing from the graph.
    pub fn output_size(&mut self) -> Result<(i32, i32)> {
        self.output_size_at(0)
    }

    /// Size `(width, height)` of the `output`-th output link.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `output` is out of range, or
    /// when the sink is missing from the graph.
    pub fn output_size_at(&mut self, output: usize) -> Result<(i32, i32)> {
        let sink = self.get_sink_context(output)?;
        Ok((sink.get_w(), sink.get_h()))
    }

    /// 运行时给图里的滤镜发一条命令（`avfilter_graph_send_command`）。
    ///
    /// 图一旦 `build()`，各滤镜的选项就被冻结了；要改一个参数，要么整张图重建
    /// （断流重连，直播/交互场景不可接受），要么走这条路径。支持 `cmd` 的滤镜会
    /// 在**处理下一帧时**应用新参数，因此适合调音量、转角度、改文字这类在线调整。
    ///
    /// - `target`  —— `"all"` 发给所有滤镜；否则按滤镜名（如 `"volume"`）或滤镜
    ///   实例名匹配，命中多个就都发。
    /// - `command` —— 命令名，只能是字母数字（FFmpeg 的硬性要求，见
    ///   `ffmpeg -h filter=<name>` 的 "Commands" 段）。
    /// - `arg`     —— 命令参数，语法由滤镜自己定义（如 `volume` 的 `"0.5"`）。
    ///
    /// 成功时返回滤镜写回的响应文本（多数滤镜不写，返回空串）。滤镜不认识该命令
    /// 时 FFmpeg 报 `AVERROR(ENOSYS)`，此处映射为 [`RsmediaError::Unsupported`]；
    /// 其余失败（target 不存在、参数被拒）保留原始 FFmpeg 错误码。
    ///
    /// # Errors
    ///
    /// 图尚未 `build()`、`target`/`command`/`arg` 含内部 NUL，或滤镜拒绝了这条
    /// 命令。
    pub fn send_command(&mut self, target: &str, command: &str, arg: &str) -> Result<String> {
        if !self.is_initialized() {
            return Err(RsmediaError::invalid_config(
                "Filter graph not initialized; send_command needs a built graph",
            ));
        }

        let target_c = strutils::str_to_cstring(target)?;
        let command_c = strutils::str_to_cstring(command)?;
        let arg_c = strutils::str_to_cstring(arg)?;

        // FFmpeg 用 av_strlcpy 往 res 里写响应，保证 NUL 结尾；给足 256 字节。
        // 元素类型跟随平台的 `c_char`（aarch64-linux 是 u8、macOS 与 x86_64 是 i8），
        // 写死任一种都会在另一种平台上编译不过——binding 的签名就是 `c_char`。
        let mut res = [0 as std::ffi::c_char; 256];
        let ret = unsafe {
            ffi::avfilter_graph_send_command(
                self.graph.as_mut_ptr(),
                target_c.as_ptr(),
                command_c.as_ptr(),
                arg_c.as_ptr(),
                res.as_mut_ptr(),
                res.len() as i32,
                0,
            )
        };

        if ret < 0 {
            // FFmpeg 文档明确：命令不被支持时返回 AVERROR(ENOSYS)。
            if ret == -(ffi::ENOSYS as i32) {
                return Err(RsmediaError::unsupported(format!(
                    "Filter {target:?} does not implement the command {command:?}; \
                     see `ffmpeg -h filter=<name>` for the commands it accepts"
                )));
            }
            return Err(RsmediaError::av_error(ret).with_context(format!(
                "Failed to send command {command:?} to filter {target:?} \
                 (unknown target, or the filter rejected the argument {arg:?})"
            )));
        }

        // res 由 FFmpeg 用 av_strlcpy 写入，保证 NUL 结尾。
        Ok(strutils::cstr_to_string_lossy(unsafe {
            CStr::from_ptr(res.as_ptr())
        }))
    }
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FilterGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FilterGraph nb_filters:{}, initialized:{}, state:{:?}, inputs:{:?}, outputs:{:?}",
            self.graph.nb_filters,
            self.is_initialized(),
            self.state_at(0),
            self.input_labels,
            self.output_labels,
        )
    }
}

/// 多输入 / 多输出滤镜图的构建器（组装层）。
///
/// 滤镜图分四层：叶子 [`Filter`]（滤镜是什么）→ 节点 [`FilterNode`]（接线：每一路
/// 输入接到哪个上游、输出叫什么名字）→ 端点 [`Endpoint`]（每一路进/出时的格式）→
/// 图 [`FilterGraph`]（本构建器组装并协商出来的结果）。
///
/// 组装分四步：
/// 1. [`add_input`](Self::add_input) 声明图输入（标签自动分配为 `in0`、`in1`…）；
/// 2. [`add_node`](Self::add_node) 追加滤镜节点；多输入滤镜（`overlay`/`amix`/…）
///    必须用 [`FilterNode::with_inputs`] 指明每一路的上游标签；
/// 3. [`add_output`](Self::add_output)（接到指定节点标签）或
///    [`add_output_tail`](Self::add_output_tail)（接链尾）声明图输出；
/// 4. [`build`](Self::build) 校验接线、生成滤镜图描述并协商格式。
///
/// 五个高频形态有现成构造函数，直接返回建好的图：[`overlay`](Self::overlay)、
/// [`hstack`](Self::hstack)、[`vstack`](Self::vstack)、[`concat`](Self::concat)、
/// [`amix`](Self::amix)。需要别的形态时用上面的四步自由组装，
/// [`Filter::new`] 仍是底层逃生舱。
///
/// [`build`](Self::build) 在**触碰 FFmpeg 之前**拒绝这些接线错误（都是
/// [`RsmediaError::InvalidConfig`]）：标签非法或重名、上游标签不存在（含前向引用）、
/// 接错媒体类型（视频/音频混接）、多输入滤镜的接线数与 pad 数不符、输出标签数与
/// 输出 pad 数不符、某个输出没人消费（悬空）或被消费多次。FFmpeg 的 pad 是一对一
/// 接线：fan-out 必须显式插一个复制节点——[`video::split`] / [`audio::asplit`]，
/// 用 [`FilterNode::with_outputs`] 给每个副本标名后各接一条链路。
///
/// # Examples
///
/// 画中画：主画面 + 右下角小图，输出仍是主画面尺寸。
///
/// ```no_run
/// use rsmedia::Rational;
/// use rsmedia::filter::{FilterGraphBuilder, VideoEndpoint};
/// use rsmedia::PixelFormat;
///
/// # fn main() -> rsmedia::Result<()> {
/// let tb = Rational::new(1, 25).unwrap();
/// let fps = Rational::new(25, 1).unwrap();
/// let main = VideoEndpoint::new(1280, 720, PixelFormat::YUV420P)
///     .with_time_base(tb)
///     .with_frame_rate(fps);
/// // 叠加层不必与主画面同尺寸、同像素格式：FFmpeg 会自动插 scale。
/// let logo = VideoEndpoint::new(160, 90, PixelFormat::RGBA)
///     .with_time_base(tb)
///     .with_frame_rate(fps);
///
/// let mut graph = FilterGraphBuilder::overlay(main, logo, "W-w-20", "H-h-20", main)?;
/// // 图输入 0 = 主画面、图输入 1 = 叠加层；唯一的输出是合成结果。
/// assert_eq!((graph.input_count(), graph.output_count()), (2, 1));
/// let _ = graph.receive_frame_from(0)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct FilterGraphBuilder {
    /// 图输入：标签 + 端点，下标即逻辑输入序号。
    inputs: Vec<(String, Endpoint)>,
    /// 图输出：`None` = 接链尾（最后一个节点的输出）。
    outputs: Vec<(Option<String>, Endpoint)>,
    /// 滤镜节点，顺序即它们在图里的创建顺序（也就是自动接线的顺序）。
    nodes: Vec<FilterNode>,
}

/// 接线解析的结果：`(每个节点的上游标签, 每个标签被消费的次数)`。
///
/// 抽成别名只为让 `resolve_wiring` 的签名保持可读：分量含义见
/// `FilterGraphBuilder::build` 里的注释。
type Wiring = (Vec<Vec<String>>, HashMap<String, usize>);

impl FilterGraphBuilder {
    /// 建一个空的构建器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 声明一路图输入，标签自动分配为 `in{序号}`（用
    /// [`input_label`](Self::input_label) 读回）。
    ///
    /// 端点描述的是**这一路帧推进来时**的格式，不做任何预处理；各路不必一致，
    /// FFmpeg 会在链路协商阶段自动插入 `scale` / `aresample`。
    ///
    /// 注意"各路不一致"指的是**端点之间**可以不同，而推进来的帧应当与它对应的
    /// 端点声明一致：多输入图里基底那一路若喂了别的像素格式/尺寸，FFmpeg 只会在
    /// 取帧时回一句 `EINVAL`（错误信息里会列出各输入声明的格式以便定位）。
    /// 需要转换时先用 `MediaFrame::convert_to` / [`crate::Scaler`] 转好再推。
    pub fn add_input(&mut self, endpoint: impl Into<Endpoint>) -> &mut Self {
        let label = format!("in{}", self.inputs.len());
        self.inputs.push((label, endpoint.into()));
        self
    }

    /// 用指定标签声明一路图输入（标签会写进滤镜图描述，只能是 ASCII 字母、数字与
    /// 下划线，且不能与任何节点输出标签重名）。
    pub fn add_input_with(
        &mut self,
        label: impl Into<String>,
        endpoint: impl Into<Endpoint>,
    ) -> &mut Self {
        self.inputs.push((label.into(), endpoint.into()));
        self
    }

    /// 第 `index` 路图输入的标签（`None` = 该路不存在）。
    pub fn input_label(&self, index: usize) -> Option<&str> {
        self.inputs.get(index).map(|(label, _)| label.as_str())
    }

    /// 已声明的图输入路数。
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// 已声明的图输出路数。
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// 追加一个滤镜节点。
    ///
    /// 节点的接线为空（[`FilterNode::new`]）时自动接链尾：第一个节点接**第一路**
    /// 图输入（用 [`Self::add_input`] 声明时它的标签是 `in0`；用
    /// [`Self::add_input_with`] 时就是给定的那个标签），其余接上一个节点的**末位**
    /// 输出 pad。多输入滤镜必须用 [`FilterNode::with_inputs`] 显式接线，否则建图时
    /// 以「接线数与 pad 数不符」报错。
    pub fn add_node(&mut self, node: impl Into<FilterNode>) -> &mut Self {
        self.nodes.push(node.into());
        self
    }

    /// 声明一路图输出，接在 `upstream` 指定的节点输出标签上。
    pub fn add_output(
        &mut self,
        upstream: impl Into<String>,
        endpoint: impl Into<Endpoint>,
    ) -> &mut Self {
        self.outputs.push((Some(upstream.into()), endpoint.into()));
        self
    }

    /// 声明一路图输出，接在**最后一个节点**的输出（链尾）上。
    pub fn add_output_tail(&mut self, endpoint: impl Into<Endpoint>) -> &mut Self {
        self.outputs.push((None, endpoint.into()));
        self
    }

    /// 校验图的三个必填部分（输入、节点、输出）都非空。
    fn check_arity(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(RsmediaError::invalid_config(
                "filter graph has no input: declare one with add_input",
            ));
        }
        if self.nodes.is_empty() {
            return Err(RsmediaError::invalid_config(
                "filter graph has no node: add one with add_node",
            ));
        }
        if self.outputs.is_empty() {
            return Err(RsmediaError::invalid_config(
                "filter graph has no output: declare one with add_output/add_output_tail",
            ));
        }
        Ok(())
    }

    /// 校验图输入标签，并把它们收进 `seen`。
    ///
    /// 图输入与节点输出共用**一个**标签命名空间：重名会让描述里的 `[label]`
    /// 指代不明，所以这里回收的 `seen` 会被后续的节点标签解析继续使用。
    fn collect_input_labels(&self) -> Result<HashSet<String>> {
        let mut seen: HashSet<String> = HashSet::new();
        for (label, _) in &self.inputs {
            if !is_valid_label(label) {
                return Err(RsmediaError::invalid_config(format!(
                    "invalid filter graph label '{label}': a label must be non-empty and consist \
                     of ASCII letters, digits and underscores only"
                )));
            }
            if !seen.insert(label.clone()) {
                return Err(RsmediaError::invalid_config(format!(
                    "duplicate label '{label}': graph inputs and node outputs share one namespace"
                )));
            }
        }
        Ok(seen)
    }

    /// 解析每个节点的输出标签。
    ///
    /// 显式标签优先（多输出滤镜必须逐个标注），单输出滤镜没写标签时按顺序自动
    /// 分配 `n0`、`n1`…；标签个数必须与滤镜的输出 pad 数一致。
    fn resolve_node_labels(&self, seen: &mut HashSet<String>) -> Result<Vec<Vec<String>>> {
        let mut node_labels: Vec<Vec<String>> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let labels: Vec<String> = if node.outputs().is_empty() {
                vec![format!("n{index}")]
            } else {
                node.outputs().to_vec()
            };
            let (pads, dynamic) = filter_output_pads(node.filter().name())?;
            if !dynamic && pads != labels.len() {
                let hint = if node.outputs().is_empty() {
                    "label every output pad with FilterNode::with_outputs"
                } else {
                    "use FilterNode::with_outputs to label every output pad"
                };
                return Err(RsmediaError::invalid_config(format!(
                    "filter '{}' (node {index}) has {pads} output pad(s) but {} label(s) were given: \
                     {hint}",
                    node.filter().name(),
                    labels.len()
                )));
            }
            for label in &labels {
                if !is_valid_label(label) {
                    return Err(RsmediaError::invalid_config(format!(
                        "invalid filter graph label '{label}': a label must be non-empty and \
                         consist of ASCII letters, digits and underscores only"
                    )));
                }
                if !seen.insert(label.clone()) {
                    let auto = if node.outputs().is_empty() {
                        " (the label was auto-assigned; give the node an explicit label)"
                    } else {
                        ""
                    };
                    return Err(RsmediaError::invalid_config(format!(
                        "duplicate label '{label}': graph inputs and node outputs share one \
                         namespace{auto}"
                    )));
                }
            }
            node_labels.push(labels);
        }
        Ok(node_labels)
    }

    /// 逐节点解析接线，返回每个节点的上游标签，以及每个标签被消费的次数。
    ///
    /// 上游必须是**已经声明过**的标签（图输入或前序节点的输出），所以图是顺序
    /// 搭起来的，不存在前向引用。
    fn resolve_wiring(&self, node_labels: &[Vec<String>]) -> Result<Wiring> {
        let mut available: HashMap<String, MediaType> = self
            .inputs
            .iter()
            .map(|(label, endpoint)| (label.clone(), endpoint.media_type()))
            .collect();
        // 标签被消费的次数。每个 pad 只能接一条下游链路，因此每处都必须是 1。
        let mut consumers: HashMap<String, usize> = HashMap::new();
        let mut wires: Vec<Vec<String>> = Vec::with_capacity(self.nodes.len());

        for (index, node) in self.nodes.iter().enumerate() {
            let media_type = node.filter().media_type();
            let resolved: Vec<String> = if node.inputs().is_empty() {
                // 自动接线：链首接图输入 0，其余接上一个节点的**末位**输出 pad
                // （单输出滤镜只有这一个 pad，多输出滤镜需要显式接线）。
                vec![if index == 0 {
                    self.inputs[0].0.clone()
                } else {
                    node_labels[index - 1]
                        .last()
                        .expect("every node has at least one output label")
                        .clone()
                }]
            } else {
                node.inputs().to_vec()
            };

            for label in &resolved {
                let Some(&upstream) = available.get(label) else {
                    return Err(RsmediaError::invalid_config(format!(
                        "filter '{}' (node {index}) is wired to '{label}', which is neither a \
                         declared graph input nor the output label of an earlier node",
                        node.filter().name()
                    )));
                };
                if upstream != media_type {
                    return Err(RsmediaError::invalid_config(format!(
                        "media type mismatch: filter '{}' (node {index}) is {media_type:?} but its \
                         input '{label}' carries {upstream:?}",
                        node.filter().name()
                    )));
                }
                *consumers.entry(label.clone()).or_insert(0) += 1;
            }

            // 静态输入滤镜按 pad 数精确校验；动态输入滤镜（hstack/amix/concat…）的
            // pad 由标签数量决定，路数与滤镜自己的 `inputs=`/`n=` 选项是否一致交由
            // FFmpeg 在 config 阶段核对。
            let (pads, dynamic) = filter_input_pads(node.filter().name())?;
            if !dynamic && pads != resolved.len() {
                return Err(RsmediaError::invalid_config(format!(
                    "filter '{}' (node {index}) takes {pads} input(s) but {} label(s) were wired: \
                     use FilterNode::with_inputs to wire every input pad",
                    node.filter().name(),
                    resolved.len()
                )));
            }

            for label in &node_labels[index] {
                available.insert(label.clone(), media_type);
            }
            wires.push(resolved);
        }
        Ok((wires, consumers))
    }

    /// 解析图输出：每一路输出都必须落在**某个节点的输出标签**上。
    ///
    /// 图输入是数据入口，不能直接当输出（需要直通时插一个 `null`/`anull` 节点）。
    fn resolve_outputs(
        &self,
        node_labels: &[Vec<String>],
        consumers: &mut HashMap<String, usize>,
    ) -> Result<Vec<(String, Endpoint)>> {
        let node_output_types: HashMap<&str, MediaType> = node_labels
            .iter()
            .enumerate()
            .flat_map(|(index, labels)| {
                let media_type = self.nodes[index].filter().media_type();
                labels.iter().map(move |label| (label.as_str(), media_type))
            })
            .collect();

        let mut resolved_outputs: Vec<(String, Endpoint)> = Vec::with_capacity(self.outputs.len());
        for (index, (upstream, endpoint)) in self.outputs.iter().enumerate() {
            let label = match upstream {
                Some(label) => label.clone(),
                // `add_output_tail`：接在最后一个节点的末位输出 pad 上（nodes 非空已校验）。
                None => node_labels[node_labels.len() - 1]
                    .last()
                    .expect("every node has at least one output label")
                    .clone(),
            };
            let Some(&producer) = node_output_types.get(label.as_str()) else {
                return Err(RsmediaError::invalid_config(format!(
                    "output {index} is wired to '{label}', which no node produces: graph outputs \
                     must come from a node's output label (insert a null/anull pass-through node \
                     to expose a graph input directly)"
                )));
            };
            if producer != endpoint.media_type() {
                return Err(RsmediaError::invalid_config(format!(
                    "media type mismatch: output {index} is {:?} but '{label}' carries {producer:?}",
                    endpoint.media_type()
                )));
            }
            *consumers.entry(label.clone()).or_insert(0) += 1;
            resolved_outputs.push((label, *endpoint));
        }
        Ok(resolved_outputs)
    }

    /// 每个标签恰好被消费一次：0 次是悬空（图里有死代码，或声明了没人用的
    /// 输入），多于 1 次是 fan-out（FFmpeg 一个输出 pad 只能接一个下游）。
    fn check_consumers(
        &self,
        node_labels: &[Vec<String>],
        consumers: &HashMap<String, usize>,
    ) -> Result<()> {
        for (label, _) in &self.inputs {
            match consumers.get(label).copied().unwrap_or(0) {
                1 => {}
                0 => {
                    return Err(RsmediaError::invalid_config(format!(
                        "graph input '{label}' is never used by any node: wire it into a filter or \
                         remove it"
                    )));
                }
                times => {
                    return Err(RsmediaError::invalid_config(format!(
                        "graph input '{label}' is consumed {times} times: one pad feeds exactly one \
                         downstream input; insert a split/asplit node to fan out"
                    )));
                }
            }
        }
        for (index, labels) in node_labels.iter().enumerate() {
            for label in labels {
                match consumers.get(label).copied().unwrap_or(0) {
                    1 => {}
                    0 => {
                        let hint = if labels.len() > 1 {
                            " (every output pad of the node must be consumed)"
                        } else {
                            ""
                        };
                        return Err(RsmediaError::invalid_config(format!(
                            "filter '{}' (node {index}) produces '{label}' but nothing consumes it: \
                             wire it into a downstream node or declare it as a graph output{hint}",
                            self.nodes[index].filter().name()
                        )));
                    }
                    times => {
                        return Err(RsmediaError::invalid_config(format!(
                            "label '{label}' (node {index}) is consumed {times} times: one pad feeds \
                             exactly one downstream input; insert a split/asplit node (or add a \
                             video::split / audio::asplit node) to fan out"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// 生成带标签的滤镜图描述。
    ///
    /// 每段一个滤镜（用 `;` 分隔 filterchain，避免 `,` 把前一个滤镜未连接的输出
    /// 自动接到后一个滤镜上），段内先写这一路的输入标签、再写滤镜、最后写本节
    /// 点的输出标签。开放端点的名字就是这些标签，FFmpeg 按名字把它们与
    /// `buffer` / `buffersink` 端点配对（见 `setup_endpoints`）。
    fn build_spec(&self, node_labels: &[Vec<String>], wires: &[Vec<String>]) -> String {
        let mut segments = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let mut segment = String::new();
            for label in &wires[index] {
                segment.push('[');
                segment.push_str(label);
                segment.push(']');
            }
            segment.push_str(&node.filter().spec());
            // 输出标签按 pad 顺序写：多输出滤镜（`split`/`asplit`）在这里把每个 pad
            // 显式暴露出来，下游才引用得到。
            for label in &node_labels[index] {
                segment.push('[');
                segment.push_str(label);
                segment.push(']');
            }
            segments.push(segment);
        }
        segments.join(";")
    }

    /// 建 `buffer`/`buffersink` 端点、装好端点列表，并由
    /// `avfilter_graph_config` 完成格式协商。
    fn assemble(&self, resolved_outputs: &[(String, Endpoint)], spec: &str) -> Result<FilterGraph> {
        let endpoints_in: Vec<(String, Endpoint)> = self
            .inputs
            .iter()
            .map(|(label, endpoint)| (label.clone(), *endpoint))
            .collect();
        let endpoints_out: Vec<(String, Endpoint)> = resolved_outputs
            .iter()
            .map(|(label, endpoint)| (label.clone(), *endpoint))
            .collect();

        let mut graph = FilterGraph::new();
        graph.input_labels = self.inputs.iter().map(|(label, _)| label.clone()).collect();
        graph.input_specs = self
            .inputs
            .iter()
            .map(|(label, endpoint)| format!("{label}={}", describe_endpoint(endpoint)))
            .collect();
        graph.output_labels = resolved_outputs
            .iter()
            .map(|(label, _)| label.clone())
            .collect();
        graph.eof_sent = vec![false; self.inputs.len()];
        graph.states = vec![ProcessState::Normal; resolved_outputs.len()];

        graph
            .setup_endpoints(&endpoints_in, &endpoints_out, spec)
            .with_context(|| format!("Failed to build filter graph: {spec}"))?;
        graph
            .graph
            .config()
            .with_context(|| format!("Failed to configure filter graph: {spec}"))?;
        graph.initialized.store(true, DEFAULT_ORDERING);
        Ok(graph)
    }

    /// 组装并协商整张图：校验接线与标签、生成带标签的滤镜图描述、按标签建
    /// `buffer`/`abuffer` 源与 `buffersink`/`abuffersink` 汇，最后由
    /// `avfilter_graph_config` 完成格式协商。
    ///
    /// 校验都在触碰 FFmpeg 之前完成，接线错误报的是 [`RsmediaError::InvalidConfig`]，
    /// 而不是一句难以定位的 FFmpeg 解析错误。
    pub fn build(&self) -> Result<FilterGraph> {
        self.check_arity()?;

        // 图输入与节点输出共用一个标签命名空间：重名会让描述里的 `[label]` 指代
        // 不明。`seen` 由输入标签起步，节点标签解析接着往里加。
        let mut seen = self.collect_input_labels()?;
        let node_labels = self.resolve_node_labels(&mut seen)?;
        let (wires, mut consumers) = self.resolve_wiring(&node_labels)?;
        let resolved_outputs = self.resolve_outputs(&node_labels, &mut consumers)?;
        self.check_consumers(&node_labels, &consumers)?;

        let spec = self.build_spec(&node_labels, &wires);
        self.assemble(&resolved_outputs, &spec)
    }

    /// 画中画 / 水印：把 `over` 叠到 `base` 上（`overlay=x:y`）。
    ///
    /// 叠加层通常比基底小，两路尺寸、像素格式不同都没关系（FFmpeg 会插 `scale`）。
    /// `output` 是整图输出的端点声明，尺寸与 `base` 不同时自动补一个 `scale` 收口。
    ///
    /// 图输入 0 是基底（`base`）、图输入 1 是叠加层（`over`），唯一的输出是合成结果。
    ///
    /// ```no_run
    /// # use rsmedia::Rational;
    /// # use rsmedia::filter::{FilterGraphBuilder, VideoEndpoint};
    /// # use rsmedia::PixelFormat;
    /// # fn main() -> rsmedia::Result<()> {
    /// let tb = Rational::new(1, 25).unwrap();
    /// let fps = Rational::new(25, 1).unwrap();
    /// let main = VideoEndpoint::new(1280, 720, PixelFormat::YUV420P)
    ///     .with_time_base(tb)
    ///     .with_frame_rate(fps);
    /// let logo = VideoEndpoint::new(160, 90, PixelFormat::RGBA)
    ///     .with_time_base(tb)
    ///     .with_frame_rate(fps);
    /// let mut graph = FilterGraphBuilder::overlay(main, logo, "W-w-20", "H-h-20", main)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn overlay(
        base: VideoEndpoint,
        over: VideoEndpoint,
        x: &str,
        y: &str,
        output: VideoEndpoint,
    ) -> Result<FilterGraph> {
        let mut builder = Self::new();
        builder.add_input_with("base", base);
        builder.add_input_with("over", over);
        builder.add_node(
            FilterNode::new(video::overlay(x, y)?)
                .with_inputs(["base", "over"])
                .with_label("overlay"),
        );
        builder.finish_video_output((base.width, base.height), output)
    }

    /// 横向并排（`hstack`）：把多路视频拼成一行。
    ///
    /// 要求各路**高度一致**（FFmpeg 的硬性约束，不一致时报错并点名是哪一路）；输出宽度
    /// 是各路宽度之和，`output` 声明的尺寸不同时自动补一个 `scale` 收口。
    ///
    /// ```no_run
    /// # use rsmedia::Rational;
    /// # use rsmedia::filter::{FilterGraphBuilder, VideoEndpoint};
    /// # use rsmedia::PixelFormat;
    /// # fn main() -> rsmedia::Result<()> {
    /// let tb = Rational::new(1, 25).unwrap();
    /// let fps = Rational::new(25, 1).unwrap();
    /// let left = VideoEndpoint::new(640, 720, PixelFormat::YUV420P)
    ///     .with_time_base(tb)
    ///     .with_frame_rate(fps);
    /// let right = VideoEndpoint::new(640, 720, PixelFormat::YUV420P)
    ///     .with_time_base(tb)
    ///     .with_frame_rate(fps);
    /// // 图输入 0 / 1 分别对应 left / right，输出是 1280x720。
    /// let mut graph = FilterGraphBuilder::hstack(
    ///     &[left, right],
    ///     VideoEndpoint::new(1280, 720, PixelFormat::YUV420P)
    ///         .with_time_base(tb)
    ///         .with_frame_rate(fps),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn hstack(inputs: &[VideoEndpoint], output: VideoEndpoint) -> Result<FilterGraph> {
        Self::stack(inputs, output, true)
    }

    /// 纵向堆叠（`vstack`）：把多路视频叠成一列。
    ///
    /// 要求各路**宽度一致**；输出高度是各路高度之和，`output` 声明的尺寸不同时自动补
    /// 一个 `scale` 收口。
    pub fn vstack(inputs: &[VideoEndpoint], output: VideoEndpoint) -> Result<FilterGraph> {
        Self::stack(inputs, output, false)
    }

    /// 首尾拼接（`concat`）：把多路视频按顺序接成一路。
    ///
    /// 要求各路**尺寸一致**（FFmpeg 的硬性约束；不一致时报错并点名是哪一路，先加
    /// `scale` 统一即可）；`concat` 之后时间戳是连续的，`output` 声明的尺寸不同时
    /// 自动补一个 `scale` 收口。
    pub fn concat(inputs: &[VideoEndpoint], output: VideoEndpoint) -> Result<FilterGraph> {
        if inputs.len() < 2 {
            return Err(RsmediaError::invalid_config(format!(
                "concat needs at least 2 inputs, got {}",
                inputs.len()
            )));
        }
        let (width, height) = (inputs[0].width, inputs[0].height);
        for (index, endpoint) in inputs.iter().enumerate() {
            if (endpoint.width, endpoint.height) != (width, height) {
                return Err(RsmediaError::invalid_config(format!(
                    "concat needs every input to have the same size: input 0 is {width}x{height}, \
                     input {index} is {}x{} (insert a scale node to normalise the sizes first)",
                    endpoint.width, endpoint.height
                )));
            }
        }

        let mut builder = Self::new();
        for endpoint in inputs {
            builder.add_input(*endpoint);
        }
        let labels = builder.input_labels_upto(inputs.len());
        builder.add_node(
            FilterNode::new(video::concat(inputs.len() as u32))
                .with_inputs(labels)
                .with_label("concat"),
        );
        builder.finish_video_output((width, height), output)
    }

    /// 混音（`amix`）：把多路音频混成一路。
    ///
    /// 各路采样率 / 采样格式 / 通道布局不同也没关系，FFmpeg 会自动插 `aresample`。
    /// `duration` 取 `longest`（以最长的一路为准，最常用）、`shortest` 或 `first`。
    ///
    /// ```no_run
    /// # use rsmedia::Rational;
    /// # use rsmedia::filter::{AudioEndpoint, FilterGraphBuilder};
    /// # use rsmedia::SampleFormat;
    /// # fn main() -> rsmedia::Result<()> {
    /// let tb = Rational::new(1, 48000).unwrap();
    /// let voice = AudioEndpoint::new(2, 48000, SampleFormat::FLTP, tb);
    /// let music = AudioEndpoint::new(2, 48000, SampleFormat::FLTP, tb);
    /// let mut graph = FilterGraphBuilder::amix(&[voice, music], "longest", voice)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn amix(
        inputs: &[AudioEndpoint],
        duration: &str,
        output: AudioEndpoint,
    ) -> Result<FilterGraph> {
        check_amix_duration(duration)?;
        Self::stack_audio(inputs, output, duration)
    }

    /// 前 `count` 路图输入的标签（自动标签与显式标签都涵盖）。
    fn input_labels_upto(&self, count: usize) -> Vec<String> {
        self.inputs
            .iter()
            .take(count)
            .map(|(label, _)| label.clone())
            .collect()
    }

    /// 末尾收口：节点输出尺寸与声明的 `output` 不一致时追加一个 `scale`，再声明图输出，
    /// 最后 [`build`](Self::build)。
    fn finish_video_output(
        &mut self,
        natural: (i32, i32),
        output: VideoEndpoint,
    ) -> Result<FilterGraph> {
        if (output.width, output.height) != natural {
            let (width, height) = (output.width, output.height);
            self.add_node(
                FilterNode::new(video::scale(width as u32, height as u32, None)?)
                    .with_label("scale"),
            );
        }
        self.add_output_tail(output);
        self.build()
    }

    /// `hstack` / `vstack` 的公共实现：`horizontal = true` 取横向。
    fn stack(
        inputs: &[VideoEndpoint],
        output: VideoEndpoint,
        horizontal: bool,
    ) -> Result<FilterGraph> {
        let name = if horizontal { "hstack" } else { "vstack" };
        if inputs.len() < 2 {
            return Err(RsmediaError::invalid_config(format!(
                "{name} needs at least 2 inputs, got {}",
                inputs.len()
            )));
        }
        // hstack 要求各路等高、vstack 要求各路等宽：先把不一致的那一路点名，比让
        // config 阶段报一句 "Input 2 height does not match" 好定位。
        let axis = if horizontal { "height" } else { "width" };
        let common = if horizontal {
            inputs[0].height
        } else {
            inputs[0].width
        };
        for (index, endpoint) in inputs.iter().enumerate() {
            let got = if horizontal {
                endpoint.height
            } else {
                endpoint.width
            };
            if got != common {
                return Err(RsmediaError::invalid_config(format!(
                    "{name} needs every input to have the same {axis}: input 0 is {common}, input \
                     {index} is {got} (insert a scale node to normalise the sizes first)"
                )));
            }
        }
        let width: i32 = if horizontal {
            inputs.iter().map(|endpoint| endpoint.width).sum()
        } else {
            common
        };
        let height: i32 = if horizontal {
            common
        } else {
            inputs.iter().map(|endpoint| endpoint.height).sum()
        };

        let mut builder = Self::new();
        for endpoint in inputs {
            builder.add_input(*endpoint);
        }
        let labels = builder.input_labels_upto(inputs.len());
        let filter = if horizontal {
            video::hstack(inputs.len() as u32)
        } else {
            video::vstack(inputs.len() as u32)
        };
        builder.add_node(
            FilterNode::new(filter)
                .with_inputs(labels)
                .with_label("stack"),
        );
        builder.finish_video_output((width, height), output)
    }

    /// `amix` 的公共实现：音频侧路数与时长策略。
    fn stack_audio(
        inputs: &[AudioEndpoint],
        output: AudioEndpoint,
        duration: &str,
    ) -> Result<FilterGraph> {
        if inputs.len() < 2 {
            return Err(RsmediaError::invalid_config(format!(
                "amix needs at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        let mut builder = Self::new();
        for endpoint in inputs {
            builder.add_input(*endpoint);
        }
        let labels = builder.input_labels_upto(inputs.len());
        builder.add_node(
            FilterNode::new(audio::amix(inputs.len() as u32, duration)?)
                .with_inputs(labels)
                .with_label("amix"),
        );
        builder.add_output_tail(output);
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用的有理数构造：字面量都是常量，失败即写错，直接 `unwrap`。
    fn rat(num: i32, den: i32) -> Rational {
        Rational::new(num, den).unwrap()
    }

    /// `Result<Filter>.spec()`，供那些只关心描述字符串的测试使用。
    ///
    /// 构造器改成返回 [`Result`] 之后（值可能含 NUL，见 [`escape_filter_value`]），
    /// 这些测试里全是字面量、不可能失败，也就没必要为它们逐个改签名加 `?`：失败即测试
    /// 本身写错，直接 panic 并把错误打出来更省事。
    trait SpecOf {
        fn spec(self) -> String;
    }

    impl SpecOf for Result<Filter> {
        fn spec(self) -> String {
            match self {
                Ok(filter) => filter.spec(),
                Err(e) => panic!("filter construction failed: {e}"),
            }
        }
    }

    /// D6：可选参数统一为 `impl Into<Option<T>>` 后，"传值"与"不传"两种写法都直接
    /// 可用 —— 值不需要包 `Some`，`None` 也仍能被推断出正确的类型。这里逐个钉住
    /// 那些把 `Option<T>` 收窄成 `impl Into<Option<T>>` 的构造器，避免以后有人
    /// "顺手"改回去。
    #[test]
    fn test_optional_arguments_accept_bare_values() {
        // 值直接传（不用 `Some(...)`）
        assert!(
            video::scale(64, 64, "neighbor")
                .spec()
                .contains("flags=neighbor")
        );
        // `overlay` 不再接受"不透明度"：`alpha` 是 0~2 的格式枚举，不是透明度，
        // 所以这个位置改用 `curves` 之类的占位断言没有意义 —— 直接断言 x/y 转义。
        assert_eq!(video::overlay("0", "0").spec(), "overlay=x=0:y=0");
        assert!(
            video::gif_palette(10.0, "bayer")
                .spec()
                .contains("dither=bayer")
        );
        assert!(
            video::curves("color_negative", None::<&str>)
                .spec()
                .contains("preset=color_negative")
        );
        assert!(
            video::lutyuv("val+10", None, None)
                .spec()
                .contains("y=val+10")
        );
        assert!(
            audio::compressor(4.0, 10.0, 100.0)
                .unwrap()
                .spec()
                .contains("attack=10")
        );
        assert!(
            audio::advanced_fft_denoise(12, -50, "v", true)
                .spec()
                .contains("nt=v")
        );
        // `p` / `r` 是 `<duration>`（秒），不是样本数：2 ms / 6 ms
        assert!(
            audio::anlm_denoise(
                0.001,
                Duration::from_micros(2000),
                Duration::from_micros(6000)
            )
            .spec()
            .contains("p=0.002")
        );
        assert!(video::zoompan("1.5", "0", "0", 25).spec().contains("d=25"));

        // 不传值仍然可用
        assert!(video::scale(64, 64, None).spec().contains("flags=bicubic"));
        assert_eq!(video::overlay("0", "0").spec(), "overlay=x=0:y=0");
        assert_eq!(video::lutyuv(None, None, None).spec(), "lutyuv");
        assert_eq!(audio::anlm_denoise(None, None, None).spec(), "anlmdn");
    }

    /// 批 4：滤镜参数按 `ffmpeg -h filter=<name>` 声明的类型对齐后，**以前写不出来
    /// 的值现在能写出来**。每条断言都对应一个 FFmpeg 实际接受的域，而不是风格。
    #[test]
    fn test_filter_arguments_cover_the_domain_ffmpeg_declares() {
        // `eq` 的 `brightness`/`contrast` 是 `<string>`（FFmpeg 要求值）：表达式
        // 可达，不再是"只能传常量"。
        assert_eq!(
            video::eq("sin(t)", "1+0.2*sin(t)").spec(),
            "eq=brightness=sin(t):contrast=1+0.2*sin(t)"
        );
        // 常量写法不变，且整数/浮点字面量都能直接传
        assert_eq!(video::eq(0, 1.0).spec(), "eq=brightness=0:contrast=1");

        // `volume` 的 `<string>` 允许 dB —— 没有哪个数值类型能表达 `"-6dB"`
        assert_eq!(audio::volume("-6dB").spec(), "volume=-6dB");
        assert_eq!(audio::volume(0.5).spec(), "volume=0.5");

        // `fps` 的 `<string>` 允许精确有理数：`f32` 表示不出 30000/1001
        assert_eq!(video::fps("30000/1001").spec(), "fps=30000/1001");

        // `<double>` 选项的分数值：31.5 Hz 是标准 1/3 倍频程中心频率
        assert_eq!(audio::highpass(31.5).spec(), "highpass=f=31.5");
        assert_eq!(audio::bass(100, 2.5).spec(), "bass=f=100:g=2.5");
        // 分数 dB 曾被 `fft_denoise(i32, i32)` 挡在门外
        assert_eq!(
            audio::fft_denoise(12.5, -50.5).spec(),
            "afftdn=nr=12.5:nf=-50.5:nt=w"
        );

        // `<duration>` 用 `Duration` 表达，与 `f32` 秒的区别是"这是一段时间"
        assert_eq!(
            audio::afade("in", Duration::from_millis(1500), Duration::from_secs(1)).spec(),
            "afade=t=in:st=1.5:d=1"
        );
        assert_eq!(
            trim(
                MediaType::AUDIO,
                Duration::from_millis(500),
                Duration::from_millis(2500)
            )
            .spec(),
            "atrim=start=0.5:end=2.5"
        );
    }

    /// 批 4 的 bug 修复：`afftdn.tr` 是 **布尔**（`track_residual`），不是浮点
    /// 平滑系数。旧实现把 `Option<f32>` 直接格式化成 `tr=0.5`，FFmpeg 会拒绝整条
    /// 滤镜链（`Unable to parse "tr" option value "0.5" as boolean`）——也就是说
    /// 只要传了非整数，`advanced_fft_denoise` 就让 `FilterGraph` 建不起来。
    /// 这里钉住新的输出形态，避免退回"名字像浮点、选项是布尔"的老样子。
    #[test]
    fn test_advanced_fft_denoise_emits_a_boolean_for_tr() {
        let spec = audio::advanced_fft_denoise(12, -50, "v", true).spec();
        assert!(spec.ends_with(":tr=1"), "{spec}");
        let spec = audio::advanced_fft_denoise(12, -50, None, false).spec();
        assert!(spec.ends_with(":tr=0"), "{spec}");
        // 默认噪声类型仍是 `w`
        assert!(spec.contains("nt=w"), "{spec}");
    }

    #[test]
    fn test_escape_option_level() {
        // Test case 1: Empty string
        assert_eq!(
            escape_option_level("").unwrap(),
            "",
            "Empty string should return empty string"
        );

        // Test case 2: String with no special characters but with spaces
        // 空白也在选项级的转义集里（`AV_ESCAPE_FLAG_WHITESPACE`）。
        assert_eq!(
            escape_option_level("normal text").unwrap(),
            "normal\\ text",
            "whitespace is escaped at the option level"
        );

        // Test case 3: String with special characters
        assert_eq!(
            escape_option_level("text with [brackets]").unwrap(),
            "text\\ with\\ \\[brackets\\]",
            "Brackets should be escaped and spaces too"
        );

        // Test case 4: 选项级转义只保证"值里的 `:` 不会截断选项"，不负责图级
        // 分隔符——`file:///...` 作为**不带引号**的选项值还必须再经过图级转义
        // （见 test_escape_filter_value_two_levels）。
        assert_eq!(
            escape_option_level("file:///path/to/video.mp4").unwrap(),
            "file\\:///path/to/video.mp4",
            "Single-level (option) escaping escapes the colon"
        );

        // Test case 5: String with all special characters
        let input = "filter=value,'text',[in],[out],key=val;next:filter\\backslash";
        let expected = "filter\\=value\\,\\'text\\'\\,\\[in\\]\\,\\[out\\]\\,key\\=val\\;next\\:filter\\\\backslash";
        assert_eq!(
            escape_option_level(input).unwrap(),
            expected,
            "All special characters should be escaped"
        );

        // Test case 6: String with escaped characters already
        assert_eq!(
            escape_option_level("already\\escaped").unwrap(),
            "already\\\\escaped",
            "Backslashes should be escaped even if they're escaping something else"
        );

        // Test case 7: Complex filter string with multiple special chars
        // Note that FFmpeg also escapes the exclamation mark (!)
        let complex_filter = "drawtext=text='Hello, World!':x=10:y=10";
        let expected = "drawtext\\=text\\=\\'Hello\\,\\ World!\\'\\:x\\=10\\:y\\=10";
        assert_eq!(
            escape_option_level(complex_filter).unwrap(),
            expected,
            "Complex filter string should be properly escaped with spaces and exclamation marks escaped too"
        );

        // Test case 8: Test with exclamation marks specifically
        // Note character!
        assert_eq!(
            escape_option_level("Warning!").unwrap(),
            "Warning!",
            "Exclamation marks should be escaped"
        );

        // Test case 9: Unicode characters - using pattern matching instead of exact comparison
        let unicode_result = escape_option_level("Unicode: こんにちは").unwrap();
        assert!(
            unicode_result.contains("Unicode"),
            "Result should contain the word 'Unicode'"
        );
        assert!(
            unicode_result.contains("こんにちは"),
            "Result should contain the Japanese characters"
        );

        // Test case 10: Unicode with special characters - using pattern matching
        let unicode_special_result = escape_option_level("Unicode: こんにちは[世界]").unwrap();
        assert!(
            unicode_special_result.contains("\\[") && unicode_special_result.contains("\\]"),
            "Unicode string with special characters should have brackets escaped"
        );

        // Test case 11: Very long string - only check that it ends correctly
        let long_string = "x".repeat(1000) + "=[]:";
        let long_result = escape_option_level(&long_string).unwrap();
        assert!(
            long_result.ends_with("\\=\\[\\]\\:"),
            "Long strings should have special characters at the end properly escaped"
        );
    }

    #[test]
    fn test_escape_filter_value_two_levels() {
        // 普通值不受影响：scale/pad/yadif/adelay 等生成的 spec 依赖"简单值原样保留"。
        for plain in ["lanczos", "send_frame", "black@0.5", "16/9"] {
            assert_eq!(
                escape_filter_value(plain).unwrap(),
                plain,
                "plain value changed"
            );
        }

        // 断言**精确**的反斜杠个数，而不是 `contains`：起作用的是个数，不是"有没有"。
        // 图级字符要奇数个（第 1 遍 unescape 后必须是裸字符），选项级要 2 个
        // （第 1 遍后必须仍是转义态）。2 个反斜杠能让 `contains("\\,")` 通过，
        // 但实测会把图拆开 —— 旧断言恰好漏掉了这个坏值。
        assert_eq!(
            escape_filter_value("/tmp/a,b;c[d].mp4").unwrap(),
            r"/tmp/a\\\,b\\\;c\\\[d\\\].mp4",
            "graph-level separators need an odd number of backslashes"
        );
        assert_eq!(
            escape_filter_value("file:///path/to/video.mp4").unwrap(),
            r"file\\:///path/to/video.mp4",
            "a colon needs exactly two backslashes to survive both passes"
        );
        assert_eq!(
            escape_filter_value(r"a'b\c").unwrap(),
            r"a\\\'b\\\\c",
            "quote and backslash: 3 and 4 backslashes respectively"
        );
        assert_eq!(
            escape_filter_value("a b").unwrap(),
            r"a\\ b",
            "whitespace: escaped once, then the backslash is doubled"
        );
    }

    /// 新的 Rust 实现在**每个输入上**都必须与 `ffi::av_escape` 逐字节一致。
    ///
    /// 生产代码不再调用 `av_escape`（它会把"分配失败"变成一条不存在的可恢复错误，
    /// 而这正是之前那个 fallback 的来源），但 FFmpeg 才是转义语义的出处，所以这里保留
    /// 一次差分对照：同一组 `specials` / flags 下两者输出必须相等。`av_escape` 只在本测试里出现。
    #[test]
    fn test_rust_escaping_matches_av_escape() {
        fn reference(input: &str, specials: &str, flags: i32) -> String {
            let c_input = CString::new(input).expect("corpus has no interior NUL");
            let c_specials = CString::new(specials).unwrap();
            let mut ptr = std::ptr::null_mut();
            // SAFETY: all three arguments are valid NUL-terminated C strings that outlive
            // the call, and `ptr` is a live local; on success FFmpeg hands back an
            // `av_malloc`ed buffer that this function frees.
            let ret = unsafe {
                ffi::av_escape(
                    &mut ptr,
                    c_input.as_ptr(),
                    c_specials.as_ptr(),
                    ffi::AV_ESCAPE_MODE_BACKSLASH,
                    flags,
                )
            };
            assert!(ret >= 0 && !ptr.is_null(), "av_escape failed on {input:?}");
            let escaped = unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: `ptr` came from `av_escape`, which allocates with `av_malloc`.
            unsafe { ffi::av_free(ptr as *mut _) };
            escaped
        }

        let corpus = [
            "",
            "plain",
            "black@0.5",
            "16/9",
            "a,b",
            "a;b",
            "a[b]c",
            "a:b",
            "a=b",
            "a'b",
            r"a\b",
            "a{b}c",
            "a,b;c[d]e:f=g'h\\i",
            "file:///path/to/video.mp4",
            "/tmp/a,b;c[d].mp4",
            " leading",
            "trailing ",
            "mid dle",
            "a b\tc\nd\re",
            "Unicode: こんにちは[世界]",
            "100%",
            "%{pts:hms}",
        ];

        for input in corpus {
            assert_eq!(
                escape_backslash(input, OPTION_SPECIALS, true).unwrap(),
                reference(
                    input,
                    OPTION_SPECIALS,
                    ffi::AV_ESCAPE_FLAG_WHITESPACE as i32
                ),
                "option level diverged on {input:?}"
            );
            assert_eq!(
                escape_backslash(input, GRAPH_SPECIALS, false).unwrap(),
                reference(input, GRAPH_SPECIALS, 0),
                "graph level diverged on {input:?}"
            );
        }
    }

    /// 值里的 NUL 不再被静默丢弃，而是在**转义这一层**就报错。
    ///
    /// `av_escape` 要的是 NUL 结尾的 C 字符串，所以含 NUL 的值根本进不了转义 —— 这也
    /// 正合语义：滤镜描述本身就是 C 字符串，NUL 表示不出来。错误是
    /// [`RsmediaError::InvalidConfig`]，能一眼看出是 NUL 的问题。
    ///
    /// 旧实现把它 `replace('\0', "")` 掉然后照常成功，产出一条少几个字符、却看起来
    /// 完全正常的滤镜描述 —— 那正是"掩盖问题"。手写 spec 里的 NUL 同样被拒
    /// （`FilterGraph::build` 把描述变成 C 字符串那一步），两个入口行为一致。
    #[test]
    fn test_nul_in_a_filter_value_is_reported_not_dropped() {
        for err in [
            escape_option_level("a\0b").unwrap_err(),
            escape_filter_value("a\0b").unwrap_err(),
        ] {
            assert!(err.is_invalid_config(), "{err}");
            assert!(err.to_string().contains("NUL"), "{err}");
        }

        // 构造器亦然：返回 `Err`，既不 panic，也不静默把字符丢掉。
        let err = video::pad(0, 0, 8, 8, "black\0").unwrap_err();
        assert!(err.is_invalid_config(), "{err}");

        let params = FilterParams::Video(VideoParams {
            width: 8,
            height: 4,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });
        let poisoned = Filter::new("null", MediaType::VIDEO, "null=a\0b".to_string());
        let err = match FilterGraph::build(&params, &[poisoned]) {
            Ok(_) => panic!("a NUL byte in a filter value must be reported, not dropped"),
            Err(e) => e,
        };
        assert!(err.is_invalid_config(), "{err}");
        assert!(err.to_string().contains("NUL"), "{err}");
    }

    #[test]
    fn test_real_world_filter_strings() {
        // Test case 1: Scale filter
        assert_eq!(
            escape_option_level("scale=width=1280:height=720").unwrap(),
            "scale\\=width\\=1280\\:height\\=720",
            "Scale filter string should be properly escaped"
        );

        // Test case 2: Overlay filter
        assert_eq!(
            escape_option_level("overlay=x=10:y=10").unwrap(),
            "overlay\\=x\\=10\\:y\\=10",
            "Overlay filter string should be properly escaped"
        );

        // Test case 3: Complex drawtext filter with Unicode
        let drawtext = "drawtext=text='Copyright © 2023':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=h-th-10";

        // Instead of checking the exact string, check for key patterns
        let drawtext_result = escape_option_level(drawtext).unwrap();

        // Check presence of escaped key components
        assert!(
            drawtext_result.contains("drawtext\\="),
            "Result should contain escaped filter name"
        );
        assert!(
            drawtext_result.contains("text\\=\\'"),
            "Result should contain escaped parameter text"
        );
        assert!(
            drawtext_result.contains("Copyright"),
            "Result should preserve the copyright text"
        );
        assert!(
            drawtext_result.contains("\\:fontcolor\\="),
            "Result should escape colon and equals sign"
        );

        // Test case 4: Filter with square brackets for pad names
        assert_eq!(
            escape_option_level("[in1][in2]overlay=format=rgb[out]").unwrap(),
            "\\[in1\\]\\[in2\\]overlay\\=format\\=rgb\\[out\\]",
            "Filter with pad names should be properly escaped"
        );
    }

    #[test]
    fn test_escape_filter_expr() {
        // 保留 `%{localtime}` 展开块，其余部分正常转义
        assert_eq!(
            escape_filter_expr("%{localtime}").unwrap(),
            "%{localtime}",
            "Time expansion block should be preserved"
        );
        assert_eq!(
            escape_filter_expr("T %{pts:hms}").unwrap(),
            "T\\ %{pts:hms}",
            "Surrounding text should be escaped but block preserved"
        );
        // 多个展开块
        assert_eq!(
            escape_filter_expr("%{frame_num}/%{n}").unwrap(),
            "%{frame_num}/%{n}",
            "Multiple expansion blocks should be preserved"
        );
        // 无展开块时退化为普通转义
        assert_eq!(
            escape_filter_expr("plain: text").unwrap(),
            escape_option_level("plain: text").unwrap(),
            "Without expansion blocks it should match plain escaping"
        );
    }

    #[test]
    fn test_drawtext_time_text() {
        // 静态文本正常转义
        let static_spec = video::DrawText::new("Hello", 10, 20, 24, "white")
            .build()
            .spec()
            .to_string();
        assert!(
            static_spec.contains("drawtext=text='Hello':x=10:y=20:fontsize=24:fontcolor=white"),
            "static drawtext spec mismatch: {static_spec}"
        );

        // 时间表达式不被转义
        let time_spec = video::DrawText::new("", 10, 20, 24, "white")
            .time_text("%{localtime}")
            .build()
            .spec()
            .to_string();
        assert!(
            time_spec.contains("text='%{localtime}'"),
            "time expression should not be escaped: {time_spec}"
        );
    }

    #[test]
    fn test_filter_spec_generation() {
        use MediaType::*;

        // ---- Video filters ----
        let cases: Vec<(String, String, MediaType)> = vec![
            (
                "scale=w=640:h=360:flags=lanczos".into(),
                video::scale(640, 360, Some("lanczos")).spec(),
                VIDEO,
            ),
            // 不指定 flags 时用 FFmpeg `scale` 滤镜的默认算法（bicubic），
            // 与本 crate 的 `Scaler::default()` 一致。
            (
                "scale=w=640:h=360:flags=bicubic".into(),
                video::scale(640, 360, None).spec(),
                VIDEO,
            ),
            (
                "crop=x=10:y=20:w=100:h=50".into(),
                video::crop(10, 20, 100, 50).spec(),
                VIDEO,
            ),
            (
                "fade=t=in:start_frame=0:nb_frames=30".into(),
                video::fade_in(30).spec(),
                VIDEO,
            ),
            (
                "fade=t=out:start_frame=30:nb_frames=30".into(),
                video::fade_out(30, 30).spec(),
                VIDEO,
            ),
            ("unsharp".into(), video::unsharp().spec(), VIDEO),
            (
                "boxblur=luma_radius=2".into(),
                video::blur(2.0).spec(),
                VIDEO,
            ),
            (
                "eq=brightness=0.2:contrast=1.5".into(),
                video::eq(0.2, 1.5).spec(),
                VIDEO,
            ),
            ("fps=30".into(), video::fps(30.0).spec(), VIDEO),
            (
                "yadif=mode=send_frame".into(),
                video::yadif("send_frame").spec(),
                VIDEO,
            ),
            (
                "pad=w=1280:h=720:x=0:y=0:color=black".into(),
                video::pad(0, 0, 1280, 720, "black").spec(),
                VIDEO,
            ),
            (
                "subtitles=sub.srt".into(),
                video::subtitles("sub.srt").spec(),
                VIDEO,
            ),
            ("setdar=16/9".into(), video::setdar(16, 9).spec(), VIDEO),
            ("setsar=1/1".into(), video::setsar(1, 1).spec(), VIDEO),
            ("hue=h=30".into(), video::hue(30).spec(), VIDEO),
            ("negate".into(), video::negate().spec(), VIDEO),
            ("noise=alls=10".into(), video::noise(10).spec(), VIDEO),
            (
                "hqdn3d=luma_spatial=3:chroma_spatial=2".into(),
                video::hqdn3d(3.0, 2.0).spec(),
                VIDEO,
            ),
            ("nlmeans=s=1.5".into(), video::nlmeans(1.5).spec(), VIDEO),
            ("eq=gamma=1.2".into(), video::gamma(1.2).spec(), VIDEO),
            (
                "eq=saturation=1.5".into(),
                video::saturation(1.5).spec(),
                VIDEO,
            ),
            (
                "vibrance=intensity=0.4".into(),
                video::vibrance(0.4).spec(),
                VIDEO,
            ),
            ("deblock".into(), video::deblock().spec(), VIDEO),
            (
                "delogo=x=0:y=0:w=100:h=50".into(),
                video::delogo(0, 0, 100, 50).spec(),
                VIDEO,
            ),
            ("transpose=1".into(), video::transpose(1).spec(), VIDEO),
            (
                "drawbox=x=1:y=2:w=10:h=10:color=red:t=2".into(),
                video::drawbox(1, 2, 10, 10, "red", 2).spec(),
                VIDEO,
            ),
        ];

        for (expected, actual, media_type) in cases {
            assert_eq!(actual, expected, "spec mismatch");
            assert_eq!(
                Filter::new("x", media_type, String::new()).media_type(),
                media_type
            );
        }

        // ---- Audio filters ----
        let audio_cases: Vec<(String, String)> = vec![
            ("volume=1.5".into(), audio::volume(1.5).spec()),
            (
                "loudnorm=I=-16:TP=-1.5:LRA=11".into(),
                audio::loudnorm(-16.0).spec(),
            ),
            ("highpass=f=80".into(), audio::highpass(80).spec()),
            ("lowpass=f=4000".into(), audio::lowpass(4000).spec()),
            ("atempo=1.25".into(), audio::atempo(1.25).spec()),
            (
                "afftdn=nr=20:nf=-50:nt=w".into(),
                audio::fft_denoise(20, -50).spec(),
            ),
            ("afftdn=nr=15:nt=w".into(), audio::denoise(15.0).spec()),
        ];
        for (expected, actual) in audio_cases {
            assert_eq!(actual, expected, "audio spec mismatch");
        }

        // compressor 返回 Result
        let comp = audio::compressor(3.0, Some(30.0), Some(200.0)).unwrap();
        assert_eq!(comp.spec(), "acompressor=ratio=3:attack=30:release=200");
        // 非法 ratio 返回错误而非 panic
        assert!(audio::compressor(0.5, None, None).is_err());
    }

    #[test]
    fn test_delogo_builder() {
        use MediaType::*;

        // 单区域便捷方法
        let simple = video::delogo(0, 0, 100, 50);
        assert_eq!(simple.name(), "delogo");
        assert_eq!(simple.media_type(), VIDEO);
        assert_eq!(simple.spec(), "delogo=x=0:y=0:w=100:h=50");

        // Builder：多区域 + band + show
        let filters = video::Delogo::new()
            .add_region(10, 10, 120, 30)
            .add_region(640, 10, 120, 30)
            .band(2)
            .show()
            .build();
        assert_eq!(filters.len(), 2, "one delogo per region");
        assert_eq!(
            filters[0].spec(),
            "delogo=x=10:y=10:w=120:h=30:band=2:show=1"
        );
        assert_eq!(
            filters[1].spec(),
            "delogo=x=640:y=10:w=120:h=30:band=2:show=1"
        );
        for f in &filters {
            assert_eq!(f.media_type(), VIDEO);
        }

        // 空 Builder 生成空列表
        assert!(video::Delogo::new().build().is_empty());
    }

    #[test]
    fn test_drawtext_boxed_and_fontfile() {
        let f = video::DrawText::new("Hi", 1, 2, 20, "white")
            .fontfile("fonts/A.ttf")
            .boxed("black@0.5")
            .build()
            .spec()
            .to_string();
        assert!(
            f.contains("drawtext=text='Hi':x=1:y=2:fontsize=20:fontcolor=white"),
            "base mismatch: {f}"
        );
        assert!(
            f.contains("fontfile='fonts/A.ttf'"),
            "fontfile missing: {f}"
        );
        assert!(
            f.contains("box=1:boxcolor=black@0.5:boxborderw=0"),
            "box missing: {f}"
        );
    }

    /// 构造一个 GRAY8 单平面、已分配缓冲的测试帧。
    fn make_gray8_frame(width: i32, height: i32) -> AVFrame {
        use crate::error::Context;
        use crate::pixel::PixelFormat;
        let mut f = AVFrame::new();
        f.set_width(width);
        f.set_height(height);
        f.set_format(PixelFormat::GRAY8.into());
        f.alloc_buffer()
            .context("alloc buffer for gray8 frame")
            .unwrap();
        f
    }

    /// `FilterGraph::process_frame` 独立路径测试（此前仅经 encode 间接覆盖）。
    ///
    /// 用真实的 `hflip` 滤镜：单帧进、单帧出；输出尺寸/格式保持，但整行像素
    /// 顺序被反转。同时验证状态机 `is_initialized` 与 `output_size`。
    #[test]
    fn test_filtergraph_process_frame_hflip() -> Result<()> {
        use crate::pixel::PixelFormat;

        let (w, h) = (8, 4);
        let src = make_gray8_frame(w, h);
        // 给每行填入递增的像素值，用于检测 hflip 是否确实翻转了行内顺序。
        unsafe {
            let linesize = src.linesize[0] as usize;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    // 像素值 = (y*w + x) % 256，行内单调递增。
                    *src.data[0].cast::<u8>().add(y * linesize + x) =
                        ((y * w as usize + x) % 256) as u8;
                }
            }
        }

        let params = FilterParams::Video(VideoParams {
            width: w,
            height: h,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });
        let filter = video::hflip();

        let mut graph = FilterGraph::new();
        assert!(!graph.is_initialized(), "graph should start uninitialized");
        graph.init(&params, &[filter])?;
        assert!(
            graph.is_initialized(),
            "graph should be initialized after init"
        );

        // 输出链路尺寸应与输入一致。
        assert_eq!(
            graph.output_size()?,
            (w, h),
            "output_size should match input size"
        );

        // 单帧进 -> 单帧出，行被反转。
        let out = graph
            .process_frame(Some(src))?
            .expect("one input frame should yield one output frame");
        assert_eq!(out.width, w, "output width mismatch");
        assert_eq!(out.height, h, "output height mismatch");

        // 校验每个像素是否被水平翻转（逐行逆序）。
        let linesize = out.linesize[0] as usize;
        unsafe {
            let out_ptr = out.data[0].cast::<u8>();
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let expected = ((y * w as usize + (w as usize - 1 - x)) % 256) as u8;
                    let got = *out_ptr.add(y * linesize + x);
                    assert_eq!(
                        got, expected,
                        "hflip mismatch at ({x},{y}): got {got}, expected {expected}"
                    );
                }
            }
        }

        // 提交 EOF 后 graph 进入 drained/flushed 状态，flush 亦返回空。
        assert!(
            graph.process_frame(None)?.is_none(),
            "EOF should eventually drain to None"
        );
        let remaining = graph.flush()?;
        assert!(remaining.is_empty(), "no frames should remain after EOF");
        Ok(())
    }

    /// 音频 `abuffer`/`abuffersink` 独立路径测试：aformat 把 FLTP 转成 S16。
    #[test]
    fn test_filter_graph_process_frame_audio() -> Result<()> {
        use rsmpeg::avutil::AVChannelLayout;

        let (channels, sample_rate, nb_samples) = (2i32, 44100i32, 1024i32);

        let params = FilterParams::Audio(AudioParams {
            nb_channels: channels,
            sample_rate,
            format: SampleFormat::S16,
            src_format: SampleFormat::FLTP,
            time_base: rat(1, 44100),
        });

        // aformat 把输入转为 S16（与 sink 约束一致）。
        let mut graph = FilterGraph::new();
        graph.init(
            &params,
            &[audio::format(channels, sample_rate, SampleFormat::S16)],
        )?;

        let mut frame = AVFrame::new();
        frame.set_format(SampleFormat::FLTP as _);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(channels).into_inner());
        frame.set_sample_rate(sample_rate);
        frame.set_nb_samples(nb_samples);
        frame.alloc_buffer().context("alloc audio frame")?;

        let out = graph
            .process_frame(Some(frame))?
            .expect("audio frame should flow through");
        assert_eq!(out.sample_rate, sample_rate, "sample rate preserved");
        assert_eq!(out.nb_samples, nb_samples, "sample count preserved");

        graph.process_frame(None)?;
        assert!(graph.flush()?.is_empty());
        Ok(())
    }

    /// 构造一个 YUV420P 帧：亮度面整体填 `luma`，两个色度面填 128（中性）。
    fn make_yuv420p_frame(width: i32, height: i32, luma: u8) -> AVFrame {
        use crate::pixel::PixelFormat;

        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(PixelFormat::YUV420P.into());
        frame
            .alloc_buffer()
            .context("alloc buffer for yuv420p frame")
            .unwrap();
        let (w, h) = (width as usize, height as usize);
        unsafe {
            for y in 0..h {
                for x in 0..w {
                    *frame.data[0]
                        .cast::<u8>()
                        .add(y * frame.linesize[0] as usize + x) = luma;
                }
            }
            // 420 的色度面宽高各取一半。
            for plane in [1usize, 2] {
                for y in 0..h.div_ceil(2) {
                    for x in 0..w.div_ceil(2) {
                        *frame.data[plane]
                            .cast::<u8>()
                            .add(y * frame.linesize[plane] as usize + x) = 128;
                    }
                }
            }
        }
        frame
    }

    /// 构造一个 YUV420P 帧：亮度面按列递增（第 x 列 = `x * 10`），用于区分「原样」
    /// 与「水平翻转」——翻转后逐列亮度必须与原来相反。
    fn make_ramp_frame(width: i32, height: i32) -> AVFrame {
        use crate::pixel::PixelFormat;

        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(PixelFormat::YUV420P.into());
        frame.alloc_buffer().context("alloc ramp frame").unwrap();
        let (w, h) = (width as usize, height as usize);
        unsafe {
            for y in 0..h {
                for x in 0..w {
                    *frame.data[0]
                        .cast::<u8>()
                        .add(y * frame.linesize[0] as usize + x) = x as u8 * 10;
                }
            }
            for plane in [1usize, 2] {
                for y in 0..h.div_ceil(2) {
                    for x in 0..w.div_ceil(2) {
                        *frame.data[plane]
                            .cast::<u8>()
                            .add(y * frame.linesize[plane] as usize + x) = 128;
                    }
                }
            }
        }
        frame
    }

    /// 读 YUV420P 帧亮度面在 `(x, y)` 的像素值。
    fn luma_at(frame: &AVFrame, x: i32, y: i32) -> u8 {
        unsafe {
            *frame.data[0]
                .cast::<u8>()
                .add(y as usize * frame.linesize[0] as usize + x as usize)
        }
    }

    /// 构造一个 FLTP 音频帧，所有采样点填同一个值。
    fn make_fltp_frame(sample_rate: i32, nb_samples: i32, value: f32) -> AVFrame {
        let mut frame = AVFrame::new();
        frame.set_format(SampleFormat::FLTP as _);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(1).into_inner());
        frame.set_sample_rate(sample_rate);
        frame.set_nb_samples(nb_samples);
        frame
            .alloc_buffer()
            .context("alloc buffer for fltp frame")
            .unwrap();
        unsafe {
            let samples = frame.data[0].cast::<f32>();
            for index in 0..nb_samples as usize {
                *samples.add(index) = value;
            }
        }
        frame
    }

    /// 读 FLTP 帧第 `index` 个采样（单声道，取平面 0）。
    fn fltp_sample_at(frame: &AVFrame, index: usize) -> f32 {
        unsafe { *frame.data[0].cast::<f32>().add(index) }
    }

    fn video_endpoint(width: i32, height: i32) -> VideoEndpoint {
        VideoEndpoint::new(width, height, PixelFormat::YUV420P)
            .with_time_base(rat(1, 25))
            .with_frame_rate(rat(25, 1))
    }

    /// `with_*` 让三个有理数只能按名字设置，代价是"忘了设"变成可能 —— 那必须是
    /// **响亮失败**，不能是静默建出一个时间基为 0 的图。
    #[test]
    fn test_a_video_endpoint_without_a_time_base_is_rejected() {
        let endpoint = VideoEndpoint::new(320, 240, PixelFormat::YUV420P);
        assert_eq!(endpoint.time_base, Rational::ZERO);
        assert_eq!(endpoint.frame_rate, Rational::ZERO);
        assert_eq!(endpoint.pixel_aspect, Rational::ONE);

        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("src", endpoint);
        builder.add_node(FilterNode::new(Filter::new(
            "hflip",
            MediaType::VIDEO,
            "hflip".to_string(),
        )));
        builder.add_output_tail(endpoint);
        let err = builder
            .build()
            .expect_err("a buffer source with time_base 0/1 must not build");
        assert!(
            err.to_string().contains("buffer"),
            "the failure should name the buffer source, got: {err}"
        );
    }

    fn audio_endpoint(sample_rate: i32) -> AudioEndpoint {
        AudioEndpoint::new(
            1,
            sample_rate,
            SampleFormat::FLTP,
            Rational::new(1, sample_rate).unwrap(),
        )
    }

    /// `hstack`：两路进、一路出，**端点与标签的对应关系**必须正确。
    ///
    /// 这是多输入图最关键的一环：`buffer` 源按名字与描述里的标签配对，名字错位或配对
    /// 反了都不会报错，只会把帧接错路。左半填 10、右半填 200，错位会在像素上立刻暴露。
    #[test]
    fn test_filter_graph_builder_hstack_endpoint_order() -> Result<()> {
        let mut graph = FilterGraphBuilder::hstack(
            &[video_endpoint(4, 2), video_endpoint(4, 2)],
            video_endpoint(8, 2),
        )?;

        assert_eq!(
            (graph.input_count(), graph.output_count()),
            (2, 1),
            "hstack has 2 inputs and 1 output"
        );
        assert_eq!(graph.output_size()?, (8, 2), "stacked size");

        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 2, 10)))?;
        graph.push_frame_to(1, Some(make_yuv420p_frame(4, 2, 200)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("hstack emits a frame once both inputs have one");
        assert_eq!((out.width, out.height), (8, 2));
        for y in 0..2 {
            for x in 0..8 {
                let expected = if x < 4 { 10 } else { 200 };
                assert_eq!(
                    luma_at(&out, x, y),
                    expected,
                    "hstack wiring reversed at ({x},{y})"
                );
            }
        }
        Ok(())
    }

    /// 端点与标签按**名字**绑定，与它们在描述里的出现顺序无关。
    ///
    /// 描述里 `[b]` 先出现，但输入 0 仍是 `a`：`push_frame_to` 的序号对应"声明输入
    /// 的顺序"，不会被描述里的书写顺序带偏（按位置配对的实现会在这里静默接错路）。
    #[test]
    fn test_filter_graph_builder_matches_endpoints_by_label() -> Result<()> {
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("a", video_endpoint(4, 2));
        builder.add_input_with("b", video_endpoint(4, 2));
        builder.add_node(
            FilterNode::new(video::hstack(2))
                .with_inputs(["b", "a"])
                .with_label("stack"),
        );
        builder.add_output("stack", video_endpoint(8, 2));
        let mut graph = builder.build()?;

        // hstack 的 pad0 = "b"（输入 1，200）在左，pad1 = "a"（输入 0，10）在右。
        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 2, 10)))?;
        graph.push_frame_to(1, Some(make_yuv420p_frame(4, 2, 200)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("hstack emits a frame once both inputs have one");
        for y in 0..2 {
            for x in 0..8 {
                let expected = if x < 4 { 200 } else { 10 };
                assert_eq!(
                    luma_at(&out, x, y),
                    expected,
                    "endpoint/label binding wrong at ({x},{y})"
                );
            }
        }
        Ok(())
    }

    /// `overlay`：基底走图输入 0、叠加层走图输入 1（写反了叠加位置就会露馅）。
    #[test]
    fn test_filter_graph_builder_overlay_endpoint_order() -> Result<()> {
        let mut graph = FilterGraphBuilder::overlay(
            video_endpoint(4, 4),
            video_endpoint(2, 2),
            "2",
            "2",
            video_endpoint(4, 4),
        )?;
        assert_eq!(graph.output_size()?, (4, 4), "overlay keeps base size");

        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 4, 0)))?;
        graph.push_frame_to(1, Some(make_yuv420p_frame(2, 2, 255)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("overlay emits a frame once both inputs have one");
        for y in 0..4 {
            for x in 0..4 {
                // 叠加层贴在 (2,2)-(3,3)。
                let inside = (2..4).contains(&x) && (2..4).contains(&y);
                assert_eq!(
                    luma_at(&out, x, y),
                    if inside { 255 } else { 0 },
                    "overlay mismatch at ({x},{y})"
                );
            }
        }
        Ok(())
    }

    /// `overlay` 的 x/y 是表达式，可能含 `,`（如 `if(eq(a,b),c,d)`）。不转义时
    /// 这个 `,` 会被图级解析当成滤镜分隔符，报出来的错与真实原因无关；转义后
    /// 整图必须配置成功，且表达式仍按原意求值。
    #[test]
    fn test_overlay_expression_with_comma_is_escaped_and_runs() -> Result<()> {
        // 基底高 4 → x 取 2，叠加层落在 (2,0)-(3,1)。
        let expr = "if(eq(main_h,4),2,0)";

        let spec = video::overlay(expr, "0").spec();
        assert!(
            spec.contains(r"\,"),
            "comma must stay escaped for the graph-level parse: {spec}"
        );

        let mut graph = FilterGraphBuilder::overlay(
            video_endpoint(4, 4),
            video_endpoint(2, 2),
            expr,
            "0",
            video_endpoint(4, 4),
        )?;
        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 4, 0)))?;
        graph.push_frame_to(1, Some(make_yuv420p_frame(2, 2, 255)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("overlay emits a frame once both inputs have one");
        for y in 0..4 {
            for x in 0..4 {
                let inside = (2..4).contains(&x) && (0..2).contains(&y);
                assert_eq!(
                    luma_at(&out, x, y),
                    if inside { 255 } else { 0 },
                    "overlay mismatch at ({x},{y}), spec: {spec}"
                );
            }
        }
        Ok(())
    }

    /// `EAGAIN` 只是"这一刻没帧可取"，不等于已经在排空：`Drained` 的判据是
    /// **EOF 已送到每一路输入**（见 `state::ProcessState`）。多输入图里只喂了
    /// 一路时的 `EAGAIN` 若被记成 `Drained`，`is_drained()` 会在流中段永久为真。
    #[test]
    fn test_filter_graph_eagain_midstream_is_not_drained() -> Result<()> {
        let mut graph = FilterGraphBuilder::overlay(
            video_endpoint(4, 4),
            video_endpoint(2, 2),
            "2",
            "2",
            video_endpoint(4, 4),
        )?;

        // 只喂基底：overlay 还在等叠加层，取帧必得 EAGAIN，但这仍是流中段。
        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 4, 0)))?;
        assert!(
            graph.receive_frame_from(0)?.is_none(),
            "overlay must wait for its second input"
        );
        assert!(
            !graph.is_drained(),
            "EAGAIN with input still to come is not draining: {graph:?}"
        );
        assert!(!graph.is_flushed());

        // 补齐叠加层后必须能出帧——证明上一步只是"暂时没帧"，图仍能继续接收输入。
        graph.push_frame_to(1, Some(make_yuv420p_frame(2, 2, 255)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("overlay emits a frame once both inputs have one");
        assert_eq!(luma_at(&out, 2, 2), 255);
        assert!(!graph.is_drained(), "still mid-stream: {graph:?}");

        // 推完 EOF 才真正进入排空，直至流末尾。
        graph.push_frame_to(0, None)?;
        graph.push_frame_to(1, None)?;
        graph.drain_output(0)?;
        assert!(
            graph.is_flushed(),
            "after EOF the graph must end up flushed: {graph:?}"
        );
        Ok(())
    }

    /// `Drained` 在**最后一路 EOF 推送成功**的那一刻置位，而不等取帧拿到 `EAGAIN`
    /// —— 判据是"EOF 已送到每一路输入"（`state::ProcessState` 里 `Drained` 的定义）。
    ///
    /// 旧实现把推进放在 `receive_frame_from` 的 `EAGAIN` 分支里，于是"刚推完 EOF、
    /// 一次 `EAGAIN` 都还没遇到"的图仍报 `Normal`，与文档里的判据不符。
    #[test]
    fn test_eof_marks_the_graph_drained_before_any_eagain() -> Result<()> {
        let (w, h) = (8, 4);
        let params = FilterParams::Video(VideoParams {
            width: w,
            height: h,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });
        let null = Filter::new("null", MediaType::VIDEO, "null".to_string());
        let mut graph = FilterGraph::build(&params, &[null])?;

        graph.push_frame_to(0, Some(make_gray8_frame(w, h)))?;
        assert!(!graph.is_drained(), "mid-stream is not draining");

        graph.push_frame_to(0, None)?;
        assert!(
            graph.is_drained(),
            "EOF sent to every input means draining — the state must not wait for an EAGAIN"
        );
        assert!(!graph.is_flushed(), "draining is not the end of the stream");

        // 重复推 EOF 是 no-op，不会把阶段推过头（也不会报 AVERROR_EOF）。
        graph.push_frame_to(0, None)?;
        assert!(graph.is_drained());

        let frames = graph.drain_output(0)?;
        assert_eq!(frames.len(), 1, "the frame pushed before EOF comes out");
        assert!(graph.is_flushed(), "after the drain the graph is flushed");
        Ok(())
    }

    /// `init` 这条路径（解码/编码流水线用的单输入线性链）也必须登记输入描述：
    /// 取帧失败时 FFmpeg 只回一句 `EINVAL`，错误信息全靠这份描述定位"声明的
    /// 格式 vs 实际喂进去的格式"。以前只有 `FilterGraphBuilder::build` 登记它，
    /// 于是解码/编码路径上的诊断恒为空。
    #[test]
    fn test_init_records_declared_input_specs() -> Result<()> {
        let (w, h) = (8, 4);
        let params = FilterParams::Video(VideoParams {
            width: w,
            height: h,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });
        let null = Filter::new("null", MediaType::VIDEO, "null".to_string());
        let mut graph = FilterGraph::build(&params, &[null])?;

        assert_eq!(
            graph.input_specs.len(),
            1,
            "a linear chain has exactly one input: {:?}",
            graph.input_specs
        );
        let spec = &graph.input_specs[0];
        assert!(spec.contains("8x4"), "must name the declared size: {spec}");
        assert!(
            spec.contains("GRAY8"),
            "must name the declared format: {spec}"
        );

        // rebuild 不能留下旧图的描述（旧代码漏了这一步 ⇒ 描述会越攒越多并指向上一张图）
        let null = Filter::new("null", MediaType::VIDEO, "null".to_string());
        graph.rebuild(&params, &[null])?;
        assert_eq!(
            graph.input_specs.len(),
            1,
            "rebuild must re-register, not append: {:?}",
            graph.input_specs
        );
        Ok(())
    }

    /// `concat`：按输入顺序首尾相接——前两帧来自输入 0、后两帧来自输入 1。
    #[test]
    fn test_filter_graph_builder_concat_order() -> Result<()> {
        let mut graph = FilterGraphBuilder::concat(
            &[video_endpoint(4, 2), video_endpoint(4, 2)],
            video_endpoint(4, 2),
        )?;

        for _ in 0..2 {
            graph.push_frame_to(0, Some(make_yuv420p_frame(4, 2, 10)))?;
            graph.push_frame_to(1, Some(make_yuv420p_frame(4, 2, 200)))?;
        }

        // 拉到 EAGAIN（输入 0 的两帧出完、还没推 EOF）后补 EOF 收尾。
        let mut values = Vec::new();
        while let Some(frame) = graph.receive_frame_from(0)? {
            values.push(luma_at(&frame, 0, 0));
        }
        for frame in graph.drain_output(0)? {
            values.push(luma_at(&frame, 0, 0));
        }
        assert_eq!(
            values,
            vec![10, 10, 200, 200],
            "concat must keep the declared input order"
        );
        Ok(())
    }

    /// `amix`：两路等长音频混音，输出应覆盖最长的一路。
    #[test]
    fn test_filter_graph_builder_amix_two_inputs() -> Result<()> {
        let endpoint = audio_endpoint(48000);
        let mut graph = FilterGraphBuilder::amix(&[endpoint, endpoint], "longest", endpoint)?;
        assert_eq!(graph.output_count(), 1);

        for _ in 0..2 {
            graph.push_frame_to(0, Some(make_fltp_frame(48000, 1024, 0.5)))?;
            graph.push_frame_to(1, Some(make_fltp_frame(48000, 1024, -0.5)))?;
        }

        let mut frames = Vec::new();
        while let Some(frame) = graph.receive_frame_from(0)? {
            frames.push(frame);
        }
        frames.extend(graph.drain_output(0)?);

        let samples: i32 = frames.iter().map(|frame| frame.nb_samples).sum();
        assert_eq!(
            samples, 2048,
            "amix should emit every sample of the longest input"
        );
        assert!(
            frames.iter().all(|frame| frame.sample_rate == 48000),
            "sample rate preserved"
        );
        Ok(())
    }

    /// `send_command`：图一旦建成，滤镜选项就被冻结，运行时调参只能走命令通道。
    ///
    /// `volume` 的 `volume` 命令改的是增益表达式，而表达式要**逐帧求值**才会作用到
    /// 后续帧（`eval=frame`）——默认 `eval=once` 只在初始化时算一次，改了表达式也
    /// 不会重算。这里显式带上 `eval=frame`，用幅度 0.5 的帧验证 1.0 → 0.1 的衰减。
    #[test]
    fn test_filter_graph_send_command_changes_volume() -> Result<()> {
        let endpoint = audio_endpoint(48000);
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("src", endpoint);
        builder.add_node(FilterNode::new(Filter::new(
            "volume",
            MediaType::AUDIO,
            "volume=1.0:eval=frame".to_string(),
        )));
        builder.add_output_tail(endpoint);
        let mut graph = builder.build()?;

        // 增益 1.0：输出与输入同幅。
        graph.push_frame_to(0, Some(make_fltp_frame(48000, 512, 0.5)))?;
        let out = graph
            .receive_frame_from(0)?
            .context("volume passes the first frame through")?;
        let before = fltp_sample_at(&out, 0);
        assert!(
            (before - 0.5).abs() < 1e-3,
            "gain 1.0 must leave the amplitude alone, got {before}"
        );

        // 运行时把增益改成 0.1，无需重建图。
        let response = graph.send_command("volume", "volume", "0.1")?;
        assert!(
            response.is_empty(),
            "volume writes no response, got {response:?}"
        );

        graph.push_frame_to(0, Some(make_fltp_frame(48000, 512, 0.5)))?;
        let out = graph
            .receive_frame_from(0)?
            .context("volume passes the second frame through")?;
        let after = fltp_sample_at(&out, 0);
        assert!(
            (after - 0.05).abs() < 1e-3,
            "gain 0.1 must attenuate the amplitude to 0.05, got {after}"
        );
        Ok(())
    }

    /// `send_command` 遇到滤镜不认识的命令要报 `Unsupported`（FFmpeg 的 ENOSYS），
    /// 而不是当作普通 FFmpeg 错误码糊过去——调用方可据此换滤镜或放弃该能力。
    #[test]
    fn test_filter_graph_send_command_unknown_command_is_unsupported() -> Result<()> {
        let endpoint = audio_endpoint(48000);
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("src", endpoint);
        builder.add_node(FilterNode::new(Filter::new(
            "volume",
            MediaType::AUDIO,
            "volume=1.0:eval=frame".to_string(),
        )));
        builder.add_output_tail(endpoint);
        let mut graph = builder.build()?;

        let err = graph
            .send_command("volume", "definitelynotacommand", "1")
            .expect_err("volume does not implement this command");
        assert!(err.is_unsupported(), "{err}");
        Ok(())
    }

    /// `split`：**单滤镜多输出 pad** 的 fan-out——一路输入复制成两路，各接一条链路。
    ///
    /// 用逐列递增的亮度图验证：`copy_a` 必须与输入逐像素相同、`copy_b` 经 `hflip`
    /// 后必须逐列反转。两路标签写反、或 pad 顺序错位，都会在像素上立刻暴露。
    #[test]
    fn test_filter_graph_builder_split_fanout() -> Result<()> {
        let endpoint = video_endpoint(4, 2);
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("src", endpoint);
        builder.add_node(
            FilterNode::new(video::split(2))
                .with_inputs(["src"])
                .with_outputs(["copy_a", "copy_b"]),
        );
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["copy_b"])
                .with_label("mirrored"),
        );
        builder.add_output("copy_a", endpoint);
        builder.add_output("mirrored", endpoint);
        let mut graph = builder.build()?;

        assert_eq!(
            (graph.input_count(), graph.output_count()),
            (1, 2),
            "one input fans out to two outputs"
        );

        graph.push_frame_to(0, Some(make_ramp_frame(4, 2)))?;
        let straight = graph.receive_frame_from(0)?.context("copy_a frame")?;
        let mirrored = graph.receive_frame_from(1)?.context("copy_b frame")?;
        for y in 0..2 {
            for x in 0..4 {
                assert_eq!(
                    luma_at(&straight, x, y),
                    x as u8 * 10,
                    "copy_a must be the untouched original at ({x},{y})"
                );
                assert_eq!(
                    luma_at(&mirrored, x, y),
                    (3 - x) as u8 * 10,
                    "copy_b must be horizontally flipped at ({x},{y})"
                );
            }
        }
        Ok(())
    }

    /// `asplit`：音频 fan-out 后两路各自进 `amix` 混回来，采样值应保持不变
    /// （两路内容相同，`amix` 又按路数归一化）。
    #[test]
    fn test_filter_graph_builder_asplit_two_outputs() -> Result<()> {
        let endpoint = audio_endpoint(48000);
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("src", endpoint);
        builder.add_node(
            FilterNode::new(audio::asplit(2))
                .with_inputs(["src"])
                .with_outputs(["dry", "wet"]),
        );
        builder.add_node(
            FilterNode::new(audio::amix(2, "longest")?)
                .with_inputs(["dry", "wet"])
                .with_label("mixed"),
        );
        builder.add_output("mixed", endpoint);
        let mut graph = builder.build()?;

        for _ in 0..2 {
            graph.push_frame_to(0, Some(make_fltp_frame(48000, 512, 0.5)))?;
        }
        let mut frames = Vec::new();
        while let Some(frame) = graph.receive_frame_from(0)? {
            frames.push(frame);
        }
        frames.extend(graph.drain_output(0)?);

        let samples: i32 = frames.iter().map(|frame| frame.nb_samples).sum();
        assert_eq!(samples, 1024, "asplit must feed amix both copies");
        for frame in &frames {
            unsafe {
                for index in 0..frame.nb_samples as usize {
                    let value = *frame.data[0].cast::<f32>().add(index);
                    assert!(
                        (value - 0.5).abs() <= 1e-6,
                        "asplit+amix must preserve the sample, got {value}"
                    );
                }
            }
        }
        Ok(())
    }

    /// 自由组装：两路输入（视频 + 音频）各自成链、各自输出，验证 **m 路输出**的
    /// 端点映射（输出 0 是视频、输出 1 是音频，配错会在 `config` 阶段就报类型不符）。
    #[test]
    fn test_filter_graph_builder_two_outputs() -> Result<()> {
        let video_in = video_endpoint(4, 2);
        let audio_in = audio_endpoint(48000);

        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video_in); // 自动标签 in0
        builder.add_input(audio_in); // 自动标签 in1
        assert_eq!(builder.input_label(0), Some("in0"));
        assert_eq!(builder.input_label(1), Some("in1"));
        // 第一个节点不写接线 → 自动接 in0；第二个节点显式接 in1。
        builder.add_node(FilterNode::new(video::hflip()).with_label("flipped"));
        builder.add_output("flipped", video_in);
        // 第二个节点显式接 in1。音频侧用 `volume`（逐帧直通）而不是 `areverse`：
        // areverse 要缓存整个流、EOF 前吐不出帧，验证不了「推一帧取一帧」。
        builder.add_node(
            FilterNode::new(audio::volume(1.0)?)
                .with_inputs(["in1"])
                .with_label("level"),
        );
        builder.add_output("level", audio_in);
        let mut graph = builder.build()?;

        assert_eq!(graph.output_count(), 2);
        assert_eq!(graph.input_count(), 2);
        // 输出 1 的时间基来自音频端点，说明 sink 与逻辑输出的映射正确。
        let audio_tb = graph.output_time_base_at(1)?;
        assert_eq!(
            (audio_tb.num(), audio_tb.den()),
            (1, 48000),
            "output 1 must be the audio sink"
        );

        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 2, 40)))?;
        let video_out = graph
            .receive_frame_from(0)?
            .expect("filtered video frame from output 0");
        assert_eq!((video_out.width, video_out.height), (4, 2));

        graph.push_frame_to(1, Some(make_fltp_frame(48000, 512, 0.25)))?;
        let audio_out = graph
            .receive_frame_from(1)?
            .expect("filtered audio frame from output 1");
        assert_eq!(audio_out.sample_rate, 48000);
        assert_eq!(audio_out.nb_samples, 512);
        Ok(())
    }

    /// 缺失的滤镜名报 [`RsmediaError::Unsupported`]（本构建没编入它，如
    /// `drawtext` 需要 libfreetype ⇒ 调用方只能换构建或换滤镜，跳过/降级即可），
    /// 且加了 context 之后仍能按变体识别。
    #[test]
    fn test_missing_filter_reports_unsupported() -> Result<()> {
        let params = FilterParams::Video(VideoParams {
            width: 8,
            height: 4,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });
        let filters = vec![Filter::new(
            "rsmedia_no_such_filter",
            MediaType::VIDEO,
            "rsmedia_no_such_filter".to_string(),
        )];

        let err = FilterGraph::build(&params, &filters).unwrap_err();
        assert!(
            err.is_unsupported(),
            "a filter this build does not have is a capability gap: {err}"
        );
        assert!(!err.is_invalid_config(), "{err}");
        assert!(
            err.to_string().contains("rsmedia_no_such_filter"),
            "the message must name the missing filter: {err}"
        );
        Ok(())
    }

    /// 空滤镜列表是**调用方配置错**：拼出的描述串为空，FFmpeg 只会回一句无法定位的
    /// 解析错误，所以在触到 FFmpeg 之前就报 [`RsmediaError::InvalidConfig`]。
    /// 同时验证文档给的替代方案——显式传一个 `null` 滤镜做直通——确实建得起图。
    #[test]
    fn test_empty_filter_list_is_invalid_config() -> Result<()> {
        let (w, h) = (8, 4);
        let params = FilterParams::Video(VideoParams {
            width: w,
            height: h,
            format: PixelFormat::GRAY8,
            src_format: PixelFormat::GRAY8,
            time_base: rat(1, 25),
            frame_rate: rat(25, 1),
            pixel_aspect: Rational::ONE,
        });

        let err = FilterGraph::build(&params, &[]).unwrap_err();
        assert!(
            err.is_invalid_config(),
            "an empty filter list is the caller's own mistake: {err}"
        );
        assert!(!err.is_unsupported(), "{err}");
        assert!(
            err.to_string().contains("no filter"),
            "the message must say what is missing: {err}"
        );

        // 承诺的替代方案必须真的能用：`null` 直通建图，并原样放行一帧。
        let null = Filter::new("null", MediaType::VIDEO, "null".to_string());
        let mut graph = FilterGraph::build(&params, &[null])?;
        let out = graph
            .process_frame(Some(make_gray8_frame(w, h)))?
            .expect("a passthrough yields one frame per input frame");
        assert_eq!((out.width, out.height), (w, h));
        assert!(graph.process_frame(None)?.is_none());

        Ok(())
    }

    /// 接线错误在建图前（未触碰 FFmpeg 的解析/协商）就被拒，且错误信息可定位。
    #[test]
    fn test_filter_graph_builder_rejects_bad_wiring() {
        let video = video_endpoint(4, 2);
        let audio = audio_endpoint(48000);

        // 双输入滤镜塞进线性链：接线数（1）≠ pad 数（2）。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(FilterNode::new(
            video::overlay("0", "0").expect("literal arguments cannot fail"),
        ));
        builder.add_output_tail(video);
        let err = builder.build().unwrap_err();
        assert!(err.is_invalid_config(), "{err}");
        assert!(err.to_string().contains("takes 2 input(s)"), "{err}");

        // 上游标签不存在（也涵盖前向引用：后面的节点标签此时还没声明）。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(FilterNode::new(video::hflip()).with_inputs(["nope"]));
        builder.add_output_tail(video);
        let err = builder.build().unwrap_err();
        assert!(
            err.to_string().contains("neither a declared graph input"),
            "{err}"
        );

        // 前向引用：节点 0 引用节点 1 的标签。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["later"])
                .with_label("early"),
        );
        builder.add_node(FilterNode::new(video::hflip()).with_label("later"));
        builder.add_output("early", video);
        let err = builder.build().unwrap_err();
        assert!(
            err.is_invalid_config(),
            "forward reference must be rejected: {err}"
        );

        // fan-out：同一个输出标签被两处消费。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(FilterNode::new(video::hflip()).with_label("once"));
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["once"])
                .with_label("a"),
        );
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["once"])
                .with_label("b"),
        );
        builder.add_output("a", video);
        builder.add_output("b", video);
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("insert a split"), "{err}");

        // 输出标签数与 pad 数不符：单输出滤镜标了两个名字。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["in0"])
                .with_outputs(["a", "b"]),
        );
        builder.add_output("a", video);
        builder.add_output("b", video);
        let err = builder.build().unwrap_err();
        assert!(
            err.to_string()
                .contains("has 1 output pad(s) but 2 label(s)"),
            "{err}"
        );

        // 多输出滤镜只标了一个 pad：`split` 的 pad 数由它自己的 `outputs=` 选项决定
        // （动态 pad，构建期看不到），所以这里由 FFmpeg 在协商阶段拒掉——错误信息里
        // 带着出错的滤镜图描述，足以定位。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(
            FilterNode::new(video::split(2))
                .with_inputs(["in0"])
                .with_label("only_one"),
        );
        builder.add_output("only_one", video);
        let err = builder.build().unwrap_err();
        assert!(
            err.to_string().contains("Failed to configure filter graph")
                && err.to_string().contains("split=2[only_one]"),
            "{err}"
        );

        // 多输出滤镜的某个 pad 没人消费（split 的两路里只接了一路）。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(
            FilterNode::new(video::split(2))
                .with_inputs(["in0"])
                .with_outputs(["used", "dead"]),
        );
        builder.add_node(FilterNode::new(video::hflip()).with_inputs(["used"]));
        builder.add_output_tail(video);
        let err = builder.build().unwrap_err();
        assert!(
            err.to_string()
                .contains("every output pad of the node must be consumed"),
            "{err}"
        );

        // 悬空：节点输出既没被下游消费、也没声明为图输出。
        // （自动接线会把节点串成链，所以这里必须显式接线才能造出真正的悬空输出。）
        let mut builder = FilterGraphBuilder::new();
        builder.add_input_with("a", video);
        builder.add_input_with("b", video);
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["a"])
                .with_label("dead"),
        );
        builder.add_node(
            FilterNode::new(video::hflip())
                .with_inputs(["b"])
                .with_label("alive"),
        );
        builder.add_output("alive", video);
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("nothing consumes it"), "{err}");

        // 媒体类型混接：把音频端点接进视频滤镜。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(audio);
        builder.add_node(FilterNode::new(video::hflip()).with_inputs(["in0"]));
        builder.add_output_tail(video);
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("media type mismatch"), "{err}");

        // 图输入没人用。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_input(video);
        builder.add_node(FilterNode::new(video::hflip()).with_inputs(["in1"]));
        builder.add_output_tail(video);
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("never used"), "{err}");

        // 图输出直接引用图输入（需要显式的直通节点）。
        let mut builder = FilterGraphBuilder::new();
        builder.add_input(video);
        builder.add_node(FilterNode::new(video::hflip()).with_label("flipped"));
        builder.add_output("in0", video);
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("no node produces"), "{err}");
    }

    /// 组合形态的尺寸约束在构建期报错，并点名是哪一路。
    #[test]
    fn test_filter_graph_builder_rejects_bad_sizes() {
        // hstack 要求等高。
        let err = FilterGraphBuilder::hstack(
            &[video_endpoint(4, 2), video_endpoint(4, 4)],
            video_endpoint(8, 2),
        )
        .unwrap_err();
        assert!(
            err.is_invalid_config() && err.to_string().contains("input 1"),
            "{err}"
        );

        // vstack 要求等宽。
        let err = FilterGraphBuilder::vstack(
            &[video_endpoint(4, 2), video_endpoint(6, 2)],
            video_endpoint(4, 4),
        )
        .unwrap_err();
        assert!(err.to_string().contains("same width"), "{err}");

        // concat 要求尺寸一致。
        let err = FilterGraphBuilder::concat(
            &[video_endpoint(4, 2), video_endpoint(2, 2)],
            video_endpoint(4, 2),
        )
        .unwrap_err();
        assert!(err.to_string().contains("same size"), "{err}");

        // amix 的 duration 只认三个档位。
        let endpoint = audio_endpoint(48000);
        let err = FilterGraphBuilder::amix(&[endpoint, endpoint], "long", endpoint).unwrap_err();
        assert!(err.to_string().contains("duration"), "{err}");
    }

    /// 输出尺寸与合成结果的天然尺寸不符时自动补 `scale` 收口。
    #[test]
    fn test_filter_graph_builder_hstack_scales_to_declared_output() -> Result<()> {
        let mut graph = FilterGraphBuilder::hstack(
            &[video_endpoint(4, 2), video_endpoint(4, 2)],
            video_endpoint(2, 2),
        )?;
        assert_eq!(graph.output_size()?, (2, 2), "output收口到声明尺寸");

        graph.push_frame_to(0, Some(make_yuv420p_frame(4, 2, 10)))?;
        graph.push_frame_to(1, Some(make_yuv420p_frame(4, 2, 200)))?;
        let out = graph
            .receive_frame_from(0)?
            .expect("scaled frame should come out");
        assert_eq!((out.width, out.height), (2, 2));
        Ok(())
    }

    /// 输出索引越界必须是**错误**，不能被折叠成"这一路没有帧率/时间基/尺寸"。
    ///
    /// 这三个查询原先都用 `.ok()?` 吞掉 `get_sink_context` 的失败，于是
    /// "调用方写错了索引"和"汇没协商出帧率"在返回类型上完全同形（`None`），
    /// 前者被静默忽略。改成 `Result` 之后两者必须可区分，这里钉住这一点。
    #[test]
    fn test_output_index_out_of_range_is_an_error_not_a_missing_value() -> Result<()> {
        let mut graph = FilterGraphBuilder::hstack(
            &[video_endpoint(4, 2), video_endpoint(4, 2)],
            video_endpoint(8, 2),
        )?;
        assert_eq!(graph.output_count(), 1, "hstack has exactly one output");

        // 合法索引必须仍然拿到真实值（同时钉住"正常路径没被改坏"）。
        assert_eq!(
            graph.output_size_at(0)?,
            (8, 2),
            "sink 0 is the stacked output"
        );
        assert_eq!(
            graph.output_frame_rate_at(0)?,
            rat(25, 1),
            "sink frame rate follows the endpoint's declared frame rate"
        );
        assert_eq!(
            graph.output_time_base_at(0)?,
            rat(1, 25),
            "sink time base follows the endpoint's declared time base"
        );

        // 越界：三处都必须是 InvalidConfig，而且点名 out of range。
        for err in [
            graph.output_frame_rate_at(1).unwrap_err(),
            graph.output_time_base_at(1).unwrap_err(),
            graph.output_size_at(1).unwrap_err(),
        ] {
            assert!(err.is_invalid_config(), "not InvalidConfig: {err}");
            assert!(err.to_string().contains("out of range"), "{err}");
        }
        Ok(())
    }

    /// 一张还没有任何输出的图，主输出查询同样报错而不是返回 `None`。
    #[test]
    fn test_output_queries_on_a_graph_without_outputs_are_an_error() {
        let mut graph = FilterGraph::new();
        assert_eq!(graph.output_count(), 0, "a fresh graph declares no output");
        assert!(graph.output_size().unwrap_err().is_invalid_config());
        assert!(graph.output_frame_rate().unwrap_err().is_invalid_config());
        assert!(graph.output_time_base().unwrap_err().is_invalid_config());
    }

    /// `transpose` 只接受 0..=3。
    ///
    /// FFmpeg 的 `dir` 选项声明 0..=7：4..=7 能解析通过，但**静默直通** —— 实测
    /// `transpose=4..7` 的输出与"不加滤镜"逐字节相同。把它们当作可用滤镜交出去，
    /// 就会把"视频根本没转"伪装成一个正常工作的 `Filter`，所以必须在这里挡掉。
    #[test]
    fn test_transpose_rejects_the_modes_ffmpeg_applies_as_a_no_op() {
        for mode in 0..=3 {
            assert_eq!(
                video::transpose(mode).unwrap().spec(),
                format!("transpose={mode}"),
                "mode {mode} is a real direction"
            );
        }
        for mode in [-1, 4, 5, 6, 7, 8, i32::MAX] {
            let err = video::transpose(mode).unwrap_err();
            assert!(err.is_invalid_config(), "mode {mode}: {err}");
            assert!(err.to_string().contains("0..=3"), "mode {mode}: {err}");
        }
    }

    /// 闭集滤镜选项：名字与 FFmpeg 的数字写法都收，越界/拼错在**构造期**就拒绝。
    ///
    /// `yadif`/`bwdif`/`afade`/`amix` 原先**完全不校验**：拼错的名字被原样写进滤镜
    /// 串，直到建图才变成一句 FFmpeg 的解析错误（"Unable to parse \"mode\" option
    /// value \"…\""），既没有本 crate 的上下文也定位不到是哪个参数。`amix` 更糟——
    /// 两个入口一个校验（`FilterGraphBuilder::amix`）一个不校验（`audio::amix`），
    /// 同一个值在两个入口行为不同。
    ///
    /// 值集按 `ffmpeg -h filter=<name>` 实测：`yadif.mode` 声明 `from 0 to 3` 并给
    /// 四个名字，`bwdif.mode` 是 `0..=1`（**没有** `*_nospatial`），`afade.t` 是
    /// `in`/`out`，`amix.duration` 是三个名字对应 0/1/2。名字和数字**都**要收。
    #[test]
    fn test_closed_set_options_are_validated_at_construction_time() {
        for mode in [
            "send_frame",
            "send_field",
            "send_frame_nospatial",
            "send_field_nospatial",
            "0",
            "3",
        ] {
            video::yadif(mode).unwrap_or_else(|err| panic!("yadif({mode:?}) rejected: {err}"));
        }
        for mode in ["4", "-1", "bogus", "send_frames", ""] {
            let err = video::yadif(mode).unwrap_err();
            assert!(err.is_invalid_config(), "yadif({mode:?}): {err}");
            assert!(err.to_string().contains("yadif mode"), "{err}");
        }

        for mode in ["send_frame", "send_field", "0", "1"] {
            video::bwdif(mode).unwrap_or_else(|err| panic!("bwdif({mode:?}) rejected: {err}"));
        }
        // `send_frame_nospatial` 是 yadif 的值，bwdif 没有这一档。
        for mode in ["2", "send_frame_nospatial", "bogus"] {
            let err = video::bwdif(mode).unwrap_err();
            assert!(err.is_invalid_config(), "bwdif({mode:?}): {err}");
        }

        for fade in ["in", "out", "0", "1"] {
            audio::afade(fade, Duration::from_secs(0), Duration::from_secs(1))
                .unwrap_or_else(|err| panic!("afade({fade:?}) rejected: {err}"));
        }
        for fade in ["2", "In", "fadein", ""] {
            let err =
                audio::afade(fade, Duration::from_secs(0), Duration::from_secs(1)).unwrap_err();
            assert!(err.is_invalid_config(), "afade({fade:?}): {err}");
        }

        // `amix` 的两个入口必须对同一个值给出同一个结论。
        for duration in ["longest", "shortest", "first", "0", "1", "2"] {
            audio::amix(2, duration)
                .unwrap_or_else(|err| panic!("amix({duration:?}) rejected: {err}"));
        }
        for duration in ["long", "3", "-1", ""] {
            let from_filter = audio::amix(2, duration).unwrap_err();
            let endpoint = audio_endpoint(48000);
            let from_builder =
                FilterGraphBuilder::amix(&[endpoint, endpoint], duration, endpoint).unwrap_err();
            for err in [&from_filter, &from_builder] {
                assert!(err.is_invalid_config(), "amix({duration:?}): {err}");
                assert!(err.to_string().contains("amix duration"), "{err}");
            }
        }
    }
}
