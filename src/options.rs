use crate::error::{Result, RsmediaError};
use crate::strutils;

use rsmpeg::avutil::{AVDictionary, AVDictionaryRef};
use rsmpeg::ffi;

use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;

/// A wrapper type for ffmpeg options.
///
/// Internally backed by a [`BTreeMap<String, String>`] (key-sorted, so the
/// type derives a deterministic `Hash`) and lazily materialized into an
/// [`AVDictionary`] at the FFI boundary
/// ([`Options::into_dict`]/[`Options::to_dict`]).
///
/// This avoids a subtle defect of building AVDictionary directly: rsmpeg
/// cannot represent an *empty* dictionary (FFmpeg uses a NULL pointer), so
/// previous eager construction always carried a junk `key=""` entry that made
/// libav log `Option '' not found` on every `open()`. With lazy
/// materialization an empty `Options` converts to `None`.
///
/// FFmpeg Documentation: <https://ffmpeg.org/doxygen/trunk/>
///
/// `libavformat/options_table.h`: <https://www.ffmpeg.org/doxygen/trunk/libavformat_2options__table_8h-source.html>
/// `libavcodec/options_table.h`: <https://www.ffmpeg.org/doxygen/trunk/libavcodec_2options_table_8h_source.html>
///
/// # Example
///
/// ```ignore
/// let mut opts = Options::new();
/// opts.set("threads", "4");
/// opts.merge(Options::preset_h264()); // preset keys win
/// ```
#[derive(Clone, Default, Hash, PartialEq, Eq)]
pub struct Options(BTreeMap<String, String>);

/// Alias of [`Options`] for the string key/value metadata carried by media
/// structures — `AVFrame.metadata`, container-level and per-stream metadata.
///
/// Use it in field/parameter positions to express "this is a metadata map",
/// reserving the [`Options`] name for encoder/muxer option sets.
pub type Metadata = Options;

impl Options {
    /// Creates an empty options set.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Sets a key-value pair, replacing any existing entry (same overwrite
    /// semantics as `av_dict_set` with flags 0).
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.0.insert(key.into(), value.into());
        self
    }

    /// Looks up a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Removes a key, returning its previous value.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }

    /// Returns true if the key is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterates over key-value pairs in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.0.iter()
    }

    /// Merges `other` into `self`; entries from `other` win on conflicts
    /// (same overwrite semantics as `av_dict_copy` with flags 0).
    pub fn merge(&mut self, other: Options) {
        self.0.extend(other.0);
    }

    /// Materializes into an [`AVDictionary`], or `None` when empty so that
    /// `open(None)` is called instead of feeding libav a junk dictionary.
    ///
    /// Keys/values containing interior NUL bytes are skipped with a warning
    /// (they cannot be represented in a C string).
    pub fn to_dict(&self) -> Option<AVDictionary> {
        Self::build(self.0.iter())
    }

    /// Consuming variant of [`Options::to_dict`].
    pub fn into_dict(self) -> Option<AVDictionary> {
        Self::build(self.0)
    }

    /// Shared materialization core: fold an owning or borrowed entry iterator
    /// into an [`AVDictionary`] without an intermediate copy of the whole map.
    fn build<K, V>(entries: impl IntoIterator<Item = (K, V)>) -> Option<AVDictionary>
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut dict: Option<AVDictionary> = None;
        for (k, v) in entries {
            let (k, v) = (k.as_ref(), v.as_ref());
            if k.contains('\0') || v.contains('\0') {
                tracing::warn!("Skip option with interior NUL: {k:?}={v:?}");
                continue;
            }
            let (key, value) = (
                strutils::str_to_cstring(k).unwrap(),
                strutils::str_to_cstring(v).unwrap(),
            );
            dict = match dict {
                Some(dict) => Some(dict.set(&key, &value, 0)),
                None => Some(AVDictionary::new(&key, &value, 0)),
            };
        }
        dict
    }

    /// Reads entries back from an [`AVDictionary`].
    pub fn from_dict(dict: &AVDictionary) -> Self {
        dict.into_iter()
            .filter_map(|entry| {
                match (
                    strutils::cstr_to_string(entry.key()),
                    strutils::cstr_to_string(entry.value()),
                ) {
                    (Ok(key), Ok(value)) => Some((key, value)),
                    (bad_key, bad_value) => {
                        tracing::warn!("Skip non-UTF8 option: {bad_key:?}={bad_value:?}");
                        None
                    }
                }
            })
            .collect()
    }

    /// Reads entries back from a raw `AVDictionary` pointer, the form FFmpeg
    /// structs expose (e.g. `AVFrame.metadata`, `AVStream.metadata`).
    ///
    /// FFmpeg represents an empty dictionary as a null pointer, so a null `dict`
    /// yields an empty `Options`. Non-UTF-8 entries are skipped with a warning.
    ///
    /// # Safety
    ///
    /// `dict` must be null or point to a valid `AVDictionary` that outlives the call.
    pub(crate) unsafe fn from_raw_dict(dict: *mut ffi::AVDictionary) -> Self {
        let Some(ptr) = NonNull::new(dict) else {
            return Self::new();
        };
        // SAFETY: the reference is non-owning (`wrap_ref_pure` wraps the pointer in
        // `ManuallyDrop`) and is dropped before this function returns; the caller
        // guarantees the dictionary outlives the call.
        let borrowed = unsafe { AVDictionaryRef::from_raw(ptr) };
        Self::from_dict(&borrowed)
    }

    /// Replaces the `AVDictionary` at `*dest` with `self`, freeing whatever was
    /// stored there.
    ///
    /// This is the write counterpart of [`from_raw_dict`](Self::from_raw_dict) for
    /// metadata slots embedded in FFmpeg structs (`AVFrame.metadata`,
    /// `AVStream.metadata`, ...). An empty `Options` just frees the slot and leaves
    /// a null pointer — FFmpeg's representation of "no metadata". Ownership of the
    /// freshly built dictionary is transferred to `*dest`.
    ///
    /// # Safety
    ///
    /// `dest` must point at a live `AVDictionary` slot owned by an FFmpeg struct,
    /// or be null.
    pub(crate) unsafe fn write_into_raw_dict(&self, dest: &mut *mut ffi::AVDictionary) {
        unsafe { ffi::av_dict_free(dest) };
        if let Some(dict) = self.to_dict() {
            *dest = dict.into_raw().as_ptr();
        }
    }

    /// Creates options such that ffmpeg will prefer TCP transport when reading RTSP stream (over
    /// the default UDP format). It also adds some options to reduce the socket and I/O timeouts to
    /// 4 seconds.
    ///
    /// This sets the `rtsp_transport` to `tcp` in ffmpeg options,
    /// it also sets `rw_timeout` and `stimeout` to lower (more sane) values.
    pub fn preset_avformat_rtsp_transport_tcp() -> Self {
        let mut opts = Self::new();
        opts
            // These can't be too low because ffmpeg takes its sweet time
            .set("rtsp_transport", "tcp")
            .set("rw_timeout", "16000000")
            .set("stimeout", "16000000");
        opts
    }

    /// Creates options such that ffmpeg is instructed to fragment output and mux to fragmented mp4
    /// container format.
    ///
    /// This modifies the `movflags` key to supported fragmented output. The muxer output will not
    /// have a header and each packet contains enough metadata to be streamed without the header.
    /// Muxer output should be compatiable with MSE.
    pub fn preset_avformat_fragmented_mov() -> Self {
        let mut opts = Self::new();
        opts.set(
            "movflags",
            "faststart+frag_keyframe+frag_custom+empty_moov+omit_tfhd_offset",
        );
        opts
    }

    /// Creates options for a FLV muxer.
    pub fn preset_avformat_flv() -> Self {
        let mut opts = Self::new();
        opts.set("flvflags", "no_duration_filesize")
            .set("fflags", "nobuffer+flush_packets")
            // 添加实时流标志
            .set("live", "1")
            // 完全禁用元数据更新
            .set("write_metaf", "0")
            // 设置较小的chunk大小以减少延迟
            .set("chunk_size", "4096");
        opts
    }

    /// Default avcodec options for a libx264 encoder.
    pub fn preset_h264() -> Self {
        let mut opts = Self::new();
        opts
            // ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow,placebo
            .set("preset", "medium")
            // baseline,main,high
            .set("profile", "high")
            // 场景切换敏感度
            .set("scenecut", "0");
        opts
    }

    /// Options for a libx264 encoder that are tuned for low-latency encoding such as for real-time streaming.
    pub fn preset_h264_realtime() -> Self {
        // 基址复用 preset_h264()，只覆盖低延迟所需的差异键，避免重复定义公共键值。
        let mut opts = Self::preset_h264();
        opts
            // baseline,main,high，低延迟用 main 而非 high
            .set("profile", "main")
            // film,animation,grain,stillimage,psnr,ssim,fastdecode,zerolatency
            .set("tune", "zerolatency")
            // 设置比特率控制,视频比特率
            .set("b", "3000k")
            // 最大比特率
            .set("maxrate", "3500k")
            // 缓冲区大小
            .set("bufsize", "3000k")
            // 恒定质量因子
            .set("crf", "23")
            // 周期内部刷新替代关键帧
            .set("intra-refresh", "1")
            // 参考帧数量
            .set("refs", "3")
            // GOP=60（2秒@30fps）
            .set("g", "60")
            // 禁用 B 帧
            .set("bf", "0")
            // 最小量化参数
            .set("qmin", "4")
            // 最大量化参数
            .set("qmax", "51")
            // 启用中等强度去块滤波
            .set("deblock", "1:1")
            // 自适应量化模式
            .set("aq-mode", "2")
            // 量化优化, 0: 禁用, 1: 仅用于最终编码, 2: 用于所有模式决策
            .set("trellis", "1")
            .set("threads", "auto")
            // 使用所有可用的分区模式
            .set("partitions", "all")
            // 最小关键帧间隔
            .set("keyint_min", "30")
            // 强制恒定帧率
            .set("force-cfr", "1")
            // 启用切片线程
            .set("sliced_threads", "1")
            // 禁用前瞻同步
            .set("sync-lookahead", "0")
            // 减少前瞻帧数
            .set("rc-lookahead", "10");
        opts
    }

    /// h264_nvenc options only
    ///
    /// FFMpeg with NVENC:
    /// <https://superuser.com/questions/1296374/best-settings-for-ffmpeg-with-nvenc>
    ///
    /// NVENC Preset Migration Guide:
    /// <https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/nvenc-preset-migration-guide/index.html>
    ///
    pub fn preset_h264_nvenc() -> Self {
        let mut opts = Self::new();
        opts
            // p1-p7, default(p4), slow, medium, fast, hp, hq, bd, ll, llhq, llhp, lossless
            .set("preset", "p5")
            // baseline, main, high, high444p, high10, high422
            .set("profile", "high")
            // ll, ull, lossless, film, animation, grain, fastdecode, zerolatency, hq
            .set("tune", "ll")
            // 设置比特率 4Mbps
            .set("b", "4000k")
            .set("maxrate", "5000k")
            .set("bufsize", "8000k")
            // constqp, ll_2pass_size, ll_2pass_quality
            // vbr, vbr_hq, vbr_minqp, vbr_2pass
            // cbr, cbr_hq, cbr_ld_hq
            .set("rc", "cbr")
            // 量化参数
            .set("qmin", "10")
            .set("qmax", "18")
            // 启用自适应量化
            .set("spatial-aq", "1")
            .set("temporal-aq", "1")
            .set("aq-strength", "8")
            // GOP设置，较小的GOP有利于快速恢复和低延迟
            .set("g", "30")
            // 禁用B帧以，避免出现画面闪烁
            .set("bf", "0")
            .set("b_ref_mode", "middle")
            // 启用场景切换检测，允许在场景变化时插入I帧
            .set("no-scenecut", "0")
            // 低延时
            .set("delay", "0")
            .set("zerolatency", "1")
            // NVENC特有的参数
            // 增加表面缓冲区数量
            .set("surfaces", "32")
            // 加权预测，改善低光照
            .set("weighted_pred", "1");
        opts
    }
}

/// `HashMap<String, String>` -> `Options`
impl From<HashMap<String, String>> for Options {
    fn from(item: HashMap<String, String>) -> Self {
        Self(item.into_iter().collect())
    }
}

/// `BTreeMap<String, String>` -> `Options` (zero-copy: shares the internal
/// representation).
impl From<BTreeMap<String, String>> for Options {
    fn from(item: BTreeMap<String, String>) -> Self {
        Self(item)
    }
}

/// `Options` -> `BTreeMap<String, String>` (zero-copy: yields the internal
/// representation).
impl From<Options> for BTreeMap<String, String> {
    fn from(item: Options) -> Self {
        item.0
    }
}

/// `Options` -> `HashMap<String, String>`
impl From<Options> for HashMap<String, String> {
    fn from(item: Options) -> Self {
        item.0.into_iter().collect()
    }
}

/// Borrowing conversion (clones the entries).
impl From<&Options> for HashMap<String, String> {
    fn from(item: &Options) -> Self {
        item.0.clone().into_iter().collect()
    }
}

impl FromIterator<(String, String)> for Options {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// 单一配置源检查：builder 的 typed setter 与 [`Options`] 透传不得同时配置同一项。
///
/// builder 把"有 setter 的设置项"（码率、质量、profile、线程数……）视作自己独占的
/// 键；[`Options`] 只承载 builder 未建模的私有参数。同一项若同时出现在两处，以谁为准
/// 只能靠隐含的先后顺序，是个静默陷阱，因此这里直接报
/// [`RsmediaError::InvalidConfig`]，并在消息里指出该用哪个 setter。
///
/// `owned` 是 `(AVOption key, 对应的 builder setter)` 列表，**只包含调用方显式设置过
/// 的项**：默认值不算配置冲突（否则"用透传设 `threads`"这种合法用法会被误伤）。
pub(crate) fn ensure_single_source(
    passthrough: Option<&Options>,
    owned: &[(&str, &str)],
) -> Result<()> {
    let Some(passthrough) = passthrough else {
        return Ok(());
    };
    let conflicts: Vec<String> = owned
        .iter()
        .filter(|(key, _)| passthrough.contains_key(key))
        .map(|(key, setter)| format!("'{key}' (also set by {setter})"))
        .collect();
    if conflicts.is_empty() {
        return Ok(());
    }
    Err(RsmediaError::invalid_config(format!(
        "options conflict with builder setters: {}; configure each setting in exactly one \
         place — builder setters for what they model, `with_options` only for codec-private \
         parameters without a setter",
        conflicts.join(", ")
    )))
}

/// Video encoders whose FFmpeg wrapper exposes a `crf` private option.
///
/// Single source of truth: used by [`EncoderBuilder::with_quality`](crate::EncoderBuilder::with_quality) /
/// [`Quality::Crf`] (fall back to bit-rate control for other codecs) and the
/// [`Quality::Crf`] capability documentation.
pub const CRF_CAPABLE_CODECS: &[&str] = &[
    "libx264",
    "libx265",
    "libvpx",
    "libvpx-vp9",
    "libaom-av1",
    "libsvtav1",
    "libopenh264",
];

/// Rate control strategy for video encode, set via
/// [`crate::EncoderBuilder::with_quality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Constant Rate Factor — quality-targeted, file size varies.
    /// Lower value = higher quality (x264 scale, sane range 0..=51,
    /// defaults around 18-28).
    ///
    /// Only the codecs in [`CRF_CAPABLE_CODECS`] take the `crf` option; for any
    /// other encoder the builder logs a warning and rate control falls back to the
    /// configured bit rate ([`EncoderBuilder::with_bit_rate`], else the
    /// per-media-type default shown on [`EncoderBuilder`]) — it does **not**
    /// silently become `Quality::Bitrate`.
    ///
    /// [`CRF_CAPABLE_CODECS`]: crate::options::CRF_CAPABLE_CODECS
    /// [`EncoderBuilder`]: crate::EncoderBuilder
    /// [`EncoderBuilder::with_bit_rate`]: crate::EncoderBuilder::with_bit_rate
    Crf(u8),
    /// Explicit target bit rate in bits per second.
    Bitrate(i64),
}

/// H.264-style encoding profile, set via [`crate::EncoderBuilder::with_profile`].
///
/// The variant name is passed as the codec's `profile` private option
/// string (`"baseline"`, `"main"`, `"high"`, ...). `libx264` supports all
/// variants directly; other encoders map what they support and ignore
/// unknown names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoProfile {
    Baseline,
    Main,
    High,
    High10,
    High422,
    High444,
}

impl VideoProfile {
    /// FFmpeg private-option string for this profile (x264 naming).
    pub fn as_option_str(&self) -> &'static str {
        match self {
            VideoProfile::Baseline => "baseline",
            VideoProfile::Main => "main",
            VideoProfile::High => "high",
            VideoProfile::High10 => "high10",
            VideoProfile::High422 => "high422",
            VideoProfile::High444 => "high444",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_debug() {
        let opts = Options::preset_h264_realtime();
        println!("{:?}", opts);
    }

    #[test]
    fn test_options_basic_ops() {
        let mut opts = Options::new();
        assert!(opts.is_empty());

        opts.set("threads", "4").set("preset", "fast");
        assert_eq!(opts.len(), 2);
        assert_eq!(opts.get("threads"), Some("4"));
        assert_eq!(opts.get("missing"), None);
        assert!(opts.contains_key("preset"));

        // overwrite semantics
        opts.set("threads", "8");
        assert_eq!(opts.get("threads"), Some("8"));

        assert_eq!(opts.remove("preset"), Some("fast".to_string()));
        assert_eq!(opts.remove("preset"), None);
        assert_eq!(opts.len(), 1);
    }

    #[test]
    fn test_options_merge_other_wins() {
        let mut base = Options::new();
        base.set("preset", "medium").set("crf", "23");

        let mut overlay = Options::new();
        overlay.set("crf", "18").set("tune", "film");

        base.merge(overlay);
        assert_eq!(base.get("preset"), Some("medium"));
        assert_eq!(base.get("crf"), Some("18"));
        assert_eq!(base.get("tune"), Some("film"));
    }

    #[test]
    fn test_options_empty_to_dict_is_none() {
        // 根治旧实现：空 Options 物化为 None，不再携带 key="" 的脏条目。
        let opts = Options::new();
        assert!(opts.into_dict().is_none());
    }

    #[test]
    fn test_options_dict_roundtrip() {
        let mut opts = Options::new();
        opts.set("crf", "23").set("profile", "high");

        let dict = opts.to_dict().expect("non-empty must materialize");
        let back = Options::from_dict(&dict);
        assert_eq!(back.get("crf"), Some("23"));
        assert_eq!(back.get("profile"), Some("high"));
        assert_eq!(back.len(), 2);
    }

    #[test]
    fn test_options_hashmap_roundtrip() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), "1".to_string());

        let opts: Options = map.clone().into();
        assert_eq!(opts.get("a"), Some("1"));

        let back: HashMap<String, String> = opts.into();
        assert_eq!(back, map);
    }

    #[test]
    fn test_options_btreemap_roundtrip() {
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), "1".to_string());
        map.insert("b".to_string(), "2".to_string());

        let opts: Options = map.clone().into();
        assert_eq!(opts.get("b"), Some("2"));

        let back: BTreeMap<String, String> = opts.into();
        assert_eq!(back, map);
    }

    #[test]
    fn test_options_from_raw_dict_null_is_empty() {
        // FFmpeg 的空字典就是 NULL 指针：from_raw_dict(NULL) 必须得到空 Options。
        let opts = unsafe { Options::from_raw_dict(std::ptr::null_mut()) };
        assert!(opts.is_empty());
    }

    #[test]
    fn test_options_raw_dict_read() {
        let mut dict = AVDictionary::new(c"title", c"hello", 0);
        // SAFETY: `dict` 是合法字典，且在本调用期间存活。
        let opts = unsafe { Options::from_raw_dict(dict.as_mut_ptr()) };
        assert_eq!(opts.get("title"), Some("hello"));
        assert_eq!(opts.len(), 1);
    }

    #[test]
    fn test_options_write_into_raw_dict_roundtrip() {
        let mut opts = Options::new();
        opts.set("title", "hello").set("artist", "rsmedia");

        let mut dest: *mut ffi::AVDictionary = std::ptr::null_mut();
        // SAFETY: `dest` 为 NULL，函数按"空槽位"处理。
        unsafe { opts.write_into_raw_dict(&mut dest) };
        assert!(!dest.is_null(), "non-empty options must materialize");

        // SAFETY: `dest` 刚由 to_dict 物化，合法且存活。
        let back = unsafe { Options::from_raw_dict(dest) };
        assert_eq!(back.get("title"), Some("hello"));
        assert_eq!(back.get("artist"), Some("rsmedia"));

        // SAFETY: 释放测试自建的字典，避免泄漏。
        unsafe { ffi::av_dict_free(&mut dest) };
        assert!(dest.is_null());
    }

    #[test]
    fn test_options_write_into_raw_dict_empty_stores_null() {
        let mut dest: *mut ffi::AVDictionary = std::ptr::null_mut();
        // SAFETY: `dest` 为 NULL。
        unsafe { Options::new().write_into_raw_dict(&mut dest) };
        assert!(
            dest.is_null(),
            "empty Options must store NULL (no metadata)"
        );

        // 已有内容时写入空 Options：旧字典被释放并清回 NULL。
        let owned = AVDictionary::new(c"title", c"hello", 0);
        let mut dest = owned.into_raw().as_ptr();
        // SAFETY: `dest` 指向刚转移的合法字典。
        unsafe { Options::new().write_into_raw_dict(&mut dest) };
        assert!(dest.is_null());
    }

    #[test]
    fn test_ensure_single_source() {
        let mut passthrough = Options::new();
        passthrough.set("threads", "4");

        // 没有透传选项 → 无冲突
        assert!(ensure_single_source(None, &[("threads", "with_thread_count")]).is_ok());
        // 透传的键与 setter 无关 → 无冲突
        assert!(ensure_single_source(Some(&passthrough), &[("b", "with_bit_rate")]).is_ok());
        // 未显式设置过的 setter 不在表里 → "用透传设 threads" 合法
        assert!(ensure_single_source(Some(&passthrough), &[]).is_ok());

        // 同一项两个来源 → InvalidConfig，消息指出键与对应 setter
        let err = ensure_single_source(Some(&passthrough), &[("threads", "with_thread_count")])
            .expect_err("conflicting key must be rejected");
        assert!(matches!(err, RsmediaError::InvalidConfig(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'threads'") && msg.contains("with_thread_count"),
            "message must name the key and its setter: {msg}"
        );
    }

    #[test]
    fn test_options_presets_fixed_keys() {
        // "profile:v"/"b:v" 是 ffmpeg CLI 语法，AVOption 查找不到会被静默
        // 忽略；preset 必须使用真正的 AVOption 名。
        assert_eq!(Options::preset_h264().get("profile"), Some("high"));
        assert_eq!(Options::preset_h264().get("profile:v"), None);

        let rt = Options::preset_h264_realtime();
        assert_eq!(rt.get("profile"), Some("main"));
        assert_eq!(rt.get("b"), Some("3000k"));
        assert_eq!(rt.get("b:v"), None);
        // x264opts 是 CLI 组合器，AVOption 中不存在
        assert_eq!(rt.get("x264opts"), None);

        assert_eq!(Options::preset_h264_nvenc().get("b"), Some("4000k"));
    }
}
