use crate::error::{Context, Result, RsmediaError};
use crate::pixel::PixelFormat;
use crate::{Options, imgutils, strutils};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use once_cell::sync::Lazy;
use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext, AVPixFmtDescriptorRef};
use rsmpeg::{UnsafeDerefMut, ffi};

use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

/// Hardware device configuration.
/// This struct contains all the necessary information to create a hardware device context.
///
/// The sw / hw frames conversion process includes the following steps:
///
/// CPU(NV12) -> GPU(CUDA) -> transform -> GPU(CUDA) -> CPU(NV12)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HWDeviceConfig {
    pub device_type: HWDeviceType,
    pub hw_pixel_format: PixelFormat,
    pub sw_pixel_format: PixelFormat,
    pub device_id: Option<String>,
    pub options: Option<Options>,
}

impl HWDeviceConfig {
    /// create a new HWDeviceConfig with the given parameters
    ///
    /// # Arguments
    ///
    /// * `device_type` - The type of hardware device
    /// * `hw_pixel_format` - The pixel format of the hardware device
    /// * `sw_pixel_format` - The pixel format of the software device
    /// * `device_id` - The type-specific string identifying of the GPU device,
    ///   e.g. for NVIDIA CUDA, device_id should be explicitly the GPU ID  "0" or "1",
    ///   for VAAPI: device_id should be set like "/dev/dri/renderD128"
    /// * `options` - Additional (type-specific) options to use in opening the device
    pub fn new(
        device_type: HWDeviceType,
        hw_pixel_format: PixelFormat,
        sw_pixel_format: PixelFormat,
        device_id: Option<String>,
        options: Option<Options>,
    ) -> Self {
        Self {
            device_type,
            hw_pixel_format,
            sw_pixel_format,
            device_id,
            options,
        }
    }

    /// 按设备类型的**默认格式映射**构造配置。
    ///
    /// [`Self::default_hw_pixel_format`]/[`Self::default_sw_pixel_format`]（定义在
    /// [`HWDeviceType`] 上）是格式映射的唯一真相源：构造器与平台自动选择
    /// （[`Self::auto_platform`]）都走这里，避免同一设备类型出现两套说法。
    fn default_for(device_type: HWDeviceType, device_id: Option<String>) -> Self {
        Self::new(
            device_type,
            device_type.default_hw_pixel_format(),
            device_type.default_sw_pixel_format(),
            device_id,
            None,
        )
    }

    /// build CUDA HWDeviceConfig
    ///
    /// `device_id` 为 GPU 编号字符串（如 `"0"`、`"1"`），与其他设备构造器
    /// 的类型保持一致（VAAPI 传 DRM 设备路径、QSV 传设备序号等）。
    pub fn cuda(device_id: Option<String>) -> Self {
        Self::default_for(HWDeviceType::CUDA, device_id)
    }

    /// build VAAPI HWDeviceConfig
    pub fn vaapi(device_id: Option<String>) -> Self {
        Self::default_for(HWDeviceType::VAAPI, device_id)
    }

    /// build VULKAN HWDeviceConfig
    pub fn vulkan(device_id: Option<String>) -> Self {
        Self::default_for(HWDeviceType::VULKAN, device_id)
    }

    /// build QSV (Intel Quick Sync Video) HWDeviceConfig
    pub fn qsv(device_id: Option<String>) -> Self {
        Self::default_for(HWDeviceType::QSV, device_id)
    }

    /// build AMD AMF HWDeviceConfig（Windows 平台，基于 D3D11 设备）。
    ///
    /// FFmpeg 的 AMF 编码器（`h264_amf`/`hevc_amf`/`av1_amf`）没有独立的
    /// hw_context 类型，挂在 `AV_HWDEVICE_TYPE_D3D11VA` 下：软件帧（NV12）
    /// 先上传到 D3D11 surface，再由 AMF 编码。
    #[cfg(target_os = "windows")]
    pub fn amf(device_id: Option<String>) -> Self {
        Self::default_for(HWDeviceType::D3D11VA, device_id)
    }

    /// 按当前平台自动选择最佳可用的硬件加速配置。
    ///
    /// 依 [`HWDeviceType::platform_preference`] 的平台优先级依次**真实探测**
    /// （会为每个候选建立一次设备，见 [`HWDeviceType::is_usable`]），返回第一个
    /// 在本机能真正建起来的配置；一个都建不起来（无 GPU / 无驱动 / 无 FFmpeg
    /// 支持编译）时返回错误，**不会**回退到随机设备 —— 需要软件路径时由调用方
    /// 显式省略 hw 配置。
    ///
    /// 返回值可以直接交给 `with_hardware_device`，不会再出现"拿到的配置要到
    /// 建编解码器时才失败"的两段式错误。
    pub fn auto_platform() -> Result<Self> {
        HWDeviceType::auto_platform_config(None)
    }

    /// [`Self::auto_platform`] 的可定制版本：传入自定义候选顺序（如只想在
    /// CUDA 与 QSV 之间选择）；空切片返回错误（等价于无候选可探测）。
    pub fn auto_platform_with(candidates: &[HWDeviceType]) -> Result<Self> {
        HWDeviceType::auto_platform_config(Some(candidates))
    }
}

impl std::fmt::Display for HWDeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// 该设备类型在 `device_id` 未给出时应当传给 FFmpeg 的 device 字符串。
///
/// # 为什么绝大多数后端返回 `None`（即 `device = NULL`）
///
/// 传 `NULL` 让后端自己选默认设备，是 FFmpeg 对**除 DRM 外**所有后端的既定约定，
/// 逐一看过后端源码可以确认（FFmpeg 6.1 / 7.1 / 8.1 / 9.0 一致）：
///
/// | 后端 | `device == NULL` 时的行为 |
/// |------|--------------------------|
/// | VAAPI | `if (device) {…} else {…}`：自己扫描 `/dev/dri/renderD128..135` |
/// | VDPAU | `XOpenDisplay(NULL)`（合法，取 `$DISPLAY`）；失败有 `!priv->dpy` 检查 |
/// | CUDA | `if (device) device_idx = strtol(…);` → 默认 0 号卡 |
/// | Vulkan / OpenCL | `if (device && device[0])` → 自动选择物理设备 / 平台 |
/// | D3D11VA | `if (device) {…} else {…}` → 默认适配器（或按 `vendor_id` 选项）|
/// | DXVA2 / D3D12VA | `device ? atoi(device) : 0` → 默认适配器 |
/// | QSV | 用 `child_device_type` 选项派生，device 串不参与主路径 |
/// | VideoToolbox / MediaCodec / OHCodec | 只用 device 作名字过滤，NULL 即"不过滤" |
/// | AMF | 完全不用 device 串，走自身枚举 |
///
/// 而且**没有任何一个后端**会因此 `exit()`/`abort()`：全部 15 个 `hwcontext_*.c`
/// 里都没有这类调用。实测也一致（macOS arm64 + Linux aarch64 + Linux x86_64
/// × FFmpeg 6.1/7.1/8.1/9.0 × 全部 15 种类型）：除 DRM 外 `NULL` 要么建成设备，
/// 要么返回干净的错误码（未编入该后端时为 `ENOMEM`，无驱动时为其它负值）， 无一崩溃。
///
/// # 为什么只有 DRM 例外
///
/// `libavutil/hwcontext_drm.c` 的 `drm_device_create()` 是唯一一个把 device
/// **不加判断**地交给 `open()` 的后端：
///
/// ```c
/// hwctx->fd = open(device, O_RDWR);   /* device 为 NULL 时即 open(NULL, …) */
/// ```
///
/// 这是 FFmpeg 侧的疏漏（6.1 到 9.0 的代码完全相同，未修）。后果依平台而异：
/// 原生 aarch64 上内核返回 `EFAULT`，FFmpeg 干净地报 `AVERROR(EFAULT)`；
/// 但在 **Rosetta 转译的 x86_64** 上直接 SIGSEGV —— 实测退出码 139，
/// 最小 C 探针（只调 libavutil、不经 CLI）同样复现。DRM 恰恰是
/// [`HWDeviceType::platform_preference`] 在 Linux 上的候选之一，所以它必须由
/// 我们自己补一个真实节点，不能把这个坑留给 FFmpeg。
///
/// 注意这只覆盖 rsmedia 自己的创建路径。若调用方通过 `with_options` 把
/// `hwaccel` / `hwaccel_device` 之类的 **AVCodecContext 选项**传给 FFmpeg，
/// 设备将由 FFmpeg 内部创建，那时仍要显式给出
/// `hwaccel_device=/dev/dri/renderD128`，否则会踩到同一处 `open(NULL)`。
fn default_device_string(kind: HWDeviceType) -> Result<Option<CString>> {
    if kind == HWDeviceType::DRM {
        return drm_node_from(Path::new("/dev/dri")).map(Some);
    }
    // 见上方说明：其余后端一律交给 FFmpeg 自动选择。
    Ok(None)
}

/// 在 `dir` 下挑一个 DRM 节点，返回它的路径。
///
/// 优先**渲染节点**（`renderD*`，通常无需额外权限即可打开）其次**显示节点**
/// （`card*`）。同类之间按名字排序后取第一个：同一台机器上多次调用必须挑到同一个
/// 节点，否则 [`HWDeviceConfig`] 相同（`device_id` 都是 `None`）却指向不同设备，
/// 缓存的键就名不副实。
///
/// 目录不存在（非 Linux 平台、容器里没映射设备）或没有可用节点时返回
/// [`RsmediaError::unsupported`] —— **绝不返回 `NULL`**，也不去调 FFmpeg。
fn drm_node_from(dir: &Path) -> Result<CString> {
    let mut rendered: Vec<std::path::PathBuf> = Vec::new();
    let mut cards: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // 只认节点文件名前缀：实机的 `/dev/dri` 下还会有 `by-path/`（符号链接
            // 目录）与 `controlD64` 之类的控制节点，它们都不是可打开的渲染设备。
            if name.starts_with("renderD") {
                rendered.push(entry.path());
            } else if name.starts_with("card") {
                cards.push(entry.path());
            }
        }
    }
    rendered.sort();
    cards.sort();
    let node = rendered.into_iter().chain(cards).next().ok_or_else(|| {
        RsmediaError::unsupported(format!(
            "DRM hardware device needs a node under {}, but none exists",
            dir.display()
        ))
    })?;
    strutils::os_str_to_cstring(node.as_os_str()).context("Invalid DRM device node path")
}

/// `HWContext` cache for safe sharing of hardware device contexts.
/// Uses DashMap for better concurrent read/write performance.
static HW_CTX_CACHE: Lazy<DashMap<HWDeviceConfig, Arc<HWContext>>> = Lazy::new(DashMap::new);

/// 缓存容量上限：超出时自动驱逐**未被使用**（引用计数为 1）的条目。
///
/// 硬件设备上下文持有 GPU 资源，长驻进程中持续切换配置（device_id / 选项 /
/// 设备类型组合不同）会不断产生新条目；不设上限会导致 GPU 资源泄漏。
/// 典型应用只使用 1~2 种配置，8 已留足余量。仍在使用的条目不会被驱逐，
/// 全部在使用中时可超额容纳（等待 [`release_unused_hw_contexts`] 后续清理）。
const HW_CTX_CACHE_MAX_ENTRIES: usize = 8;

/// 无锁地收集缓存中**当前未被使用**（引用计数为 1，仅缓存自身持有）的条目键。
fn unused_hw_ctx_configs() -> Vec<HWDeviceConfig> {
    HW_CTX_CACHE
        .iter()
        .filter(|entry| Arc::strong_count(entry.value()) <= 1)
        .map(|entry| entry.key().clone())
        .collect()
}

/// 逐个移除 `candidates` 并返回被移除的上下文。
///
/// 移除用 `remove_if`：判定与移除在同一分片锁内原子完成，迭代期间又有使用者
/// 拿到引用（`strong_count` 变大）的条目会被跳过而不是被强行移除。
///
/// 被移除的 `Arc` 一律**交还给调用方**（`remove_if` 把值返回，而不是在锁内析构），
/// 由调用方在锁外 drop：`Arc<HWContext>` 的析构会 unref 底层 `AVBufferRef`，
/// 可能触发 FFmpeg 日志回调与驱动调用，在 DashMap 的写锁内做会阻塞其它线程。
fn remove_cached_hw_ctxs(candidates: Vec<HWDeviceConfig>) -> Vec<Arc<HWContext>> {
    let mut removed = Vec::new();
    for config in candidates {
        if let Some((_, ctx)) =
            HW_CTX_CACHE.remove_if(&config, |_, ctx| Arc::strong_count(ctx) <= 1)
        {
            removed.push(ctx);
        }
    }
    removed
}

/// 容量超限时驱逐未使用条目（保留使用中的与新创建的 `keep` 条目）。
fn prune_hw_ctx_cache(keep: &HWDeviceConfig) {
    if HW_CTX_CACHE.len() <= HW_CTX_CACHE_MAX_ENTRIES {
        return;
    }
    let candidates = unused_hw_ctx_configs()
        .into_iter()
        .filter(|config| config != keep)
        .collect::<Vec<_>>();
    // 锁外析构（见 `remove_cached_hw_ctxs`）
    let removed = remove_cached_hw_ctxs(candidates).len();
    if HW_CTX_CACHE.len() > HW_CTX_CACHE_MAX_ENTRIES {
        tracing::warn!(
            "HW context cache still holds {} entries (> {HW_CTX_CACHE_MAX_ENTRIES}) \
             after pruning {removed}: all in use, will shrink once released.",
            HW_CTX_CACHE.len()
        );
    } else if removed > 0 {
        tracing::debug!("Pruned {removed} unused hardware device context(s) (cache over cap).");
    }
}

/// 释放所有**当前未被使用**的硬件设备上下文，返回被释放的条目数。
///
/// 缓存会保留最近使用的设备（上限见 `HW_CTX_CACHE_MAX_ENTRIES`），好让同一配置的
/// 解码器/编码器反复复用同一个 GPU 设备；代价是条目与其设备上下文会常驻到进程结束。
/// 长驻进程在结束一批转码、切换设备配置或需要立刻回收 GPU 资源时，可以调用本函数：
/// 仅被缓存自身持有（引用计数为 1）的上下文会被移除并释放，仍被解码器/编码器使用的
/// 条目会保留下来，待其释放后再调用一次即可回收。
///
/// 日常驱逐由 `prune_hw_ctx_cache`（容量超限时）负责，本函数是显式的确定性释放入口。
pub fn release_unused_hw_contexts() -> usize {
    let removed = remove_cached_hw_ctxs(unused_hw_ctx_configs()).len();
    if removed > 0 {
        tracing::debug!("Released {removed} unused hardware device context(s).");
    }
    removed
}

/// 硬件帧池的默认预分配表面数。
///
/// 这个数**直接决定显存占用**：1080p NV12 一张面约 3MB，4K 约 12MB，8K 约 50MB。
/// 20 张对 1080p（约 60MB）是安全且够用的启发值（解码器 DPB + 滤镜缓冲 + 编码
/// 上传面都从同一个池里取），但对 4K/8K 会白白占掉数百 MB 显存。
///
/// 需要按分辨率/内存预算调整时用
/// [`EncoderBuilder::with_hw_pool_size`](crate::encode::EncoderBuilder::with_hw_pool_size)
/// 或 [`DecoderBuilder::with_hw_pool_size`](crate::decode::DecoderBuilder::with_hw_pool_size)；
/// 传 `0` 表示交给后端自己决定（FFmpeg 的默认行为：按需分配，不预占）。
pub(crate) const DEFAULT_HW_POOL_SIZE: u32 = 20;

/// 串行化所有触碰进程级 `HW_CTX_CACHE` 的测试。
///
/// 缓存是**进程级**静态，而 lib 测试在同一进程里并行跑：任何创建或持有
/// [`HWContext`] 的测试（无论写在哪个模块）都必须先拿这把锁，否则
/// [`release_unused_hw_contexts`] 那类按引用计数断言的测试会被别的测试正好持有的
/// 上下文干扰（`strong_count > 1` → 该条目"仍在使用"，不会被释放）。
#[cfg(test)]
pub(crate) fn hw_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static HW_CACHE_TEST_LOCK: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));
    // 某个测试 panic 后锁会被标记为 poisoned；这里恢复内部值继续用（`into_inner`）——
    // 被破坏的只是那个测试留下的状态，与本测试的断言无关，没必要让后续测试连锁失败。
    HW_CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A live hardware device context, plus the frame setup derived from it.
///
/// Crate-internal: a caller configures [`HWDeviceConfig`] and hands it to
/// [`EncoderBuilder::with_hardware_device`](crate::encode::EncoderBuilder::with_hardware_device)
/// or the decoder equivalent; the context itself is the plumbing between those
/// builders and FFmpeg, never something a user holds.
///
/// 所有方法只做共享访问（`&self`）：底层 `av_hwframe_ctx_alloc` / `av_buffer_ref`
/// 均为 FFmpeg 保证的线程安全原子操作，因此 [`Send`]/[`Sync`] 实现成立，
/// 相同配置的多个解码器/编码器可跨线程共享同一 `Arc<HWContext>`。
pub(crate) struct HWContext {
    config: HWDeviceConfig,
    device_ctx: AVHWDeviceContext,
}

impl HWContext {
    /// create a new HWContext with the given HWDeviceConfig
    pub(crate) fn new(config: HWDeviceConfig) -> Result<Arc<HWContext>> {
        // 快路径：命中缓存直接复用，无需（也不应）再次打开设备。
        if let Some(ctx) = HW_CTX_CACHE.get(&config) {
            tracing::debug!("Reusing existing hardware device context. config:{config:?}");
            return Ok(ctx.clone());
        }

        // 创建设备上下文**不持缓存锁**：打开 GPU 设备可能耗时并触发 FFmpeg 日志，
        // 不应阻塞同一分片上的其它线程。
        let hw_device_ctx = {
            // device_id 来自调用者（CUDA 是 GPU 编号、VAAPI/DRM 是设备路径），
            // 含内部 NUL 的输入只能是调用者的错误：报错而不是 panic。
            // 未给出时按设备类型取默认——DRM 必须拿到一个真实节点，见 [`default_device_string`]。
            let device = match config.device_id.as_deref() {
                Some(device_id) => Some(
                    strutils::os_str_to_cstring(device_id).context("Invalid hardware device id")?,
                ),
                None => default_device_string(config.device_type)?,
            };
            let opts = config.options.as_ref().and_then(|opts| opts.to_dict());
            AVHWDeviceContext::create(
                config.device_type.into(),
                device.as_deref(),
                opts.as_ref(),
                0,
            )
            .context("Failed to create hardware device context")?
        };

        tracing::debug!("Created hardware device context successfully. config:{config}");

        let ctx = Arc::new(Self {
            config: config.clone(),
            device_ctx: hw_device_ctx,
        });

        // 抢占式插入：`entry()` 在同一分片锁内原子地"存在则取用、不存在则插入"。
        // 若像以前那样先 `get()` 后 `insert()`，两个线程可能同时 miss、各自打开一次
        // 设备，后插入的会覆盖先插入的条目，被覆盖的设备再无人引用 —— 重复创建且泄漏。
        // 这里并发时只有第一个线程的上下文进入缓存，其余线程复用同一个 `Arc`。
        let mut duplicate = None;
        let cached = {
            // 整个 match 放在块中：`entry` 守卫（分片写锁）随块一起析构。
            match HW_CTX_CACHE.entry(config.clone()) {
                Entry::Occupied(entry) => {
                    tracing::debug!("Reusing existing hardware device context. config:{config:?}");
                    let existing = entry.get().clone();
                    duplicate = Some(ctx);
                    existing
                }
                Entry::Vacant(entry) => {
                    entry.insert(ctx.clone());
                    ctx
                }
            }
        };
        // 锁已释放：此刻才析构落败的重复上下文（`Arc<HWContext>` 析构会 unref 硬件
        // 设备，可能触发 FFmpeg 日志回调/驱动调用），不在写锁内做。
        drop(duplicate);

        // 超过容量上限时自动驱逐未使用的旧条目（本条目刚插入、且被局部 `cached`
        // 持有，不会被驱逐）
        prune_hw_ctx_cache(&cached.config);

        Ok(cached)
    }

    /// Initialize the hardware frames context for a **decoder**.
    ///
    /// Besides creating and attaching the `AVHWFramesContext`, this also:
    /// - installs the `hwaccel_get_format` callback so the decoder picks the
    ///   hardware surface format during `avcodec_open2`;
    /// - sets `sw_pix_fmt` to the configured software format;
    /// - holds an independent reference (`av_buffer_ref`) to the hardware
    ///   device context, so the decoder owns its own ref and unrefs it on
    ///   close — no manual teardown needed in `Decoder::Drop`.
    ///
    /// # Arguments
    ///
    /// * `codec_ctx` - The decoder codec context to initialize
    /// * `width` - The width of the decoded frames
    /// * `height` - The height of the decoded frames
    /// * `pool_size` - Preallocated surface count, see [`DEFAULT_HW_POOL_SIZE`]
    pub(crate) fn setup_decoder_frames(
        &self,
        codec_ctx: &mut AVCodecContext,
        width: i32,
        height: i32,
        pool_size: u32,
    ) -> Result<()> {
        let hw_frames_ctx = self.create_hw_frames_ctx(width, height, pool_size)?;
        codec_ctx.set_hw_frames_ctx(hw_frames_ctx);
        codec_ctx.set_pix_fmt(self.get_format(true));

        // SAFETY: rsmpeg's wrap types do not implement `DerefMut`, so field writes
        // must go through `UnsafeDerefMut::deref_mut`; this access is exclusive —
        // `codec_ctx` is a `&mut` borrow held for the whole block, so no other
        // reference to the context exists. `get_format` is set to an `extern "C"`
        // function whose signature is exactly `AVCodecContext.get_format` (so the
        // ABI matches and FFmpeg may call it), and `sw_pix_fmt` is the software
        // format that same callback falls back to.
        unsafe {
            let ctx_mut_ptr = codec_ctx.deref_mut();
            ctx_mut_ptr.get_format = Some(hwaccel_get_format);
            ctx_mut_ptr.sw_pix_fmt = self.get_format(false);
        }
        // clone 即 av_buffer_ref：codec_ctx 拥有独立引用，析构时正确 unref，
        // 无需 Decoder::Drop 手动置空防 double-free。
        codec_ctx.set_hw_device_ctx(self.device_ctx.clone());

        Ok(())
    }

    /// Initialize the hardware frames context for an **encoder**.
    ///
    /// Encoders upload software frames into surfaces allocated from this
    /// frames context (see [`HWContext::hw_upload`]); they do not need a
    /// `get_format` callback or a separate device ref (the frames context
    /// already references the device).
    ///
    /// # Arguments
    ///
    /// * `codec_ctx` - The encoder codec context to initialize
    /// * `width` - The width of the frames to encode
    /// * `height` - The height of the frames to encode
    /// * `pool_size` - Preallocated surface count, see [`DEFAULT_HW_POOL_SIZE`]
    pub(crate) fn setup_encoder_frames(
        &self,
        codec_ctx: &mut AVCodecContext,
        width: i32,
        height: i32,
        pool_size: u32,
    ) -> Result<()> {
        let hw_frames_ctx = self.create_hw_frames_ctx(width, height, pool_size)?;
        codec_ctx.set_hw_frames_ctx(hw_frames_ctx);
        codec_ctx.set_pix_fmt(self.get_format(true));

        Ok(())
    }

    /// Allocate and initialize an `AVHWFramesContext` bound to this device.
    ///
    /// 仅共享访问 device_ctx：`hwframe_ctx_alloc` 内部只做 av_buffer_ref（原子），
    /// 每次调用都新建独立的 AVHWFramesContext，由调用方（codec_ctx）独占持有。
    pub(crate) fn create_hw_frames_ctx(
        &self,
        width: i32,
        height: i32,
        pool_size: u32,
    ) -> Result<rsmpeg::avutil::AVHWFramesContext> {
        let mut hw_frames_ctx = self.device_ctx.hwframe_ctx_alloc();
        hw_frames_ctx.data().format = self.get_format(true);
        hw_frames_ctx.data().sw_format = self.get_format(false);
        hw_frames_ctx.data().width = width;
        hw_frames_ctx.data().height = height;
        hw_frames_ctx.data().initial_pool_size = pool_size as i32;

        hw_frames_ctx
            .init()
            .context("Failed to initialize hardware frame context")?;
        Ok(hw_frames_ctx)
    }

    /// 把一帧硬件帧搬进 `codec_ctx` 自己的 frames context。
    ///
    /// 编码器的 frames context 是它自己那份（每个编解码器各建一个），上游交来的
    /// 硬件帧却属于**别人的** frames context（解码器、另一台设备、或调用者自建）。
    /// 两者直接混用不可靠：`av_hwframe_map` 才是 FFmpeg 认可的搬运方式，同设备
    /// 且格式兼容时它只做一次 `av_buffer_ref`，**零拷贝**。
    ///
    /// 已经在同一 frames context 里的帧原样返回（`av_hwframe_map` 对同源同缓冲
    /// 反而报 `EINVAL`）。
    ///
    /// 后端没实现 `map_to`/`map_from`（如 VideoToolbox）时 `av_hwframe_map` 返回
    /// `AVERROR(ENOSYS)`——FFmpeg 文档称之为“以当前 hwframe 配置无法映射”，此时
    /// 退回 [`Self::hw_download`] + [`Self::hw_upload`]（download → upload，两次
    /// 全帧拷贝），结果帧一样落在 `codec_ctx` 自己的 frames context 里，只是不再
    /// 零拷贝。
    pub(crate) fn map_hw_frame(
        &self,
        codec_ctx: &mut AVCodecContext,
        src: AVFrame,
    ) -> Result<AVFrame> {
        let dst_ref = unsafe { (*codec_ctx.as_ptr()).hw_frames_ctx };
        if dst_ref.is_null() {
            return Err(RsmediaError::invalid_config(
                "Codec context has no hardware frames context to map into",
            ));
        }
        // 比较的是 AVHWFramesContext 对象本身（`AVBufferRef::data`），不是 buffer_ref
        // 结构体地址：`av_buffer_ref`/`av_hwframe_get_buffer` 每次都会新建一个
        // AVBufferRef 指向同一对象，结构体地址几乎总是不等。
        if !src.hw_frames_ctx.is_null() && unsafe { (*src.hw_frames_ctx).data == (*dst_ref).data } {
            return Ok(src);
        }

        let mut dst = AVFrame::new();
        let map_ret = unsafe {
            let dst_ptr = dst.as_mut_ptr();
            let src_ptr = src.as_ptr();
            (*dst_ptr).format = (*src_ptr).format;
            (*dst_ptr).width = (*src_ptr).width;
            (*dst_ptr).height = (*src_ptr).height;
            // av_buffer_ref：dst 持有独立引用，随 AVFrame 一起 unref，不影响 codec_ctx 那份。
            (*dst_ptr).hw_frames_ctx = ffi::av_buffer_ref(dst_ref);
            // flags 按 FFmpeg 文档传 0（当前未使用）。失败时 dst 由 Drop 负责
            // unref 上面那个 buffer ref，无需手工清理。
            ffi::av_hwframe_map(dst_ptr, src_ptr, 0)
        };
        if map_ret < 0 {
            if map_ret != -(ffi::ENOSYS as i32) {
                return Err(RsmediaError::av_error(map_ret).with_context(
                    "Failed to map the hardware frame into this codec's frames context \
                     (the two frames contexts must live on the same device and agree on \
                     format and size)",
                ));
            }
            // 该后端只能 transfer，不能 map：先下载到系统内存，再上传到编码器的
            // frames context。`hw_download`/`hw_upload` 内部会搬运帧属性（含 pts）。
            tracing::debug!(
                "av_hwframe_map is not implemented for {:?}; falling back to download + upload",
                self.config.device_type
            );
            let sw = self.hw_download(&src)?;
            return self.hw_upload(codec_ctx, &sw);
        }

        tracing::debug!(
            "Mapped HW frame into the codec frames context: {:?} {}x{}",
            PixelFormat::from(dst.format),
            dst.width,
            dst.height
        );
        Ok(dst)
    }

    /// Download frame from hardware acceleration device to system memory.
    ///
    /// This method transfers the frame data from GPU memory to CPU memory,
    /// converting from hardware pixel format to software pixel format.
    ///
    /// 纯函数：transfer 使用的 `AVHWFramesContext` 取自 `hw_frame` 自身的
    /// `hw_frames_ctx`（解码器输出帧自带），无需解码器上下文参与。
    ///
    /// # Arguments
    /// * `hw_frame` - The source frame in hardware memory
    ///
    /// # Returns
    /// * `Result<AVFrame>` - A new frame in system memory with transferred data
    pub(crate) fn hw_download(&self, hw_frame: &AVFrame) -> Result<AVFrame> {
        let hw_down_start = std::time::Instant::now();

        // Check if input frame is actually in hardware memory
        if !self.is_hw_frame(hw_frame) {
            return Err(RsmediaError::msg(format!(
                "Input frame is not a valid hardware frame: format={:?}, expected={:?}, hw_frames_ctx={:p}",
                hw_frame.format, self.config.hw_pixel_format, hw_frame.hw_frames_ctx
            )));
        }

        // 创建软件帧
        let mut sw_frame = AVFrame::new();
        sw_frame.set_width(hw_frame.width);
        sw_frame.set_height(hw_frame.height);
        sw_frame.set_format(self.get_format(false));
        sw_frame
            .alloc_buffer()
            .context("Failed to allocate software frame buffer")?;

        // 从硬件帧传输数据到软件帧
        sw_frame
            .hwframe_transfer_data(hw_frame)
            .context("Failed to transfer data from hardware frame to software frame")?;

        // 复制帧属性
        self.copy_frame_props(hw_frame, &mut sw_frame)?;

        tracing::debug!(
            "Downloaded from GPU: format={:?}, size={}x{}, linesize=[{}, {}], cost={:?}ms",
            PixelFormat::from(sw_frame.format),
            sw_frame.width,
            sw_frame.height,
            sw_frame.linesize[0],
            sw_frame.linesize[1],
            hw_down_start.elapsed()
        );

        Ok(sw_frame)
    }

    /// Upload frame to hardware acceleration device.
    ///
    /// This method transfers the frame data from CPU memory to GPU memory,
    /// converting from software pixel format to hardware pixel format.
    ///
    /// # Arguments
    /// * `sw_frame` - The source frame in system memory
    ///
    /// # Returns
    /// * `Result<AVFrame>` - A new frame in hardware memory with transferred data
    pub(crate) fn hw_upload(
        &self,
        encoder: &mut AVCodecContext,
        sw_frame: &AVFrame,
    ) -> Result<AVFrame> {
        let hw_up_start = std::time::Instant::now();

        // Check if input frame format matches our software format
        if !self.is_sw_frame(sw_frame) {
            return Err(RsmediaError::msg(format!(
                "Input frame format ({:?}) doesn't match expected software format ({:?})",
                sw_frame.format, self.config.sw_pixel_format
            )));
        }

        // 确保编码器上下文有硬件帧上下文
        let mut hw_frames_ctx = encoder
            .hw_frames_ctx_mut()
            .ok_or_else(|| RsmediaError::msg("Encoder has no hardware frames context"))?;

        // 创建硬件帧
        let mut hw_frame = AVFrame::new();
        hw_frame.set_width(sw_frame.width);
        hw_frame.set_height(sw_frame.height);
        hw_frame.set_format(self.get_format(true));
        // 注意：这里不要再手动把 hw_frames_ctx 赋给 hw_frame。
        // av_hwframe_get_buffer 会在 frame->hw_frames_ctx 为 NULL 时自动 av_buffer_ref；
        // 若预先赋成与编码器共享同一 ref，则 hw_frame.unref 会把共享 ref 减到 0，
        // 造成编码器 hw_frames_ctx 悬空 → double-free。

        // 分配硬件缓冲区
        hw_frames_ctx
            .get_buffer(&mut hw_frame)
            .context("Failed to allocate hardware frame buffer")?;

        // 从软件帧传输数据到硬件帧
        hw_frame
            .hwframe_transfer_data(sw_frame)
            .context("Failed to transfer data from software frame to hardware frame")?;

        // 复制帧属性
        self.copy_frame_props(sw_frame, &mut hw_frame)?;

        tracing::debug!(
            "Uploaded to GPU: format={:?}, size={}x{}, linesize=[{}, {}], cost={:?}ms",
            PixelFormat::from(hw_frame.format),
            hw_frame.width,
            hw_frame.height,
            hw_frame.linesize[0],
            hw_frame.linesize[1],
            hw_up_start.elapsed()
        );

        Ok(hw_frame)
    }

    /// 复制视频帧属性（时间戳/画面类型/宽高比等）与 side-data 元数据。
    ///
    /// # Arguments
    /// * `src` - The source frame from which properties will be copied.
    /// * `dst` - The destination frame to which properties will be copied.
    fn copy_frame_props(&self, src: &AVFrame, dst: &mut AVFrame) -> Result<()> {
        dst.set_pts(src.pts);
        dst.set_time_base(src.time_base);
        dst.set_pict_type(src.pict_type);

        unsafe {
            let dst_ptr = dst.as_mut_ptr();
            (*dst_ptr).flags = src.flags;
            // 刻意**不**拷贝 `opaque`：它是 FFmpeg 留给应用层的私有指针，本 crate
            // 从不用它（见 `hwaccel_get_format`），而逐帧共享同一个 opaque 会让两个
            // 帧都指向调用者的同一份数据——新帧既不拥有它、也无法在其生命周期结束
            // 时做任何处理，调用者释放后即悬空。新帧的 `opaque` 保持 NULL。
            (*dst_ptr).quality = src.quality;
            (*dst_ptr).duration = src.duration;
            (*dst_ptr).sample_aspect_ratio = src.sample_aspect_ratio;
        }

        // 复制 side-data 与帧级元数据
        imgutils::copy_frame_metadata(src, dst, false)
    }

    /// Determine if a frame is in hardware memory
    ///
    /// # Arguments
    /// * `frame` - The frame to check
    ///
    /// # Returns
    /// * `bool` - True if the frame is in hardware memory
    pub(crate) fn is_hw_frame(&self, frame: &AVFrame) -> bool {
        // 检查硬件帧上下文是否为空
        if frame.hw_frames_ctx.is_null() {
            tracing::debug!("Frame hardware context is null");
            return false;
        }

        // 检查帧格式是否匹配硬件像素格式
        frame.format == self.get_format(true)
    }

    /// Check if a frame is in software memory format
    pub(crate) fn is_sw_frame(&self, frame: &AVFrame) -> bool {
        frame.format == self.get_format(false)
    }

    /// Helper function to get the appropriate pixel format for a frame
    pub(crate) fn get_format(&self, is_hw: bool) -> ffi::AVPixelFormat {
        if is_hw {
            self.config.hw_pixel_format.into()
        } else {
            self.config.sw_pixel_format.into()
        }
    }
}

/// SAFETY:
/// - `AVHWDeviceContext` 底层是引用计数的 `AVBufferRef`；本类型的所有方法
///   （`setup_decoder_frames`/`setup_encoder_frames`/`hw_download`/`hw_upload`）
///   只做共享访问，涉及的
///   FFI 调用（`av_buffer_ref`/`av_hwframe_ctx_alloc`/`av_hwframe_get_buffer`/
///   `av_hwframe_transfer_data`）均为 FFmpeg 保证的线程安全操作。
/// - 硬件设备本身由驱动保证并发使用安全。
unsafe impl Send for HWContext {}
unsafe impl Sync for HWContext {}

ffi_enum_wrap_from!(
    /// Hardware device type (FFmpeg `AV_HWDEVICE_TYPE_*`).
    ///
    /// Generated from one `variant => constant` table with a two-way `From`. A value the table
    /// does not list panics instead of degrading to `NONE`: an unknown device type means the
    /// caller's assumption about the source is wrong, so it should fail fast rather than silently
    /// fall back to "no hardware device".
    ///
    /// `AV_HWDEVICE_TYPE_NONE` itself is a listed value, so it still converts to `NONE`.
    HWDeviceType => ffi::AVHWDeviceType,
    repr = u32,
    fallback = panic {
        /// ffi definition NONE: 0
        NONE => ffi::AV_HWDEVICE_TYPE_NONE;
        /// Video Decode and Presentation API for Unix (VDPAU)
        VDPAU => ffi::AV_HWDEVICE_TYPE_VDPAU;
        /// NVIDIA CUDA
        CUDA => ffi::AV_HWDEVICE_TYPE_CUDA;
        /// Video Acceleration API (VA-API)
        VAAPI => ffi::AV_HWDEVICE_TYPE_VAAPI;
        /// DirectX Video Acceleration 2.0
        DXVA2 => ffi::AV_HWDEVICE_TYPE_DXVA2;
        /// Quick Sync Video
        QSV => ffi::AV_HWDEVICE_TYPE_QSV;
        /// VideoToolbox
        VIDEOTOOLBOX => ffi::AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
        /// Direct3D 11 Video Acceleration
        D3D11VA => ffi::AV_HWDEVICE_TYPE_D3D11VA;
        /// Linux Direct Rendering Manager
        DRM => ffi::AV_HWDEVICE_TYPE_DRM;
        /// OpenCL
        OPENCL => ffi::AV_HWDEVICE_TYPE_OPENCL;
        /// MediaCodec
        MEDIACODEC => ffi::AV_HWDEVICE_TYPE_MEDIACODEC;
        /// Vulkan
        VULKAN => ffi::AV_HWDEVICE_TYPE_VULKAN;
        /// Direct3D 12 Video Acceleration
        #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
        D3D12VA => ffi::AV_HWDEVICE_TYPE_D3D12VA;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        AMF => ffi::AV_HWDEVICE_TYPE_AMF;
        #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
        OHCODEC => ffi::AV_HWDEVICE_TYPE_OHCODEC;
    }
);

impl HWDeviceType {
    /// 该设备类型是否被**当前的 FFmpeg 构建**编入。
    ///
    /// 判据是 `av_hwdevice_iterate_types`，也就是说它回答的是"构建期支持哪些
    /// 设备类型"，**不是**"本机能不能真的用起来"。两者在有 GPU 的机器上通常一致，
    /// 但在 CI / 虚拟机 / 无显卡驱动的机器上差距很大：实测一台无 GPU 的 Ubuntu 上
    /// [`Self::list_available`] 报出 6 种类型，真正能建立设备的只有 1 种。
    ///
    /// 需要"能不能真的用"时请用 [`Self::is_usable`]；需要"帮我挑一个能用的"时
    /// 用 [`Self::auto_platform_config`]。本方法保留廉价语义（纯枚举、无副作用），
    /// 适合做能力展示或快速筛选。
    pub fn is_available(self) -> bool {
        Self::list_available().contains(&self)
    }

    /// 该设备类型在**本机**是否真的能用：实际建立一次设备上下文再释放。
    ///
    /// 与 [`Self::is_available`] 的区别见对方的文档。实现里也复用了
    /// `default_device_string`，因此 DRM 会去 `/dev/dri` 找一个真实节点
    /// （没有节点直接判为不可用），不会踩到 `open(NULL)` 那条路径 —— 那条路径在
    /// Rosetta 转译的 x86_64 上会 SIGSEGV，探测本身不能把调用方带走。
    ///
    /// **有副作用与开销**：会真的打开 GPU 设备（`/dev/dri`、Vulkan loader、
    /// CUDA 驱动等），单次调用开销在毫秒级。不要放进热路径。
    pub fn is_usable(self) -> bool {
        let Ok(device) = default_device_string(self) else {
            return false;
        };
        AVHWDeviceContext::create(self.into(), device.as_deref(), None, 0).is_ok()
    }

    /// 当前平台的硬件加速优先级（从高到低）。
    ///
    /// 排序依据与 FFmpeg CLI / 主流转码器的默认习惯一致：
    /// - macOS: VideoToolbox（Apple Silicon/Intel 均原生支持）
    /// - Windows: D3D11VA（承载 AMD AMF 及通用 D3D11 hwaccel）> QSV > CUDA > Vulkan
    /// - Linux: VAAPI（Intel/AMD 开箱即用）> CUDA > Vulkan
    /// - Android: MediaCodec
    pub fn platform_preference() -> Vec<HWDeviceType> {
        match std::env::consts::OS {
            "macos" => vec![HWDeviceType::VIDEOTOOLBOX, HWDeviceType::VULKAN],
            "windows" => vec![
                HWDeviceType::D3D11VA,
                HWDeviceType::QSV,
                HWDeviceType::CUDA,
                HWDeviceType::VULKAN,
                HWDeviceType::DXVA2,
            ],
            "linux" => vec![
                HWDeviceType::VAAPI,
                HWDeviceType::CUDA,
                HWDeviceType::VULKAN,
                HWDeviceType::VDPAU,
                HWDeviceType::OPENCL,
                HWDeviceType::DRM,
            ],
            "android" => vec![HWDeviceType::MEDIACODEC],
            _ => vec![],
        }
    }

    /// 平台自动选择：按 [`Self::platform_preference`]（或调用方自定义候选）
    /// 顺序探测，返回第一个可用设备的配置。
    ///
    /// # Arguments
    ///
    /// * `candidates` - 自定义候选顺序（空切片必报错）；`None` 使用平台默认优先级。
    pub fn auto_platform_config(candidates: Option<&[HWDeviceType]>) -> Result<HWDeviceConfig> {
        let preference: Vec<HWDeviceType> = match candidates {
            Some(list) => list.to_vec(),
            None => Self::platform_preference(),
        };
        if preference.is_empty() {
            return Err(RsmediaError::unsupported(format!(
                "No hardware acceleration preference defined for platform: {}",
                std::env::consts::OS
            )));
        }
        // 逐个候选**真实探测**（会建立一次设备），避免只用首个候选的枚举结果去
        // 匹配其它候选；更重要的是：枚举只说明"构建编入了"，无 GPU 的机器上
        // 建不起来的设备也会被枚举到，只查枚举就会返回一个注定失败的配置。
        let device = preference
            .iter()
            .find(|ty| ty.is_usable())
            .copied()
            .ok_or_else(|| {
                RsmediaError::unsupported(format!(
                    "No usable hardware acceleration device on {} (candidates probed: {preference:?})",
                    std::env::consts::OS
                ))
            })?;
        tracing::info!("Auto-selected hardware device: {device:?}");
        Ok(HWDeviceConfig::new(
            device,
            device.default_hw_pixel_format(),
            device.default_sw_pixel_format(),
            None,
            None,
        ))
    }

    /// List available hardware acceleration device types on this system.
    ///
    /// Uses `av_hwdevice_iterate_types` internally.
    pub fn list_available() -> Vec<HWDeviceType> {
        let mut hw_device_types = Vec::new();
        unsafe {
            let mut hwdevice_type = ffi::av_hwdevice_iterate_types(ffi::AV_HWDEVICE_TYPE_NONE);
            while hwdevice_type != ffi::AV_HWDEVICE_TYPE_NONE {
                // FFmpeg 可能报出本 crate 未建模的设备类型（新版本新增的类型，或
                // 平台特有的取值）。这里跳过它，而不是走会 panic 的 `From`：探测
                // 列表来自 FFmpeg，属于外部数据，不能中止进程。
                match HWDeviceType::from_ffi_checked(hwdevice_type) {
                    Some(device_type) => hw_device_types.push(device_type),
                    None => tracing::debug!(
                        "Skipping hardware device type not modelled by rsmedia: {hwdevice_type}"
                    ),
                }
                hwdevice_type = ffi::av_hwdevice_iterate_types(hwdevice_type);
            }
            hw_device_types
        }
    }

    /// 获取硬件设备对应的像素格式
    pub fn default_hw_pixel_format(&self) -> PixelFormat {
        match self {
            HWDeviceType::NONE => PixelFormat::NONE,
            HWDeviceType::VDPAU => PixelFormat::VDPAU,
            HWDeviceType::CUDA => PixelFormat::CUDA,
            HWDeviceType::VAAPI => PixelFormat::VAAPI,
            HWDeviceType::DXVA2 => PixelFormat::DXVA2_VLD,
            HWDeviceType::QSV => PixelFormat::QSV,
            HWDeviceType::VIDEOTOOLBOX => PixelFormat::VIDEOTOOLBOX,
            HWDeviceType::D3D11VA => PixelFormat::D3D11,
            HWDeviceType::DRM => PixelFormat::DRM_PRIME,
            HWDeviceType::OPENCL => PixelFormat::OPENCL,
            HWDeviceType::MEDIACODEC => PixelFormat::MEDIACODEC,
            HWDeviceType::VULKAN => PixelFormat::VULKAN,
            #[cfg(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9"))]
            HWDeviceType::D3D12VA => PixelFormat::D3D12,
            #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
            HWDeviceType::AMF => PixelFormat::AMF_SURFACE,
            #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
            HWDeviceType::OHCODEC => PixelFormat::OHCODEC,
        }
    }

    /// 获取硬件设备默认支持的软件像素格式
    pub fn default_sw_pixel_format(&self) -> PixelFormat {
        match self {
            // OpenCL/Vulkan 默认使用 RGBA
            HWDeviceType::OPENCL | HWDeviceType::VULKAN => PixelFormat::RGBA,
            // 其他设备默认使用 NV12
            _ => PixelFormat::NV12,
        }
    }

    /// 注意：该方法只适用于硬件解码，用于查找解码器输出到硬件表面所需的像素格式。
    /// 对于硬件编码，应检查编码器 AVCodec 的 pix_fmts 字段来确定支持的输入像素格式。
    pub fn find_hw_pixel_format_with_codec(&self, codec: &AVCodec) -> Option<ffi::AVPixelFormat> {
        let mut i = 0;
        loop {
            unsafe {
                let hw_config = ffi::avcodec_get_hw_config(codec.as_ptr(), i);
                if !hw_config.is_null() {
                    #[allow(clippy::unnecessary_cast)]
                    let hw_config_supports_codec = ((*hw_config).methods as i32
                        & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32)
                        != 0;
                    if hw_config_supports_codec
                        // `device_type` 来自 FFmpeg 的 codec 配置表，未收录的类型
                        // 直接视为不匹配（而不是走会 panic 的 `From`）。
                        && HWDeviceType::from_ffi_checked((*hw_config).device_type) == Some(*self)
                    {
                        break Some((*hw_config).pix_fmt);
                    }
                } else {
                    break None;
                }
            }
            i += 1;
        }
    }
}

/// `get_format` 回调：在解码器给出的候选像素格式列表中选择硬件表面格式。
///
/// 采用 FFmpeg 官方 `hw_decode` 示例的做法 —— 候选列表中带
/// `AV_PIX_FMT_FLAG_HWACCEL` 标志的格式即为本设备可用的硬件格式
/// （列表由解码器结合已设置的 `hw_device_ctx` / `hw_frames_ctx` 在
/// `avcodec_open2` 阶段生成）。因此**不需要**通过 `AVCodecContext.opaque`
/// 传递每实例数据：`opaque` 是 FFmpeg 留给应用层的私有字段，库占用会与
/// 用户代码互相破坏。
///
/// 该函数仅以函数指针形式安装到 codec context，无需导出符号，故不使用
/// `#[no_mangle]`，避免污染全局符号表。
unsafe extern "C" fn hwaccel_get_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        let mut p = pix_fmts;
        while *p != ffi::AV_PIX_FMT_NONE {
            if let Some(desc) = AVPixFmtDescriptorRef::get(*p)
                && (desc.flags & ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0
            {
                return *p;
            }
            p = p.add(1);
        }
        ffi::AV_PIX_FMT_NONE
    }
}

/// hwaccel 模块的单元测试。
///
/// # 组织方式
///
/// 按被测对象分成五组，每组前有分隔注释：
///
/// - **A 设备串解析与 DRM 安全** —— `default_device_string` / `drm_node_from`，
///   以及"用公开 API 探测任何设备类型都不许崩进程"这条底线。
/// - **B 能力查询契约** —— `is_usable` / `is_available` / `list_available` 之间的
///   不变量。
/// - **C 平台策略与自动选择** —— `platform_preference` / `auto_platform[_with]` /
///   `HWDeviceConfig::amf`。
/// - **D 枚举映射** —— `HWDeviceType` ↔ `ffi::AVHWDeviceType`。
/// - **E 上下文生命周期与并发** —— `HW_CTX_CACHE` 的释放语义、`HWContext` 的 `Send`/`Sync`。
///
/// # 环境策略
///
/// **能写成纯逻辑断言的就不要依赖硬件**：显式传入被测值（如 `drm_node_from` 收目录参数、
/// `default_device_string` 收类型），这样在 CI / 无 GPU 机器上也能严格断言，而不是"跳过"。
/// 确实需要真实设备的用例（C 组的自动选择、E 组）走 `try_auto_hw_context()`：探不到设备时
/// 打印原因并**静默跳过**，不把"这台机器没有 GPU"报成失败。
///
/// 跨 FFmpeg 版本运行：
/// `cargo test --lib --no-default-features --features ffmpegN,link_system_ffmpeg hwaccel::`
#[cfg(test)]
mod tests {
    use super::*;

    /// 取缓存测试锁（定义见 [`hw_cache_test_lock`]，其它模块的硬件测试也用同一把）。
    fn cache_lock() -> std::sync::MutexGuard<'static, ()> {
        hw_cache_test_lock()
    }

    /// 本 crate 建模的**全部**硬件设备类型。
    ///
    /// 与 [`HWDeviceType::list_available`] 的语义不同：后者问"**这个 FFmpeg 构建**
    /// 编入了哪些"（随构建与平台变化 —— 无 GPU 的 CI 上可能是空列表），这里问
    /// "rsmedia **认**哪些类型"，是固定的。凡是"必须覆盖每一个建模类型"的测试都用它，
    /// 新增变体时只需在这里补一处。
    fn all_modeled_types() -> Vec<HWDeviceType> {
        let mut types = vec![
            HWDeviceType::NONE,
            HWDeviceType::VDPAU,
            HWDeviceType::CUDA,
            HWDeviceType::VAAPI,
            HWDeviceType::DXVA2,
            HWDeviceType::QSV,
            HWDeviceType::VIDEOTOOLBOX,
            HWDeviceType::D3D11VA,
            HWDeviceType::DRM,
            HWDeviceType::OPENCL,
            HWDeviceType::MEDIACODEC,
            HWDeviceType::VULKAN,
        ];
        types.extend(version_gated_types());
        types
    }

    #[cfg(not(any(feature = "ffmpeg7", feature = "ffmpeg8", feature = "ffmpeg9")))]
    fn version_gated_types() -> Vec<HWDeviceType> {
        Vec::new()
    }

    /// D3D12VA 需要 ffmpeg7+，AMF / OHCODEC 需要 ffmpeg8+。
    #[cfg(feature = "ffmpeg7")]
    fn version_gated_types() -> Vec<HWDeviceType> {
        vec![HWDeviceType::D3D12VA]
    }

    #[cfg(any(feature = "ffmpeg8", feature = "ffmpeg9"))]
    fn version_gated_types() -> Vec<HWDeviceType> {
        vec![
            HWDeviceType::D3D12VA,
            HWDeviceType::AMF,
            HWDeviceType::OHCODEC,
        ]
    }

    /// 自动探测并创建硬件上下文；探不到就返回 `None` 让调用方跳过。
    ///
    /// `auto_platform()` 本身已经是**真探测**（内部走 [`HWDeviceType::is_usable`]，
    /// 会为每个候选真正建立一次设备），所以走到这里仍失败通常只剩竞态或驱动状态
    /// 变化之类的边缘原因 —— 无论是哪种，结论都是"本机此刻没有可用 GPU"，
    /// 打印原因并跳过，**不要**当成测试失败。
    fn try_auto_hw_context() -> Option<Arc<HWContext>> {
        let config = match HWDeviceConfig::auto_platform() {
            Ok(config) => config,
            Err(e) => {
                println!("skip: no hardware acceleration device probed: {e}");
                return None;
            }
        };
        println!("config: {config}");
        match HWContext::new(config) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                println!("skip: hardware device not usable on this machine: {e}");
                None
            }
        }
    }

    // ======================================================================
    // A 组 · 设备串解析与 DRM 安全
    // ======================================================================

    /// `default_device_string()` 对**除 DRM 外**的每个类型都必须交回 `NULL`
    /// （让 FFmpeg 自己选默认设备）。
    ///
    /// 这是一条**设计断言**：它把"`NULL` 对哪些后端是安全的"钉在测试里。若将来有人
    /// 给别的后端也加一个默认设备串，这里会失败，提醒同时更新
    /// [`default_device_string`] 的文档与实测依据 —— 免得把某天新出现的崩溃路径
    /// 悄悄引进来。
    ///
    /// 依据：15 个后端里只有 `hwcontext_drm.c` 的 `drm_device_create()` 会把 device
    /// 不加判断地交给 `open()`，其余都显式处理 `NULL`（详见上面函数的文档）。
    #[test]
    fn test_default_device_string_defers_to_ffmpeg_for_every_type_but_drm() {
        for t in all_modeled_types() {
            if t == HWDeviceType::DRM {
                continue;
            }
            assert_eq!(
                default_device_string(t).unwrap(),
                None,
                "{t:?} 应当把 device 交给 FFmpeg 自动选择（传 NULL）"
            );
        }
    }

    /// ✅ DRM 是 Linux 内核子系统（Direct Rendering Manager）
    /// ❌ Windows：不存在 DRM 内核子系统，完全不支持
    /// ❌ macOS：无 DRM，不支持
    /// 注意：WSL2（Linux 子系统）可以使用，前提是 WSL 启用 GPU passthrough，内核带 DRM、宿主机驱动支持。
    ///
    /// DRM 是唯一例外：必须给出一个真实节点，绝不能是 `NULL`。
    ///
    /// `open(NULL, O_RDWR)` 在 Rosetta 转译的 x86_64 上会 SIGSEGV（原生 aarch64 只是干净地返回 `EFAULT`）
    /// 所以本机没有 `/dev/dri` 节点时正确行为是 **干净地报错**，而不是退化成 `NULL`
    #[test]
    #[cfg(target_os = "linux")]
    fn test_drm_never_falls_back_to_null_device() {
        match default_device_string(HWDeviceType::DRM) {
            Ok(Some(node)) => {
                // 有节点：必须是真实存在的路径（拿它去 open 才可能成功）。
                use std::os::unix::ffi::OsStrExt;
                let path = std::path::Path::new(std::ffi::OsStr::from_bytes(node.as_bytes()));
                assert!(
                    path.starts_with("/dev/dri"),
                    "DRM 节点应落在 /dev/dri 下，实际 {path:?}"
                );
                assert!(path.exists(), "挑到的 DRM 节点不存在: {path:?}");
            }
            Ok(None) => panic!("DRM 不允许回退到 NULL：open(NULL) 会在部分平台崩进程"),
            // 本机没有 /dev/dri（macOS / Windows / 容器）：干净报错即为正确。
            Err(_) => {}
        }
    }

    // ======================================================================
    // B 组 · 能力查询契约
    // ======================================================================

    /// `is_usable()` 只能比 `is_available()` 更严格：能用的一定是构建里编入的。
    ///
    /// 反过来不成立 —— 编入 ≠ 本机能用（无 GPU 机器上实测枚举出 6 种、真能建起来的
    /// 只有 1 种），这正是 [`HWDeviceType::is_usable`] 存在的理由。
    #[test]
    fn test_is_usable_implies_is_available() {
        for t in HWDeviceType::list_available() {
            if t.is_usable() {
                assert!(t.is_available(), "{t:?} 报告可用却不在 list_available() 里");
            }
        }
    }

    /// `list_available()` / `is_available()` 的契约。
    ///
    /// 三条断言各管一件事：
    ///
    /// ① **`NONE` 不得出现在列表里** —— 迭代以 `AV_HWDEVICE_TYPE_NONE` 为终止符，
    ///    它不可能被合法报出。这条是**给未来准备的守卫**：一旦 FFmpeg 报出本 crate
    ///    未建模的类型、而实现把它降级成 `NONE`（`macros.rs` 明确拒绝过的做法），
    ///    这里会失败。（当前 FFmpeg 枚举出的类型都在建模集合内，所以它今天是空跑通过。）
    /// ② 报出的类型必须都在本 crate 的建模集合内（即实现里 `from_ffi_checked` 的
    ///    跳过契约：FFmpeg 报出未建模的类型时丢掉，而不是 panic 或降级）。
    /// ③ `is_available()` 与 `list_available()` 结果一致（同一定义的两种形式，钉住
    ///    的是契约：若有人把 `is_available` 换成另一种探测方式，这里会失败）。
    #[test]
    fn test_list_available_contract() {
        let listed = HWDeviceType::list_available();

        assert!(
            !listed.contains(&HWDeviceType::NONE),
            "list_available() 不应包含 NONE（它是迭代终止符，也不是可用的设备类型）"
        );

        let modeled = all_modeled_types();
        for t in &listed {
            assert!(
                modeled.contains(t),
                "{t:?} 由 FFmpeg 报出，但不在本 crate 的建模集合里"
            );
        }

        for t in &modeled {
            assert_eq!(
                t.is_available(),
                listed.contains(t),
                "{t:?} 的 is_available() 与 list_available() 结果不一致"
            );
        }
    }

    // ======================================================================
    // C 组 · 平台策略与自动选择
    // ======================================================================

    /// 平台优先级：每个已知平台都应定义非空列表，且首选设备符合平台惯例。
    ///
    /// 未知平台不校验（`_` 分支）—— `platform_preference()` 对它们返回空列表，
    /// 由 `test_auto_platform` 覆盖"空优先级 ⇒ 描述性错误"这条路径。
    #[test]
    fn test_platform_preference() {
        let preference = HWDeviceType::platform_preference();
        match std::env::consts::OS {
            "macos" => {
                assert_eq!(preference[0], HWDeviceType::VIDEOTOOLBOX);
            }
            "windows" => {
                assert_eq!(preference[0], HWDeviceType::D3D11VA);
                assert!(preference.contains(&HWDeviceType::QSV));
            }
            "linux" => {
                assert_eq!(preference[0], HWDeviceType::VAAPI);
                assert!(preference.contains(&HWDeviceType::CUDA));
            }
            "android" => {
                assert_eq!(preference, vec![HWDeviceType::MEDIACODEC]);
            }
            _ => {}
        }
    }

    /// 平台自动选择：有可用设备时返回**平台优先级之内**的配置；无设备时返回
    /// 描述性错误（CI / 无 GPU 环境），两种结局都不允许 panic。
    #[test]
    fn test_auto_platform() {
        match HWDeviceConfig::auto_platform() {
            Ok(config) => {
                assert!(
                    HWDeviceType::platform_preference().contains(&config.device_type),
                    "auto-selected device {:?} not in platform preference",
                    config.device_type
                );
            }
            Err(err) => {
                assert!(err.is_unsupported(), "unexpected error: {err:#}");
            }
        }
    }

    /// `auto_platform_config()` 一旦返回 `Ok`，那个设备就必须是本机**真能建起来**的。
    ///
    /// 这是"返回的配置可以直接用"这条承诺的检查点（用 `is_usable()` 反查），
    /// 杜绝"拿到的配置要到开编解码器时才失败"的两段式错误。
    /// 无 GPU 的机器上返回 `Err` 是正确结果，故那时不检查。
    #[test]
    fn test_auto_platform_config_returns_usable_device() {
        if let Ok(config) = HWDeviceType::auto_platform_config(None) {
            assert!(
                config.device_type.is_usable(),
                "auto_platform_config 返回了建不起来的设备: {config:?}"
            );
        }
    }

    /// 自定义候选：只允许 D3D11VA（AMD AMF 的承载设备类型）。
    ///
    /// 可用时应给出 D3D11 硬件格式 + NV12 软件格式（AMF 编码器的输入约定）；
    /// 不可用（非 Windows、无 D3D11 适配器等）时应优雅报错而不是 panic。
    #[test]
    fn test_auto_platform_amf_candidate() {
        match HWDeviceConfig::auto_platform_with(&[HWDeviceType::D3D11VA]) {
            Ok(config) => {
                assert_eq!(config.device_type, HWDeviceType::D3D11VA);
                assert_eq!(config.hw_pixel_format, PixelFormat::D3D11);
                assert_eq!(config.sw_pixel_format, PixelFormat::NV12);
            }
            Err(err) => {
                assert!(err.is_unsupported(), "unexpected error: {err:#}");
            }
        }
    }

    /// 空候选列表应报错而不是 panic（未知平台与显式空列表走同一条路径）。
    #[test]
    fn test_auto_platform_empty_candidates() {
        let result = HWDeviceConfig::auto_platform_with(&[]);
        assert!(result.is_err());
    }

    /// AMF builder 仅在 Windows 上编译：
    /// 验证字段映射（D3D11VA 设备 + D3D11 硬件格式 + NV12 软件格式）
    #[cfg(target_os = "windows")]
    #[test]
    fn test_amf_config_builder() {
        let config = HWDeviceConfig::amf(Some("0".to_string()));
        assert_eq!(config.device_type, HWDeviceType::D3D11VA);
        assert_eq!(config.hw_pixel_format, PixelFormat::D3D11);
        assert_eq!(config.sw_pixel_format, PixelFormat::NV12);
        assert_eq!(config.device_id.as_deref(), Some("0"));
    }

    // ======================================================================
    // D 组 · 枚举映射
    // ======================================================================

    /// 所有建模变体与 ffi 值双向映射后应保持自身。
    ///
    /// 变体列表取自 [`all_modeled_types()`]，因此按 FFmpeg 版本门控的
    /// `D3D12VA` / `AMF` / `OHCODEC` 也会在对应 feature 下被测到
    /// （此前手写列表漏掉了这三个，等于它们的 `From`/`Into` 从未被验证）。
    #[test]
    fn test_hw_device_type_roundtrip() {
        for v in all_modeled_types() {
            let ffi_value: ffi::AVHWDeviceType = v.into();
            assert_eq!(
                HWDeviceType::from(ffi_value),
                v,
                "roundtrip failed for {v:?}"
            );
        }
    }

    // ======================================================================
    // E 组 · 上下文生命周期与并发
    // ======================================================================

    /// 缓存释放语义：只释放**仅被缓存自身持有**（引用计数为 1）的条目。
    ///
    /// 无 GPU 机器上只验证"空缓存释放返回 0 且不 panic"；有可用设备时进一步验证
    /// 三段语义：持有期间释放应为 0 → `drop` 后释放应为 1 → 再释放应为 0。
    #[test]
    fn test_release_unused_hw_contexts() {
        let _guard = cache_lock();

        // 基线：先释放其它测试可能遗留的条目，再验证空缓存释放返回 0
        let _leftovers = release_unused_hw_contexts();
        let removed = release_unused_hw_contexts();
        assert_eq!(removed, 0, "empty cache should release nothing");

        // 若本机有可用 GPU 设备：创建后释放，缓存条目应可被释放
        let Some(ctx) = try_auto_hw_context() else {
            return;
        };
        // 持有期间释放不应移除
        assert_eq!(release_unused_hw_contexts(), 0);
        drop(ctx);
        // 引用释放后（仅缓存持有），释放应移除该条目
        assert_eq!(release_unused_hw_contexts(), 1);
        // 再次释放：缓存已空
        assert_eq!(release_unused_hw_contexts(), 0);
    }

    /// 帧池表面数**必须**是调用方要的那个值。
    ///
    /// `initial_pool_size` 是**预分配**的，直接决定显存占用（4K NV12 一张约 12MB），
    /// 因此 `with_hw_pool_size` 的值不能在路上被默认值悄悄顶掉——`0` 也必须是 `0`
    /// （表示"交给后端按需分配"），而不是被换成默认的 20。
    #[test]
    fn test_hw_frames_pool_size_reaches_context() {
        let _guard = cache_lock();

        let Some(ctx) = try_auto_hw_context() else {
            return; // 无 GPU 环境跳过
        };
        for pool_size in [0u32, 1, 7, DEFAULT_HW_POOL_SIZE] {
            let mut frames = ctx
                .create_hw_frames_ctx(64, 64, pool_size)
                .unwrap_or_else(|e| panic!("pool_size {pool_size} must be accepted: {e}"));
            assert_eq!(
                frames.data().initial_pool_size,
                pool_size as i32,
                "requested pool size {pool_size} must reach AVHWFramesContext"
            );
        }
    }

    /// `map_hw_frame` 的第一道闸门是"编码器上下文必须已经有自己的 frames context"。
    ///
    /// 没有就说明调用方把硬件帧塞给了软件编码路径——这是**配置错误**，不是环境差异，
    /// 必须 fail-fast 报 `InvalidConfig`，而不是丢帧或崩在 FFmpeg 内部。
    #[test]
    fn test_map_hw_frame_without_frames_context_is_invalid_config() {
        let _guard = cache_lock();

        let Some(ctx) = try_auto_hw_context() else {
            return; // 无 GPU 环境跳过
        };
        // 编解码器只用来提供一个"还没有 frames context"的上下文，因此只要求它在本
        // 构建里存在（mpeg4 是内置编码器，不依赖外部库）。
        let Some(codec) = AVCodec::find_encoder_by_name(c"mpeg4") else {
            return; // 该 FFmpeg 构建没有 mpeg4 编码器
        };
        let mut codec_ctx = AVCodecContext::new(&codec);

        let mut src = AVFrame::new();
        src.set_width(64);
        src.set_height(64);
        src.set_format(ctx.get_format(true));

        let err = ctx
            .map_hw_frame(&mut codec_ctx, src)
            .expect_err("a codec without hw frames context must reject hardware frames");
        assert!(
            err.is_invalid_config(),
            "missing frames context must be InvalidConfig, got: {err}"
        );
    }

    /// 帧已经在**编码器自己那份** frames context 里时必须原样返回。
    ///
    /// 这是零拷贝路径的兜底：`av_hwframe_map` 对同源同缓冲反而报 `EINVAL`（文档），
    /// 后端不支持 map 时更会白白跑一遍 download + upload。真跑一遍并比对帧指针，
    /// 保证这条捷径没有被误删。
    #[test]
    fn test_map_hw_frame_is_noop_for_frame_already_in_codec_context() {
        let _guard = cache_lock();

        let Some(ctx) = try_auto_hw_context() else {
            return; // 无 GPU 环境跳过
        };
        let Some(codec) = AVCodec::find_encoder_by_name(c"mpeg4") else {
            return;
        };
        let mut codec_ctx = AVCodecContext::new(&codec);
        codec_ctx.set_hw_frames_ctx(
            ctx.create_hw_frames_ctx(64, 64, 2)
                .expect("frames context must be allocated"),
        );

        let mut src = AVFrame::new();
        src.set_width(64);
        src.set_height(64);
        src.set_format(ctx.get_format(true));
        codec_ctx
            .hw_frames_ctx_mut()
            .expect("frames context was just set")
            .get_buffer(&mut src)
            .expect("frame must be allocated from the codec's own frames context");
        let src_ptr = src.as_ptr();

        let mapped = ctx
            .map_hw_frame(&mut codec_ctx, src)
            .expect("a frame already in the codec frames context must be accepted");
        assert!(
            std::ptr::addr_eq(mapped.as_ptr(), src_ptr),
            "a frame already in the codec frames context must be returned as-is"
        );
    }

    /// `HWContext` 跨线程共享同一 `Arc` 不应触发数据竞争。
    ///
    /// 回归来源：`setup_decoder_frames` 并发调用曾通过 `UnsafeCell` 做可变访问；
    /// `HWContext` 的所有方法现在都只做共享访问（`&self`），配合 FFmpeg 保证的
    /// 原子引用计数，`Send`/`Sync` 才成立。
    #[test]
    fn test_hw_context_shared_across_threads() {
        let _guard = cache_lock();

        let Some(ctx) = try_auto_hw_context() else {
            return; // 无 GPU 环境跳过
        };
        let ctx2 = Arc::clone(&ctx);
        let handle = std::thread::spawn(move || {
            // 共享引用上的只读方法跨线程调用
            let _ = ctx2.get_format(true);
        });
        let _ = ctx.get_format(false);
        handle.join().expect("thread should not panic");
    }
}
