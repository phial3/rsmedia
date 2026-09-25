# rsmedia 媒体格式测试覆盖审计

- 审计对象：`/Users/admin/Workspace/rust/github.com/rsmedia`（HEAD `aff24d1`，v0.10.1）
- 审计日期：2026-09-23
- 审计范围：用户给出的 25 个扩展名（图片 7 / 视频 11 / 音频 6，去重后 24 个 + `mpeg`）
- 方法：① 静态清点仓内测试与外部 harness 的格式引用；② 在本机（macOS arm64 + Homebrew FFmpeg 9.0.1）跑一次性探针脚本，实测"能写/能读/能否回读校验"，区分**测试缺口**与**功能缺口**。探针脚本用后即删，仓内未留痕。

---

## 一、结论速览

| 类别 | 判定 | 一句话 |
|---|---|---|
| **音频** | ✅ 充分 | 6/6 全覆盖，且几乎全部是"写→读→断言采样率/声道/样本数/峰值"的闭环。是三类里最扎实的。 |
| **视频** | ✅ 基本充分，有 3 处小缺口 | 主流容器全覆盖且带回读校验；`wmv`、`m4v` 只有"写入"没有回读校验，`m4p` 零覆盖。 |
| **图片** | ⚠️ **最大短板** | 只有 `png`（序列，8 帧）和 `cat.jpg`（读）有测试；`bmp`/`tiff`/`webp` **零覆盖**；无"单张图片经 `Muxer` 写盘"的测试。 |

实测补充（重要）：缺口**主要是测试缺口，不是功能坏了**。本机实测 `png / jpeg / bmp / tiff / gif` 的"单图写盘 + 回读"全部通过，`png/jpeg/bmp/tiff/gif` 的读取也全部通过 —— 也就是说这些路径现在**能用但没人验**，属于回归风险敞口。唯一的真功能边界是 `webp` 写入依赖 `libwebp` 编码器，本机 Homebrew FFmpeg 9.0.1 没编它（报 `InvalidConfig`，消息正确）。

---

## 二、测试设施地图

rsmedia 的格式覆盖由三层构成，评估时必须一起算，只看 `tests/` 会低估：

| 层 | 规模 | 跑在哪 | 说明 |
|---|---|---|---|
| 仓内单元测试 | `src/**` 约 **285** 个 `#[test]` | 本机 + CI 全平台 | 像素/采样格式、io、seek、mux 等就地断言 |
| 仓内集成测试 | `tests/**` 约 **97** 个 `#[test]`（17 个文件、约 7.8k 行） | 同上 | 格式覆盖主力：`container_roundtrip` / `encode_pipeline` / `codec_matrix` / `encode_video` / `transcode` 等 |
| 示例自带测试 | `examples/**` 约 **17** 个 `#[test]` | 同上（`cargo test --all-targets`） | `examples/mod.rs → misc` 会被编译为示例目标并**执行其中的 `#[test]`**，`thumbnail_test0/1`、`image_dump_test` 属于这一层（产物落在 `output/`） |
| 外部 harness（`github.com/rsmedia_test`） | **38 模式 / 每个 FFmpeg 版本 237 case** | Linux amd64+arm64、Windows amd64+arm64、macOS arm64 × FFmpeg 6.1/7.1/8.1/9.0 | CI 矩阵，`--features …,image` 时图片相关代码才编入 |

核心的"格式矩阵"测试有三个：

- `tests/container_roundtrip.rs` —— **23 个容器规格**（`mp4 mov mkv webm ts flv avi mpg 3gp gif y4m h264 h265 m4a aac mp3 ogg opus ac3 wma flac wav caf`），每个都做 `编码 → 落盘 → demux 结构断言 → 解码断言`（视频帧数/尺寸/非空白像素、音频采样率/声道/样本数±容差/峰值、字幕文本与时间戳逐字比对）。**这是最有含金量的一层。**
- `tests/encode_pipeline.rs` —— 视频 **14** 个容器 + 音频 **15** 个容器，滤镜链 + 时间基 + 编码参数，逐容器解码回读。
- `tests/codec_matrix.rs` —— 视频 5 编解码器、音频 4 编解码器，编码后回读校验帧数/尺寸/采样数，缺编码器则 `SKIP`（先探测可用性，不拿错误变体当跳过标记）。

harness 侧与格式相关的模式：`videocodecs`（10 编解码器）、`audiocodecs`（7）、`containers`（webm/mov/avi/flv/gif/wav/adts）、`video10b`、`videohw`、`thumbnail`，以及 `remux`/`remuxext`/`lossless`/`scenarios`。**但没有独立的"图片"模式**（harness 全仓只出现 `.png`×1、`.gif`×2，`.png` 还只是缩略图的输出名）。

---

## 三、逐格式覆盖矩阵

图例：✅ 有测试且带回读断言 · ⚠️ 有测试但缺回读/断言过弱 · ❌ 零覆盖

### 图片（7）

| 格式 | 写入 | 读取 | 回读断言 | 覆盖位置 | 判定 |
|---|:---:|:---:|:---:|---|---|
| `jpg` / `jpeg` | ⚠️ | ✅ | 写：无 | 写：`examples/misc/thumbnail.rs`（`mp4.mp4→jpg`、`cat.jpg→jpg`，**直接写 packet 字节到 `File`，绕过 `Muxer`**）；读：`src/io.rs::test_read_single_image`（`assets/cat.jpg`） | ⚠️ 写路径绕过 rsmedia 封装；读断言过弱（只查 `frames>0` 与 `dims>0`，不断言 `2000x1333`） |
| `png` | ✅ | ✅ | ✅ | `src/io.rs::test_write_image_sequence`（image2 + png，8 帧，逐文件存在+非空）、`test_read_image_sequence`（序列回读 8 帧 + 尺寸）；`src/mux.rs` 封面图（`mjpeg`/`png`）；harness `videocodecs` 里 png 当 mp4 内视频编码器 | ✅ 但**只覆盖序列，不覆盖单图** |
| `gif` | ✅ | ✅ | ✅ | `container_roundtrip`（`spec("gif", Some("gif"))`）、harness `containers/gif`（动画） | ✅ |
| `bmp` | ❌ | ❌ | ❌ | 无 | ❌ **零覆盖** |
| `tiff` | ❌ | ❌ | ❌ | 无 | ❌ **零覆盖** |
| `webp` | ❌ | ❌ | ❌ | 无 | ❌ **零覆盖**；写还依赖 `libwebp`（平台差异） |

> 另注：`src/imgutils.rs::test_image_text` 名为图片测试，实际只用 `image` crate 的 `rgb.save("*.png")` 存盘，**不经过 rsmedia 任何图片路径**，属于"名字像测试、实则无断言网络"的一例。

### 视频（11）

| 格式 | 写入 | 读取 | 回读断言 | 覆盖位置 | 判定 |
|---|:---:|:---:|:---:|---|---|
| `mp4` | ✅ | ✅ | ✅ | `container_roundtrip`、`encode_pipeline`、`codec_matrix`、harness `containers`/`videocodecs`（默认容器） | ✅ 最深 |
| `avi` | ✅ | ✅ | ✅ | `container_roundtrip`（libx264+aac）、`encode_pipeline`、harness `containers/avi_mpeg4` + `videocodecs/mjpeg→avi` | ✅ |
| `mkv` | ✅ | ✅ | ✅ | `container_roundtrip`（含 ass 字幕）、`encode_pipeline`、`codec_matrix`（hevc/ffv1）、harness | ✅ |
| `mov` | ✅ | ✅ | ✅ | `container_roundtrip`（mov_text 字幕）、`encode_pipeline`、harness `containers/mov_h264` | ✅ |
| `flv` | ✅ | ✅ | ✅ | `container_roundtrip`、`encode_pipeline`、`encode_video`、harness `containers/flv_h264` | ✅ |
| `webm` | ✅ | ✅ | ✅ | `container_roundtrip`（vp9+opus）、`encode_pipeline`（vp9 / opus 纯音频）、harness `containers/webm_vp9` + `videocodecs/libvpx-vp9` | ✅ |
| `mpeg` / `mpg` | ✅ | ✅ | ✅ | `encode_pipeline` `vc("mpg")`、`container_roundtrip` `spec("mpg", mpeg2video+mp2)`；`ts` 覆盖 MPEG-TS | ✅（`.mpeg` 扩展名本身未被用过，但同一 muxer，风险低） |
| `m4v` | ⚠️ | ❌ | ❌ | 仅 `encode_video.rs` `("m4v","libx264")` **只编码不读回**；`encode_pipeline` 明确把它精简掉；`container_roundtrip`/harness 均无 | ⚠️ 写而不验 |
| `wmv` | ⚠️ | ❌ | ❌ | 仅 `encode_video.rs` `("wmv","wmv2")` **只编码不读回**（注释说明为缩短测试耗时被精简）；`container_roundtrip` 只有 `wma`（纯音频） | ⚠️ 写而不验，Windows Media 容器端到端为零 |
| `m4p` | ❌ | ❌ | ❌ | 无 | ❌ 零覆盖（本质是音频型 MPEG-4，与 `m4a` 同族） |

### 音频（6）

| 格式 | 写入 | 读取 | 回读断言 | 覆盖位置 | 判定 |
|---|:---:|:---:|:---:|---|---|
| `mp3` | ✅ | ✅ | ✅ | `container_roundtrip`、`encode_pipeline`、`encode_audio`、`codec_matrix`、harness `audiocodecs/libmp3lame` | ✅ |
| `wav` | ✅ | ✅ | ✅ | `container_roundtrip`（pcm_s16le）、`encode_pipeline`、`encode_audio`、harness；资产 `assets/wav.wav` 还被 `decode_audio`/`transcode_aac` 当输入 | ✅ |
| `flac` | ✅ | ✅ | ✅ | `container_roundtrip`、`encode_pipeline`、`encode_audio`、`codec_matrix`、harness | ✅（无损，另有 global-header 变体用例） |
| `aac` | ✅ | ✅ | ✅ | `container_roundtrip`（`m4a`/裸 `aac`）、`encode_pipeline`（m4a/aac/adts）、`encode_audio`、`codec_matrix`、harness | ✅ |
| `ogg` | ✅ | ✅ | ✅ | `container_roundtrip`（ogg/opus）、`encode_pipeline`、harness `audiocodecs`（opus/vorbis） | ✅ |
| `wma` | ✅ | ✅ | ✅ | `container_roundtrip`（`wmav2`，含 priming 剪帧容差注释）、`encode_pipeline`（`wma`）、`encode_audio::test_encode_audio_wmav2` | ✅（harness 无 `wmav2`，仓内已够） |

---

## 四、本机实测（Homebrew FFmpeg 9.0.1 / macOS arm64）

一次性探针（`EncoderBuilder` + `Muxer::new(path)` 按扩展名推 muxer，320×240 单帧，写后立即 `StreamReader`+`DecoderBuilder` 回读）：

| 格式 | 写盘 | 回读 | 备注 |
|---|:---:|:---:|---|
| `.png` | ✅ 840 B | ✅ 1 帧 320×240 | |
| `.jpg`（`mjpeg`） | ✅ 3186 B | ✅ 1 帧 320×240 | 日志有 `deprecated pixel format` + image2 警告（见下） |
| `.bmp` | ✅ 307254 B | ✅ 1 帧 320×240 | |
| `.tiff` | ✅ 232774 B | ✅ 1 帧 320×240 | |
| `.gif` | ✅ 34715 B | ✅ 1 帧 320×240 | |
| `.webp` | ❌ | 未测 | `libwebp` 编码器不在本机构建内，报 `invalid configuration: encoder 'libwebp' is not available in this FFmpeg build`（分类与措辞都正确） |

读取（`StreamReader`+`DecoderBuilder`，文件由 ffmpeg CLI 生成）：`png / jpeg / bmp / tiff / gif` 全部 ✅ 1 帧 320×240；`assets/cat.jpg` ✅ 2000×1333。

两个实测副产物（值得记录，非本次审计主线）：

1. **`Muxer` 写单张图片可用，但 FFmpeg 会告警**：`image2` muxer 对不含 `%03d` 模式的文件名会打印 *"does not contain an image sequence pattern … use the -update option"*。单帧仍正常落盘（见上表）。
2. **同一文件名写多于一帧会失败**：写 2 帧到 `/tmp/…/two.png` 时 muxer 报
   `AVERROR(-22): 'Invalid argument'`，FFmpeg 原文 *"Cannot write more than one file with the same name. Are you missing the -update option or a sequence pattern?"*
   —— 单帧场景没问题，但若想让"写图片"更像用户预期（传 N 帧只留最后一张 / 报更友好的错），需要给 image2 传 `update=1` 或在 rsmedia 侧拦一下。

---

## 五、缺口清单与建议（按优先级）

**P0 —— 图片三格式零覆盖 + 单图 mux 无测试**

1. 新增 `tests/image_formats.rs`：对 `png / jpeg / bmp / tiff / gif / webp` 做"`Muxer::new(out.<ext>)` 写 1 帧 → demux+decode 读回 1 帧"的矩阵，断言尺寸、像素非空白（`max-min > 阈值`）、回读帧数 = 1；`webp`（以及任何本构建缺编码器的情况）按**预检编码器可用性**跳过 —— 沿用 `container_roundtrip::encoder_available()` 的既有写法，不要拿 `InvalidConfig` 变体当跳过标记（记忆与代码注释都明确禁止）。这一条同时补掉 `bmp/tiff/webp` 与"单图 mux"两个空白。
2. 强化 `src/io.rs::test_read_single_image`：现在是 `assert!(!decoded.is_empty())` + `dims > 0`，等于没断言。`assets/cat.jpg` 已知真值 **2000×1333**，应精确断言尺寸 + 像素非空白。
3. 图片解码真值资产缺失：`assets/` 只有 `cat.jpg` 一张。建议补 `png/bmp/tiff/webp` 各一张小图（或让测试运行时用 image2 先生成再自读），否则"读"的方向永远只测到 jpeg。

**P1 —— 视频两个"写而不验"**

4. `wmv`：把 `spec("wmv", Some("wmv2"), Some("wmav2"), None, 44_100)` 加进 `container_roundtrip::CONTAINERS`（asf muxer 支持 wmv2+wmav2），让它获得与其它容器同等的回读校验；或至少在 `encode_video.rs` 里补读回。Windows Media 现在是端到端零覆盖。
5. `m4v`：同理并入 `container_roundtrip`（`spec("m4v", Some("libx264"), Some("aac"), None, 44_100)`）。
6. `m4p`：与 `m4a` 同族，建议直接用 `m4a` 的既有覆盖说明"同 muxer"，或在矩阵里加一行低成本条目，避免清单上留着❌。

**P2 —— 外围**

7. harness 加一个 `images` 模式：38 个模式里没有任何图片矩阵，而图片路径恰恰是当前最弱的一环；做成模式后即可在 3 平台 × 4 FFmpeg 版本上自动跑。
8. `examples/misc/thumbnail.rs` 的 jpg 写盘绕过了 `Muxer`（手工写 `packet.data` 到 `File`），不计入 rsmedia 图片输出路径的覆盖；建议把 `examples/id_photo.rs` 那条"jpg 进 → 图片出"管线（自带 `verify()`）抽成集成测试，让 jpeg 单图往返进入 CI。
9. `src/imgutils.rs::test_image_text` 建议改名或补断言 —— 它现在只验证 `image` crate 能存盘。

---

## 六、顺带发现：工作区有一处未提交改动会挂 CI

审计过程中发现 `git status` 出现一处**我没有做过的**改动：`src/lib.rs`（mtime `2026-09-23 14:22`，不在任何已提交版本里，`git log -S CRATE_NAME` 查无记录）：

```diff
@@ -22,6 +22,7 @@ pub mod pixel;
 pub mod resample;
 pub mod resize;
 pub mod scale;
+pub mod state;
 pub mod stream;
@@ -52,11 +53,8 @@ pub use stream::MediaType;
 pub use subtitle::SubtitleSegment;
 pub use time::Time;
 
-pub(crate) mod state;
 pub(crate) const MAX_DRAIN_ITERATIONS: usize = 1_000;
-
-#[cfg(feature = "image")]
-pub use imgutils::thumbnail;
+pub(crate) const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
 
 /// re-exported under the name `ffmpeg`
 pub use rsmpeg as ffmpeg;
```

两点影响，都已本机复核：

1. `RUSTFLAGS="-D warnings" cargo check --lib` **编译失败**：`error: constant CRATE_NAME is never used`（CI 三个 job 都带 `-D warnings`，会红）。全仓 `CRATE_NAME` 无任何使用点。
2. `pub use imgutils::thumbnail`（`image` feature 门）被删，属公开 API 回退；`thumbnail` 仍可用 `imgutils::thumbnail` 全路径访问。

**用户答复（2026-09-23）：此改动是本人有意为之，保留。** 本节仅作记录，整改未触碰该文件；`CRATE_NAME` 的死代码警告仍在，CI 的 `-D warnings` 会因此在修完之前保持红色。

---

## 七、整改记录（2026-09-23，按用户三项要求执行）

### 7.1 需求 3：错误分类 —— "本构建没有的能力"一律 `Unsupported`

原 `error.rs` 的规则是"codec/filter **名字**在本构建里不存在 ⇒ `InvalidConfig`"（理由：换个名字就行）。用户判定这不对：**调用方改自己的调用改不动它**，只能换构建 —— 正是 `Unsupported` 的定义（"skip or degrade gracefully"）。全仓翻转并统一构造点：

新增 `RsmediaError::unsupported(...)` 作为**唯一**的"本构建缺能力"构造点（原 `pub(crate) not_in_build(kind, name)` 已按用户要求合并进来并删除，避免同一件事有两个入口），统一措辞 `{kind} '{name}' is not available in this FFmpeg build`，改造 12 处：

| 位置 | 原来 | 现在 |
|---|---|---|
| `encode.rs` 编码器名查找 | `InvalidConfig` | `unsupported("encoder '…' is not available in this FFmpeg build")` |
| `codec.rs::new`（按 id） / `new_with_name`（按名） | `InvalidConfig` | `unsupported("codec '…' …")` |
| `decode.rs` 解码器名查找 | `Option::context` ⇒ 无类型 `Other` | `unsupported("decoder '…' …")` |
| `mux.rs` 某 codec_id 无解码器 | `InvalidConfig` | `unsupported("decoder for codec_id … …")` |
| `filter.rs` 滤镜名 ×3（输入/输出 pad 数、图校验） | `InvalidConfig` | `unsupported("filter '…' …")` |
| `filter.rs` 内部端点滤镜 `buffer`/`buffersink`/`abuffer`/`abuffersink` | `Option::context` ⇒ `Other` | `unsupported("filter '…' …")` |
| `bsf.rs` bitstream filter 名 | `InvalidConfig` | `unsupported("bitstream filter '…' …")` |
| `pixel.rs::get_pix_fmt_loss` 负值（硬件/未建模格式无可评分描述） | `InvalidConfig` | `Unsupported` |
| `subtitle.rs`（测试内）mov_text / ass 解码器缺失 | `InvalidConfig` | `unsupported("decoder '…' …")` |

同时把"调用顺序"类错误的措辞里的 "is not supported" 去掉（`add_cover_art` header 之后 = 顺序问题，不是本构建不支持），避免与 `Unsupported` 撞概念。

**刻意保留为 `InvalidConfig` 的两处**（并在代码里写明理由，防止下次审计再翻）：

- `decode.rs::ensure_pix_fmt_storable` —— 格式是调用方用 `with_pix_fmt` **显式要求**的产出格式，换个格式就是正确调用（与 `with_sample_fmt` 不可用输出格式同一族）。
- `scale.rs::pooled_frame_buffer_size` —— 尺寸/格式都来自调用方在本轮 `scale_frame` 里点的目标参数，属于入参口径错误。

规则边界因此是：**能力缺口（构建/平台/crate 未建模）⇒ `Unsupported`；调用本身有问题（取值、顺序、来源冲突、显式点了做不到的东西）⇒ `InvalidConfig`**。

连带更新：3 个断言旧分类的单元测试（`encode`/`bsf`/`filter`）改名并改断言；7 处"分类合并后无法区分所以必须预检"的注释改写为"预检更早更准，且 `Unsupported` 还含其它能力缺口，拿它当跳过标记会吞掉真问题"（预检本身仍保留，未改成靠变体跳过）。

### 7.2 需求 1：图片格式覆盖（bmp / tiff / webp 从零覆盖）

新增 `tests/image_formats.rs`：

- `test_single_image_roundtrip` —— `png / jpg / jpeg / bmp / tiff / gif / webp` 七行，`Muxer::new("out.<ext>")` 写 1 帧 → demux + decode 回读，断言**恰好 1 帧**、精确几何、`codecpar().codec_id`、非空白像素；缺编码器按**预检**跳过（先 `find_encoder_by_name`）。
- `test_webp_without_libwebp_reports_unsupported` —— 把用户报的那个 case 钉成回归测试：没有 `libwebp` 时必须 `is_unsupported()`、且 `!is_invalid_config()`、消息点名 `libwebp`。
- `reference_decoding::test_decode_reference_images`（`#[cfg(feature = "image")]`）—— 用 `image` crate 生成 png/jpg/bmp/tiff/gif/**webp** 真值图，再让 rsmedia 解码。**解码侧因此不依赖本构建有没有该编码器**（`webp` 解码器是内置的，与 `libwebp` 编码器无关）。

`src/io.rs::test_read_single_image` 从"`frames>0` 且尺寸非零"改为精确断言 `2000×1333` + 恰好 1 帧 + 非空白。

### 7.3 需求 2：容器全部"写完必验"

`tests/container_roundtrip.rs` 由 23 行扩到 **27 行**，新增的 4 行都走完整的结构/视频/音频回读断言：

| 容器 | 编码器 | 结果 | 说明 |
|---|---|---|---|
| `m4v` | libx264 + aac | ✅ | 实测 FFmpeg 把 `.m4v` 猜成 MP4 家族（不是只收 mpeg4 的 `m4v` muxer），所以音频也在 |
| `wmv` | wmv2 + wmav2 | ✅ | 补上 Windows Media 的端到端（原来只有纯音频的 `.wma`） |
| `mpeg` | mpeg2video + mp2 | ✅ | 与 `.mpg` 同 muxer，但走的是扩展名猜测这条路径 |
| `m4p` | aac（`ipod`） | ✅ | ⚠️ FFmpeg **没有任何 muxer/demuxer 认领 `.m4p`**，读写两侧都必须显式钉格式，为此给矩阵加了 `ContainerSpec::format` + `spec_as(...)` |

本机全绿：`containers: 27 passed [...], 0 skipped, 0 failed`。

### 7.4 验证

- `cargo test --lib`：**279 passed / 0 failed**（含 error 分类新测试、`pixel` 分类测试）。
- `cargo test --tests --features image`：全绿（container_roundtrip 27 行、image_formats 3 个用例、codec_matrix、encode_video/audio、encoder_options、buffer_pool …）。
- `cargo fmt --check` 干净；`cargo clippy --all-targets [--features image]` 只剩 `CRATE_NAME` 那一条（用户自己的改动，见第六节）。

### 7.5 仍未做（建议，非本次范围）

1. `examples`/`tests/encode_video.rs` 的 `ogv`（libtheora）仍是"只写不读"：它不在用户给的 11 个视频扩展名里，但同一毛病。一行即可收口 —— 往 `container_roundtrip::CONTAINERS` 加 `spec("ogv", Some("libtheora"), Some("libvorbis"), None, 44_100)`（缺编码器自动 SKIP）。
2. `Muxer::new("x.m4p")`（不给格式名）现在的报错是裸 `FFmpeg error: AVERROR(-22): 'Invalid argument'`。建议在 `io.rs` 建输出上下文前用 `av_guess_format` 探一次，给一句 `InvalidConfig`："该扩展名没有对应 muxer，请用 `with_format` 显式指定"。
3. 无 `libwebp` 的构建上，图片矩阵的 webp 行是 SKIP（本机即如此）；写入侧覆盖依赖 CI 里带 `libwebp` 的 FFmpeg。解码侧已由 `image` crate 真值图补齐。
4. 外部 harness 仍无图片模式（38 模式里一个都没有），跨 3 平台 × 4 版本的图片覆盖只有仓内这一层。

