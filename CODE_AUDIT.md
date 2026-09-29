# rsmedia 代码审读：架构与现存缺陷

审读时间：2026-09-27 ｜ 基线：`dbad3f3`（`dev`，与 `origin/dev` 同步，工作区干净）
审查范围：`src/` 29 个文件 / 32555 行（含 `#[cfg(test)]`）＋ 新增提交 `1f6c004`→`dbad3f3`
方法：模块通读＋三条正交检查＋ `rsmedia-codebase-audit` 脚本量化。**每条缺陷都标了复核状态**。

---

## 一、架构速览

### 1.1 分层与职责

| 层 | 模块 | 职责 |
|---|---|---|
| I/O 与容器 | `io.rs` `mux.rs` `location.rs` | `Reader`/`Writer` trait（`Stream*`/`Buffer*`/`Io*` 三种后端）、`Demuxer`/`Muxer`、章节/封面/metadata、`Options` 注入 |
| 编解码 | `decode.rs` `encode.rs` `codec.rs` `stream.rs` `bsf.rs` `subtitle.rs` | `Decoder`/`Encoder` 构建器与排空状态机、能力查询、码流过滤器、字幕 |
| 媒体表示 | `frame.rs` `fmt.rs` `pixel.rs` `scale.rs` `resample.rs` `resize.rs` `imgutils.rs` `pcm.rs` `colors.rs` | `MediaFrame`/`FrameData`（自己持有 ndarray）、像素/采样格式布局推导、swscale/swresample、PCM 分块 |
| 滤镜 | `filter.rs` | 线性链 `FilterGraph::init` + 多入多出 `FilterGraphBuilder`，pad 校验、标签接线、`send_command` |
| 硬件 | `hwaccel.rs` | `ProbeDepth{CompiledIn,DeviceOpen,SurfaceAlloc}`、平台自动选择、DRM 节点解析、hw↔sw 传输、进程级设备缓存 |
| 基础设施 | `error.rs` `state.rs` `init.rs` `time.rs` `strutils.rs` `options.rs` `macros.rs` | 错误分类、`ProcessState`（`Normal→Drained→Flushed`）、日志初始化、时间基 |

### 1.2 数据流

```
bytes → Reader(io.rs) → Demuxer(mux.rs) → Decoder(decode.rs)
                                              ↓  AVFrame → MediaFrame
                       Scaler / FilterGraph / Resampler（表示层，frame.rs 统一承载）
                                              ↓  编码前归一
       bytes ← StreamWriter(io.rs) ← Muxer(mux.rs) ← Encoder(encode.rs)
```

- **跨界一律拷贝**：`MediaFrame` 自己持有 ndarray，`from_avframe`/`to_avframe` 每次都 `alloc_buffer` + 逐平面复制；没有 ndarray-over-AVFrame 的零拷贝视图（唯一例外：`FilterData::with_normalized_layout` 用 `av_frame_clone` 引用计数共享，`Resizer` 纯整数计算）。
- **所有 FFmpeg 句柄都是 RAII**：`ReaderCore`/`WriterCore` 持 `AVFormatContext{,Input,Output}`；`Decoder`/`Encoder` 持 `AVCodecContext` + `FilterGraph` + 复用型 `Scaler`/`SwrContext`；`MuxerStream` 持 `Bsf`。只能 `Send` 不能 `Sync`。
- **排空语义**：`Decoder::flush_buffers`（**丢弃**，seek 后用） vs `Encoder::flush`（**排空到 writer**）；`ProcessState` 只在 EOS 真正送出后前进，`EAGAIN` 不回退；`MAX_DRAIN_ITERATIONS=1000` 兜底。
- **错误分类**（2026-09-23 定稿，`error.rs`）：`Unsupported`＝本构建/本平台/本 crate 做不到（构造点唯一：`RsmediaError::unsupported`），`InvalidConfig`＝调用本身有问题，`Other`＝数据形状/不变量，`FFmpeg(AVError)`＝FFmpeg 调用失败；`is_*` 谓词穿透 `Context` 链。

### 1.3 正交检查结果（全部通过，无退化）

| 检查 | 结果 |
|---|---|
| 特性门 vs `data/ffmpeg-*/binding.rs` | ✅ `every feature gate matches the per-version bindings` |
| `cargo check --all-targets --no-default-features --features ffmpeg9,link_system_ffmpeg` | ✅ 无警告 |
| `cargo clippy --all-targets ... -- -D warnings` | ✅ 干净 |
| `cargo fmt --check` | ✅ 干净 |

---

## 二、缺陷清单

> **修复记录（本轮，2026-09-27）**：已按 §三 的建议顺序修完 **A5 / A4 / A2 / A1、B4 / B1、B2、C2 / C1**，每条都带回归测试，并逐条验证过"回退实现则测试失败"。明细见下；未动的项在表内以 `〔未修〕` 标注。
>
> | 编号 | 结论 | 改动位置 | 回归测试 |
> |---|---|---|---|
> | A5 | 已修 | `encode.rs` `Encoder::flush` 守卫改 `is_flushed()`，EOS/FIFO 段移进 `if self.state.is_normal()` | `test_flush_retries_drain_after_a_failed_attempt` |
> | A2 | 已修 | `frame.rs` `yuv_matrix` 阈值 `< 1080` → `< 2160`（1080p 归 HD） | `test_yuv_matrix_heuristic_boundaries` |
> | A1 | 已修 | `frame.rs` `yuv420p_to_rgb24` 色度尺寸改 `div_ceil(2)`（与 `data_layout` 一致）；**实测症状是报错而非静默错位** | `test_yuv420p_to_rgb24_odd_dimensions` |
> | A4 | 已修 | `imgutils.rs` `plane_geom` 色度位移限 `1 \| 2`，平面 3（alpha）保持全分辨率 | `test_get_plane_buffer_alpha_plane_is_full_resolution` |
> | A3 | 〔未修〕 | `MediaFrame` 增 `AVChannelLayout` 字段属 API 变更，单独一轮 | — |
> | A6 | 〔未修·2026-09-28 新增〕 | 音频重采样尾部延迟未排空：`Resampler::flush` 只有 `pcm.rs:351` 与单测调用，`encode.rs`/`decode.rs` 从未调用 ⇒ 流末尾若干采样被静默丢弃。**行为取舍，待裁决**（详见 §二 A6） | `test_streaming_resampler_carries_delay_and_flushes`（证明尾巴存在） |
> | B1 | 已修 | `mux.rs` `unsafe impl<R: Reader + Send> Send for Demuxer<R>` | `test_demuxer_is_send_for_every_reader_impl` |
> | B2 | 已修 | `decode.rs` 新增 `send_packet_with_retry` + `drain_decoder_frames` + `pending_frames` 队列；`decode_raw_packet`/`drain_raw` 改走重试入口 | `test_send_packet_with_retry_recovers_from_a_full_decoder`、`test_decode_raw_packet_recovers_from_a_full_decoder` |
> | B3 | 〔未修〕 | rsmpeg 侧 `hwframe_ctx_alloc().unwrap()`，需上游改 | — |
> | B4 | 已修 | `filter.rs` `init` 前置拒绝空 `filters`（`InvalidConfig`）；顺手修正 `build` 的过时文档（缺滤镜是 `Unsupported`） | `test_empty_filter_list_is_invalid_config` |
> | B5 | 已修（2026-09-28） | `stream.rs::find_{de,en}coder_name` 的 c-name 转换失败处理为"未注册"并 `warn!`；`options.rs::build` 由转换自身决定跳过，去掉 check/unwrap 漂移 | `options::tests::test_interior_nul_entries_are_skipped_not_panicking` |
> | B6 | 已修（2026-09-28） | `Time::from_nth_of_a_second`/`from_units` 改 `Result<Self>`，参数改用 FFmpeg 自己的宽度（`i32` 分母 / `i64` 刻度）⇒ 原 `one_over` 的"折叠成无值时基"与 `as i32` 回绕都失去立足点；没有新增任何范围判断，剩下的只有 `Rational::new` 本就有的 `den != 0` | `time::tests::test_from_nth_of_a_second_rejects_a_zero_denominator`、`time::tests::test_from_units_rejects_a_zero_denominator` |
> | B7 | 已修（2026-09-28） | `scale.rs` 对齐改查 `av_cpu_max_align()`（留白 ≥ align）；`hwaccel.rs` 的 `pool_size` 一路改 `i32`（= `AVHWFramesContext::initial_pool_size` 的宽度），直接赋值、不做转换判断；`frame.rs::plane_stride` 对所有平面判整除 | `scale::tests::test_pooled_buffer_holds_the_alignment_offset`、`scale::tests::test_scaler_pool_frame_alignment`（断言改随 `pool_align`）、`frame::tests::test_plane_stride_validates_one_row_planes` |
> | C1 | 已修（2026-09-28 随类型统一而简化） | `width`/`height` 已统一为 `i32`（对齐 `AVFrame`/`AVCodecContext` 的 `int`），`encode.rs` `build()` 的守卫从"`u32` ∈ `1..=i32::MAX`"简化为 `<= 0`：负值与 `0` 都在 `build()` 报 `InvalidConfig`，而"`u32` 超出 `i32` 回绕成负数"这**第二种**非法值已不可表达 | `test_video_size_out_of_range_is_invalid_config` |
> | C2 | 已修 | 新增 `codec.rs` `apply_thread_count`（超 `i32` 打 `warn!`）；`rc_max_rate`/`rc_buffer_size` 负值 `warn!`、显式 0 只 `debug!` | `test_builder_thread_count_beyond_i32_is_ignored`、`test_non_positive_rate_control_is_not_applied` |
> | C3 | 〔未修〕 | 同上，`b68f5ed` 的有意设计（typed setter 与 `with_options` 同名键冲突时字典静默胜出；文档已写明但无重建期诊断） | — |
> | C4/C5 | 已修（更早批次） | **更正**：`thread_count` 两侧现已都是 `i32`；位掩码 setter 由批次 5 的 `FlagSet<E>` 强类型化，`src/` 下再无 `impl Into<u32>` | — |
> | D | 〔未修〕 | 纯增量：`# Errors` 优先 | — |
>
> 验证：`cargo fmt --check` ✅、`clippy -D warnings`（含 / 不含 `image`）✅、
> `check --all-targets --no-default-features` ✅、特性门脚本 ✅、
> **（2026-09-28 复测）lib 340 + 集成 19 个 binary + 68 doctest 全绿**；
> VM 6.1 / 7.1 / 8.1 / 9.0 = 339 / 340 / 341 / 341；harness 38 模式 293 pass / 0 fail / 2 xfail。

> **修复记录（分支 / 参数 / 命名 专项，2026-09-29）**
>
> | 编号 | 结论 | 改动位置 | 回归测试 |
> |---|---|---|---|
> | A7 | 已修（**真 bug**） | `io.rs` `InterruptData` 的 deadline 互斥量被 poison 后，`triggered()`/`interrupt_callback` 用 `.unwrap_or(false)` 把它当成"没到期" ⇒ 已过期的 abort/timeout 请求被静默丢弃，阻塞读永远等下去。抽出唯一的 `is_set()` 谓词，`poison` 时 `unwrap_or_else(\|e\| e.into_inner())` 照读里面的值 | `io::tests::test_a_poisoned_deadline_mutex_still_reports_an_expired_timeout`、`test_abort_is_honoured_even_when_the_deadline_mutex_is_poisoned` |
> | C6 | 已修 | `filter.rs` `output_frame_rate{,_at}` / `output_time_base{,_at}` / `output_size{,_at}` 原来 `self.get_sink_context(output).ok()?` 把 `InvalidConfig`（输出索引越界）吞成 `None`。实测三个 rsmpeg getter 都是**无失败**的（`get_w`/`get_h -> i32`、`get_frame_rate`/`get_time_base -> AVRational`），故 `Option` 从不表示"值不可用"，只是在撒谎；六个方法改 `Result<T>` | `test_output_index_out_of_range_is_an_error_not_a_missing_value`、`test_output_queries_on_a_graph_without_outputs_are_an_error` |
> | C7 | 已修 | `filter.rs::video::transpose` 的守卫**方向反了**：原代码对 `!(0..=7)` 只 `warn!`，而 `4..=7` —— 实测 `transpose=4..7` 的输出与"不加滤镜"逐字节相同（静默直通）—— 反而不报警。改为 `Result<Filter>`，`0..=3` 之外一律 `InvalidConfig` | `test_transpose_rejects_the_modes_ffmpeg_applies_as_a_no_op` |
> | D11 | 已修 | `mux.rs::mux_packet` 三次 `get_stream*` 查找 + `.expect("checked above")` ⇒ 改一次 `get_stream_mut` + `Option::transpose()` | 既有 mux 测试覆盖 |
> | D12 | 已修 | `mux.rs` 两份逐字重复的"header 之后不准加流"守卫 ⇒ 收敛为一个 `ensure_streams_open()`（`have_written_header` 与 `Writer::is_header_written()` 由 `ensure_header_written` 同步置位，本就不存在不一致窗口） | 既有 `io.rs` `test_writer_rejects_header_and_add_stream_after_header_written` 等 |
> | D13 | 已修 | 合并等价 match arm：`encode.rs::drain_encoder_packets`、`bsf.rs::drain` | — |
> | D14 | 已修 | 命名：`let raw = X.deref_mut()` ⇒ 按所指对象命名（`frame_raw`/`ctx_raw`/`dst_raw`）；名为 `_ptr` 实为引用/切片的 `ctx_mut_ptr`/`dst_ptr`/`src_ptr`/`data_ptr` 改名；`CodecConfig::is_support_*` ⇒ `supports_*`（`pub(crate)`）；`DrawText::raw_text` ⇒ `text_is_expression` | — |
>
> 验证（2026-09-29）：`cargo fmt --check` ✅、`clippy --all-targets -D warnings`（含 / 不含 `image`）✅、
> `RUSTDOCFLAGS="-D warnings" cargo doc` ✅、**lib 345 / doctest 68 / 集成 19 个 binary 全绿**；
> VM 6.1 / 7.1 / 8.1 = 344 / 345 / 346（各 = 基线 + 本轮 5 个新测试）；
> harness 38 模式 **293 pass / 0 fail / 2 xfail**（与基线一致，`vf_transpose` 实测 320×180 → 180×320）。

> **修复记录（A6 尾部延迟 + 破坏性签名，2026-09-29）**
>
> **A6** 见上一节的落地记录：只有 `Encoder::encode_resampler` 需要排空（另两个上下文不做
> 重采样、没有延迟线，已在文档与测试里写明）。
>
> | 位置 | 原签名 | 改后 | 理由 |
> |---|---|---|---|
> | `filter.rs::video::pad` | `pad(w, h, x, y, color)` | `pad(x, y, w, h, color)` | 与 `crop` / `delogo` / `Delogo::add_region` 统一为"先位置、后尺寸" |
> | `filter.rs::VideoEndpoint::new` | `new(w, h, fmt, time_base, frame_rate)` | `new(w, h, fmt)` + `with_time_base` / `with_frame_rate` / `with_pixel_aspect` | 三个相邻同类型 `Rational` 写反了编译得过；改成按名字设置。"忘了设"不会静默：`buffer` 源直接拒 `time_base=0/1`（实测 `Invalid time base 0/1`） |
> | `hwaccel.rs::HWDeviceConfig::new` | `new(device_type, hw_pixel_format, sw_pixel_format, device_id, options)` | `new(device_type)` + `with_hw_pixel_format` / `with_sw_pixel_format` / `with_device_id` / `with_options` | 两个相邻 `PixelFormat` 成对出现、写反了要等设备初始化才报错；默认取设备类型的默认格式映射（唯一真相源） |
> | `subtitle.rs::copy_subtitle_stream` | `(reader, writer, src_index, out_index)` | `(reader, src_index, writer, out_index)` | 两个 `usize` 流索引挨着可互换；中间隔着类型不同的 `writer` 就换不动 |
> | `codec.rs`（`CodecConfig::new` / `id` / `encoders_for` / `decoders_for`） | `ffi::AVCodecID` | `u32` | 该 FFI 别名就是 `c_uint`；换掉后公开 API 不再漏 `rsmpeg::ffi`，且与 `StreamInfo::codec_id: u32` 同一写法（**类型未变，非破坏性**） |
> | `encode.rs::Encoder::flush` | `flush(writer, interleaved: bool, index, tb)` | `flush(writer, mode: WriteMode, index, tb)` | 新增 `io::WriteMode { Interleaved, Direct }`：调用处原来是 `flush(&mut w, true, 0, tb)` |
> | `imgutils::apply_cropping` | `apply_cropping(frame, flags: i32)` | ~~`impl Into<FlagSet<CropFlag>>`~~ **已回退，仍是 `flags: i32`** | 见下一节：用户裁定"一个位不值得为它定义类型"，中间形态 `CropAlignment` 也被删了 |
>
> 连带更新：`examples/{encoding,filter_compose,filter_demo,id_photo}.rs`、`benches/encode_mux_container.rs`、
> `tests/encode_pipeline.rs`、`src/mux.rs`（由 `Muxer::interleaved` 推出 `WriteMode`），
> 以及 harness `../rsmedia_test`（5 处调用点）。
>
> **有意未动**：`filter.rs::VideoParams` 也有同样的三个相邻 `Rational`（`time_base` /
> `frame_rate` / `pixel_aspect`），但它不是构造入口 —— `From<VideoParams> for VideoEndpoint`
> 与 `FilterParams` 内部才建它，且 `VideoEndpoint` 已经能从它派生。若要一并改成按名字设置，
> 需要连带改 `VideoParams` 的构造（当前全靠 struct 字面量），属更大范围的 API 变更，留待后续。
>
> 验证（2026-09-29）：`cargo fmt --check` ✅、`clippy --all-targets -D warnings`
> （含 / 不含 `image`）✅、`RUSTDOCFLAGS="-D warnings" cargo doc` ✅、
> **lib 350 / doctest 70 / 集成 19 个 binary 全绿**（各 = 基线 + 本轮 4 个新测试）；
> VM 6.1 / 7.1 / 8.1 = 349 / 350 / 351；
> harness 38 模式 **293 pass / 0 fail / 2 xfail**（与基线一致）。

> **修复记录（复核后的三处回退 + `alloc_buffer` 陷阱，2026-09-29）**
>
> 上一节落地后用户逐条复核，四处按裁决改回 / 简化：
>
> | 项 | 上一节的做法 | 本轮 | 理由 |
> |---|---|---|---|
> | `codec.rs::CodecConfig` | `ffi::AVCodecID` → `u32` | **整体还原** | 该别名本来就是 `c_uint`，换签名只是"看起来不漏 FFI"，收益为零，却多一次全仓调用点改动 |
> | `encode.rs::Encoder::flush` | 新增 `io::WriteMode{Interleaved,Direct}` 取代 `bool` | **删掉 `WriteMode`**，回到 `interleaved: bool` | 一个二元开关套一层枚举是过度设计；`flush(&mut w, true, 0, tb)` 比 `flush(&mut w, WriteMode::Interleaved, 0, tb)` 更短且无需 import |
> | `imgutils::apply_cropping` | `impl Into<FlagSet<CropFlag>>`（`ffi_enum!` 位集） | 中间形态 `impl Into<CropAlignment>` → **最终两个都删，回到 `flags: i32`** | 核对 `data/ffmpeg-{5_1,6_1,7_1,8_1,9_0}/binding.rs`：**`AV_FRAME_CROP_UNALIGNED` 是唯一的 `AV_FRAME_CROP_*` 常量**（5.1/6.1 上是 `_bindgen_ty_2`、7.1+ 是 `_bindgen_ty_1`，值均为 1）。**一个位既不成"位集"、也不值得为它单独定义类型** —— 用户裁定：`apply_cropping(&mut f, ffi::AV_FRAME_CROP_UNALIGNED as i32)` 已经够清楚，多一个枚举只是多一层要记的名字。调用方直接传 FFI 常量，`as i32` 处加 `#[allow(clippy::unnecessary_cast)]`（Windows/vcpkg 绑定里它已是 `i32`，Linux 上是 `u32`） |
>
> **真 BUG（本轮调试中踩到，已修）**：`AVFrame::alloc_buffer()` 给的是**未初始化**内存 —— 它走
> `av_frame_get_buffer` → `av_buffer_alloc` → `av_malloc`，**没有 memset**。把这样的帧直接喂编码器，
> 里面的随机浮点会触发 `avcodec_send_frame` 返回 `EINVAL(-22)`，而且是**间歇性**的（单独跑过、
> 整套跑挂），极易被误判成并发 bug。
>
> | 位置 | 问题 | 修法 |
> |---|---|---|
> | `src/decode.rs::write_test_clip`（测试夹具） | `noise=false` 时"留空"= 直接送未初始化像素进编码器，同一份输入产出不同码流，依赖"输入相同 ⇒ 输出相同"的损坏实验间歇性失败 | 分配后先 `imgutils::fill_black(&mut frame)` 整帧填黑，再决定是否叠噪声。用 `fill_black` 而非 `fill_color`：后者底层 `av_image_fill_color` 自 FFmpeg 7.0 才有（`#[cfg(any(ffmpeg7,ffmpeg8,ffmpeg9))]`），VM 上的 6.1 会编不过 |
> | `src/scale.rs:580-583`、`src/scale.rs:1270`、`src/macros.rs:840` | 三处文档声称池化缓冲"清零，与 `alloc_buffer` 一致" —— **是错的** | 改为：池只清零 swscale 不会写的字节（对齐偏移 / stride 余量 / 平面间隙 / 尾部留白），可见像素由 swscale 整体覆写；并明确写出 `alloc_buffer` 不清零、`av_malloc` 链路 |
>
> 顺带修的非本轮问题：`../rsmedia_test/examples/{seek_probe,seekflag_probe,stream_probe}.rs`
> 仍在按 `u32` 传 `MediaFrame::new_video_frame` / `EncoderBuilder::new_video`（这两个参数本轮前已改
> `i32`），`cargo clippy --all-targets` 因此编不过。共 7 处：`W`/`H` 常量与 `solid_frame` 形参各
> 2 + 1，以及 `stream_probe` 里 `box_of` / `label_bar_of` 的 `w/h: u32`（调用方传的是
> `MediaFrame::width`，本来就是 `i32`，`run_all.sh` 只编 bin 所以一直没暴露）。
> 全部改 `i32`，索引处仍就地转 `usize`。
>
> ⚠️ **自查更正**：这一小节第一次报"harness `clippy --all-targets` 编得过"是**错的** —— 那次命令
> 以 `| tail` 结尾，`$?` 取的是 `tail` 的退出码，把 cargo 的非零冲掉了；真实情况是还剩 4 个
> `E0308`。已改为**重定向到文件再取 `$?`**，重跑后 `CLIPPY_EXIT=0`。教训：**要断言"通过"
> 就不能把编译器的输出接进管道**。
>
> 验证（2026-09-29，退出码均为直接取到的真值）：`cargo fmt --check` ✅（0）、
> `clippy --all-targets -D warnings`（含 / 不含 `image`）✅（0 / 0）、
> `RUSTDOCFLAGS="-D warnings" cargo doc` ✅（0）、**lib 350 / doctest 68 / 集成 19 个 binary 全绿**
> （doctest 从上一节的 70 回到 68：那 +2 是 `WriteMode` 的文档示例，跟着枚举一起删了）；
> VM 6.1 / 7.1 / 8.1 = 349 / 350 / 351（与上一节一致，`VM_EXIT` 均为 0）；
> harness 38 模式 **293 pass / 0 fail / 2 xfail**（与基线一致，`EXIT=0`），
> harness `clippy --all-targets` **真正编得过**（`CLIPPY_EXIT=0`，仅剩 21 条既有风格 warning）。
>
> 上面四行在**再删掉 `CropAlignment` 之后**原样复跑过一遍，数字完全一致（lib 350 / doctest 68 /
> VM 349-350-351 / harness 293-0-2），退出码仍全为 0。

### A. 会**静默产出错误数据**（最高优先级，建议先修）　**（A5 / A1 / A2 / A4 已修；A3 / A6 未修）**

**A1〔已修·中〕奇数尺寸 `YUV420P` → `RGB24` 静默错位**
`src/frame.rs:456`
```rust
let (uv_width, uv_height) = (width / 2, height / 2);   // 向下取整
...
u_stride: uv_width as u32,                              // 真实色度行宽是 ceil(width/2)
```
`PixelFormat::data_layout` 对奇数尺寸按 **ceil** 推导（`pixel.rs:450-457`，65×49 → 色度 33×25），而这里用 floor。`rgb24_to_yuv420p` 却在 `frame.rs:403-408` 明确拒绝奇数尺寸 —— 两个方向不对称。修法：奇数尺寸要么同样拒绝（`InvalidConfig`），要么按 ceil 传 stride。

> **⚠️ 复核更正（2026-09-27，实测）**：原文写"静默错位且不报错"是**错的**。写单测实测后，floor 的 stride 会在进入转换前就被 `yuv` crate 拒绝：`check_chroma_channel` 以 `width.div_ceil(2) = 33` 为最小行宽，`stride 32 × chroma_height 25 = 800 < 33 × 25 = 825` ⇒ `ChromaPlaneMinimumSizeMismatch { expected: 825, received: 800 }`，被 `.context()` 包成一句 `Other`（`"Failed to convert YUV420P to RGB24"`）。所以真实症状是**这条快路径对整个奇数尺寸不可用 + 错误信息不可定位**，不是静默错数据。修法取 **ceil**（与 `data_layout`、`imgutils` 一致），保留能力。
> 写侧（`rgb24_to_yuv420p` 拒绝奇数）经查 `yuv` crate 的 `YuvPlanarImageMut::alloc` 其实用 `div_ceil`，本可支持奇数；本轮**不改**，因为拒绝属于对调用方配置的正确分类，且已有测试锁定，避免扩大范围。

**A2〔已修·高〕`1080p` 被判成 BT.2020**
`src/frame.rs:1331-1338`
```rust
let height = self.height;
if height < 720 { Bt601 } else if height < 1080 { Bt709 } else { Bt2020 }
```
`height == 1080` 落进 `else` ⇒ **BT.2020**，而 1080p 属于 HD（BT.709），UHD 从 2160 起。文档还写着"与 ffmpeg 的 `sws_getCoefficients` 缺省行为一致"。仅当帧没有 `colorspace` 标记时触发（`yuv` crate 快路径 `convert_rgb24_to_yuv420p` / `convert_yuv420p_to_rgb24`），所以最常见的 1080p 素材 + 无标记正是命中场景。修法：阈值改成 `< 2160`（或分成 720/2160 两档）。

**A3〔未修·中高〕声道布局被"规范化"成默认布局**
`src/frame.rs:618`（只存 `nb_channels`）→ `src/frame.rs:1256-1258`
```rust
frame.set_ch_layout(AVChannelLayout::from_nb_channels(self.nb_channels as i32).into_inner());
```
`from_nb_channels` 即 `av_channel_layout_default`：3 通道的 2.1（FL/FR/LFE）来回一趟变成默认的 FL/FR/FC，**声道被静默重排**；`resample.rs:59-62` 同样用它。`MediaFrame` 上根本没有承载 `AVChannelMask` 的字段，所以这不是"某处写错"，而是表示层缺一个字段。影响仅限非默认多声道布局，但完全没有提示。

**A4〔已修·中〕`imgutils` 把 YUVA 的 alpha 平面当色度平面**
`src/imgutils.rs:255-259`
```rust
let (shift_w, shift_h) = if plane_idx > 0 { (desc.log2_chroma_w, desc.log2_chroma_h) } else { (0, 0) };
```
FFmpeg 只对平面 1/2 做色度下采样，平面 3（alpha）是全分辨率 —— 本项目自己的 `pixel.rs:450-457` 就是这么写的（`match plane { 1 | 2 => shift, _ => 0 }`）。于是 `YUVA420P`/`YUVA422P`（及 9~16 bit 变体）的平面 3 被算成 `ceil(w/2)×ceil(h/2)`，`get_plane_buffer` 返回截断缓冲、`fill_plane_from_buffer` 只写部分 alpha。两处必须统一。

**A5〔已修·高〕`Encoder::flush` 失败后重试会**静默截断**并被当成成功**
`src/encode.rs:1795` + `1825` + `1873-1880`
```rust
if !self.state.is_normal() {           // ← 把 Drained 也当成"已经 flush 过"
    return Ok(W::Accum::default());     // 直接返回空，什么都不排
}
...
self.state = ProcessState::Drained;    // 先置位，再进排空循环
loop { ... Err(e) => return Err(e.with_context("... output is truncated")) }
```
排空循环里 `receive_packet()` 报错、或撞上 `MAX_DRAIN_ITERATIONS` 时返回 `Err`，但 `state` 已经停在 `Drained` 且不会回退。此时**第二次**调用 `flush()`（用户重试、或 `Muxer::flush_if_needed`/`Drop` 再次驱动，`mux.rs:1190`）命中这个守卫 → 返回 `Ok(默认累加器)` → `finish()` 照常写 trailer → **截断的容器以成功返回**。守卫的判定应该是 `is_flushed()`（真的排空完成）而不是 `!is_normal()`。这是"把可恢复的 I/O 错误升级成静默数据丢失"，值得优先修。

**A6〔未修·中高〕音频重采样器的尾部延迟从未排空 ⇒ 流末尾采样被静默丢弃**
`src/resample.rs:517`（`Resampler::flush`）+ `src/encode.rs` / `src/decode.rs`

```rust
// src/resample.rs:405 —— 类型自己的文档已经写明要求
/// ... Call [`Self::flush`] after the last frame to drain what is left.
```

但 `flush` 在整个 `src/` 下只有两个调用点：`pcm.rs:351`（PCM 写出路径）和
`resample.rs:933`（单元测试）。**`Encoder` 的 `filter_resampler` / `encode_resampler`
与 `Decoder::resampler` 都从未调用它。**

`swr` 内部保留采样率换算的余数（delay line）。非整数倍重采样（如 48kHz → 44.1kHz）时，
流结束时仍有若干采样留在上下文里（数量由重采样比决定，FFmpeg 侧可用 `swr_get_delay()`
查询 —— **本 crate 从未调用它**），不 `flush` 就不会产出 ⇒
**末尾的采样被静默截掉**：无报错、无 `warn!`，只是样本数与时长对不上。
`test_streaming_resampler_carries_delay_and_flushes` 已经证明这条尾巴真实存在、且
`flush` 能把它取出来 —— 只是主链路没接上。

> **为什么算 A 类**：症状是"输出静默少了一段"，与 A1/A5 同类；只是它属于**继承的取舍**
> 而非本轮引入的回归 —— 旧的 `StreamingConverter` 同样不排空。

修法（**待用户裁决**，两条路互斥）：
1. **真的排空** —— `Encoder::flush` 先把两个 resampler 排空、把尾巴作为末帧送进编码器；
   `Decoder` 侧在 EOS（`None`）时排空并补出最后一帧。**行为变更**：所有音频产物的样本数
   会变多，需要 harness 音频组复验。
2. **维持现状并明写** —— 在 `Resampler` 及编码/解码侧的文档里写明"主链路不排空，末尾
   的延迟采样会被丢弃"，把这个取舍从"未记录的隐式行为"变成"有文档的已知限制"。

> **已修（2026-09-29，用户裁决"只处理 2 3"后按"该排空的排空、该写明的写明"落地）**
>
> 先分清三个 `Resampler` 谁真的有延迟线 —— 这是"排空 vs 写明"的分界：
>
> | 位置 | 输出规格取自 | 会不会重采样 | 处理 |
> |---|---|---|---|
> | `Encoder::encode_resampler` | **编码器**（`audio_spec()`，采样率可能与输入不同） | 会 | **排空** |
> | `Encoder::filter_resampler` | 帧自己（`from_frame(&f).with_sample_fmt(..)`） | 不会 | 写明（无需排空） |
> | `Decoder::resampler` | 帧自己（同上，见 `convert_decoded_audio`） | 不会 | 写明（无需排空） |
>
> - 新增 `Resampler::flush_frames()`（`resample.rs`）：按一秒容量分配、循环取空、循环
>   次数以 `MAX_DRAIN_ITERATIONS` 为上限（与 `pcm.rs::drain_resampler` 同一做法）。
> - `Encoder::flush` 的"输入侧收尾"抽成 `finish_input()`，顺序为
>   **滤镜冲刷 → 排空重采样延迟线 → 冲刷 audio_fifo → 送 EOS**；尾帧走新的
>   `send_frame_ready()`（而非 `send_frame_post_filter`）—— 它们已出过重采样、pts 也在
>   编码器时间基上，再走一遍滤镜输出时间基的换算会换算两次。
> - 回归测试：`encode::tests::test_finish_input_drains_the_resampler_delay_line`
>   （48kHz → 44.1kHz，8×1024 样本；**实测排空前 7510、期望 7526，少 16 个样本**，
>   排空后相符）、`resample::tests::test_flush_frames_returns_the_delay_line`、
>   `resample::tests::test_a_format_only_resampler_keeps_no_delay_line`（钉住"只换格式的
>   上下文没有延迟线"，即另两个不需要排空的依据）。
>
> ⚠️ 顺带记录一个**测试侧**的坑：`AVFrame::alloc_buffer()` 给的是**未初始化**内存；把
> 里面的随机浮点喂给 aac 会以 `SendFrameError(-22)` **间歇性**失败。填真实样本才稳定。

### B. 健壮性 / 健全性　**（B1 / B2 / B4 / B5 / B6 / B7 已修；B3 未修）**

| # | 位置 | 问题 | 复核 |
|---|---|---|---|
| B1 | `mux.rs:1732` | `unsafe impl<R: Reader> Send for Demuxer<R> {}` **缺 `R: Send` 约束**（`Muxer` 那边写的是 `W: Writer + Send`）。自定义一个含 `Rc`/线程亲和状态的 `Reader` 就能构造出"假 Send"，跨线程移动是 UB。类型级问题，现有 Reader 实现都 Send | 已复核 |
| B2 | `decode.rs:1029-1035` | `decode_raw_packet` = 送 1 包 + 收 **1** 帧；`encode` 侧有 `send_frame_with_retry` + 全量排空，解码侧没有对应物。H.264 场编码 / MPEG-2 field picture 这类"一个包出多帧"的流，下一次 `send_packet` 会返回 `EAGAIN`，被 rsmpeg 映射成 `DecoderFullError` 抛出，而不是先把已就绪的帧交出来 | 已复核（`EAGAIN` 映射为推理，未实测） |
| B3 | `hwaccel.rs:865` | `hw_device_ctx.hwframe_ctx_alloc()` 内部是 rsmpeg 的 `av_hwframe_ctx_alloc(...).upgrade().unwrap()`：FFmpeg 返回 NULL（OOM/未知类型）时 **panic**，而 `is_available()` 是 `-> bool` 的探测 API，不该 unwind | 已复核 |
| B4 | `filter.rs:1952-1988` | `with_filters(Some(vec![]))` 不被拒绝：`filter_spec = ""` → `setup_endpoints` → `graph.config()` 失败，报的是一句**不透明 FFmpeg 错误**，而不是像 `FilterGraphBuilder::build`（`filter.rs:2717`）那样给 `InvalidConfig`。`decode.rs:547`/`encode.rs:708` 的注释还承诺"校验在 `init` 里做" | 已复核 |
| B5 | `stream.rs:452` `stream.rs:476` `options.rs:122-123` | 对 `str_to_cstring(...)` 的 `.unwrap()`。当前输入分别是静态表和"NUL 已预检过"的键值，**都不可达**；但这是唯一一类"未来改动即 panic"的写法，建议顺手换成 `?`/`InvalidConfig` | **已修**（2026-09-28）：`stream.rs` 两处改为带 `warn!` 的"不可表示 ⇒ 未注册"（回退软件编解码器，`find_*_name` 仍返回 `Option<String>` 不改 API）；`options.rs::build` 去掉"先 `contains('\0')` 预检、再 `unwrap()`"的漂移形状，改成转换自身决定，且能分别指出是 key 还是 value |
| B6 | `time.rs:56` `time.rs:92-93` | `from_nth_of_a_second(0)` → 时间基 `1/0`（退化）；`from_nth_of_a_second(usize::MAX)`/`from_units(_, usize::MAX)` 在 `as i32` 处回绕成负数。且 `has_value()` 对 `1/0` 返回 true，`seconds_or_none()` 返回 `None`，两个"可用"定义不一致 | **已修**（2026-09-28）：签名改为 `from_nth_of_a_second(nth: i32)` / `from_units(time: i64, base_den: i32)` —— **用 FFmpeg 自己的宽度承接**，越界值因此根本不可表达，无需任何范围判断；原 `one_over` 的"折叠成无值时基"删除（那正是"悄悄换掉调用方的值"）。剩下的唯一拒绝是 `Rational::new` 本就有的 `den != 0`。`has_value`/`seconds_or_none` 的不一致在更早一轮已统一 |
| B7 | `scale.rs:728` `hwaccel.rs:486` `frame.rs:1591` | 池化缓冲固定 32 字节对齐（`av_frame_get_buffer` 默认是 `av_cpu_max_align()`，AVX-512 上是 64）；`pool_size as i32` 对 `u32::MAX` 回绕成 -1；`plane_stride` 对单行平面跳过全部校验。都是"只在畸形输入下出问题" | **已修**（2026-09-28）：对齐改查 `av_cpu_max_align()`（`OnceLock` 缓存，实测 x86 8/16/32/64 随 CPU 变、aarch64 16/8），留白改为 `max(64, align)` 以兜住偏移；`hw_pool_size` 用 `i32::try_from` 并把校验拆成自由函数（无 GPU 也能测）；`plane_stride` 对**所有**平面判整除，只对单行平面免"行放得下"检查（音频平面靠这条豁免） |

### C. 静默接受矛盾/越界配置（多数是 `b68f5ed` 的有意设计，但缺诊断）　**（C1 / C2 / C4 / C5 已修；C3 未修）**

| # | 位置 | 问题 |
|---|---|---|
| C1 | `encode.rs:720-728`（校验点） | **〔已修〕** 原问题（`width/height` 改成 `u32` 后没有范围校验，`new_video(1 << 31, 720)` 回绕成负数，最终只报 `avcodec_open2` 的不透明错误）在 2026-09-28 随类型统一**从根上消失**：两者现在是 `i32`，守卫是 `<= 0`（`encode.rs:724`），"`u32` 超出 `i32`"这第二类非法值已不可表达 |
| C2 | `codec.rs:18-21` + `encode.rs:506` + `decode.rs:326` | `thread_count` 超出 `i32` 会**下溢成负数被忽略**，静默退回 FFmpeg 默认；`with_buffer_size(0)`/`with_max_bit_rate(0)` 从"build 报错"变成"视为未设置"（doc 已同步，但没有 `warn!`）。三处都只在代码注释里说明 |
| C3 | `codec.rs:130-147` | 删掉 `owned_option_keys` 守卫后，typed setter 与 `with_options` 同名键（`threads`/`flags`/`b`/`crf`/`g`/`bf`…）冲突时**由 dict 静默胜出**，build 期无任何提示。**〔未修〕** |
| C4 | `encode.rs:1707` vs `decode.rs:703` | 同一个 `thread_count` 在 `Encoder` 上返回 `u32`、在 `Decoder` 上返回 `i32`。**〔已修〕** 两侧现在都是 `i32`（`encode.rs:1867`、`decode.rs:737`） |
| C5 | `codec.rs:104-128` + `init.rs:80-95` | `impl Into<u32>` 的 flag setter 接受任意 u32（`u32::MAX`）直接写进 `flags/flags2/thread_type`，不再受枚举约束。**〔已修〕** 批次 5 引入 `FlagSet<E>`（`src/flags.rs`），setter 收 `impl Into<FlagSet<_>>`；`src/` 下已无 `impl Into<u32>`。注：这条原不顺 §五 P4 的"`impl Into<AVCodecFlag>`"修法——那样的签名会让 `A \| B`（无名组合）和 `0` 都编译不过 |

### D. 工程质量债（量化）

> **§D 的数字是 2026-09-28 的实测值**（`rsmedia-codebase-audit/scripts/audit_quality.py`），与 2026-09-27 那版有明显漂移，
> 主要来自其间的"文档/实现一致性"一轮（`148a2c6`）与 B5–B7 修复。

| 维度 | 现状（2026-09-28） | 说明 |
|---|---|---|
| 公开文档语言 | 英文 **2166** 行 / 中文 **2057** 行 | 项目规则要求公开项英文；集中在 `filter.rs`(775)、`mux.rs`(261)、`decode.rs`(170)、`hwaccel.rs`(124)、`io.rs`(114) |
| `# Errors` 段 | **145** 个返回 `Result` 的公开函数缺 | 用户靠它知道该 match `is_invalid_config` 还是 `is_unsupported`。**优先级最高、纯增量** |
| 公开项无文档 | **90** 个 | |
| `unsafe` 缺 `SAFETY:` | **94 / 144** | 优先补 `unsafe impl Send` 这类健全性声明（`io.rs` 17、`scale.rs` 15、`mux.rs` 12、`imgutils.rs` 11） |
| 缺 `# Panics` 段 | **23** 个可能 panic 的公开函数缺 | |
| 泛化 `Other` | **41** 处 | 26 内部不变量（应保留）、11 疑似该是 `InvalidConfig`、3 数据形状（保留）、1 该用 `.context()`（`io.rs:600`） |
| 长函数 | `encode.rs:705 build` **294 行**、`filter.rs:3107 build` 279、`decode.rs:379 build_from_reader` 217、`stream.rs:207 from_stream` 141、`decode.rs:1248 receive_normalized_frame` 138 | `filter.rs` 单文件 **5260** 行 |
| 重复守卫 | "header 之后不准加流"在 `io.rs:1125`、`mux.rs:364`、`mux.rs:443` **三份**（两份逐字相同、消息不一致）；`get_stream`/`get_stream_mut` 在 `Muxer`/`Demuxer` 各一份 | 该不变量坏掉会 SIGSEGV，漂移有实际风险 |

---

## 三、建议修复顺序

**4 未完成；A3 / B3 / C3 未动（B5–B7 已于 2026-09-28 修完）。**

1. ~~**A5**（`flush` 守卫，1 行改动 + 一个回归测试）→ **A2**（阈值改 `< 2160`）→ **A1**（奇数尺寸按 ceil）→ **A4**（对齐 `pixel.rs` 的规则）~~ ✅ 已完成
   → **A3**（`MediaFrame` 增 `AVChannelLayout` 字段，是 API 变更，单独一轮）**未修**。
2. ~~**B4**（空 filters 前置拒绝）、**B1**（补 `R: Send`）、**B2**（解码侧补全量排空）~~ ✅ 已完成。
3. ~~**C2/C1**：给静默路径补 `tracing::warn!` 或范围校验~~ ✅ 已完成。
4. **D**：`# Errors` 优先（信息量最大、纯增量），再翻译公开文档，`SAFETY:` 按文件清。**未动用**。
5. ~~**B5–B7**（仅畸形输入）~~ ✅ 已于 2026-09-28 修完（见 §二 表内各行）。
6. 仍未动：**A3**（声道布局字段）、**B3**（rsmpeg 侧 `unwrap`，需上游）、**C3**（`b68f5ed` 的有意设计）、§D 全部。

## 四、已确认健康（勿重复报警）

- 特性门与每版 binding 完全一致；`--no-default-features` + `clippy -D warnings` + `fmt` 全绿。
- `MAX_DRAIN_ITERATIONS` 与各处裸 `loop` 的终止性可证；`ProcessState` 的 EAGAIN 语义正确。
- FFI 镜像枚举表（含零调用点）、`#[allow(non_camel_case_types)]`、`#[allow(unused_macros)]` 都是有意保留。
- `Unsupported`/`InvalidConfig` 边界（2026-09-23 定稿）在本次新增代码里没有回退。

### 本轮修复后的复检（2026-09-27）

- 回归测试都做了**反向验证**：回退实现后测试确实失败（A1 报 `ChromaPlaneMinimumSizeMismatch`、A4 报 alpha 缓冲 768 ≠ 3072、A5 报 `Ok(0)`、B2 报 `Decoder isn't accepting input`）。
- 新增 10 个测试：`frame.rs` ×2、`imgutils.rs` ×1、`filter.rs` ×1、`mux.rs` ×1、`encode.rs` ×3、`decode.rs` ×2。
- 规模：lib **302**（`--no-default-features`）/ **306**（+`image`）、集成 21 个 binary、doctest 44，全绿且 `clippy -D warnings` 干净。
- 唯一的量级回退：§D 的泛化 `Other` **39 → 40**，新增那条已判定为**应保留**（防挂死不变量）。

---

## 五、公开 API 参数类型一致性（2026-09-27）

方法：提取 `src/*.rs` 全部 `pub fn` 的 **589** 个参数，按**参数名**分组，找同名参数出现多种类型的点
（脚本 `/tmp/audit_api_types.py`）。

**根因**：crate 里并存两种取向 —— 编解码层**照抄 FFmpeg 字段宽度**（`i32`/`i64`），
表示层**用域类型**（`u32`，非负）。同一概念跨层时两边都得 cast，**设置端与读取端还各选一边**。

| # | 概念 | 现状 | 问题 |
|---|---|---|---|
| P1 | 尺寸 `width`/`height` | **〔2026-09-28 定案：全链路 `i32`，本节原描述已作废〕** 设：`EncoderBuilder::new_video(i32,i32)`/`with_width(i32)`/`with_height(i32)`、`MediaFrame::new_video(i32,i32)`/`new_video_frame(i32,i32)`；读：`Encoder/Decoder::width()/height() -> i32`；`MediaFrame.width/height`、`VideoParams`/`VideoEndpoint`、`StreamInfo`、`scale::scale_frame`、`imgutils::fill_linesizes` 全是 `i32` —— 逐字段镜像 `AVFrame`/`AVCodecContext` 的 `int`，**设与读同宽，往返不再需要 cast** | 剩下的**三处 `u32` 孤岛是有意保留**的：`filter::video::{scale,crop,pad,drawbox,delogo,tile,add_region}` 的 `w`/`h`（产的是滤镜**选项串**而非结构体字段，FFmpeg 侧还接受表达式）、`Resize::Exact(u32,u32)` + `Dims=(u32,u32)`、`image` crate 边界。<br>⚠️ **遗留**：三处孤岛目前**没有文档说明为什么和主体不同宽**（＝ P10 的同款问题） |
| P2 | 音频 `nb_channels`/`sample_rate` | **〔2026-09-28 定案：全链路 `i32`，本节原描述已作废〕** 实测现状：`EncoderBuilder::new_audio(.., nb_channels: i32, sample_rate: i32)`/`with_nb_channels(i32)`/`with_sample_rate(i32)`、`MediaFrame.nb_channels/sample_rate: i32`、`filter::audio::{resample,format}(nb_channels: i32, sample_rate: i32, ..)`、`AudioParams`/`AudioEndpoint`、`PcmSpec::new(sample_rate: i32, channels: i32)`、`StreamInfo.sample_rate: i32`、读侧 `Encoder/Decoder::sample_rate() -> i32` | 原"声道数有 i32/u32/**u16** 三种写法、采样率两种"的漂移**已消除**；`PcmSpec` 那个全 crate 唯一的 `u16` 通道数也已改为 `i32` |
| P3 | `thread_count` | **〔已修·两侧都是 `i32`〕** 原状：设 `with_thread_count(u32)`（`codec.rs` 宏）；读 **`Encoder::thread_count() -> u32`**（`encode.rs:1738`）但 **`Decoder::thread_count() -> i32`**（`decode.rs:715`）；FFmpeg 字段是 `int` | 一个概念三种宽度；且 u32 让"超出 `i32`"变成**可表示但无意义**的值，逼出 `i32::try_from` + 丢弃分支 |
| P4 | 位掩码 setter | `with_flags`/`with_flags2`/`with_thread_type`/`with_err_recognition`/`with_scale_quality` 全是 `impl Into<u32>` | 任意 `u32` 都能塞，**枚举约束被绕开**（＝ §C5）。旁边 `with_quality(Quality)` 却是强类型，风格不一致。想保留 `A\|B` 组合能力的话，正解是给枚举实现 `BitOr` 并把泛型收紧（如 `impl Into<AVCodecFlag>`） |
| P5 | `with_level(impl ToString)`（`encode.rs:267`） | 全 crate 唯一的 `impl ToString` | `with_level(42u8)` 都能编译，校验推到运行时；对照 `with_profile(VideoProfile)` |
| P6 | `Encoder::flush(writer, interleaved: bool, index: usize, out_stream_time_base: ffi::AVRational)`（`encode.rs:1822`） | 4 个位置参数，含裸 `bool` + 裸 FFI 类型 | 调用处 `flush(&mut w, true, 0, tb)` 完全不可读；`bool` 应换枚举、时间基应换 `Time`（或收进结构体） |
| P7 | hwaccel 同义不同形 | `auto_platform_with(candidates: &[HWDeviceType])`（`hwaccel.rs:123`） vs `HWDeviceType::auto_platform_config(candidates: Option<&[HWDeviceType]>)`（`910`）；`HWDeviceConfig::new(.., options: Option<Options>)`（`43`） vs 全 crate 的 `impl Into<Option<Options>>` | 同一概念两种签名；`None`/空 slice 语义还不一样 |
| P8 | 秒 | `Time::from_secs(f32)` vs `from_secs_f64(f64)`（`time.rs:65/77`）；`Chapter::new(title, start: f64, end: f64)`（`mux.rs:43`） | f32 在时长 > ~4.6h 时已丢到毫秒以下；"秒"到底是 f32 还是 f64 没有统一 |
| P9 | `SampleFormat::data_layout(channels: usize, samples: usize)`（`fmt.rs:100`） | `usize`，因为要喂 ndarray 形状 | 量纲上是 usize 合理，但和 u32 通道数互通时又要 cast；属"可接受，但应写明为什么是 usize" |
| P10 | `filter::video::crop(x:i32, y:i32, w:u32, h:u32)`（`filter.rs:433`，`delogo`/`drawbox`/`add_region` 同形） | 一条签名里**坐标有符号、尺寸无符号** | 设计上说得通（坐标可为负），但**没有文档说明**，看起来像笔误 |

**关于 P3 的结论：`i32` 更合适。** 理由：
1. **镜像 FFmpeg 字段**（`AVCodecContext.thread_count` 是 `int`）。类型上就穷尽了合法域，
   `i32::try_from` + "超出范围"分支**整条消失** —— 正是这轮为了简化才删掉的那种复杂度，用 i32 后压根不需要。
2. **消除 Encoder/Decoder 不对称**：`Decoder::thread_count()` 已经是 `i32`，`Encoder` 那个 `u32` 是唯一的异类。
3. **与 crate 既有惯例一致**：`with_gop_size(i32)`/`with_max_b_frames(i32)`/`with_buffer_size(i32)`/`nb_frames() -> i64` 都是照抄 FFmpeg 宽度。
4. `0` = 自行推导的语义不受影响；负数无意义，`count <= 0 → return` 这个守卫**本来就有**（为 `0` 而设），不新增成本。

反向考虑（不选 u32 的理由要弱）：u32 能在类型上表达"非负"，但既然 `0` 已经占用了"自动"，
"正数"并不能用一个类型精确表达，u32 只挡住了负数这一种非法值，却引入了"超出 i32"这第二种非法值。

> ### ⚠️ 本节的"统一方向"已于 2026-09-28 **反转**（用户裁决）
>
> 上面"高层一律 `u32`、仅 FFI 直通函数允许 `i32`"的方向**已被否决**。最终采用的判据是：
>
> > **要写进 FFmpeg 结构体的值，公开 API 就用 FFmpeg 那个字段自己的类型来承接
> > （通常 `i32`/`i64`）；类型一致就直接传下去 —— 不加转换、不加范围判断。**
>
> 否决"用域类型 `u32` 表非负"的关键理由，正是上面第 4 点里那个真实案例：`u32` 挡住了
> 负数，却引入了"超出 `i32`"这**第二种**非法值；而它的真实症状不是变负数，而是**回绕**
> （`nth = 2^31+5` ⇒ `5`，时基变成 `1/5` —— 看似合理却全错）。这只能靠"参数就用 `i32`"
> 从类型上根治，靠运行时检查治不了。§六 D4 的"自我更正"因此被**再次推翻**（见 §6.3）。
>
> **落地结果**：`width`/`height`、`nb_channels`/`sample_rate`、`thread_count`、`frame_size`
> 全部 `i32`；`PixelFormat::data_layout(width: usize, height: usize)` 是唯一的 `usize` 尺寸
> 入口（喂 ndarray 形状，与 `SampleFormat::data_layout(channels, samples)` 对齐）。
> 保留 `u32` 的只剩 P1 行列出的三处孤岛。

### 落地记录（2026-09-27 P3 → P1/P2；**2026-09-28 反转**）

按上面的统一方向落地，三个都是**破坏性签名改动**（调用方少写 cast）。判据是：

> 公开项用 `i32` **仅当**该值原样写进 FFmpeg 结构体字段、且负半轴有含义（哨兵/方向）。
> （2026-09-28 起改为：**只要值原样写进 FFmpeg 结构体字段就用 FFmpeg 字段的宽度**，
> 不再要求负半轴有含义。）

| 项 | 2026-09-27 的改动 | **2026-09-28 的当前状态** |
|---|---|---|
| **P3** | `with_thread_count(i32)`、`Encoder::thread_count() -> i32`；`set_thread_count(ctx, i32)` 非正值不写字段（负数另打 `warn!`） | 不变。`0` = 自行推导、镜像 `int`，整条 `i32` |
| **P1** | `Encoder/Decoder::width()/height() -> u32`（原 `i32`） | **已改回 `i32`**：`MediaFrame.width/height`、`new_video(i32,i32)`/`new_video_frame`、`EncoderBuilder.width/height`、`Encoder/Decoder::width()/height()`、`VideoParams`/`VideoEndpoint`/`StreamInfo`/`scale::scale_frame`/`imgutils::fill_linesizes` 全链路 `i32`；`build()` 守卫简化为 `<= 0` |
| **P2** | `new_audio(nb_channels: u32, sample_rate: u32)`、`with_nb_channels/with_sample_rate(u32)`、`Encoder/Decoder::sample_rate() -> u32`、`PcmSpec::new(channels: u32)` | **已改为 `i32`**（不是 u32）：`new_audio(nb_channels: i32, sample_rate: i32)`、两个 setter、`PcmSpec::new(sample_rate: i32, channels: i32)`、`filter::audio::{resample,format}`、`AudioParams`/`AudioEndpoint`、`StreamInfo`、`Encoder/Decoder::sample_rate() -> i32` |
| — | — | **新增**：`PixelFormat::data_layout(width: usize, height: usize)`（`usize`，用户指定） |

**附带发现并修掉的缺陷**：工作区的 `codec::set_thread_count` 守卫被写成 `if thread_count == 0 { return; }`，
与它自己的文档（"两者都不写字段"）和两个单元测试矛盾 ⇒ 负数会被真的写进 `AVCodecContext.thread_count`。
已改回 `<= 0`。**这是本轮唯一一处"静默错值"**，由 `cargo test --lib` 抓出（harness 因为当时只喂 `u32::MAX`
没覆盖负值，没有报警）。

**`frame_size()` 有意保留 `i32`**：`0` 在 FFmpeg 里是"可变帧长"这个**有意义**的值，与被测量的正数共用一个
值域，属于上表判据的"负半轴/哨兵有含义"一类，已补注释说明为什么它和 `width()` 不同宽。



## 六、公开 API 类型逐个复审（2026-09-27，第一性原理）

上一节（§五）的出发点是"**同一个概念的类型有没有漂移**"，所以它天然只会得出"统一到某一侧"的结论，
而"哪一侧"是靠**项目惯例**决定的 —— 那正是"适配以前的 API"。本节换判据：对**每一个**公开字段 /
参数 / 返回，问"这个值实际能取哪些值 / 是什么量纲 / 调用方需要什么"，再决定类型。

### 6.0 方法与规模

脚本 `rsmedia-codebase-audit/scripts/dump_api_surface.py`（本轮的产物）：

```bash
python3 ~/.workbuddy/skills/rsmedia-codebase-audit/scripts/dump_api_surface.py            # 全量
python3 ~/.workbuddy/skills/rsmedia-codebase-audit/scripts/dump_api_surface.py --flat     # 可 grep
```

盘点结果：**公开字段 155 / 公开函数参数 591 / 公开函数返回 549**。

参数类型分布（前几）：`u32` 70、`f32` **51**、`i32` 50、`&str` 43、`usize` 40、`u8` 15、
`impl Into<u32>` 8、`bool` 11、`i64` 8、`ffi::AVRational` 8。

> **（2026-09-28 复测，脚本 `/tmp/audit_api_types_2026.py`，610 个参数）**
> 上表的口径把 `impl Into<T>` 摊平成 `T`（所以 `f32 51` ≈ 裸 `f32` 15 + `impl Into<f64>` 37），
> 复测脚本**保留 `impl Into<T>` 原样**，因此两类数字不能逐项直接比较。**可比较的关键项**：
> `u32` **70 → 33**、`i32` **50 → 87**、`usize` 40 → 37、`&str` 43 → 48、
> `ffi::AVRational` **8 → 0**（批次 6a 的 `Rational` 收口）、`impl Into<u32>` **8 → 0**（批次 5 的
> `FlagSet<E>`）。`i32` 反超成为第一大类，正是 P1 / P2 方向反转后的直接结果。

三类**外部可验证**证据（不是推测）：
- `ffmpeg -h filter=<name>` 打印的选项类型（`<float>` / `<double>` / `<duration>` / `<string>`）。
- FFmpeg 的 `av_d2q`/`av_reduce` 实测（见 A1）。
- 调用点计数（见 D2）。

### 6.1 判据（按优先级；第 6 条故意排在最后）

1. **值域忠实** —— 非法值能否在类型上不可表示（`NonZero`/`Option`/枚举），代价低时就该做。
2. **量纲与单位** —— 时长 / 速率 / 计数 / 下标 / 比值 / 位掩码是六种不同的量，不能共用一个原语。
3. **哨兵语义** —— `0`/`-1`/`i64::MIN` 有意义就保留原语并写文档，否则不该让它可表示。
4. **能力不丢失** —— 类型不能比 FFmpeg 实际接收的域更窄（窄了就是**功能缺失**，不只是风格）。
5. **调用点代价** —— 样板/cast 的**出现次数**是硬证据，不是主观感受。
6. **一致性** —— **仅作平票时的 tie-breaker**，不能作为主理由。

### 6.2 问题清单

#### A 级：类型选错 ⇒ 行为错误 / 能力丢失（可复现）

**A1 `with_fps(fps: f32)` 会把标准帧率算错。**
帧率是**有理数**（NTSC 是 `30000/1001`、电影转 NTSC 是 `24000/1001`）。`f32` 只有 24 位尾数，
而 `with_fps` 把它 `as f64` 后交给 `av_d2q(fps, FPS_MAX=100_000)` 做连分数逼近 —— 逼近的是**被 f32 舍入过的值**，
于是可能选中一个**不同于标准值**的有理数。实测（真链接 FFmpeg，非推测）：

| 调用 | 期望 | 实得 |
|---|---|---|
| `with_fps(24000.0 / 1001.0)` | `24000/1001` | **`86002/3587`** ✘ |
| `with_fps(30000.0 / 1001.0)` | `30000/1001` | `30000/1001` ✔ |
| `with_fps(60000.0 / 1001.0)` | `60000/1001` | `60000/1001` ✔ |
| `with_fps(25.0)` / `(30.0)` | `25/1` / `30/1` | ✔ |

`86002/3587` 与 `24000/1001` 相差 2/(3587·1001) ≈ 5.6e-7 fps：短片段看不出来，长片会累积成可见偏差，
且容器里写的是**非标准**时基。
**这不是精度洁癖，是"标准值表达不出来"**。修法：补一个精确入口
`with_frame_rate(num: u32, den: u32)`（或 `FrameRate` 类型，带 `FrameRate::ntsc(30)` / `film()` 之类的构造器），
`with_fps` 保留为便捷方法并在文档里写明它经由 `av_d2q` 逼近。

**A2 `filter::video::{afade, trim}` 把"时长"当 `f32` 秒。**
`ffmpeg -h filter=afade`：`start_time <duration>`、`duration <duration>`；`-h filter=trim`：
`start/end/duration <duration>`。`<duration>` 是**时长类型**，接受 `"1.5s"`、`"00:00:01.5"` 等。
而 crate 的实现是 `format!("afade=t={fade_type}:st={start}:d={duration}")` —— `f32` 经 `Display`
只能吐出裸小数，**既丢精度（f32 在 10⁴ s 量级 ULP≈1 ms）又丢单位语法**，而且类型本身不表达"这是时长"。
修法：`start`/`duration` 收 `std::time::Duration`（或 crate 的 `Time`）并格式化成时长字面量。

**A3 `filter::video::eq` / `boxblur` 的 `f32` 让"表达式"能力失效。**
`ffmpeg -h filter=eq`：`brightness/contrast/saturation/gamma <string>`；`-h filter=boxblur`：
`luma_radius <string>`。这几个选项在 FFmpeg 里是**可求值表达式**（`"sin(t)"`、`"iw/2"`、按帧变化）。
crate 收 `f32` ⇒ 只能传常量，**表达式形式在 API 上不可达**。
修法：`&str`（表达式）或一个 `Expr::{Const(f64), Expr(&str)}` 枚举。至少不要用 `f32` 把能力锁死。

**A4 `time::new_rational(num: i32, den: i32)` 允许 `den == 0`。**
它直接转 `avutil::ra(num, den)`，不校验；退化的 `0/0` 有理数会顺着 `Time`/`StreamInfo`/滤镜端点传播
（`Time` 甚至专门为它写了 `0/0` 分支）。`den: NonZeroI32` 能让非法值**不可表示**，删掉一整类防御代码。

**A5（实施批 4 时新发现）`filter::audio::advanced_fft_denoise` 把浮点写进了布尔选项。**
参数叫 `time_smoothing: Option<f32>`、文档写"Temporal smoothing factor. Default 0."，实现是
`format!("…:tr={tr}")`。但 `ffmpeg -h filter=afftdn` 里 `tr` 是 **`track_residual <boolean>`** ——
既不是"时间平滑系数"，也不是浮点。实测（真链接 FFmpeg 9.0.2）：

```
$ ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -af "afftdn=nr=12:nf=-50:nt=w:tr=0.5" -f null -
[Parsed_afftdn_0] Unable to parse "tr" option value "0.5" as boolean
Error applying option 'tr' to filter 'afftdn': Invalid argument
```

即**只要传任何非整数，整条滤镜链都建不起来**（`FilterGraph` 初始化直接失败）；而 `tr=0`（默认路径）
恰好能解析成 `false`，所以问题一直被默认值掩盖着。
这是 A 级（行为错误）而不是 B 级（收窄）：类型和名字指向"一个浮点系数"，掩盖了选项真实语义。

**A6（同批发现，属"反例"，不改）`anlm_denoise` 的 `patch_size`/`search_range` 对应 `anlmdn.patch`/`research`，
FFmpeg 9.0 声明它们是 `<duration>`。** 看似"应该改成 `Duration`"，但 `<duration>` 的无单位数值在 FFmpeg 里
就是**微秒**量纲：`p=7` 与 CLI 的 `p=7` 完全等价，改成 `Duration::from_micros(7)` 反而扭曲语义。
⇒ 保持 `Option<i32>`。记录在此，说明判据不是"见到 `<duration>` 就得换类型"。

#### B 级：类型比 FFmpeg 窄（narrowing，无行为错误）

**B1 10 处 filter 参数 FFmpeg 声明 `<double>`，crate 收 `f32`。**
逐条 `ffmpeg -h filter=<name>` 可验：`hqdn3d{luma,chroma}_spatial/tmp`、`nlmeans.s`、`loudnorm.{I,LRA,TP}`、
`equalizer.{frequency,width,gain}`、`bass/treble/lowshelf/highshelf.gain`、`atempo.tempo`、`acompressor.ratio`。

对照 —— 这些 crate 用 `f32` 是**正确**的（FFmpeg 就是 `<float>`）：`smartblur.{luma,chroma}_{radius,strength}`、
`chromakey`/`colorkey.{similarity,blend}`、`vibrance.intensity`、`gblur.sigma`。
⇒ 结论不是"全部改 f64"，而是**按 FFmpeg 声明的类型对齐**：`<float>`→`f32` 保持不变，`<double>`→`f64`。

**B2 "历史 API" 被当成保留理由。**
`time.rs` 里 `as_secs() -> f32` 的文档原话是 *"Single-precision on purpose (**the historical API**)"*，
并且 `from_secs(f32)` / `from_secs_f64(f64)` / `as_secs() -> f32` / `as_secs_f64() -> f64` 四者并存 ——
同一个量两种精度、调用方没有依据可选。crate 版本 **0.10.1（pre-1.0）**，破坏性改动是预期内的，
"历史 API" 不构成理由。且 `Time` 缺 `From<Duration>` / `From<f64>`，没有惯用入口。
⇒ 让 `f64`/`Duration` 成为唯一规范入口，`f32` 若保留必须显式命名（如 `as_secs_f32_lossy`）。

#### C 级：FFI 类型泄漏到高层 API

> **已实施（批次 6 前半，`Rational` 收口）。** `Rational` 已是 crate 对有理数的**唯一**出口：
> `ffi::AVRational` 在 `src/` 下只剩 `time.rs` 里的两个 `From` impl（双向）与它自己的测试 ——
> `src/rational.rs` 已并入 `src/time.rs`（有理数与时间是同一件事，两个模块反而让人找不到
> `Rational` 在哪；`pub use time::{Rational, Time}` 保持顶层再导出不变）。
> 高层一律用 `Rational`，只在**调用 rsmpeg/FFmpeg 的那一行**写 `.into()`。四处值得记：
> ① `CodecConfig::supported_frame_rates()` 由 `Result<Option<&[ffi::AVRational]>>` 改为
> `Result<Option<Vec<Rational>>>` —— `Rational` 与 `AVRational` 布局不同，借不出切片，只能拥有；
> ② `Writer::add_stream` / `Writer::stream_time_base`（含 `DynWriter` 转发）同步改 `Rational`，
> 于是 `Muxer` / `subtitle` / `PcmSink` 三条写包路径不再有裸类型；
> ③ `mux.rs` 的 `AVChapter.time_base`（毫秒）是唯一必须写裸 FFI 字段的地方，
> 值仍由 `Rational::new(1, 1000)` 在边界上生成，不在该文件里拼字面量；
> ④ 顺带修掉一个**实测到的回归**：`time::TIME_BASE` 曾被写成 `Rational::integer(1_000_000)`
> （＝ 1000000/1），而 `AV_TIME_BASE_Q` 是 1/1000000 —— 症状是 `Time::as_secs()` 差 10^12 倍、
> `seek_to_timestamp` 的毫秒→微秒换算系数为 0。为能在 `const` 里正确表达 1/n，
> 给 `Rational` 加了 `const fn unit(den)`（`den > 0`，`const` 场景下非法值即编译错误）。

**C1 `ffi::AVRational` 出现在 30 处公开位置**（16 个字段 + 8 个参数 + 6 个返回）：
`VideoParams.time_base/frame_rate/pixel_aspect`、`VideoEndpoint.*`、`AudioParams.time_base`、
`StreamInfo.time_base`、`MediaFrame.time_base`、`Encoder/Decoder::time_base() -> ffi::AVRational`、
`bsf::new(time_base)`、`mux::new_copy(src_time_base)`、`Encoder::flush(.., out_stream_time_base)`、
`CodecConfig::supported_frame_rates() -> &[ffi::AVRational]` 等。
有理数是媒体领域的一等概念（时基 / 帧率 / 像素宽高比），crate 却**没有公开的 `Rational` 类型** ⇒
调用方要构造一个 `VideoEndpoint` 就必须自己依赖 `rsmpeg::ffi`（或 `AVRational { num, den }` 字面量）。
⇒ 引入公开的 `Rational`（`num: i32`, `den: NonZeroI32`，见 A4）并让高层结构体用它；
`ffi::` 只应出现在**显式低层**的模块（`scale`/`imgutils`/`resample` 的裸 `AVFrame` 路径）。

**C2 9 类 FFI 枚举直接作为公开字段**（15 处）：`AVColorSpace`/`AVColorRange`/`AVColorPrimaries`/
`AVColorTransferCharacteristic`/`AVChromaLocation`/`AVFieldOrder`/`AVPictureType`/`AVAlphaMode`/`AVFrameSideDataType`。
crate 对 `PixelFormat`/`SampleFormat` 都做了自己的枚举，颜色却没有 —— 不一致，且都要求用户懂 FFmpeg 头文件。

**C3 参数/返回里的 FFI 类型**：`ffi::AVSampleFormat`(4)、`ffi::AVChannelLayout`(4)、`ffi::AVCodecID`(3)、
`ffi::AVPixelFormat`(2)。其中 `AVChannelLayout` 尤其值得包一层（`nb_channels` 与 `ch_layout` 两个概念在 crate 里反复互相转换）。

**C4 同一概念两种形态，且更差的那个是 `u32`**：`StreamInfo.codec_id: u32` vs `Decoder::codec_id() -> ffi::AVCodecID`。
`codec_id` 本质是 `AVCodecID` 枚举，用 `u32` 连类型信息都没有。同理 `StreamInfo.codec_tag: u32` 应为 `[u8; 4]` 或 newtype。

#### D 级：一致性与可读性

**D1 `impl Into<u32>` 的位掩码 setter ×8** —— `with_flags`/`with_flags2`/`with_thread_type`/`with_scale_quality`/
`with_err_recognition`/`init_with`/`init_logging`/`new_with_options`。任意 `u32` 都能进，`AVCodecFlag`/
`ThreadType`/`ScaleQuality` 的枚举约束**完全被绕开**（＝ §C5 / P4）。最能说明问题的是
`init_with(level: AVLogLevel, flag: impl Into<u32>)`：同一次调用里一个参数强类型、另一个裸 `u32`。

> **已实施（批次 5）＋ 本条前提更正。** 原写的"正解：给枚举实现 `BitOr`，setter 收
> `impl Into<AVCodecFlag>`"**两半都不成立**：① `ffi_enum!` 早已生成 `BitOr`/`BitAnd`（`macros.rs`），
> 无需再补；② 但它的 `BitOr` 产出的是**裸 `repr`**，而无字段枚举表示不了 `LOW_DELAY | CLOSED_GOP`
> 这种无名组合，所以 setter 收 `impl Into<AVCodecFlag>` 会让 `A | B`（`encode.rs:2094`、`decode.rs:1621`
> 都在用）和 `0`（"无质量位"，`encode.rs:2950`）**编译不过** —— 为满足判据 1 而破坏判据 4。
>
> 真正的正解是引入**集合类型**：[`FlagSet<E>`](src/flags.rs)（`u32` + `PhantomData<E>`）。
> `ffi_enum!` 的 `BitOr` 改为产出 `FlagSet<Enum>`（5 种操作数组合：`A|B`、`A|raw`、`raw|A`、`set|A`、`A|set`，
> 另有 `|=`/`&=`），setter 收 `impl Into<FlagSet<_>>`。于是：`A | B` 照写、`A | B | C` 链条保持类型、
> 空集用 `FlagSet::EMPTY`、裸掩码只能显式走 `FlagSet::from_bits` ——**任意 `u32` 再也进不来，且一项能力都没丢**。
> 读取侧 `FlagSet::contains` / `set & Enum`；`Encoder`/`Decoder`/`Scaler` 的位掩码 getter 同步改为 `FlagSet`。
> 唯一保持 `u32` 的是 `Scaler::flags`：它是 `ScaleAlgorithm` 与 `ScaleQuality` **两种**位集的并集，
> 没有哪**一个** `FlagSet` 能描述它。

**D2 `with_codec_name(impl Into<Option<String>>)` 根本不接受 `&str`。**
它只能收 `String` 或 `Option<String>`，所以 **38 个调用点全部要 `.to_string()`，其中 16 个还要额外包 `Some(...)`**：
`with_codec_name(Some("mov_text".to_string()))` ×9、`with_codec_name("mjpeg".to_string())` ×4 …
"让调用方写样板"是明确要避免的。⇒ `with_codec_name(name: impl Into<String>)` + 一个显式的
"恢复默认选择"方法（`None` 表达的是**另一种意图**，不是一个值）。

**D3 bool 位置参数**：`Encoder::flush(writer, interleaved: bool, index: usize, out_stream_time_base)`、
`imgutils::copy_frame_metadata(.., copy_data: bool)`、`pixel::find_best_pix_fmt(.., has_alpha: bool)`。
调用处 `(w, true, 0, tb)` 不可读（P6）。注意区分：`with_global_header(enabled: bool)` 这类 **setter 收 bool 没问题**
（方法名承载了语义），坏的是**多参数函数里的裸 bool**。

**D4 ⚠️⚠️ 本条"自我更正"已于 2026-09-28 被**再次推翻**（用户裁决）—— 方向是 `i32`，不是 `u32`。**

> **原文本**（2026-09-27，保留以存史，但**结论已作废**）：
> "`VideoParams`/`VideoEndpoint`/`AudioParams`/`AudioEndpoint`/`StreamInfo` 的尺寸用 `i32`，
> 我上一轮的理由不成立。… 这些结构体是公开可写字段 + crate 自己构造、面向调用方的；
> 负宽度 / 负声道数 / 负采样率在任何边界都无意义，'镜像'只发生在写进 `AVFrame` 的那一刻
> —— 那是边界，不是 API 该承担的义务。⇒ 应改为 `u32`。"

**推翻的理由**（与 §五 的反转同源）：上面的论证只考虑了"负值无意义"，却漏了 `u32` 引入的
**第二种**非法值 —— 超出 `i32` 的部分。而它的真实症状**不是**"变成一个很大的正数"，
而是 `as i32` **回绕**成看似合理的小值（`2^31+5 ⇒ 5`），一路静默走到底。这类错误
运行时检查抓不住（回绕后的值合法），只有让**参数本身就取 FFmpeg 字段的宽度**才能根治。

**实测现状（2026-09-28）**：`VideoParams`/`VideoEndpoint`/`AudioParams`/`AudioEndpoint`/
`StreamInfo` 的 `width`/`height`/`nb_channels`/`sample_rate` **全为 `i32`**，与
`AVFrame.width/height`、`AVCodecContext.width/height/sample_rate`、`AVChannelLayout.nb_channels`
逐字段同宽 ⇒ **构造时不再有任何转换**，C1 那类"`u32` 越界"与"回绕"同时失去立足点。
上表末尾 `StreamInfo.{block_align, initial_padding, …, pts_wrap_bits}` 那一串**尚未复核**
（`frame_size` 已确认保留 `i32`，因为 `0` = "可变帧长"是有意义的值）。

**真正该保留有符号 / 原语的**（这才是判据 3 的正例）：
| 项 | 为什么 |
|---|---|
| `StreamInfo.profile` / `level` | `AV_PROFILE_UNKNOWN = -99` 是真实值 |
| `StreamInfo.duration` / `start_time` / `nb_frames` | `AV_NOPTS_VALUE = i64::MIN` 哨兵 |
| `MediaFrame.pts` / `pkt_dts` / `duration` / `pkt_duration` / `best_effort_timestamp` | 同上 |
| `filter::crop`/`delogo`/`drawbox` 的 `x`/`y` | 坐标**可为负**（让内容移出画面），`w`/`h` 已是非负 |
| `Encoder::frame_size() -> i32` | `0` = "可变帧长"是**有意义**的值 |
| `Encoder/Decoder::thread_count() -> i32` | 见 6.4 |

**D5 `Option<i32>` 用于计数**：`zoompan(duration: Option<i32>)`、`anlm_denoise(patch_size/search_range: Option<i32>)`
⇒ `Option<u32>`（`0` 无意义时甚至 `Option<NonZeroU32>`）。`Option` 本身用得对（`None` = 用 FFmpeg 默认），
错的只是内层宽度。

**D6 两种"可选值"写法并存**：`impl Into<Option<T>>`（`with_options` ×3、`with_filters`、`with_interrupt`）
vs 裸 `Option<T>`（`HWDeviceConfig::new(options: Option<Options>)`、`with_hardware_device(Option<HWDeviceConfig>)`、
`new_from_reader(filters: Option<Vec<Filter>>)`、`hwaccel::{cuda,vaapi,...}(Option<String>)`）。
前者能直接 `with_options(opts)` 而不用包 `Some`，**样板更少**，应统一到它。

**D7 四种时间表示**：`Chapter{start, end: f64 秒}`、`SubtitleSegment{start_ms, end_ms: i64}`、
`Time{time: Option<i64>, time_base}`、`Muxer::duration() -> f64`。至少 `Chapter` 与 `SubtitleSegment` 应统一
（`Chapter` 用 `Time`/`Duration` 更自然：章节是容器时间，毫秒整数其实比 f64 秒更精确）。

**D8 `MediaFrame` 内部两种像素量宽度**：`crop_{top,bottom,left,right}: usize` vs `width`/`height: i32`
（原文写的是 `u32`，2026-09-28 后 `width`/`height` 已改 `i32`）。

> **（2026-09-28 复核）这条其实已经符合判据，不再是缺陷** —— 因为 `AVFrame` 里
> `crop_top/bottom/left/right` 的类型正是 `size_t`，`width`/`height` 正是 `int`。
> 两者不同宽不是漂移，而是**各自镜像了 FFmpeg 对应字段的宽度**，与 P9
> （`SampleFormat::data_layout(usize, usize)`）同理。
> **遗留只有一条**：`MediaFrame` 上**没有注释说明**这一点，读起来仍像笔误（＝ P9/P10 同类）。

**D9 `MediaFrame` 的位掩码字段**：`flags: i32`、`decode_error_flags: i32` 都是位集 ⇒ 应 bitflags；
`quality`/`repeat_pict` 是普通整数，`i32` 合理。

**D10 闭集选项用 `&str`**：
- `filter::video::afade(fade_type: &str)` —— 只有 `in`/`out` 两个值，应收窄为枚举（传错字符串现在会静默走 FFmpeg 默认）。
  **〔2026-09-29 部分修〕** 枚举化仍未做（破坏性），但已与 `yadif`/`bwdif`/`amix` 一起走
  `check_closed_set` 前置校验：名字与 FFmpeg 的数字写法都收，越界/拼错在构造期即
  `InvalidConfig` 并点名候选值（值集按 `ffmpeg -h` + 运行时实测核对：
  `yadif.mode` 0..=3、`bwdif.mode` 0..=1、`afade.t` 0..=1、`amix.duration` 0..=2）。
  `amix` 的两个入口（`audio::amix` 与 `FilterGraphBuilder::amix`）此前一个校验一个不校验，
  现共用 `check_amix_duration`。
- `filter::video::scale(flags: Option<&str>)` —— crate **已经有** `ScaleAlgorithm`/`ScaleQuality` 强类型枚举，
  这里却绕过它们收裸字符串。
- `curves(preset)`/`gif_palette(dither)`/`advanced_fft_denoise(noise_type)`/`lutyuv(y,u,v)` —— 这些是**有意**的
  "FFmpeg 语法逃逸舱口"，本身合理，但应在文档里统一声明为"直接透传 filtergraph 语法"，
  而不是与强类型 API 混在一起看不出区别。

### 6.3 与 §五 的差异（自我更正汇总）

| §五 结论 | §六 复审（2026-09-27） | **2026-09-28 用户裁决（现行）** |
|---|---|---|
| P1 保留 `VideoParams`/`VideoEndpoint`/`StreamInfo` 的 `i32`（"FFI 镜像层"） | **推翻**（判据 1/2 优先于"镜像"；见 D4） | **推翻之推翻**：维持 `i32`，但理由换成了更硬的一条 —— `u32` 的越界值经 `as i32` 会**回绕**成看似合法的小值（`2^31+5 ⇒ 5`），运行时检查抓不住，只能靠"参数取 FFmpeg 字段宽度"从类型上根治。见 D4 |
| P3 `thread_count` 用 `i32` | **结论保留，理由更换**（`i32` 值域完整包含 FFmpeg `int` ⇒ 转换无损且不可能失败；最精确的 `Option<NonZeroU32>` 因调用代价过高而不选） | 不变 |
| P2 `nb_channels`/`sample_rate` → `u32` | 维持（判据 2/3 支持） | **推翻**：改为 **`i32`**（含 `PcmSpec` 那个全 crate 唯一的 `u16` 通道数） |
| P4/P6/P9/P10 | 维持并升级为 D1/D3/D8/D10 | 维持。**D8 复核后已不算缺陷**：`AVFrame.crop_*` 是 `size_t`、`width`/`height` 是 `int`，两者正是各自镜像 FFmpeg 字段 ⇒ 只剩"缺注释说明"这一条 |

### 6.4 建议的修复顺序（按 收益/风险 排序）

1. **A4 + A1**（`NonZeroI32` 的有理数 + 精确帧率入口）—— 小而纯增量，直接修掉一个**实测到的错误**。
2. **D2**（`with_codec_name`）—— 纯 ergonomics，38 个调用点立即受益，无行为变化。
3. **D6**（统一 `impl Into<Option<T>>`）—— 同上，改动零散但机械。
4. **A2 + A3 + B1**（filter 参数按 `ffmpeg -h` 的类型对齐：`<duration>`→时长、`<double>`→f64、`<string>`→表达式）——
   面广，需要 harness 同步；建议按 filter 分组提交。
5. **D1**（位掩码强类型）—— 需要先给 4 个枚举实现 `BitOr`。
6. **D4 + C1 + C2**（尺寸统一 —— **方向已于 2026-09-28 定为 `i32`，不是 `u32`**）+ 公开 `Rational`/颜色类型）——
   **尺寸这一半已完成**（见 §6.5 批次 6b）；**未做的是"9 类 FFI 枚举封箱"**这一半。
7. D5/D7/D8/D9/D10 收尾（D8 复核后只剩"补注释"）。
8. **A6 音频尾部延迟** —— 行为取舍，**待用户裁决**后再动（见 §二 A6）。

> 全部都是破坏性签名改动。crate 当前 **0.10.1（pre-1.0）**，这类改动是预期内的；
> 每一批都应同步 `rsmedia_test` 并跑 §二 的正交检查 + 全量 harness。

### 6.5 落地进度

| # | 批次 | 状态 | 落点 |
|---|---|---|---|
| 1 | A4 + A1 | ✅ | `Rational{num: i32, den: NonZeroI32}`（现住 `src/time.rs`）+ `EncoderBuilder::with_frame_rate`；`with_fps` 文档标注 `av_d2q` 逼近实情 |
| 2 | D2 | ✅ | `with_codec_name(impl Into<String>)`；清掉 47 处 `.to_string()`/`Some(...)` |
| 3 | D6 | ✅ | 10 处裸 `Option<T>` 参数统一为 `impl Into<Option<T>>` |
| 4 | A2 + A3 + B1 | ✅ | filter 参数按 `ffmpeg -h filter=<name>` 对齐（`impl Into<f64>` ×23、`Expr` 表达式参数、`afade`/`trim` 收 `Duration`）；顺带发现并修掉 **A5**（`afftdn.tr` 写浮点给布尔位），记录 **A6**（`anlmdn.patch` 虽是 `<duration>` 但裸数按微秒，故不改） |
| 5 | D1 | ✅ | 新增 `src/flags.rs` `FlagSet<E>`；`ffi_enum!` 的 `BitOr` 改为产出 `FlagSet<Enum>`（+`contains`/`From`/`Into<repr>`/`|=`/`&=`）；8 个位掩码 setter 收 `impl Into<FlagSet<_>>`；`Encoder`/`Decoder`/`Scaler` 的掩码 getter 同步强类型。**前提更正见 §6.2 D1** |
| 6a | C1 | ✅ | 30 处公开 `ffi::AVRational` → `Rational`；`Rational` 成为唯一出口（`ffi::AVRational` 只存在于 `time.rs`）；`supported_frame_rates()` 改拥有 `Vec<Rational>`；`Writer` 的时间基接口同步；新增 `Rational::unit(den)` 供 `const` 场景；修掉 `TIME_BASE` 分子分母颠倒的回归（细节见 §6.2 C1）。**`src/rational.rs` 已并入 `src/time.rs`** |
| 6b | D4 + C2 | ⏸️ **部分完成** | **尺寸统一已完成，但方向是 `i32` 而非原计划的 `u32`**：`width`/`height`、`nb_channels`/`sample_rate` 在 `MediaFrame`/`Encoder`/`Decoder`/`EncoderBuilder`/`VideoParams`/`VideoEndpoint`/`AudioParams`/`AudioEndpoint`/`StreamInfo`/`PcmSpec`/`filter::audio::{resample,format}`/`scale::scale_frame`/`imgutils::fill_linesizes` 上全链路 `i32`；`PixelFormat::data_layout` 改 `usize`；`encode.rs` `build()` 守卫简化为 `<= 0`。<br>⚠️ **未做的一半**：9 类 FFI 枚举封箱 |
| 7 | D5/D7/D8/D9/D10 | ⬜ | 收尾（D8 复核后只剩"补注释"） |
| — | **A6 尾部延迟** | ⬜ **待裁决** | 音频重采样尾部未排空（§二 A6）；修 vs 记录两条路互斥 |

> 批次 1–6a 已过的正交检查（每批都跑）：`cargo fmt --check`、`check --all-targets`、
> `clippy -D warnings`（含 / 不含 `image`）、`RUSTDOCFLAGS="-D warnings" cargo doc`（两种特性组合）、
> `ffmpeg-feature-audit` 脚本、`rsmedia-codebase-audit` 脚本、lib 322 + 集成 17 个 binary + 65 doctest 全绿。
>
> **（2026-09-28 更新）** 批次 6b 的尺寸部分已完成，并同步了 `rsmedia_test`（`9b518b5`）。
> 复测：lib **340** + 集成 **19** 个 binary + **68** doctest 全绿；
> VM 6.1 / 7.1 / 8.1 / 9.0 = 339 / 340 / 341 / 341；harness 38 模式 **293 pass / 0 fail / 2 xfail**。
>
> **仍未做**：9 类 FFI 枚举封箱、批次 7、`videodecode` 目录扫描应跳过非视频文件、
> harness 的 `tracing-subscriber` 未使用依赖。
>
> 注：批次 4 行里的 **A5/A6** 是**该批次内部的滤镜参数编号**（`afftdn.tr` / `anlmdn.patch`），
> 与 §二 的缺陷编号**不是同一套**。
