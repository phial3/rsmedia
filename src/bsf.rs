//! Bitstream filters (`AVBSFContext`)
//!
//! Where the transcoding path through [`Encoder`](crate::Encoder) re-encodes,
//! a bitstream filter works at the **packet level**: it only adjusts container
//! conventions such as parameter-set placement, NALU boundary representation
//! and audio frame headers. The coded data itself is unchanged — zero quality
//! loss, negligible cost. Typical uses:
//!
//! - `h264_mp4toannexb` / `hevc_mp4toannexb`: MP4/MKV (avcC/hvcC) → TS/raw
//!   streams (Annex B start codes + inlined parameter sets) — the standard fix
//!   for remuxing to TS/IPTV;
//! - `aac_adtstoasc`: ADTS (TS) → raw + ASC (MP4/M4A); without it FFmpeg fails
//!   with `Malformed AAC bitstream detected`.
//!
//! Note: unlike the `ffmpeg` CLI (which inserts the filter it needs while
//! muxing), a library user must create the filter and drive it on the packet
//! path explicitly; the output stream's codecpar should be taken from
//! [`Bsf::par_out`] before being handed to [`Muxer`](crate::Muxer).
//!
//! # Usage
//!
//! ```no_run
//! use rsmedia::bsf::Bsf;
//! use rsmedia::mux::Demuxer;
//!
//! fn main() -> rsmedia::Result<()> {
//!     let mut demuxer = Demuxer::new("in.mp4")?;
//!
//!     // The source codecpar tells the filter whether it applies to this codec,
//!     // and the time base goes into `AVBSFContext.time_base_in`.
//!     let info = demuxer.stream_info(0)?;
//!     let mut bsf = Bsf::new("h264_mp4toannexb", &info.codec_parameters, info.time_base)?;
//!
//!     // Filter packet by packet: one input packet yields zero or more output
//!     // packets. `packet` is consumed - FFmpeg unrefs it on send.
//!     while let Some((index, mut packet)) = demuxer.demux_packet()? {
//!         if index != 0 {
//!             continue;
//!         }
//!         for mut out in bsf.filter_packet(&mut packet)? {
//!             let _ = &mut out; // hand `out` to `Muxer::mux_packet`
//!         }
//!     }
//!
//!     // Flush the buffered tail. From here on the output stream's parameters
//!     // are `bsf.par_out()` - the avcC extradata is gone.
//!     for mut out in bsf.flush_packets()? {
//!         let _ = &mut out;
//!     }
//!     Ok(())
//! }
//! ```

use crate::error::{Context, Result, RsmediaError};
use crate::options::AVOptions;
use crate::time::Rational;

use rsmpeg::avcodec::{
    AVBSFContext, AVBSFContextUninit, AVBitStreamFilter, AVCodecParameters, AVPacket,
};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

use std::ffi::CString;

/// 已初始化、可直接收发包的 bitstream filter 上下文。
///
/// 由 [`Bsf::new`] 创建；一个实例对应一路流的包通路。内部维护 FFmpeg 的
/// "send 后必须把 receive 抽干"的复用语义（见 [`Self::filter_packet`]）。
pub struct Bsf {
    inner: AVBSFContext,
    /// `true` once EOF has been sent ([`Self::flush_packets`] ran): a second
    /// `send_packet(None)` would make FFmpeg return EINVAL.
    flushed: bool,
}

impl std::fmt::Debug for Bsf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bsf").field("filter", &self.name()).finish()
    }
}

impl Bsf {
    /// 列出当前 FFmpeg 构建支持的全部 bitstream filter 名字。
    pub fn list() -> Vec<String> {
        AVBitStreamFilter::iterate()
            .map(|f| f.name().to_string_lossy().into_owned())
            .collect()
    }

    /// 创建并初始化一个 bitstream filter。
    ///
    /// # Arguments
    ///
    /// * `name` - filter 名（如 `h264_mp4toannexb`，完整列表见 [`Bsf::list`]）
    /// * `codecpar` - 输入流的 codec 参数（决定 filter 是否适用该 codec）
    /// * `time_base` - 输入包的时间基（写入 `AVBSFContext.time_base_in`）
    ///
    /// 需要给 filter 传私有选项时（`h264_metadata`、`hevc_metadata`、`setts`
    /// 这类 filter 的全部行为都由选项描述）改用 [`Self::new_with_options`]。
    pub fn new(name: &str, codecpar: &AVCodecParameters, time_base: Rational) -> Result<Self> {
        Self::new_with_options(name, codecpar, time_base, AVOptions::new())
    }

    /// [`Self::new`] 的带私有选项版本。
    ///
    /// `options` 是 **filter 自己的 AVOption**，写进 `AVBSFContext.priv_data`；
    /// 必须在 `av_bsf_init` 之前设置，`init` 之后上下文即冻结。这是让
    /// `h264_metadata` / `hevc_metadata` / `setts` 这类 filter 真正可用的唯一入口 ——
    /// 它们没有任何无选项的默认行为，能创建而不能配置等于不可用。
    ///
    /// 认不出的键**不会**被静默丢弃：FFmpeg 的 `av_opt_set_dict` 只消费认得的键，
    /// 其余的留在字典里；此处据此返回 [`RsmediaError::InvalidConfig`]，消息里点出
    /// 全部未生效的键名（传错键名是这类 API 最常见的错误，静默忽略会让用户以为
    /// 配置已生效）。
    pub fn new_with_options(
        name: &str,
        codecpar: &AVCodecParameters,
        time_base: Rational,
        options: AVOptions,
    ) -> Result<Self> {
        let name_c = CString::new(name).context(format!("bsf name {name:?}"))?;
        let filter = AVBitStreamFilter::find_by_name(&name_c).ok_or_else(|| {
            RsmediaError::unsupported(format!(
                "bitstream filter '{name}' is not available in this FFmpeg build"
            ))
        })?;
        let mut ctx = AVBSFContextUninit::new(&filter);
        ctx.set_par_in(codecpar);
        ctx.set_time_base_in(time_base.into());
        Self::apply_private_options(&mut ctx, name, options)?;
        let inner = ctx.init()?;
        Ok(Self {
            inner,
            flushed: false,
        })
    }

    /// 把 filter 的私有选项写进 `AVBSFContext.priv_data`。
    ///
    /// 必须在 `av_bsf_init` 之前调用：init 会把上下文冻结，之后再写 `priv_data`
    /// 不会重新配置 filter。
    fn apply_private_options(
        ctx: &mut AVBSFContextUninit,
        name: &str,
        options: AVOptions,
    ) -> Result<()> {
        if options.is_empty() {
            return Ok(());
        }
        let Some(dict) = options.into_dict() else {
            return Ok(());
        };
        let mut raw = dict.into_raw().as_ptr();

        // SAFETY: `ctx` 由 `av_bsf_alloc` 创建，`priv_data` 是它拥有的可写上下文，
        // 且此处仍在 `init` 之前（尚未冻结）；`raw` 是刚物化的合法字典。
        let priv_data = unsafe { (*ctx.as_mut_ptr()).priv_data };
        let ret = unsafe { ffi::av_opt_set_dict(priv_data, &mut raw) };

        // `av_opt_set_dict` 只消费认得的键，剩下的留在 `raw` 里。无论成败都必须
        // 释放，否则会留下一处只在"传错键名"时才出现的泄漏。
        // SAFETY: `raw` 由 `into_raw` 转移而来，所有权归本函数。
        let unmatched = unsafe { AVOptions::from_raw_dict(raw) };
        // SAFETY: 同上，`raw` 仍是一个有效（可能为空的）字典。
        unsafe { ffi::av_dict_free(&mut raw) };

        if !unmatched.is_empty() {
            let keys: Vec<&str> = unmatched.iter().map(|(key, _)| key.as_str()).collect();
            return Err(RsmediaError::invalid_config(format!(
                "bitstream filter '{name}' does not take option(s): {}",
                keys.join(", ")
            )));
        }
        if ret < 0 {
            return Err(RsmediaError::av_error(ret)
                .with_context(format!("Failed to configure bitstream filter '{name}'")));
        }
        Ok(())
    }

    /// 本 filter 的名字（如 `h264_mp4toannexb`）。
    pub fn name(&self) -> String {
        self.inner.filter().name().to_string_lossy().into_owned()
    }

    /// 送入一个输入包。EOF 后（[`Self::flush_packets`] 之后）再送入会报错。
    ///
    /// 低层步骤：外部入口是 [`Self::filter_packet`]（送入并抽干，一次完成）。
    ///
    /// 注意：包所有权在 `send` 时转移给 filter（FFmpeg 内部会 unref 这个包），
    /// 调用方传入的 `packet` 在调用后即失效。
    fn send_packet(&mut self, packet: &mut AVPacket) -> Result<()> {
        // 已发过 EOF 的通路不能再收包：FFmpeg 此时返回 EINVAL，直接给出可读的
        // 状态错误，而不是把陌生的 AVERROR 透传给调用方。
        if self.flushed {
            return Err(RsmediaError::invalid_config(
                "bitstream filter is already flushed: no more packets can be filtered",
            ));
        }
        match self.inner.send_packet(Some(packet)) {
            Ok(()) => Ok(()),
            Err(RsmpegError::BitstreamFullError) => Err(RsmediaError::msg(
                "bitstream filter is full: drain received packets first",
            )),
            Err(e) => Err(e.into()),
        }
    }

    /// 发送 EOF 并抽出所有剩余输出包（每个输入流的通路结束时调用一次）。
    ///
    /// 幂等：第二次调用不再向 FFmpeg 发送 EOF（`AVBSFContext` 会返回 EINVAL），
    /// 而是直接返回空列表——剩余输出在第一次调用时已经抽干。
    pub fn flush_packets(&mut self) -> Result<Vec<AVPacket>> {
        if self.flushed {
            return Ok(Vec::new());
        }
        match self.inner.send_packet(None) {
            Ok(()) => {
                self.flushed = true;
                let mut holder = AVPacket::new();
                self.drain(&mut holder)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// 便捷入口：送入一个包并抽干全部输出包（顺序与输入一致）。
    ///
    /// 一个输入包可能产出零或多个输出包；时间戳由 filter 保持不变。
    /// 通路结束时改用 [`Self::flush_packets`]。
    ///
    /// **输入包在调用后失效**：`send` 把 `packet` 交给 filter，FFmpeg 随后
    /// 将它复用为接收输出的落点，其内容会被覆写。因此 `packet` 只能使用一次
    /// ——重发或在调用后继续读用它都是未定义数据。返回的输出包是本 crate 另建
    /// 的引用拷贝，可以安全留存。
    pub fn filter_packet(&mut self, packet: &mut AVPacket) -> Result<Vec<AVPacket>> {
        self.send_packet(packet)?;
        self.drain(packet)
    }

    /// 过滤后输出流的 codec 参数（以 `avcC` → Annex B 为例：extradata 被移除）。
    ///
    /// 转封装时应以它替换输出流的 codecpar，目标 muxer 才能写出正确的容器头。
    pub fn par_out(&self) -> AVCodecParameters {
        let mut out = AVCodecParameters::new();
        out.copy(&self.inner.par_out());
        out
    }

    /// 抽干当前可得的输出包。
    ///
    /// FFmpeg 依赖"receive 复用被 send 的那个包"，因此这里在收到每个输出后
    /// 通过 `av_packet_ref` 立即产出一份引用拷贝（共享 refcounted 缓冲，
    /// 零拷贝），再继续 receive。
    fn drain(&mut self, holder: &mut AVPacket) -> Result<Vec<AVPacket>> {
        let mut out = Vec::new();
        loop {
            match self.inner.receive_packet(holder) {
                Ok(()) => {
                    let mut owned = AVPacket::new();
                    // SAFETY: `holder` was filled in by `receive_packet` above.
                    // `av_packet_ref` can still fail (OOM); ignoring that would
                    // push an empty packet downstream as if it were real data.
                    let ret = unsafe { ffi::av_packet_ref(owned.as_mut_ptr(), holder.as_ptr()) };
                    if ret < 0 {
                        return Err(RsmediaError::av_error(ret)
                            .with_context("Failed to reference a bitstream-filter output packet"));
                    }
                    out.push(owned);
                }
                // 要更多输入，或已排空。
                Err(RsmpegError::BitstreamDrainError) | Err(RsmpegError::BitstreamFlushedError) => {
                    return Ok(out);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::StreamReader;
    use crate::mux::{Demuxer, Muxer};
    use crate::{DecoderBuilder, MediaType};

    /// MP4 → TS 纯复制（不重编码）：视频包经 `h264_mp4toannexb` 转为 Annex B
    /// 后与音频一起写入 TS；读回 TS 断言视频流可解码（证明 SPS/PPS 内联正确），
    /// 且帧数与源一致。
    #[test]
    fn test_bsf_mp4_to_ts_copy() -> Result<()> {
        let out = crate::test_support::test_output_path("bsf", "mp4_to_ts.ts");
        crate::test_support::remove_test_output(&out);

        let mut demuxer = Demuxer::new("assets/mp4.mp4")?;
        let nb = demuxer.nb_streams();
        let infos: Vec<_> = (0..nb).map(|i| demuxer.stream_info(i).unwrap()).collect();
        let v_in = infos
            .iter()
            .position(|i| i.media_type == MediaType::VIDEO)
            .expect("mp4 asset has a video stream");
        let a_in = infos.iter().position(|i| i.media_type == MediaType::AUDIO);

        let mut muxer = Muxer::new(&out)?;

        // 视频流：经 bsf 转换，输出流参数以 par_out 为准（avcC 已被移除）
        let mut vinfo = infos[v_in].clone();
        let mut bsf = Bsf::new("h264_mp4toannexb", &vinfo.codec_parameters, vinfo.time_base)?;
        vinfo.codec_parameters = bsf.par_out();
        let v_idx = muxer.add_copy_stream(&vinfo)?;
        // 音频流：MP4 的 raw AAC 由 mpegts muxer 自行补 ADTS，无需 bsf
        let a_idx = a_in.map(|i| muxer.add_copy_stream(&infos[i]).unwrap());

        let mut video_packets = 0usize;
        let mut filtered = 0usize;
        while let Some((idx, mut pkt)) = demuxer.demux_packet()? {
            if idx == v_in {
                video_packets += 1;
                for mut out in bsf.filter_packet(&mut pkt)? {
                    filtered += 1;
                    muxer.mux_packet(&mut out, v_idx)?;
                }
            } else if Some(idx) == a_in {
                muxer.mux_packet(&mut pkt, a_idx.unwrap())?;
            }
        }
        for mut out in bsf.flush_packets()? {
            filtered += 1;
            muxer.mux_packet(&mut out, v_idx)?;
        }
        // flush 幂等：二次调用不再向 FFmpeg 发 EOF（会报 EINVAL），返回空列表
        assert!(
            bsf.flush_packets()?.is_empty(),
            "flush_packets must be idempotent"
        );
        muxer.finish()?;
        assert_eq!(video_packets, filtered, "annexb 转换应 1:1 产出包");

        // 读回 TS：视频流必须可解码——Annex B 的 SPS/PPS 内联是解码前提
        let mut reader = StreamReader::new(&out)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut frames = 0usize;
        while decoder.decode_raw(&mut reader)?.is_some() {
            frames += 1;
        }
        drop(reader);
        assert_eq!(frames, video_packets, "TS 中解码出的帧数应与源视频包数一致");

        crate::test_support::remove_test_output(&out);
        Ok(())
    }

    /// [`Muxer::add_copy_stream_with_bsf`] 是同一件事的一站式入口：调用方不持有
    /// [`Bsf`]、不手工搬运 filter 的输出包，产物与上面的手工接线一致。
    ///
    /// 顺带覆盖两处只在库内路径上才会走到的逻辑：输出流 codecpar 取 `par_out()`、
    /// `Muxer::finish` 自动冲刷 filter 里缓冲的尾部包。
    #[test]
    fn test_bsf_via_muxer_copy_stream() -> Result<()> {
        let out = crate::test_support::test_output_path("bsf", "mp4_to_ts_via_muxer.ts");
        crate::test_support::remove_test_output(&out);

        let mut demuxer = Demuxer::new("assets/mp4.mp4")?;
        let nb = demuxer.nb_streams();
        let infos: Vec<_> = (0..nb).map(|i| demuxer.stream_info(i).unwrap()).collect();
        let v_in = infos
            .iter()
            .position(|i| i.media_type == MediaType::VIDEO)
            .expect("mp4 asset has a video stream");
        let a_in = infos.iter().position(|i| i.media_type == MediaType::AUDIO);

        let mut muxer = Muxer::new(&out)?;
        let v_idx = muxer.add_copy_stream_with_bsf(&infos[v_in], "h264_mp4toannexb")?;
        let a_idx = a_in.map(|i| muxer.add_copy_stream(&infos[i])).transpose()?;

        let mut video_packets = 0usize;
        while let Some((idx, mut pkt)) = demuxer.demux_packet()? {
            if idx == v_in {
                video_packets += 1;
                muxer.mux_packet(&mut pkt, v_idx)?;
            } else if Some(idx) == a_in {
                muxer.mux_packet(&mut pkt, a_idx.expect("audio stream registered"))?;
            }
        }
        muxer.finish()?;

        // 读回 TS：视频流必须可解码（Annex B 的 SPS/PPS 内联是解码前提），
        // 帧数与源视频包数一致。
        let mut reader = StreamReader::new(&out)?;
        let mut decoder = DecoderBuilder::new(MediaType::VIDEO).build_from_reader(&reader)?;
        let mut frames = 0usize;
        while decoder.decode_raw(&mut reader)?.is_some() {
            frames += 1;
        }
        drop(reader);
        assert_eq!(frames, video_packets, "TS 中解码出的帧数应与源视频包数一致");

        crate::test_support::remove_test_output(&out);
        Ok(())
    }

    /// list() 应包含常用 filter。
    #[test]
    fn test_bsf_list_contains_common_filters() {
        let list = Bsf::list();
        assert!(list.contains(&"h264_mp4toannexb".to_string()), "{list:?}");
        assert!(list.contains(&"aac_adtstoasc".to_string()), "{list:?}");
    }

    /// 不存在的 filter 名报 `Unsupported`（本构建没编入它 ⇒ 调用方改自己的
    /// 调用改不动，只能换构建或换一个 filter 名），而不是 panic、也不是
    /// `InvalidConfig`。
    #[test]
    fn test_bsf_unknown_name_rejected() {
        let codecpar = AVCodecParameters::new();
        let err = Bsf::new("no_such_bsf", &codecpar, Rational::new(1, 30).unwrap())
            .expect_err("unknown filter must be rejected");
        assert!(err.is_unsupported(), "{err}");
        assert!(!err.is_invalid_config(), "{err}");
        assert!(err.to_string().contains("no_such_bsf"), "{err}");
    }

    /// filter 的私有选项必须在 `av_bsf_init` 之前写进 `priv_data` —— 否则
    /// `h264_metadata` / `hevc_metadata` / `setts` 这类"行为完全由选项描述"的
    /// filter 根本不可用（能创建、不能配置）。
    #[test]
    fn test_bsf_private_options_are_applied() {
        let codecpar = AVCodecParameters::new();
        let time_base = Rational::new(1, 30).unwrap();

        // 认得的选项：正常创建。
        let mut options = AVOptions::new();
        options.set("ts", "0");
        match Bsf::new_with_options("setts", &codecpar, time_base, options) {
            Ok(bsf) => assert_eq!(bsf.name(), "setts"),
            Err(e) if e.is_unsupported() => {
                println!("SKIP: 'setts' is not in this FFmpeg build");
                return;
            }
            Err(e) => panic!("a recognised option must not fail: {e}"),
        }

        // 认不得的选项：报 `InvalidConfig` 并点出键名，而不是静默丢掉。
        let mut bad = AVOptions::new();
        bad.set("definitely_not_an_option", "1");
        let err = Bsf::new_with_options("setts", &codecpar, time_base, bad)
            .expect_err("unknown private option must be rejected");
        assert!(err.is_invalid_config(), "{err}");
        assert!(
            err.to_string().contains("definitely_not_an_option"),
            "the error must name the offending option: {err}"
        );
    }
}
