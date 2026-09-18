//! 编解码器上下文与滤镜图共用的推进状态。

/// FFmpeg 处理单元（`AVCodecContext` / `AVFilterGraph`）的推进阶段。
///
/// 两者走的是同一套生命周期：接收输入 → 收到 EOF 后排空内部缓冲 → 不再产出，
/// 因此用同一个类型描述。[`Decoder`](crate::Decoder)、[`Encoder`](crate::Encoder)
/// 与 [`FilterGraph`](crate::FilterGraph) 共用这里的谓词——「一个事实只有一处定义」，
/// 三者对 drained / flushed 的判断不会各自漂移。
///
/// - `Normal`：正常接收输入。
/// - `Drained`：EOF 已送出（`send_frame(None)`、给 buffersrc 推 NULL），正在排空内部
///   缓冲，不再接收新输入。
/// - `Flushed`：排空完成，所有输出已取出，此后不会再产出。
///
/// 阶段只由**真正送出 EOF** 的那一步推进到 `Drained`：`EAGAIN` 在流中段同样会出现
/// （编码器的 B 帧/lookahead 缓冲、解码器缺包），把它记成 `Drained` 会让
/// [`is_drained`](Self::is_drained) 在流中段就永久为真，排空循环随之空转。
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ProcessState {
    /// 正常接收输入。
    Normal,
    /// EOF 已送出，正在排空内部缓冲。
    Drained,
    /// 排空完成，不会再产出。
    Flushed,
}

impl ProcessState {
    /// 是否仍在 `Normal` 阶段（可以继续接收输入）。
    pub(crate) fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }

    /// 是否处于 `Drained` 阶段（EOF 已送出，仍在产出缓冲数据）。
    pub(crate) fn is_drained(self) -> bool {
        matches!(self, Self::Drained)
    }

    /// 是否处于 `Flushed` 阶段（排空完成，不会再产出）。
    pub(crate) fn is_flushed(self) -> bool {
        matches!(self, Self::Flushed)
    }
}

#[cfg(test)]
mod tests {
    use super::ProcessState;

    /// 三个谓词互斥且完备：每个状态恰好命中一个谓词；且谓词与变体一一对应
    /// （后者防止谓词之间写反了顺序却仍然"恰好命中一个"）。
    #[test]
    fn predicates_are_exclusive_and_match_their_variant() {
        for state in [
            ProcessState::Normal,
            ProcessState::Drained,
            ProcessState::Flushed,
        ] {
            let hits = [state.is_normal(), state.is_drained(), state.is_flushed()]
                .into_iter()
                .filter(|hit| *hit)
                .count();
            assert_eq!(hits, 1, "{state:?} 应恰好命中一个谓词，实际命中 {hits} 个");
        }

        assert!(ProcessState::Normal.is_normal());
        assert!(ProcessState::Drained.is_drained());
        assert!(ProcessState::Flushed.is_flushed());
    }
}
