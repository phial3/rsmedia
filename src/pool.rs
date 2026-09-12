//! 引用计数缓冲池（`AVBufferPool`）的安全封装。
//!
//! 池化的动机：热路径（逐帧缩放/编码）里反复为同样尺寸的帧做分配/释放，
//! 每帧一次大块 `malloc` 的代价在高帧率下不可忽略。`AVBufferPool` 把这些
//! 缓冲做成"归还即复用"：[`BufferPool::get`] 优先取回上一个归还的缓冲，
//! 取不到才真正分配；缓冲的所有引用归零时自动回到池里。
//!
//! 分配语义：**首次**分配一律清零（`av_buffer_allocz`，与
//! `av_frame_get_buffer` 的缓冲清零行为对齐）；FFmpeg 的池在**复用**时
//! 不会重新清零——需要"每次取用都是零"的调用方（如帧组装）必须在使用前
//! 自行清零，`crate::scale` 的池化帧组装正是这样做的。
//!
//! # 用法
//!
//! ```no_run
//! use rsmedia::BufferPool;
//!
//! // 为 1920x1080 的 YUV420P 帧建池（尺寸通常由 imgutils 计算得出）
//! let mut pool = BufferPool::new(3_138_048)?;
//! let buf = pool.get()?;                  // refcount = 1
//! assert_eq!(buf.size, 3_138_048);
//! drop(buf);                              // 归还池中，下次 get 复用
//! assert_eq!(pool.allocations(), 1);      // 复用不再计数
//! # Ok::<(), rsmedia::RsmediaError>(())
//! ```

use crate::error::{Result, RsmediaError};
use rsmpeg::avutil::AVBufferRef;
use rsmpeg::ffi;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

/// 池的分配统计：`allocations` 记录**真实分配**次数（归还后的复用不计入）。
/// 由 `av_buffer_pool_init2` 的 opaque 指针承载，C 回调直接更新。
struct BufferPoolCounter(AtomicU64);

/// 池的分配回调：清零分配（与 `av_frame_get_buffer` 的缓冲清零语义一致），
/// 并递增统计计数器。
///
/// # Safety
///
/// `opaque` 必须是 [`BufferPool::new`] 放入的 `Box<BufferPoolCounter>` 裸指针；
/// FFmpeg 保证回调只在池存活期间被调用，而池存活期间 opaque 由
/// `pool_free` 负责释放。
unsafe extern "C" fn pool_alloc(opaque: *mut c_void, size: usize) -> *mut ffi::AVBufferRef {
    // Safety: 见函数文档。
    let counter = unsafe { &*(opaque as *const BufferPoolCounter) };
    counter.0.fetch_add(1, Ordering::Relaxed);
    // Safety: av_buffer_allocz 只在 OOM 时返回 NULL；NULL 由 get() 侧统一报错。
    unsafe { ffi::av_buffer_allocz(size) }
}

/// 池析构回调：释放 opaque 承载的计数器。
///
/// # Safety
///
/// `opaque` 必须是 [`BufferPool::new`] 放入的 `Box<BufferPoolCounter>` 裸指针，
/// 且此前未被释放（FFmpeg 保证每个池只调用一次 `pool_free`）。
unsafe extern "C" fn pool_free(opaque: *mut c_void) {
    // Safety: 见函数文档。
    drop(unsafe { Box::from_raw(opaque as *mut BufferPoolCounter) });
}

/// 一个引用计数缓冲池，持有 `buffer_size` 尺寸的复用缓冲。
///
/// 线程模型：FFmpeg 文档称 `av_buffer_pool_get` 可多线程并发调用，但
/// rsmedia 将 [`BufferPool`] 设计为被 [`Scaler`](crate::Scaler) 独占持有
/// （`&mut self` 访问），因此只实现 `Send`、不实现 `Sync`。
pub struct BufferPool {
    ptr: *mut ffi::AVBufferPool,
    counter: *mut BufferPoolCounter,
    buffer_size: usize,
}

// Safety: AVBufferPool 自身由 C 侧管理生命周期；本句柄只拥有裸指针，
// 所有权整体随 BufferPool 移动是安全的（内部状态不因移动失效）。
unsafe impl Send for BufferPool {}

impl BufferPool {
    /// 创建一个缓冲尺寸固定为 `buffer_size` 字节的池。
    ///
    /// 尺寸必须与后续写入的帧布局匹配（见
    /// [`crate::scale`] 的池化帧构造——几何变化时 `Scaler` 会重建池）。
    pub fn new(buffer_size: usize) -> Result<Self> {
        if buffer_size == 0 {
            return Err(RsmediaError::invalid_config(
                "frame pool buffer size must be positive",
            ));
        }
        let counter = Box::into_raw(Box::new(BufferPoolCounter(AtomicU64::new(0))));
        // Safety: counter 是有效的 Box 裸指针，由 pool_free 在池析构时回收。
        let ptr = unsafe {
            ffi::av_buffer_pool_init2(
                buffer_size,
                counter as *mut c_void,
                Some(pool_alloc),
                Some(pool_free),
            )
        };
        if ptr.is_null() {
            // Safety: 尚未移交给 FFmpeg，由这里回收。
            drop(unsafe { Box::from_raw(counter) });
            return Err(RsmediaError::custom(format!(
                "av_buffer_pool_init2 failed for size {buffer_size}"
            )));
        }
        Ok(Self {
            ptr,
            counter,
            buffer_size,
        })
    }

    /// 从池中取一个缓冲：优先复用已归还的缓冲，取不到才新分配。
    ///
    /// 返回的 [`AVBufferRef`] 引用计数为 1；所有引用归零时缓冲自动归还池
    /// （若池已析构则直接释放）。
    pub fn get(&mut self) -> Result<AVBufferRef> {
        // Safety: self.ptr 在 Drop 前始终指向有效池。
        let raw = unsafe { ffi::av_buffer_pool_get(self.ptr) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            RsmediaError::custom(format!(
                "av_buffer_pool_get failed for size {}",
                self.buffer_size
            ))
        })?;
        // Safety: raw 来自 FFmpeg 的成功返回，引用计数已置 1。
        Ok(unsafe { AVBufferRef::from_raw(raw) })
    }

    /// 池内缓冲的固定尺寸（字节）。
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// 池至今**真实分配**的缓冲数（归还复用不计数）。
    ///
    /// 稳态流水线里该值应在若干帧后停止增长并远小于总帧数——这是池化
    /// 生效的直接证据。
    pub fn allocations(&self) -> u64 {
        // Safety: counter 在 Drop 前始终有效（释放由 pool_free 承担，
        // 而 pool_free 只在池析构时调用，此时 self 已不可用）。
        unsafe { (*self.counter).0.load(Ordering::Relaxed) }
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("buffer_size", &self.buffer_size)
            .field("allocations", &self.allocations())
            .finish()
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        // Safety: ptr 由 init2 成功返回且未被重复释放；
        // uninit 将池标记为待析构，未归还的缓冲释放时不会回到池里。
        unsafe { ffi::av_buffer_pool_uninit(&mut self.ptr) };
        self.ptr = std::ptr::null_mut();
    }
}

/// 供 `c_int` 侧回调/常量使用的别名占位（保持与本 crate 其余 FFI 代码一致）。
const _: () = {
    // 编译期确认 c_int 与 i32 同宽，池内长度换算安全。
    assert!(std::mem::size_of::<c_int>() == std::mem::size_of::<i32>());
};

#[cfg(test)]
mod tests {
    use super::*;

    /// 基本生命周期：取两个缓冲 → refcount 各自独立 → 归还后复用。
    #[test]
    fn test_frame_pool_recycle_and_counter() -> Result<()> {
        let mut pool = BufferPool::new(1024)?;
        assert_eq!(pool.buffer_size(), 1024);

        let a = pool.get()?;
        assert_eq!(a.size, 1024);
        assert_eq!(a.get_ref_count(), 1);
        assert_eq!(pool.allocations(), 1);

        let b = pool.get()?;
        assert_eq!(pool.allocations(), 2, "两个缓冲都在持有中，必须新分配");
        assert_ne!(a.data, b.data, "不同的缓冲应有不同的数据指针");

        // 归还 a；下一次 get 必须复用它（同指针、不再计数）。
        let a_ptr = a.data;
        drop(a);
        let c = pool.get()?;
        assert_eq!(c.data, a_ptr, "归还的缓冲应被池复用");
        assert_eq!(pool.allocations(), 2);

        // 引用计数：clone 后 +1，归位后恢复。
        let cloned = c.clone();
        assert_eq!(c.get_ref_count(), 2);
        drop(cloned);
        assert_eq!(c.get_ref_count(), 1);
        Ok(())
    }

    /// 缓冲是清零分配的（与 `av_frame_get_buffer` 的语义一致）。
    #[test]
    fn test_frame_pool_zeroed_allocation() -> Result<()> {
        let mut pool = BufferPool::new(4096)?;
        let buf = pool.get()?;
        let slice = unsafe { std::slice::from_raw_parts(buf.data, buf.size) };
        assert!(slice.iter().all(|&b| b == 0), "池缓冲应清零分配");
        Ok(())
    }

    /// 池析构后，未归还的缓冲直接释放而不是归还（由 Drop 语义保证，
    /// 这里验证 Drop 不 panic 且统计在 Drop 前可读）。
    #[test]
    fn test_frame_pool_drop_with_outstanding_buffers() {
        let mut pool = BufferPool::new(512).expect("pool");
        let buf = pool.get().expect("buffer");
        assert_eq!(pool.allocations(), 1);
        drop(pool); // buf 仍活着，但池已标记析构
        drop(buf); // 不回到池（池已亡），直接释放
    }

    /// 缓冲数据指针按 FFmpeg 的 `av_malloc` 保证至少 32 字节对齐
    /// （多数平台实际为 64；32 是跨平台可断言的下限）——这是
    /// `av_image_fill_arrays` 按对齐排布平面指针、编码器 SIMD 读取的前提。
    #[test]
    fn test_buffer_pool_alignment_at_least_32() -> Result<()> {
        for _ in 0..8 {
            let mut pool = BufferPool::new(4096)?;
            let buf = pool.get()?;
            let addr = buf.data as usize;
            assert_eq!(
                addr % 32,
                0,
                "buffer data at {addr:#x} is not 32-byte aligned"
            );
        }
        Ok(())
    }

    /// 压力测试：128 个缓冲全部持有（指针两两不同、计数正确），全部
    /// 归还后再取 64 个——必须全部复用（计数不增长）、指针两两不同且
    /// 都落在原始分配集合内。
    #[test]
    fn test_buffer_pool_churn_uniqueness_and_reuse() -> Result<()> {
        let mut pool = BufferPool::new(2048)?;
        let held: Vec<_> = (0..128).map(|_| pool.get().expect("get")).collect();
        assert_eq!(pool.allocations(), 128);

        let originals: std::collections::HashSet<usize> =
            held.iter().map(|b| b.data as usize).collect();
        assert_eq!(originals.len(), 128, "持有中的缓冲指针必须两两不同");

        // 全部归还，然后重新取 64 个：全部来自池（复用），不新分配。
        drop(held);
        let reacquired: Vec<_> = (0..64).map(|_| pool.get().expect("get")).collect();
        assert_eq!(pool.allocations(), 128, "复用不应计数");
        let mut ra: Vec<usize> = reacquired.iter().map(|b| b.data as usize).collect();
        ra.sort_unstable();
        ra.dedup();
        assert_eq!(ra.len(), 64, "复用的缓冲指针必须两两不同");
        for p in &ra {
            assert!(originals.contains(p), "{p:#x} 不是原始分配的缓冲");
        }
        Ok(())
    }

    /// 零尺寸构造报 invalid_config，而不是交给 FFmpeg 失败。
    #[test]
    fn test_frame_pool_rejects_zero_size() {
        assert!(BufferPool::new(0).is_err());
    }
}
