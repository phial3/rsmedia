//! Test-support helpers compiled only for library unit tests (`#[cfg(test)]`).

/// 串行化所有触碰进程级 `HW_CTX_CACHE` 的测试。
///
/// 缓存是**进程级**静态（[`crate::hwaccel::release_unused_hw_contexts`] 的复用池），
/// 而 lib 测试在同一进程里并行跑：任何创建或持有硬件上下文的测试（无论写在哪个模块）
/// 都必须先拿这把锁，否则按引用计数断言的缓存释放测试会被别的测试正好持有的上下文
/// 干扰（`strong_count > 1` → 该条目"仍在使用"，不会被释放）。
pub fn hw_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static HW_CACHE_TEST_LOCK: once_cell::sync::Lazy<std::sync::Mutex<()>> =
        once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));
    // 某个测试 panic 后锁会被标记为 poisoned；这里恢复内部值继续用（`into_inner`）——
    // 被破坏的只是那个测试留下的状态，与本测试的断言无关，没必要让后续测试连锁失败。
    HW_CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// # Arguments
/// * `category` - The subdirectory name (e.g., "encode", "mux", "pcm")
/// * `filename` - The output filename
pub fn test_output_path(category: &str, filename: &str) -> std::path::PathBuf {
    const CRATE_NAME: &str = env!("CARGO_PKG_HOMEPAGE");
    let output_dir = std::path::PathBuf::from("output").join(category);
    std::fs::create_dir_all(&output_dir).ok();
    println!("output_dir: {}/{:?}", CRATE_NAME, output_dir);
    output_dir.join(filename)
}

/// Removes a test output file if it exists.
/// Most unit-test modules use this, but the API is kept generally usable.
pub fn remove_test_output(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}
