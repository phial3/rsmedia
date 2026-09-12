//! FFmpeg 内置 MD5 摘要（`AVMD5`）。
//!
//! 典型用途：配合 `framehash`/`framemd5` muxer 做逐帧校验，或对流/文件做
//! 快速一致性比对（非密码学用途——AVMD5 即普通 MD5）。
//!
//! # 用法
//!
//! ```
//! use rsmedia::md5::Md5;
//!
//! let hex = Md5::hex(b"abc");
//! assert_eq!(hex, "900150983cd24fb0d6963f7d28e17f72");
//!
//! // 流式更新
//! let mut md5 = Md5::new();
//! md5.update(b"ab");
//! md5.update(b"c");
//! assert_eq!(md5.to_hex(), hex);
//! ```

use rsmpeg::avutil::AVMD5;

/// 流式 MD5 摘要器。
pub struct Md5 {
    inner: AVMD5,
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    /// 创建并初始化摘要器。
    pub fn new() -> Self {
        let mut inner = AVMD5::new();
        inner.init();
        Self { inner }
    }

    /// 追加数据（可分任意多次调用）。
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// 结束并取出 16 字节摘要。
    pub fn finalize(mut self) -> [u8; 16] {
        self.inner.finalize()
    }

    /// 结束并取出小写十六进制摘要（32 字符）。
    pub fn to_hex(self) -> String {
        self.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// 一次性计算 `data` 的 MD5 摘要。
    pub fn sum(data: &[u8]) -> [u8; 16] {
        AVMD5::sum(data)
    }

    /// 一次性计算 `data` 的 MD5 小写十六进制摘要（32 字符）。
    pub fn hex(data: &[u8]) -> String {
        Self::sum(data).iter().map(|b| format!("{b:02x}")).collect()
    }
}
