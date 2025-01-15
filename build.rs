fn main() {
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    #[cfg(target_arch = "x86_64")]
    {
        // 仅在 x86_64 架构下添加特定链接配置
        println!("cargo:rustc-link-search=/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=ld-linux-x86-64");
    }
}
