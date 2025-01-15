fn main() {
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
}