fn main() {
    println!("cargo:rerun-if-changed=native/kernels.cc");
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(2)
        .flag("-fno-fast-math")
        .flag("-ffp-contract=on")
        .file("native/kernels.cc")
        .compile("reference_math");
}
