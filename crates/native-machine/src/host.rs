//! Read-only host capability inspection.

pub fn print_report() {
    println!("architecture: {}", std::env::consts::ARCH);
    println!("operating_system: {}", std::env::consts::OS);
    println!("pointer_width: {}", usize::BITS);
    println!("built_in_kernels: scalar");
    let features = crate::cpu::features();
    println!(
        "cpu_features: avx2={} neon={}",
        features.avx2, features.neon
    );
    println!("runtime: ready");
}
