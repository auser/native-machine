//! One-time CPU capability detection for immutable runtime dispatch.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub neon: bool,
    pub fma: bool,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

pub fn features() -> CpuFeatures {
    *FEATURES.get_or_init(detect)
}

fn detect() -> CpuFeatures {
    CpuFeatures {
        avx2: detect_avx2(),
        neon: detect_neon(),
        fma: detect_fma(),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx2() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn detect_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_neon() -> bool {
    false
}

/// FMA3 (x86) only; AArch64 fused multiply-add is covered by the NEON bit.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_fma() -> bool {
    std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_fma() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_stable() {
        assert_eq!(features(), features());
    }
}
