//! Transport-bound throughput accounting.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportTier {
    Memory,
    Storage,
}

impl std::fmt::Display for TransportTier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory => formatter.write_str("memory"),
            Self::Storage => formatter.write_str("storage"),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum TransportError {
    #[error("transport bandwidth must be finite and greater than zero")]
    Bandwidth,
    #[error("transport bytes per pass must be greater than zero")]
    BytesPerPass,
    #[error("tokens per pass must be greater than zero")]
    TokensPerPass,
}

/// A bandwidth ceiling for one execution pass.
///
/// `bytes_per_pass` is a declared logical lower bound. Hardware traffic and
/// sustained bandwidth remain measurements, so this reports a ceiling rather
/// than predicted performance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportBound {
    pub tier: TransportTier,
    pub bandwidth_bytes_per_second: f64,
    pub bytes_per_pass: u64,
    pub tokens_per_pass: u32,
}

impl TransportBound {
    pub fn new(
        tier: TransportTier,
        bandwidth_bytes_per_second: f64,
        bytes_per_pass: u64,
        tokens_per_pass: u32,
    ) -> Result<Self, TransportError> {
        if !bandwidth_bytes_per_second.is_finite() || bandwidth_bytes_per_second <= 0.0 {
            return Err(TransportError::Bandwidth);
        }
        if bytes_per_pass == 0 {
            return Err(TransportError::BytesPerPass);
        }
        if tokens_per_pass == 0 {
            return Err(TransportError::TokensPerPass);
        }
        Ok(Self {
            tier,
            bandwidth_bytes_per_second,
            bytes_per_pass,
            tokens_per_pass,
        })
    }

    pub fn tokens_per_second_ceiling(self) -> f64 {
        f64::from(self.tokens_per_pass) * self.bandwidth_bytes_per_second
            / self.bytes_per_pass as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_bandwidth_ceiling() {
        let bound = TransportBound::new(TransportTier::Memory, 614_000_000_000.0, 7_500_000_000, 1)
            .expect("bound is valid");
        assert!((bound.tokens_per_second_ceiling() - 81.866_666_666_7).abs() < 1e-9);
    }

    #[test]
    fn rejects_invalid_contracts() {
        assert_eq!(
            TransportBound::new(TransportTier::Storage, f64::NAN, 1, 1),
            Err(TransportError::Bandwidth)
        );
        assert_eq!(
            TransportBound::new(TransportTier::Storage, 1.0, 0, 1),
            Err(TransportError::BytesPerPass)
        );
        assert_eq!(
            TransportBound::new(TransportTier::Storage, 1.0, 1, 0),
            Err(TransportError::TokensPerPass)
        );
    }
}
