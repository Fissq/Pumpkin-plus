//! Byte counters for the chunk IO path.
//!
//! The atomics are compiled in only for unit tests or with the
//! `io-bench-counters` feature; without them `reset`/`snapshot` are no-ops
//! and call sites are compiled out, so release behaviour is unchanged.
//! Used by `benches/chunk_io.rs` to report written/read bytes per scenario.

#[cfg(any(test, feature = "io-bench-counters"))]
pub(crate) mod imp {
    use std::sync::atomic::AtomicU64;

    pub static BYTES_WRITTEN: AtomicU64 = AtomicU64::new(0);
    pub static BYTES_READ: AtomicU64 = AtomicU64::new(0);
}

/// Record `n` bytes written to disk by the chunk IO path.
#[cfg(any(test, feature = "io-bench-counters"))]
#[inline]
pub(crate) fn add_written(n: u64) {
    imp::BYTES_WRITTEN.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Record `n` bytes read from disk by the chunk IO path.
#[cfg(any(test, feature = "io-bench-counters"))]
#[inline]
pub(crate) fn add_read(n: u64) {
    imp::BYTES_READ.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Reset both counters to zero.
#[cfg(any(test, feature = "io-bench-counters"))]
pub fn reset() {
    use std::sync::atomic::Ordering;
    imp::BYTES_WRITTEN.store(0, Ordering::Relaxed);
    imp::BYTES_READ.store(0, Ordering::Relaxed);
}

/// No-op: the counters are compiled out (enable `io-bench-counters`).
#[cfg(not(any(test, feature = "io-bench-counters")))]
pub const fn reset() {}

/// `(written, read)` bytes since the last [`reset`].
#[cfg(any(test, feature = "io-bench-counters"))]
#[must_use]
pub fn snapshot() -> Option<(u64, u64)> {
    use std::sync::atomic::Ordering;
    Some((
        imp::BYTES_WRITTEN.load(Ordering::Relaxed),
        imp::BYTES_READ.load(Ordering::Relaxed),
    ))
}

/// `None`: the counters are compiled out (enable `io-bench-counters`).
#[cfg(not(any(test, feature = "io-bench-counters")))]
#[must_use]
pub const fn snapshot() -> Option<(u64, u64)> {
    None
}
