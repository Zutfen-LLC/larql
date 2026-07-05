//! Persistent device-resident weight cache — the first slice of the
//! per-projection round-trip collapse.
//!
//! Each native CUDA quant/dense projection launcher historically re-uploaded
//! its full weight matrix via `CudaStream::clone_htod` on **every** call. For
//! decode that is the dominant host→device (htod) cost: the same per-layer
//! `wq`/`wk`/`wv`/`wo`/`gate`/`up`/`down` weight matrices (GB-scale on a 4-26B
//! model) are re-uploaded for every generated token, even though the weights
//! never change.
//!
//! This module uploads each immutable weight matrix once, keyed on the host
//! slice's `(pointer, byte length)`, and reuses the device buffer (`Arc<CudaSlice<T>>`)
//! on every subsequent call with the same slice address. Only **weights** are
//! cached — activations (`x`, projections, KV rows, the per-token Q8
//! quantization of the input) are uploaded fresh each call, exactly as before.
//!
//! ## Cache-key soundness
//!
//! LARQL weights are zero-copy `mmap`'d slices that keep a stable address for
//! the lifetime of the loaded vindex (see AGENTS.md: "Storage is mmap-first").
//! Within one `CudaBackend` instance serving weights from **one vindex** the
//! same weight matrix is always the same `(ptr, len)` pair, so the cache lookup
//! is exact — a hit returns the same bytes that would have been uploaded,
//! bit-for-bit. That is the normal case (a backend is bound to one vindex for
//! its lifetime), so the cache is unconditionally correct for it.
//!
//! ## ABA scope (and its limit)
//!
//! Pointer-identity keying can't detect a *different* model's weights mapped
//! at a recycled `(ptr, len)` (a freed+reallocated mmap address). The cache is
//! flushed at each `DecodeBackend::reset_kv_cache` (the generation boundary) so
//! decode/prefill backends can't carry a stale buffer into a later generation.
//! **That flush does NOT cover the browse path** — gate/lm-head KNN
//! (`f16_gemv`/`f32_gemv` from `larql-vindex` DESCRIBE/WALK/SELECT) never resets
//! the KV cache, so a single backend reused across two vindex loads for browse
//! could read the prior model's cached weights at a recycled address. Reusing a
//! backend across vindex weight sets is therefore the caller's responsibility:
//! call `CudaBackend::flush_weight_cache()` at the vindex-rebind boundary. The
//! per-`reset_kv_cache` flush still captures the dominant reuse win (weights
//! uploaded during prefill's projections are reused across every decode token
//! of that generation).
//!
//! ## What this is NOT
//!
//! This collapses only the **weight** htod round-trips. Per-projection
//! activation uploads + device→host (dtoh) readbacks remain (the
//! host-orchestrated pipeline still reads each projection back to host for the
//! next elementwise op). Folding those into a single-command-buffer fused
//! pipeline (mirroring Metal's shape) is the follow-on work — this slice
//! removes the largest, most repeated transfer first.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaSlice, CudaStream, DriverError};

/// Cache key: the host slice's base pointer + element count. The two together
/// uniquely identify an immutable weight matrix within one backend instance.
/// `len` is stored in **elements** (not bytes) so a `&[u8]` and a `&[f32]`
/// view of the same backing memory can't collide across the typed maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct WeightKey {
    ptr: usize,
    len: usize,
}

impl WeightKey {
    /// Key for a byte slice (quant weights: Q4_K / Q6_K / Q4_0, f16 byte
    /// slices). Empty slices are never passed (upstream validates weight
    /// lengths before launch), but an empty key still hashes safely.
    pub(crate) fn of_bytes(b: &[u8]) -> Self {
        Self {
            ptr: b.as_ptr() as usize,
            len: b.len(),
        }
    }

    /// Key for an f32 slice (dense f32 weight matrices).
    pub(crate) fn of_f32(f: &[f32]) -> Self {
        Self {
            ptr: f.as_ptr() as usize,
            len: f.len(),
        }
    }

    /// Element count used for the key (bytes for the byte map, f32s for the
    /// float map). Exposed for tests.
    #[cfg(test)]
    pub(crate) fn len(self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Cumulative hit/miss counters, surfaced via [`WeightCache::stats`] for tests
/// and debugging (a runtime-gated test asserts a second launch with the same
/// weight slice registers a hit and skips the upload). Test-only snapshot —
/// the live counters are the `AtomicU64`s on [`WeightCache`]; this struct only
/// exists to return a coherent snapshot from `stats()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub bytes_hits: u64,
    pub bytes_misses: u64,
    pub float_hits: u64,
    pub float_misses: u64,
    /// Cumulative bytes uploaded to the device across all misses (the raw
    /// transfer volume, for the pipeline bench's `htod_bytes/token` metric).
    pub bytes_uploaded: u64,
}

/// Persistent device-resident weight cache. Two typed maps (`u8` for quant/f16
/// byte weights, `f32` for dense weight matrices) keyed on [`WeightKey`].
/// `Arc<CudaSlice<T>>` values so a cached handle clones out of the lock cheaply
/// and outlives the per-launch scope (each `CudaSlice<T>` already owns an
/// `Arc<CudaStream>` internally, mirroring the KV cache's storage discipline).
///
/// Counters are `AtomicU64` (not `Mutex<CacheStats>`) so the per-launch hit/miss
/// bookkeeping adds no lock acquisition to the decode hot path — the counters
/// are only ever read under `#[cfg(test)]`, but a write-only-under-lock counter
/// would still pay the lock cost on every one of the ~7-weights×N-layers
/// lookups per token.
#[derive(Debug, Default)]
pub(crate) struct WeightCache {
    bytes: Mutex<HashMap<WeightKey, Arc<CudaSlice<u8>>>>,
    floats: Mutex<HashMap<WeightKey, Arc<CudaSlice<f32>>>>,
    bytes_hits: AtomicU64,
    bytes_misses: AtomicU64,
    float_hits: AtomicU64,
    float_misses: AtomicU64,
    bytes_uploaded: AtomicU64,
}

impl WeightCache {
    /// Upload `bytes` once and cache by `(ptr, len)`; reuse on subsequent
    /// calls with the same slice address. Only call this for **immutable
    /// weight** slices — per-token activations must keep the fresh
    /// `clone_htod` path. Returns `DriverError` (the caller maps it into its
    /// kernel-specific `RuntimeError` context).
    ///
    /// On a miss the upload happens **while still holding the map lock**: a
    /// concurrent miss of the same key then finds the entry populated instead
    /// of racing a duplicate upload (and a transient second device copy).
    /// Decode is single-stream today so this is latent-correctness, but it
    /// keeps the de-dup sound if concurrency is ever added.
    pub(crate) fn get_or_upload_bytes(
        &self,
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, DriverError> {
        let key = WeightKey::of_bytes(bytes);
        let dev = {
            let mut guard = lock_or_recover(&self.bytes);
            if let Some(dev) = guard.get(&key) {
                let dev = Arc::clone(dev);
                self.bytes_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(dev);
            }
            let dev = Arc::new(stream.clone_htod(bytes)?);
            guard.insert(key, Arc::clone(&dev));
            self.bytes_uploaded
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            dev
        };
        self.bytes_misses.fetch_add(1, Ordering::Relaxed);
        Ok(dev)
    }

    /// f32 twin of [`get_or_upload_bytes`] for dense weight matrices.
    pub(crate) fn get_or_upload_f32(
        &self,
        stream: &Arc<CudaStream>,
        floats: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, DriverError> {
        let key = WeightKey::of_f32(floats);
        let dev = {
            let mut guard = lock_or_recover(&self.floats);
            if let Some(dev) = guard.get(&key) {
                let dev = Arc::clone(dev);
                self.float_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(dev);
            }
            let dev = Arc::new(stream.clone_htod(floats)?);
            guard.insert(key, Arc::clone(&dev));
            self.bytes_uploaded
                .fetch_add((floats.len() * 4) as u64, Ordering::Relaxed);
            dev
        };
        self.float_misses.fetch_add(1, Ordering::Relaxed);
        Ok(dev)
    }

    /// Snapshot the cumulative hit/miss counters (test/diagnostic/bench).
    /// Relaxed loads are fine — these are observability counters, not
    /// synchronisation.
    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            bytes_hits: self.bytes_hits.load(Ordering::Relaxed),
            bytes_misses: self.bytes_misses.load(Ordering::Relaxed),
            float_hits: self.float_hits.load(Ordering::Relaxed),
            float_misses: self.float_misses.load(Ordering::Relaxed),
            bytes_uploaded: self.bytes_uploaded.load(Ordering::Relaxed),
        }
    }

    /// Drop every cached device buffer. The counters are cumulative across the
    /// backend's lifetime (not reset by flush). See [`CudaBackend`] /
    /// `reset_kv_cache` for when this is invoked and the ABA scope.
    pub(crate) fn flush(&self) {
        lock_or_recover(&self.bytes).clear();
        lock_or_recover(&self.floats).clear();
    }

    /// Number of cached byte-weight buffers (test/diagnostic).
    #[cfg(test)]
    pub(crate) fn cached_byte_count(&self) -> usize {
        lock_or_recover(&self.bytes).len()
    }

    /// Number of cached f32-weight buffers (test/diagnostic).
    #[cfg(test)]
    pub(crate) fn cached_float_count(&self) -> usize {
        lock_or_recover(&self.floats).len()
    }
}

/// Take a mutex guard, recovering from poisoning the same way the KV cache
/// does (a poisoned lock means a prior holder panicked, not that the cache
/// data is corrupt). Keeps the documented "fall back instead of aborting"
/// contract.
fn lock_or_recover<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_key_distinguishes_distinct_slices() {
        let a = [0u8; 16];
        let b = [0u8; 16];
        assert_ne!(WeightKey::of_bytes(&a), WeightKey::of_bytes(&b));
    }

    #[test]
    fn byte_key_stable_for_same_slice() {
        let a = [0u8; 16];
        assert_eq!(WeightKey::of_bytes(&a), WeightKey::of_bytes(&a));
    }

    #[test]
    fn byte_key_distinguishes_by_length() {
        // Same base pointer (a subslice of `buf`) but different length must
        // not collide — a recycled-address ABA between models would otherwise
        // serve stale data.
        let buf = [0u8; 64];
        let full = WeightKey::of_bytes(&buf);
        let half = WeightKey::of_bytes(&buf[..32]);
        assert_ne!(full, half);
        assert_eq!(full.len(), 64);
        assert_eq!(half.len(), 32);
    }

    #[test]
    fn f32_key_uses_element_count_not_byte_count() {
        let f = [0.0f32; 8];
        let k = WeightKey::of_f32(&f);
        // Element count, not byte count, so the two typed maps can't cross-
        // collide on the same backing memory.
        assert_eq!(k.len(), 8);
    }

    #[test]
    fn empty_key_is_well_formed() {
        let k = WeightKey::of_bytes(&[]);
        assert!(k.is_empty());
        // Still hashable/equatable (used as a HashMap key).
        let mut m = HashMap::new();
        m.insert(k, 1u8);
        assert_eq!(m.get(&k), Some(&1));
    }

    #[test]
    fn flush_clears_maps_without_device() {
        // `flush` only touches the typed maps (no device needed), so it is the
        // one cache operation reachable on a no-CUDA host. The full
        // upload/hit path is covered by the runtime-gated tests in lib.rs.
        let cache = WeightCache::default();
        assert_eq!(cache.cached_byte_count(), 0);
        assert_eq!(cache.cached_float_count(), 0);
        cache.flush();
        assert_eq!(cache.cached_byte_count(), 0);
        assert_eq!(cache.cached_float_count(), 0);
        let stats = cache.stats();
        assert_eq!(stats.bytes_hits, 0);
        assert_eq!(stats.float_hits, 0);
    }
}
