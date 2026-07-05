//! Per-expert output cache for the MoE expert dispatch path.
//!
//! Distinct from [`crate::ffn_l2_cache::FfnL2Cache`], which caches the dense
//! WalkFFN output keyed by sorted gate-KNN feature IDs. This cache is keyed by
//! `(layer, expert_id, hash(residual_in))` so that repeated identical expert
//! dispatches — common when the same prompt or context recurs across requests
//! — hit the cache instead of recomputing the Q4_K/Q8_K matvec chain.
//!
//! # Key strategy
//!
//! The residual is hashed via i16 quantisation (scale x256, clamp) before
//! `DefaultHasher` — the same scheme `FfnL1Cache::residual_key` uses. This:
//!
//! - Costs O(hidden) (~1 us for hidden=2560) — negligible vs the ~100 us+
//!   per-expert matvec.
//! - Is robust to sub-1/256 floating-point noise, so residuals that are
//!   bit-identical up to FP nondeterminism collapse to the same key.
//!
//! # Determinism
//!
//! The expert FFN is a pure function of `(weights, h_norm)` at inference time
//! (no dropout, no sampling noise in the residual). The residual arriving at
//! the expert endpoint (`h_post_attn`) is itself deterministic given the same
//! attention output. So caching is sound whenever the same
//! `(layer, expert, residual)` triple recurs.
//!
//! # Opt-in
//!
//! Default-off behind `LARQL_EXPERT_OUTPUT_CACHE=1`. The hash + lookup only
//! run when enabled, so disabled builds pay zero overhead. Capacity is bounded
//! per layer by `max_entries` (default mirrors `FfnL2Cache`).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Default per-layer capacity — mirrors [`crate::ffn_l2_cache::L2_DEFAULT_MAX_ENTRIES`].
pub const EXPERT_CACHE_DEFAULT_MAX_ENTRIES: usize = 4096;

/// Capacity-capped per-expert output cache.
///
/// Keyed by `(layer, expert_id, residual_key)` -> `Arc<Vec<f32>>` (the
/// single-expert output, length = hidden). Mirrors [`FfnL2Cache`]'s shape:
/// one `RwLock<HashMap>` per layer, atomic hit/miss counters, a `stats()`
/// snapshot, and a per-layer entry-count cap.
///
/// [`FfnL2Cache`]: crate::ffn_l2_cache::FfnL2Cache
pub struct ExpertOutputCache {
    layers: Vec<RwLock<HashMap<(usize, u64), Arc<Vec<f32>>>>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ExpertOutputCache {
    pub fn new(num_layers: usize) -> Self {
        Self::with_max_entries(num_layers, EXPERT_CACHE_DEFAULT_MAX_ENTRIES)
    }

    pub fn with_max_entries(num_layers: usize, max_entries: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Stable u64 key from a residual vector. Quantises each float to i16
    /// (scale x256) before hashing so that near-identical residuals (differing
    /// by < 1/256 per dimension) collapse to the same key. Mirrors
    /// `FfnL1Cache::residual_key`.
    ///
    /// Cost: O(hidden), ~1 us at hidden=2560.
    pub fn residual_key(residual: &[f32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in residual {
            let q = (v * 256.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            q.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Look up a cached expert output by `(layer, expert_id, rkey)`.
    ///
    /// `rkey` should be pre-computed via [`residual_key`](Self::residual_key) —
    /// the layer-batch path hashes `h_norm` once per layer (shared across all K
    /// experts) rather than per expert. Returns `Some(Arc)` on hit, `None` on
    /// miss. Bumps hit/miss counters regardless.
    pub fn get(&self, layer: usize, expert_id: usize, rkey: u64) -> Option<Arc<Vec<f32>>> {
        let map = self.layers.get(layer)?.read().ok()?;
        match map.get(&(expert_id, rkey)) {
            Some(v) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(v.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert a computed expert output. Silently drops when the per-layer map
    /// is at capacity (mirrors `FfnL2Cache::insert`).
    pub fn insert(&self, layer: usize, expert_id: usize, rkey: u64, value: Vec<f32>) {
        if let Some(lock) = self.layers.get(layer) {
            if let Ok(mut map) = lock.write() {
                if map.len() < self.max_entries {
                    map.insert((expert_id, rkey), Arc::new(value));
                }
            }
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits();
        let m = self.misses();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    /// Snapshot for /v1/stats overlay (mirrors `FfnL2Cache::stats`).
    pub fn stats(&self) -> serde_json::Value {
        let h = self.hits();
        let m = self.misses();
        let total = h + m;
        let hit_rate = if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        };
        serde_json::json!({
            "hits": h,
            "misses": m,
            "total": total,
            "hit_rate": (hit_rate * 1000.0).round() / 1000.0,
            "layers": self.layers.len(),
            "max_entries_per_layer": self.max_entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_key_is_deterministic() {
        let r: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        assert_eq!(
            ExpertOutputCache::residual_key(&r),
            ExpertOutputCache::residual_key(&r)
        );
    }

    #[test]
    fn residual_key_differs_for_different_residuals() {
        let r1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let r2: Vec<f32> = vec![1.0, 2.0, 4.0];
        assert_ne!(
            ExpertOutputCache::residual_key(&r1),
            ExpertOutputCache::residual_key(&r2)
        );
    }

    #[test]
    fn residual_key_collapses_near_identical() {
        // Sub-1/256 noise -> same i16 bucket -> same key.
        let base: Vec<f32> = (0..32).map(|i| i as f32 * 0.001).collect();
        let noise: Vec<f32> = base.iter().map(|&v| v + 1e-5).collect();
        assert_eq!(
            ExpertOutputCache::residual_key(&base),
            ExpertOutputCache::residual_key(&noise)
        );
    }

    #[test]
    fn residual_key_empty_vec() {
        assert_eq!(
            ExpertOutputCache::residual_key(&[]),
            ExpertOutputCache::residual_key(&[])
        );
    }

    #[test]
    fn miss_then_hit() {
        let cache = ExpertOutputCache::new(4);
        let rkey = ExpertOutputCache::residual_key(&[1.0, 2.0, 3.0]);
        // First call: miss
        assert!(cache.get(0, 5, rkey).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        cache.insert(0, 5, rkey, vec![1.0, 2.0, 3.0]);
        // Second call: hit
        let result = cache.get(0, 5, rkey);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn hit_rate_computation() {
        let cache = ExpertOutputCache::new(2);
        let rkey = ExpertOutputCache::residual_key(&[1.0]);
        cache.insert(0, 0, rkey, vec![0.0]);
        cache.get(0, 0, rkey); // hit
        cache.get(0, 0, rkey); // hit
        let miss_rkey = ExpertOutputCache::residual_key(&[999.0]);
        cache.get(0, 0, miss_rkey); // miss (wrong key)

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn capacity_cap_drops_excess() {
        // Memory-bound test: per-layer cap of 2 entries; 3rd insert dropped.
        let cache = ExpertOutputCache::with_max_entries(1, 2);
        let rkey_a = ExpertOutputCache::residual_key(&[1.0]);
        let rkey_b = ExpertOutputCache::residual_key(&[2.0]);
        let rkey_c = ExpertOutputCache::residual_key(&[3.0]);

        cache.insert(0, 0, rkey_a, vec![0.0]);
        cache.insert(0, 1, rkey_b, vec![1.0]);
        cache.insert(0, 2, rkey_c, vec![2.0]); // at capacity — dropped

        assert!(cache.get(0, 0, rkey_a).is_some());
        assert!(cache.get(0, 1, rkey_b).is_some());
        assert!(cache.get(0, 2, rkey_c).is_none());
    }

    #[test]
    fn different_experts_same_residual_key_independently() {
        // Same residual, different expert_id -> independent entries.
        let cache = ExpertOutputCache::new(2);
        let rkey = ExpertOutputCache::residual_key(&[1.0, 2.0]);
        cache.insert(0, 3, rkey, vec![10.0]);
        cache.insert(0, 7, rkey, vec![20.0]);

        assert_eq!(*cache.get(0, 3, rkey).unwrap(), vec![10.0]);
        assert_eq!(*cache.get(0, 7, rkey).unwrap(), vec![20.0]);
    }

    #[test]
    fn layers_are_independent() {
        let cache = ExpertOutputCache::new(4);
        let rkey = ExpertOutputCache::residual_key(&[7.0]);
        cache.insert(0, 1, rkey, vec![0.0]);
        cache.insert(2, 1, rkey, vec![2.0]);

        assert_eq!(*cache.get(0, 1, rkey).unwrap(), vec![0.0]);
        assert_eq!(*cache.get(2, 1, rkey).unwrap(), vec![2.0]);
        assert!(cache.get(1, 1, rkey).is_none());
    }

    #[test]
    fn out_of_range_layer_is_safe() {
        let cache = ExpertOutputCache::new(2);
        let rkey = ExpertOutputCache::residual_key(&[1.0]);
        assert!(cache.get(99, 0, rkey).is_none());
        cache.insert(99, 0, rkey, vec![1.0]); // must not panic
    }

    #[test]
    fn arc_values_are_shared_not_cloned() {
        let cache = ExpertOutputCache::new(2);
        let rkey = ExpertOutputCache::residual_key(&[42.0]);
        cache.insert(0, 0, rkey, vec![2.5]);
        let a = cache.get(0, 0, rkey).unwrap();
        let b = cache.get(0, 0, rkey).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn stats_json_has_expected_fields() {
        let cache = ExpertOutputCache::new(3);
        let stats = cache.stats();
        assert!(stats["hits"].is_number());
        assert!(stats["misses"].is_number());
        assert!(stats["total"].is_number());
        assert!(stats["hit_rate"].is_number());
        assert_eq!(stats["layers"], 3);
        assert_eq!(
            stats["max_entries_per_layer"],
            EXPERT_CACHE_DEFAULT_MAX_ENTRIES
        );
    }

    #[test]
    fn concurrent_reads_do_not_panic() {
        let cache = Arc::new(ExpertOutputCache::new(4));
        let rkey = ExpertOutputCache::residual_key(&[1.0, 2.0, 3.0]);
        cache.insert(0, 0, rkey, vec![1.0, 2.0]);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = Arc::clone(&cache);
                std::thread::spawn(move || {
                    assert!(c.get(0, 0, rkey).is_some());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}

