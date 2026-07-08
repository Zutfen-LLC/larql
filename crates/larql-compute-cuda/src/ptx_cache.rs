//! On-disk PTX cache for the NVRTC-compiled CUDA kernel module.
//!
//! [`CudaRuntime`](crate::backend::CudaRuntime) concatenates ~20 kernels into
//! one NVRTC translation unit and compiles it on every process start. Compiling
//! a module that size is typically hundreds of milliseconds of pure startup
//! latency for every `larql run`/`bench`. This module caches the resulting PTX
//! text on disk so a warm cache skips the NVRTC round-trip entirely (the driver
//! still JITs PTX→SASS at module load, which is fast relative to NVRTC).
//!
//! ## Cache key
//!
//! [`cache_key`] hashes the combined kernel source, the target architecture,
//! the `fmad` policy, and a [`CACHE_FORMAT_VERSION`] component that pins the
//! cudarc version + on-disk layout. Any change to the kernel set, the compiled
//! arch, the numerics policy, or the cache format invalidates the entry (a key
//! mismatch is a cache miss → recompile).
//!
//! ## Failure model
//!
//! Cache failures are **never fatal**. [`try_read`] returns `None` on any miss
//! or I/O error, and [`try_write`] silently swallows any persistence error.
//! Callers always fall back to a fresh NVRTC compile, so a read-only cache
//! root, a corrupt entry, or an unavailable `$HOME`/`$XDG_CACHE_HOME` only
//! costs a recompile — it never makes CUDA unavailable.
//!
//! ## Location
//!
//! `$XDG_CACHE_HOME/larql/` when `XDG_CACHE_HOME` is set and absolute; otherwise
//! `$HOME/.cache/larql/` on Unix-like hosts. [`cache_dir`] returns `None` when
//! neither is available (cache disabled, compile normally). Entries are stored
//! as human-inspectable `cuda-<hash>.ptx` files.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Bumped whenever the on-disk PTX cache format, the cudarc version, or the
/// combined-kernel compilation contract changes. A stale entry produces a key
/// mismatch (a miss) instead of loading incompatible PTX.
const CACHE_FORMAT_VERSION: &str = "cudarc:0.19.8;larql-ptx-cache:v1";

/// Build the deterministic on-disk cache key (a 64-char hex SHA-256 digest)
/// for a combined-kernel NVRTC compile.
///
/// The key folds in: the cache-format/cudarc version, the target architecture
/// string (e.g. `compute_89`), the `fmad` policy, and the full combined source
/// text. Two compiles with identical inputs yield identical keys; any change
/// to the inputs yields a different key.
pub(crate) fn cache_key(src: &str, arch: &str, fmad: bool) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_FORMAT_VERSION.as_bytes());
    hasher.update(b"\narch=");
    hasher.update(arch.as_bytes());
    hasher.update(b"\nfmad=");
    hasher.update(if fmad { &b"true"[..] } else { &b"false"[..] });
    hasher.update(b"\nsrc_len=");
    hasher.update(src.len().to_string().as_bytes());
    hasher.update(b"\n");
    hasher.update(src.as_bytes());
    let digest = hasher.finalize();
    // 32 bytes → 64 hex chars. Used verbatim as the filename suffix.
    format!("{:x}", digest)
}

/// Resolve the cache directory from environment variables without touching the
/// process env. Pure core of [`cache_dir`]; unit-tested directly.
///
/// `$XDG_CACHE_HOME/larql/` when `xdg` is set and absolute, else
/// `$HOME/.cache/larql/` when `home` is set, else `None` (cache disabled).
fn resolve_cache_dir(xdg: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
    if let Some(xdg) = xdg.map(str::trim).filter(|s| !s.is_empty()) {
        let p = Path::new(xdg);
        if p.is_absolute() {
            return Some(p.join("larql"));
        }
    }
    home.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|h| Path::new(h).join(".cache").join("larql"))
}

/// Resolve the cache directory from the process environment.
///
/// `None` when neither `XDG_CACHE_HOME` (set + absolute) nor `HOME` (set) is
/// available — the cache is then disabled and callers compile normally.
pub(crate) fn cache_dir() -> Option<PathBuf> {
    let xdg = std::env::var_os("XDG_CACHE_HOME").and_then(|v| v.into_string().ok());
    let home = std::env::var_os("HOME").and_then(|v| v.into_string().ok());
    resolve_cache_dir(xdg.as_deref(), home.as_deref())
}

fn cache_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("cuda-{key}.ptx"))
}

/// Try to read a cached PTX blob for `key`. Returns `None` on any miss, I/O
/// error, or empty entry — callers fall back to a fresh NVRTC compile.
pub(crate) fn try_read(key: &str) -> Option<String> {
    let dir = cache_dir()?;
    let path = cache_path(&dir, key);
    match std::fs::read_to_string(&path) {
        Ok(src) if !src.is_empty() => Some(src),
        _ => None,
    }
}

/// Persist a freshly compiled PTX blob, best-effort. Any I/O error (a
/// read-only cache root, a full disk, …) is swallowed — caching is purely an
/// optimisation. Writes to a sibling temp file then renames it into place so a
/// crash mid-write can't leave a corrupt `.ptx` shadowing a good entry.
pub(crate) fn try_write(key: &str, ptx_src: &str) {
    let Some(dir) = cache_dir() else {
        return;
    };
    // Creating the dir may fail on a read-only root — just skip the write.
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = cache_path(&dir, key);
    let tmp = dir.join(format!("cuda-{key}.ptx.tmp"));
    if std::fs::write(&tmp, ptx_src).is_err() {
        return;
    }
    // Rename over the destination. Same-filesystem rename is atomic; on the
    // rare concurrent-write race the final content is identical anyway.
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRC: &str =
        "extern \"C\" __global__ void k(float* x, int n) { if (threadIdx.x < n) x[threadIdx.x] = 1.0f; }";

    #[test]
    fn cache_key_is_deterministic_for_identical_inputs() {
        let a = cache_key(SAMPLE_SRC, "compute_89", false);
        let b = cache_key(SAMPLE_SRC, "compute_89", false);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64, "key should be a 64-char hex digest");
    }

    #[test]
    fn cache_key_is_hex_only() {
        let key = cache_key(SAMPLE_SRC, "compute_89", false);
        assert!(
            key.chars().all(|c| c.is_ascii_hexdigit()),
            "key must be hex-only (it's a filename): {key}"
        );
    }

    #[test]
    fn cache_key_changes_with_source() {
        let a = cache_key(SAMPLE_SRC, "compute_89", false);
        let b = cache_key(
            "extern \"C\" __global__ void other(float* x, int n) {}",
            "compute_89",
            false,
        );
        assert_ne!(a, b, "different sources must produce different keys");
    }

    #[test]
    fn cache_key_changes_with_arch() {
        let a = cache_key(SAMPLE_SRC, "compute_89", false);
        let b = cache_key(SAMPLE_SRC, "compute_80", false);
        assert_ne!(a, b, "different archs must produce different keys");
    }

    #[test]
    fn cache_key_changes_with_fmad() {
        let a = cache_key(SAMPLE_SRC, "compute_89", false);
        let b = cache_key(SAMPLE_SRC, "compute_89", true);
        assert_ne!(a, b, "different fmad policies must produce different keys");
    }

    #[test]
    fn cache_key_includes_source_content_not_just_length() {
        // Two distinct sources of equal length must not collide.
        let src_a = "extern \"C\" __global__ void aaa() {}";
        let src_b = "extern \"C\" __global__ void bbb() {}";
        assert_eq!(src_a.len(), src_b.len());
        assert_ne!(
            cache_key(src_a, "compute_89", false),
            cache_key(src_b, "compute_89", false)
        );
    }

    #[test]
    fn resolve_cache_dir_prefers_absolute_xdg() {
        let dir = resolve_cache_dir(Some("/custom/xdg"), Some("/home/u"));
        assert_eq!(dir, Some(PathBuf::from("/custom/xdg/larql")));
    }

    #[test]
    fn resolve_cache_dir_ignores_relative_xdg() {
        // A relative XDG_CACHE_HOME is invalid per the spec; fall back to HOME.
        let dir = resolve_cache_dir(Some("relative/xdg"), Some("/home/u"));
        assert_eq!(dir, Some(PathBuf::from("/home/u/.cache/larql")));
    }

    #[test]
    fn resolve_cache_dir_falls_back_to_home() {
        let dir = resolve_cache_dir(None, Some("/home/u"));
        assert_eq!(dir, Some(PathBuf::from("/home/u/.cache/larql")));
    }

    #[test]
    fn resolve_cache_dir_is_none_without_anything() {
        assert_eq!(resolve_cache_dir(None, None), None);
        // Empty strings count as unset.
        assert_eq!(resolve_cache_dir(Some(""), Some("")), None);
        assert_eq!(
            resolve_cache_dir(Some("   "), Some("/home/u")),
            Some(PathBuf::from("/home/u/.cache/larql"))
        );
    }
}
