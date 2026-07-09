//! Bounded per-layer transform + ordered serial write, shared by the
//! attention and dense-FFN kquant writers (IMPORT-002).
//!
//! `jobs <= 1` runs the exact original serial path: transform and write
//! interleaved per layer, no thread pool constructed. `jobs > 1` transforms
//! layers concurrently in a bounded chunk of at most `jobs` layers at a
//! time (via a local `rayon` thread pool sized to `jobs`, not the global
//! default pool), then writes that chunk out in ascending layer order
//! before moving on — so a 60+ layer model never holds every layer's
//! transformed blob in memory simultaneously, only one chunk's worth.
//!
//! Writing (manifest offset assignment, file I/O, callbacks) always
//! happens on the calling thread in ascending layer order, so output
//! bytes and manifests are identical regardless of `jobs`.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::error::VindexError;

/// Run `transform` over `0..num_layers` bounded by `jobs`, then call
/// `write` for each layer's result in ascending layer order.
pub(super) fn transform_then_write<T, F, W>(
    num_layers: usize,
    jobs: usize,
    transform: F,
    mut write: W,
) -> Result<(), VindexError>
where
    T: Send,
    F: Fn(usize) -> Result<T, VindexError> + Sync,
    W: FnMut(usize, T) -> Result<(), VindexError>,
{
    if jobs <= 1 {
        for layer in 0..num_layers {
            let blob = transform(layer)?;
            write(layer, blob)?;
        }
        return Ok(());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|e| VindexError::Parse(format!("failed to build extraction thread pool: {e}")))?;

    let mut layer = 0;
    while layer < num_layers {
        let end = (layer + jobs).min(num_layers);
        let chunk: Vec<Result<T, VindexError>> =
            pool.install(|| (layer..end).into_par_iter().map(&transform).collect());
        for (offset, result) in chunk.into_iter().enumerate() {
            write(layer + offset, result?)?;
        }
        layer = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[test]
    fn serial_path_writes_in_order_for_jobs_le_1() {
        for jobs in [0usize, 1] {
            let written = Mutex::new(Vec::new());
            transform_then_write(
                5,
                jobs,
                |layer| Ok(layer * 10),
                |layer, blob| {
                    written.lock().unwrap().push((layer, blob));
                    Ok(())
                },
            )
            .unwrap();
            let got = written.into_inner().unwrap();
            assert_eq!(got, vec![(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]);
        }
    }

    #[test]
    fn parallel_path_still_writes_in_ascending_order() {
        for jobs in [2usize, 4, 8] {
            let written = Mutex::new(Vec::new());
            transform_then_write(
                11,
                jobs,
                |layer| Ok(layer * 2),
                |layer, blob| {
                    written.lock().unwrap().push((layer, blob));
                    Ok(())
                },
            )
            .unwrap();
            let got = written.into_inner().unwrap();
            let expected: Vec<(usize, usize)> = (0..11).map(|l| (l, l * 2)).collect();
            assert_eq!(got, expected, "jobs={jobs}");
        }
    }

    #[test]
    fn parallel_path_bounds_in_flight_transforms_to_chunk_size() {
        // At no point should more than `jobs` transforms be concurrently
        // "in flight" (started but not yet consumed by write) — proves the
        // chunked design doesn't materialise every layer's blob at once.
        let jobs = 3;
        let in_flight = AtomicUsize::new(0);
        let max_in_flight = AtomicUsize::new(0);
        transform_then_write(
            10,
            jobs,
            |layer| {
                let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(n, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(2));
                Ok(layer)
            },
            |_layer, _blob| {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert!(
            max_in_flight.load(Ordering::SeqCst) <= jobs,
            "max in-flight transforms must not exceed jobs={jobs}, saw {}",
            max_in_flight.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn transform_error_propagates_and_stops_further_writes() {
        let written = Mutex::new(Vec::new());
        let err = transform_then_write(
            5,
            2,
            |layer| {
                if layer == 3 {
                    Err(VindexError::Parse("boom".to_string()))
                } else {
                    Ok(layer)
                }
            },
            |layer, blob| {
                written.lock().unwrap().push((layer, blob));
                Ok(())
            },
        )
        .unwrap_err();
        assert!(format!("{err}").contains("boom"));
    }
}
