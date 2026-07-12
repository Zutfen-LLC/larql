use larql_vindex::format::filenames::*;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::InferenceModel;
use larql_vindex::IndexBuildCallbacks;

const REFERENCE_REVISION: &str = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf";
const REFERENCE_SHA256: &str = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550";
const REFERENCE_BYTES: u64 = 10_246_621_918;
const WORK_START_SHA: &str = "d5922116a1ea8967a427164365baa75b370baffc";

#[derive(Args)]
pub struct ExtractIndexArgs {
    /// Model path or HuggingFace model ID (extracts directly from weights).
    /// Not needed if --from-vectors is used.
    model: Option<String>,

    /// Output path for the .vindex directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Build from already-extracted NDJSON vector files instead of model weights.
    /// Point to the directory containing ffn_gate.vectors.jsonl, etc.
    #[arg(long)]
    from_vectors: Option<PathBuf>,

    /// Top-K tokens to store per feature in down metadata (only for model extraction).
    #[arg(long, default_value = "10")]
    down_top_k: usize,

    /// Per-expert top-K right singular vectors of `gate_proj` to store
    /// instead of the full per-expert gate matrix. Default `0` = disabled
    /// (write full per-expert gate, original behaviour). Set e.g. `64` to
    /// produce a tractable summary vindex for many-experts MoE models
    /// (DeepSeek-V4-Pro at 384 experts/layer would otherwise need ~370 GB
    /// of gate_vectors; with `--summary-features-per-expert 64` it shrinks
    /// to ~11 GB).
    #[arg(long, default_value = "0")]
    summary_features_per_expert: usize,

    /// How much of the model to include in the vindex. Each tier is a
    /// strict superset of the previous:
    ///
    ///   browse     — gate + embed + down_meta only. WALK / DESCRIBE only.
    ///   attention  — + attention + norms. Client half of `run --ffn URL`.
    ///   inference  — + FFN up/down. Full local forward pass (default).
    ///   all        — + lm_head + anything for COMPILE.
    #[arg(long, default_value = "inference", value_parser = parse_extract_level)]
    level: larql_vindex::ExtractLevel,

    /// Include full model weights. Alias for --level all (deprecated, use --level instead).
    #[arg(long)]
    include_weights: bool,

    /// Opt out of the f16 default: store side-channel tensors
    /// (gate_vectors.bin, embeddings.bin, attn/norms/lm_head when
    /// `--quant none`) at f32 instead. Doubles file sizes for
    /// negligible accuracy gain. Rarely wanted.
    #[arg(long)]
    f32: bool,

    /// Storage dtype (`f16` or `f32`). `--f32` remains as a compatibility alias.
    #[arg(long, value_parser = parse_storage_dtype)]
    dtype: Option<larql_vindex::StorageDtype>,

    /// Build a lossless BF16-source to F32 reference vindex.
    #[arg(long)]
    reference_f32: bool,

    /// Quantise model forward-pass weights inline while extracting —
    /// skips any f32 intermediate. `q4k`: Q4_K for Q/K/O/gate/up, Q6_K
    /// for V/down (Ollama-compatible). Implies `--level all` (the Q4_K
    /// writer materialises all components in one pass) and forces f16
    /// on unquantised side-channels (gate_vectors, embeddings) even if
    /// `--f32` was passed.
    #[arg(long, default_value = "none", value_parser = parse_quant)]
    quant: larql_vindex::QuantFormat,

    /// Skip writing `up_weights.bin` + `down_weights.bin`. The up/down
    /// weights are reconstructable from `up_features.bin` /
    /// `down_features.bin` which are produced separately via
    /// `build_{up,down}_features`. This saves ~3.4 GB on a 4B f16 vindex
    /// / ~14 GB on a 31B vindex.
    ///
    /// **Caveat:** a compact vindex can only be read by `WalkFfn` (the
    /// default inference path). `WeightFfn` / `larql dev walk --compare`
    /// will panic on missing FFN tensors.
    #[arg(long)]
    compact: bool,

    /// Skip writing `gate_vectors.bin`. Only valid with `--quant q4k`
    /// — the loader rebuilds the f16 gate by dequantizing
    /// `interleaved_kquant.bin` at vindex-load time. Saves ~1.7 GB on a
    /// 4B q4k vindex / ~14 GB on a 31B q4k vindex; costs ~1.6 s / ~12 s
    /// of CPU at load. See
    /// `cargo run --release -p larql-vindex --example bench_gate_dequant`
    /// for the measured trade-off.
    #[arg(long)]
    drop_gate_vectors: bool,

    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Only valid with
    /// `--quant q4k`. Default keeps the Ollama-compatible mix (Q4_K for
    /// gate/up, Q6_K for down). Enabling this saves ~30 MB/layer on 31B
    /// (~1.8 GB total) and drops down matmul cost ~1.5-1.7× at decode.
    /// Quantisation error on down is a scatter-sum over the intermediate
    /// dimension — noise averages — but quality must be validated
    /// against `walk_correctness` before adopting in production.
    #[arg(long)]
    down_q4k: bool,

    /// Emit `down_features_q4k.bin` (W2 feature-major down) so per-feature
    /// row decode can skip the `kquant_ffn_layer` cache. Adds ~14 MB / layer
    /// at Gemma 4B dims; eliminates the ~840 MB heap cache ceiling on
    /// CPU sparse walk and frees the same headroom across all grid shards.
    /// Requires `--quant q4k`.
    #[arg(long)]
    feature_major_down: bool,

    /// Skip stages that already have output files (resume interrupted builds).
    #[arg(long)]
    resume: bool,

    /// Profile extraction wall-clock and write `extract_profile.json` +
    /// a concise stderr hotspot summary. Opt-in; disabled by default (no
    /// output change, no profile file). Also enabled by
    /// `LARQL_EXTRACT_PROFILE=1`.
    #[arg(long, alias = "profile")]
    profile_extract: bool,

    /// Bounded worker count for attention + dense-FFN Q4_K/Q6_K layer
    /// transforms (`--quant q4k` only; IMPORT-002). Must be a positive
    /// integer. `1` runs the exact serial path used before this flag
    /// existed. Overrides `LARQL_EXTRACT_JOBS`. Default:
    /// `min(available_parallelism, 4)` — deliberately conservative so a
    /// large-core-count box doesn't default to all-core extraction.
    #[arg(long)]
    jobs: Option<usize>,
}

/// Resolve the bounded worker count for parallel kquant layer
/// transforms: `--jobs` takes precedence over `LARQL_EXTRACT_JOBS`,
/// which takes precedence over a conservative default. An invalid
/// `--jobs` (zero) is a hard CLI error; an invalid env value is a
/// warning that falls back to the default rather than panicking.
fn resolve_jobs(cli_jobs: Option<usize>) -> Result<usize, String> {
    if let Some(n) = cli_jobs {
        return if n > 0 {
            Ok(n)
        } else {
            Err("--jobs must be a positive integer".to_string())
        };
    }
    if let Ok(raw) = std::env::var("LARQL_EXTRACT_JOBS") {
        match raw.parse::<usize>() {
            Ok(n) if n > 0 => return Ok(n),
            _ => eprintln!(
                "  warning: LARQL_EXTRACT_JOBS={raw:?} is not a positive integer; using default"
            ),
        }
    }
    Ok(std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4))
}

fn parse_quant(s: &str) -> Result<larql_vindex::QuantFormat, String> {
    match s.to_lowercase().as_str() {
        "none" | "" => Ok(larql_vindex::QuantFormat::None),
        // `q4k` is the legacy tag preserved for back-compat; `kquant`
        // is the post-rename canonical tag. Both map to the same
        // `QuantFormat::Q4K` variant — they differ only in how the
        // value is spelled on disk in `index.json` / on the CLI.
        "q4k" | "q4_k" | "kquant" => Ok(larql_vindex::QuantFormat::Q4K),
        _ => Err(format!(
            "unknown quant format: {s} (expected: none, q4k, kquant)"
        )),
    }
}

fn parse_extract_level(s: &str) -> Result<larql_vindex::ExtractLevel, String> {
    match s.to_lowercase().as_str() {
        "browse" => Ok(larql_vindex::ExtractLevel::Browse),
        "attention" | "attn" => Ok(larql_vindex::ExtractLevel::Attention),
        "inference" | "infer" => Ok(larql_vindex::ExtractLevel::Inference),
        "all" => Ok(larql_vindex::ExtractLevel::All),
        _ => Err(format!(
            "unknown extract level: {s} \
             (expected: browse, attention, inference, all)"
        )),
    }
}

fn parse_storage_dtype(s: &str) -> Result<larql_vindex::StorageDtype, String> {
    match s.to_ascii_lowercase().as_str() {
        "f32" => Ok(larql_vindex::StorageDtype::F32),
        "f16" => Ok(larql_vindex::StorageDtype::F16),
        _ => Err(format!("unknown storage dtype: {s} (expected: f16, f32)")),
    }
}

struct CliBuildCallbacks {
    stage_start: Option<Instant>,
    feature_bar: ProgressBar,
}

impl CliBuildCallbacks {
    fn new() -> Self {
        let feature_bar = ProgressBar::new(0);
        feature_bar.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        feature_bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());

        Self {
            stage_start: None,
            feature_bar,
        }
    }
}

impl IndexBuildCallbacks for CliBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.feature_bar.finish_and_clear();
        eprintln!("\n── {stage} ──");
        self.stage_start = Some(Instant::now());
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, total: usize) {
        self.feature_bar.reset();
        self.feature_bar
            .set_message(format!("{component} L{layer} ({}/{})", layer + 1, total));
    }

    fn on_feature_progress(&mut self, component: &str, _layer: usize, done: usize, total: usize) {
        if total > 0 {
            self.feature_bar.set_length(total as u64);
        }
        self.feature_bar.set_position(done as u64);
        if total == 0 {
            self.feature_bar
                .set_message(format!("{component} {done} records"));
        }
    }

    fn on_layer_done(&mut self, component: &str, layer: usize, elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        eprintln!("  {component} L{layer:2}: {:.1}s", elapsed_ms / 1000.0);
    }

    fn on_stage_done(&mut self, stage: &str, _elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        if let Some(start) = self.stage_start.take() {
            eprintln!("  {stage}: {:.1}s", start.elapsed().as_secs_f64());
        }
    }
}

pub fn run(mut args: ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.reference_f32 {
        return run_inner(args);
    }

    validate_reference_args(&args)?;
    let reference_start = Instant::now();
    let source_dir = larql_models::resolve_model_path(
        args.model
            .as_deref()
            .ok_or("--reference-f32 requires a model directory")?,
    )?;
    validate_reference_source(&source_dir, &args.output)?;
    let final_output = args.output.clone();
    if final_output.exists() {
        return Err(format!(
            "reference output already exists; refusing to replace {}",
            final_output.display()
        )
        .into());
    }
    let name = final_output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("reference output must have a valid final component")?;
    let staging = final_output.with_file_name(format!("{name}.tmp-{}", std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    args.output = staging.clone();
    if let Err(error) = run_inner(args) {
        let cleanup = std::fs::remove_dir_all(&staging);
        eprintln!(
            "reference extraction failed in {}; cleanup: {}",
            staging.display(),
            if cleanup.is_ok() {
                "complete"
            } else {
                "failed"
            }
        );
        return Err(error);
    }
    write_reference_provenance(&staging, reference_start.elapsed())?;
    larql_vindex::load_vindex_config(&staging)
        .map_err(|e| format!("reference structural validation failed: {e}"))?;
    std::fs::rename(&staging, &final_output)?;
    eprintln!(
        "Reference artifact published atomically: {}",
        final_output.display()
    );
    Ok(())
}

fn validate_reference_source(
    source_dir: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let revision = std::env::var("LARQL_GEMMA4_ST_REVISION")
        .map_err(|_| "LARQL_GEMMA4_ST_REVISION is required by --reference-f32")?;
    if revision != REFERENCE_REVISION {
        return Err(format!(
            "unexpected source revision {revision}; expected {REFERENCE_REVISION}"
        )
        .into());
    }
    let shard = source_dir.join("model.safetensors");
    let size = std::fs::metadata(&shard)?.len();
    if size != REFERENCE_BYTES {
        return Err(
            format!("unexpected source shard size {size}; expected {REFERENCE_BYTES}").into(),
        );
    }
    let hash = larql_vindex::format::checksums::sha256_file(&shard)?;
    if hash != REFERENCE_SHA256 {
        return Err(
            format!("unexpected source shard SHA-256 {hash}; expected {REFERENCE_SHA256}").into(),
        );
    }

    let parent = output.parent().unwrap_or_else(|| std::path::Path::new("."));
    let available = available_bytes(parent)?;
    let estimated_output = REFERENCE_BYTES
        .checked_mul(2)
        .ok_or("output estimate overflow")?;
    let required = estimated_output
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(1024 * 1024 * 1024))
        .ok_or("capacity estimate overflow")?;
    eprintln!(
        "Reference disk preflight: source={REFERENCE_BYTES}, estimated_output={estimated_output}, available={available}, required={required}"
    );
    if available < required {
        return Err(
            format!("insufficient disk space: {available} available, {required} required").into(),
        );
    }
    Ok(())
}

#[cfg(unix)]
fn available_bytes(path: &std::path::Path) -> Result<u64, Box<dyn std::error::Error>> {
    use std::os::unix::ffi::OsStrExt;
    let canonical = path.canonicalize()?;
    let c_path = std::ffi::CString::new(canonical.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let stats = unsafe { stats.assume_init() };
    let available_blocks: u64 = stats.f_bavail;
    let fragment_size: u64 = stats.f_frsize;
    Ok(available_blocks.saturating_mul(fragment_size))
}

#[cfg(not(unix))]
fn available_bytes(_path: &std::path::Path) -> Result<u64, Box<dyn std::error::Error>> {
    Err("--reference-f32 disk preflight is currently supported only on Unix".into())
}

fn write_reference_provenance(
    output: &std::path::Path,
    duration: std::time::Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(output)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            files.push(serde_json::json!({
                "name": entry.file_name().to_string_lossy(),
                "size_bytes": entry.metadata()?.len(),
                "sha256": larql_vindex::format::checksums::sha256_file(&path)?,
            }));
        }
    }
    files.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let provenance = serde_json::json!({
        "schema_version": 1,
        "slice_id": "LARQL-INFERENCE-TRUST-001A-ST2",
        "larql_work_start_sha": WORK_START_SHA,
        "larql_final_head_sha": option_env!("LARQL_GIT_SHA"),
        "source_repository": "google/gemma-4-E2B-it",
        "source_revision": REFERENCE_REVISION,
        "source_safetensors_filename": "model.safetensors",
        "source_safetensors_bytes": REFERENCE_BYTES,
        "source_safetensors_sha256": REFERENCE_SHA256,
        "source_dtype": "bf16",
        "destination_dtype": "f32",
        "conversion_contract": "Every required BF16 source value is widened exactly to F32. No required reference tensor is converted through F16 or quantized.",
        "required_source_tensor_count": 600,
        "excluded_multimodal_tensor_count": 1411,
        "tied_head_contract": "embed_tokens.weight serialized once; lm_head derived by the production float loader",
        "ple_storage_policy": "reference_f32",
        "source_directory": "${LARQL_GEMMA4_ST_DIR}",
        "extraction_command": "larql extract ${LARQL_GEMMA4_ST_DIR} -o ${LARQL_GEMMA4_REFERENCE_VINDEX} --level all --quant none --dtype f32 --reference-f32 --profile",
        "extraction_duration_seconds": duration.as_secs_f64(),
        "peak_rss_bytes": peak_rss_bytes(),
        "output_files": files,
    });
    std::fs::write(
        output.join("reference_provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    Ok(())
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        (usage.ru_maxrss as u64).saturating_mul(1024)
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

fn validate_reference_args(args: &ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    let level = if args.include_weights {
        larql_vindex::ExtractLevel::All
    } else {
        args.level
    };
    if level != larql_vindex::ExtractLevel::All {
        return Err("--reference-f32 requires --level all".into());
    }
    if args.quant != larql_vindex::QuantFormat::None {
        return Err("--reference-f32 requires --quant none".into());
    }
    if args.dtype == Some(larql_vindex::StorageDtype::F16) {
        return Err("--reference-f32 conflicts with --dtype f16".into());
    }
    if args.compact {
        return Err("--reference-f32 conflicts with --compact".into());
    }
    if args.resume {
        return Err("--reference-f32 does not support --resume".into());
    }
    if args.from_vectors.is_some() {
        return Err("--reference-f32 requires a safetensors source directory".into());
    }
    if args.drop_gate_vectors || args.down_q4k || args.feature_major_down {
        return Err("--reference-f32 conflicts with Q4_K-only extraction options".into());
    }
    Ok(())
}

fn run_inner(args: ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut callbacks = CliBuildCallbacks::new();
    let build_start = Instant::now();

    // Profiling is enabled by `--profile-extract` OR `LARQL_EXTRACT_PROFILE=1`.
    let profile_enabled =
        args.profile_extract || std::env::var("LARQL_EXTRACT_PROFILE").as_deref() == Ok("1");
    let profiler = if profile_enabled {
        Some(larql_vindex::ExtractProfiler::new())
    } else {
        None
    };

    let jobs = resolve_jobs(args.jobs)?;

    // Resolve extract level: --include-weights upgrades to All (backwards compat)
    let level = if args.include_weights {
        larql_vindex::ExtractLevel::All
    } else {
        args.level
    };

    // Dtype resolution:
    //   --f16                → F16
    //   --quant q4k          → F16 (Q4K quantizes attn + FFN; pairing that
    //                          with f32 gate_vectors/embeddings doubles
    //                          the side-channel footprint for zero accuracy
    //                          benefit. The f16 browse extract already
    //                          proves f16 side-channels are correct.)
    //   default              → F32
    // f16 is the default now; --f32 opts out. `--quant q4k` always
    // forces f16 on the side-channel tensors.
    let dtype = if args.reference_f32 {
        larql_vindex::StorageDtype::F32
    } else if let Some(dtype) = args.dtype {
        if args.f32 && dtype != larql_vindex::StorageDtype::F32 {
            return Err("--f32 conflicts with --dtype f16".into());
        }
        if args.quant == larql_vindex::QuantFormat::Q4K && dtype == larql_vindex::StorageDtype::F32
        {
            return Err("--dtype f32 conflicts with --quant q4k".into());
        }
        dtype
    } else if args.f32 && args.quant != larql_vindex::QuantFormat::Q4K {
        larql_vindex::StorageDtype::F32
    } else {
        larql_vindex::StorageDtype::F16
    };

    if let Some(ref vectors_dir) = args.from_vectors {
        // Build from existing NDJSON files
        eprintln!("Building vindex from vectors: {}", vectors_dir.display());
        eprintln!("Output: {}", args.output.display());

        larql_vindex::build_vindex_from_vectors(vectors_dir, &args.output, &mut callbacks)?;

        if matches!(
            level,
            larql_vindex::ExtractLevel::Inference | larql_vindex::ExtractLevel::All
        ) {
            let model_name = args.model.as_deref().ok_or(
                "--model required with --level inference/all (need model to extract weights)",
            )?;
            eprintln!("\nLoading model for weights: {}", model_name);
            let model = InferenceModel::load(model_name)?;
            let weight_opts = larql_vindex::WriteWeightsOptions {
                level,
                ffn_compact: args.compact,
                skip_attn: false,
                skip_ffn: false,
                ple_storage: if args.reference_f32 {
                    larql_vindex::PleStoragePolicy::ReferenceF32
                } else {
                    larql_vindex::PleStoragePolicy::ProductionF16
                },
            };
            larql_vindex::write_model_weights_with_opts(
                model.weights(),
                &args.output,
                &mut callbacks,
                weight_opts,
                profiler.as_ref(),
            )?;
        }
    } else {
        // Build from model — streaming mode (mmap safetensors, no full model load)
        let model_name = args
            .model
            .as_deref()
            .ok_or("Either provide a model name or use --from-vectors")?;

        let model_path = larql_models::resolve_model_path(model_name)?;

        let level_str = match level {
            larql_vindex::ExtractLevel::Browse => "browse",
            larql_vindex::ExtractLevel::Attention => "attention",
            larql_vindex::ExtractLevel::Inference => "inference",
            larql_vindex::ExtractLevel::All => "all",
        };
        let dtype_str = match dtype {
            larql_vindex::StorageDtype::F32 => "f32",
            larql_vindex::StorageDtype::F16 => "f16",
        };
        eprintln!(
            "Extracting: {} → {} (level={}, dtype={}, quant={})",
            model_path.display(),
            args.output.display(),
            level_str,
            dtype_str,
            args.quant
        );

        let output = &args.output;

        // Detect GGUF source. `resolve_model_path` returns either a directory
        // (safetensors or GGUF) or a single `.gguf` file. We classify here so
        // we can pick the right loader and resolve sibling files (tokenizer,
        // HF metadata) from the correct directory.
        let is_gguf_file = model_path.is_file()
            && model_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
        let gguf_dir = if model_path.is_dir() {
            std::fs::read_dir(&model_path)
                .ok()
                .and_then(|entries| {
                    entries.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
                        p.extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                    })
                })
                .filter(|_| {
                    // Only treat the dir as GGUF if no safetensors are present.
                    std::fs::read_dir(&model_path)
                        .map(|entries| {
                            !entries.filter_map(|e| e.ok()).any(|e| {
                                e.path()
                                    .extension()
                                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
                            })
                        })
                        .unwrap_or(false)
                })
        } else {
            None
        };
        let is_gguf_source = is_gguf_file || gguf_dir.is_some();

        // Sibling-file lookup directory: a `.gguf` file's siblings live in
        // its parent; a model directory's siblings are itself.
        let sibling_dir = if model_path.is_file() {
            model_path
                .parent()
                .ok_or_else(|| format!("model path has no parent: {}", model_path.display()))?
                .to_path_buf()
        } else {
            model_path.clone()
        };

        // Find or load tokenizer (sibling to the GGUF file or in the model dir).
        let tok_path = sibling_dir.join(TOKENIZER_JSON);
        let tokenizer = if tok_path.exists() {
            larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| format!("failed to load tokenizer: {e}"))?
        } else {
            return Err(format!("tokenizer.json not found at {}", tok_path.display()).into());
        };

        let weight_opts = larql_vindex::WriteWeightsOptions {
            level,
            ffn_compact: args.compact,
            skip_attn: false,
            skip_ffn: false,
            ple_storage: if args.reference_f32 {
                larql_vindex::PleStoragePolicy::ReferenceF32
            } else {
                larql_vindex::PleStoragePolicy::ProductionF16
            },
        };
        if args.drop_gate_vectors && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--drop-gate-vectors requires --quant q4k (gate is rebuilt from Q4K at load)"
                    .into(),
            );
        }
        if args.down_q4k && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--down-q4k requires --quant q4k (only the Q4K writer honours this flag)".into(),
            );
        }
        if args.feature_major_down && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--feature-major-down requires --quant q4k (only the Q4K writer honours this flag)"
                    .into(),
            );
        }
        let q4k_opts = larql_vindex::KquantWriteOptions {
            down_proj: if args.down_q4k {
                larql_vindex::DownProjFormat::Q4K
            } else {
                larql_vindex::DownProjFormat::Q6K
            },
            feature_major_down: args.feature_major_down,
            jobs,
        };

        // Per-expert SVD-summary tier (opt-in via `--summary-features-per-expert`)
        // is threaded as a parameter to `build_vindex_streaming` below — see the
        // `summary_features_per_expert` arg. (Was an env side-channel.)

        // Dispatch:
        //
        //  - Safetensors (always) and GGUF at browse level go through the
        //    streaming pipeline — no full model in RAM.
        //  - GGUF at inference / attention / all levels (or any level
        //    with `--quant q4k`) still hits the in-memory loader: the
        //    `StreamingWeights` writer subsystem is safetensors-only,
        //    and porting it to GGUF is a follow-on PR.
        let route_gguf_through_streaming = is_gguf_source
            && matches!(level, larql_vindex::ExtractLevel::Browse)
            && args.quant == larql_vindex::QuantFormat::None;

        if is_gguf_source && !route_gguf_through_streaming {
            // GGUF + attention/inference/all (or any level with q4k) →
            // in-memory loader. `load_model_dir_validated` auto-detects
            // GGUF (single file or directory containing one) and
            // dequantises tensors to f32, producing the `ModelWeights`
            // shape the in-memory build path expects.
            let load_target: std::path::PathBuf = if let Some(gguf) = gguf_dir {
                gguf
            } else {
                model_path.clone()
            };
            eprintln!("  GGUF source detected — loading via in-memory path");
            let weights = larql_models::load_model_dir_validated(&load_target)
                .map_err(|e| format!("failed to load GGUF model: {e}"))?;

            larql_vindex::build_vindex(
                &weights,
                &tokenizer,
                model_name,
                output,
                args.down_top_k,
                level,
                dtype,
                &mut callbacks,
            )?;

            if matches!(
                level,
                larql_vindex::ExtractLevel::Attention
                    | larql_vindex::ExtractLevel::Inference
                    | larql_vindex::ExtractLevel::All
            ) {
                match args.quant {
                    larql_vindex::QuantFormat::Q4K => {
                        larql_vindex::write_model_weights_kquant_with_opts(
                            &weights,
                            output,
                            &mut callbacks,
                            q4k_opts,
                            profiler.as_ref(),
                        )?;
                    }
                    larql_vindex::QuantFormat::None => {
                        larql_vindex::write_model_weights_with_opts(
                            &weights,
                            output,
                            &mut callbacks,
                            weight_opts,
                            profiler.as_ref(),
                        )?;
                    }
                }
            }
        } else {
            // Safetensors path (any level) OR GGUF at browse level —
            // streaming mmap, no full model load. For GGUF, point the
            // pipeline at the shard-1 file (or the directory; the
            // pipeline picks the right shard internally).
            let streaming_entry: std::path::PathBuf = if let Some(gguf) = gguf_dir.as_ref() {
                gguf.clone()
            } else {
                model_path.clone()
            };
            larql_vindex::build_vindex_streaming_profiled(
                &streaming_entry,
                &tokenizer,
                model_name,
                output,
                args.down_top_k,
                args.summary_features_per_expert,
                level,
                dtype,
                args.quant,
                weight_opts,
                q4k_opts,
                args.drop_gate_vectors,
                &mut callbacks,
                profiler.as_ref(),
            )?;
        }

        // Opportunistically copy HF metadata (tokenizer_config.json,
        // special_tokens_map.json, generation_config.json) from the source
        // directory into the vindex. Chat-template-aware runtimes read
        // `tokenizer_config.json::chat_template` from here; missing files
        // are silently skipped. Use the sibling-file dir (parent of a GGUF
        // file, or the model dir itself).
        if let Err(e) = larql_vindex::snapshot_hf_metadata(&sibling_dir, output) {
            eprintln!("  warning: failed to snapshot HF metadata: {e}");
        }
    }

    callbacks.feature_bar.finish_and_clear();
    let build_elapsed = build_start.elapsed();

    // Print summary
    eprintln!("\n── Summary ──");
    eprintln!("  Output: {}", args.output.display());

    if build_elapsed.as_secs() >= 60 {
        eprintln!("  Build time: {:.1}min", build_elapsed.as_secs_f64() / 60.0);
    } else {
        eprintln!("  Build time: {:.1}s", build_elapsed.as_secs_f64());
    }

    for name in &[
        INDEX_JSON,
        GATE_VECTORS_BIN,
        EMBEDDINGS_BIN,
        "down_meta.jsonl",
        DOWN_META_BIN,
        TOKENIZER_JSON,
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        NORMS_BIN,
        LM_HEAD_BIN,
        WEIGHT_MANIFEST_JSON,
    ] {
        let path = args.output.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
            if size_mb > 1024.0 {
                eprintln!("  {name}: {:.2} GB", size_mb / 1024.0);
            } else if size_mb > 0.1 {
                eprintln!("  {name}: {:.1} MB", size_mb);
            } else {
                let size_kb = meta.len() as f64 / 1024.0;
                eprintln!("  {name}: {:.1} KB", size_kb);
            }
        } else {
            eprintln!("  {name}: (not found)");
        }
    }

    // Total: sum all files in the directory
    let total_size: u64 = std::fs::read_dir(&args.output)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    eprintln!(
        "  Total: {:.2} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // ── Profile report (only when profiling is enabled) ──
    if let Some(prof) = &profiler {
        let report_path = args.output.join("extract_profile.json");
        if let Err(e) = prof.write_json_report(&report_path) {
            eprintln!("  warning: failed to write extract_profile.json: {e}");
        }
        prof.print_summary();
    }

    eprintln!("\nUsage:");
    eprintln!(
        "  larql walk --index {} -p \"The capital of France is\"",
        args.output.display()
    );

    Ok(())
}
