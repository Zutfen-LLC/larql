use larql_inference::ComputeBackendKind;

pub fn parse_backend_kind(raw: &str) -> Result<ComputeBackendKind, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(ComputeBackendKind::Auto),
        "cpu" => Ok(ComputeBackendKind::Cpu),
        "metal" => Ok(ComputeBackendKind::Metal),
        "cuda" => Ok(ComputeBackendKind::Cuda),
        "vulkan" => Ok(ComputeBackendKind::Vulkan),
        other => Err(format!(
            "unknown backend `{other}` (expected one of: auto, cpu, metal, cuda, vulkan)"
        )),
    }
}

pub fn parse_backend_list(raw: &str) -> Result<Vec<ComputeBackendKind>, String> {
    let kinds: Result<Vec<_>, _> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(parse_backend_kind)
        .collect();
    let kinds = kinds?;
    if kinds.is_empty() {
        return Err("backend list is empty".to_string());
    }
    Ok(kinds)
}

pub fn backend_kind_from_args(
    backend: &str,
    metal_alias: bool,
) -> Result<ComputeBackendKind, String> {
    if metal_alias {
        Ok(ComputeBackendKind::Metal)
    } else {
        parse_backend_kind(backend)
    }
}

pub fn backend_kinds_from_args(
    backends: &str,
    cpu_alias: bool,
    metal_alias: bool,
) -> Result<Vec<ComputeBackendKind>, String> {
    if cpu_alias {
        return Ok(vec![ComputeBackendKind::Cpu]);
    }
    if metal_alias {
        return Ok(vec![ComputeBackendKind::Metal]);
    }
    parse_backend_list(backends)
}

pub fn primary_backend_kind(
    backends: &[ComputeBackendKind],
    fallback: ComputeBackendKind,
) -> ComputeBackendKind {
    backends
        .iter()
        .copied()
        .find(|kind| !matches!(kind, ComputeBackendKind::Cpu))
        .unwrap_or(fallback)
}

pub fn backend_label(kind: ComputeBackendKind) -> &'static str {
    match kind {
        ComputeBackendKind::Auto => "larql-auto",
        ComputeBackendKind::Cpu => "larql-cpu",
        ComputeBackendKind::Metal => "larql-metal",
        ComputeBackendKind::Cuda => "larql-cuda",
        ComputeBackendKind::Vulkan => "larql-vulkan",
    }
}

pub fn compute_backend_or_err(
    kind: ComputeBackendKind,
) -> Result<Box<dyn larql_compute::ComputeBackend>, Box<dyn std::error::Error>> {
    larql_inference::compute_backend(kind)
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })
}

pub fn engine_backend_or_err(
    kind: ComputeBackendKind,
) -> Result<Box<dyn larql_inference::EngineBackend>, Box<dyn std::error::Error>> {
    larql_inference::engine_backend(kind)
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })
}
