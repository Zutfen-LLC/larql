use std::collections::HashMap;
use std::io::Read;

use larql_vindex::{audit_safetensors_preflight, SafetensorsPreflightOptions};
use memmap2::Mmap;
use sha2::{Digest, Sha256};

const REVISION: &str = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf";
const SHA256: &str = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550";

#[test]
#[ignore = "requires the pinned 10 GB Gemma 4 source and extracted F32 reference vindex"]
fn gemma4_reference_vindex_roundtrip() {
    let validation_start = std::time::Instant::now();
    let source_dir = std::path::PathBuf::from(
        std::env::var_os("LARQL_GEMMA4_ST_DIR").expect("LARQL_GEMMA4_ST_DIR is required"),
    );
    let reference_dir = std::path::PathBuf::from(
        std::env::var_os("LARQL_GEMMA4_REFERENCE_VINDEX")
            .expect("LARQL_GEMMA4_REFERENCE_VINDEX is required"),
    );
    assert_eq!(std::env::var("LARQL_GEMMA4_ST_REVISION").unwrap(), REVISION);

    let shard_path = source_dir.join("model.safetensors");
    let mut shard_file = std::fs::File::open(&shard_path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = shard_file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    assert_eq!(format!("{:x}", hasher.finalize()), SHA256);

    let preflight =
        audit_safetensors_preflight(&source_dir, SafetensorsPreflightOptions::default()).unwrap();
    assert!(preflight.is_valid(), "{}", preflight.diagnostic());
    assert_eq!(preflight.required.len(), 600);

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let loaded = larql_vindex::load_model_weights(&reference_dir, &mut callbacks).unwrap();
    assert_eq!(loaded.num_layers, 35);
    assert_eq!(loaded.hidden_size, 1536);
    assert_eq!(loaded.vocab_size, 262_144);
    assert_eq!(loaded.embed.shape(), &[262_144, 1536]);
    assert_eq!(loaded.lm_head.shape(), loaded.embed.shape());
    assert_eq!(loaded.lm_head.as_ptr(), loaded.embed.as_ptr());
    assert!(!reference_dir.join("lm_head.bin").exists());

    let shard_file = std::fs::File::open(&shard_path).unwrap();
    let shard = unsafe { Mmap::map(&shard_file).unwrap() };
    let safetensors = safetensors::SafeTensors::deserialize(&shard).unwrap();
    let source_by_normalized = safetensors
        .names()
        .into_iter()
        .map(|name| (normalize(name), name))
        .collect::<HashMap<_, _>>();

    let mut compared_tensors = 0_u64;
    let mut compared_elements = 0_u64;
    for key in &preflight.required {
        if key == "lm_head.weight" {
            continue;
        }
        let source_name = source_by_normalized
            .get(key.as_str())
            .unwrap_or_else(|| panic!("required source tensor missing from index: {key}"));
        let view = safetensors.tensor(source_name).unwrap();
        assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{key}");
        let actual: &[f32] = if key == "embed_tokens.weight" {
            loaded.embed.as_slice().unwrap()
        } else if view.shape().len() == 2 {
            loaded
                .tensors
                .get(key)
                .unwrap_or_else(|| panic!("loaded tensor missing: {key}"))
                .as_slice()
                .unwrap()
        } else {
            loaded
                .vectors
                .get(key)
                .unwrap_or_else(|| panic!("loaded vector missing: {key}"))
        };
        let expected_elements = view.data().len() / 2;
        assert_eq!(
            actual.len(),
            expected_elements,
            "shape/length mismatch: {key}"
        );
        let mut mismatch_count = 0_u64;
        let mut samples = Vec::new();
        for (index, (bytes, value)) in view.data().chunks_exact(2).zip(actual).enumerate() {
            let bf16_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            let expected_bits = (bf16_bits as u32) << 16;
            if value.to_bits() != expected_bits {
                mismatch_count += 1;
                if samples.len() < 8 {
                    samples.push(format!(
                        "{key}[{index}]: bf16=0x{bf16_bits:04x}, expected=0x{expected_bits:08x} {:?}, actual=0x{:08x} {:?}",
                        f32::from_bits(expected_bits), value.to_bits(), value
                    ));
                }
            }
        }
        assert_eq!(mismatch_count, 0, "{}", samples.join("\n"));
        compared_tensors += 1;
        compared_elements += expected_elements as u64;
    }

    println!(
        "{}",
        serde_json::json!({
            "required_tensor_count": preflight.required.len(),
            "compared_tensor_count": compared_tensors,
            "compared_element_count": compared_elements,
            "bitwise_mismatch_count": 0,
            "missing_tensor_count": 0,
            "shape_mismatch_count": 0,
            "dtype_mismatch_count": 0,
            "tied_head": "shared embedding storage",
            "loader": "production float loader",
            "load_validation_duration_seconds": validation_start.elapsed().as_secs_f64(),
            "load_validation_peak_rss_bytes": peak_rss_bytes(),
        })
    );
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    return usage.ru_maxrss as u64;
    #[cfg(not(target_os = "macos"))]
    return (usage.ru_maxrss as u64) * 1024;
}

fn normalize(name: &str) -> &str {
    for prefix in [
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "model.",
    ] {
        if let Some(stripped) = name.strip_prefix(prefix) {
            return stripped;
        }
    }
    name
}
