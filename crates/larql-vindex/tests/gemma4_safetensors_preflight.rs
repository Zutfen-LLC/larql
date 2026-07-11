use std::io::Read;

use larql_vindex::{audit_safetensors_preflight, SafetensorsPreflightOptions};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires a pinned real Gemma 4 safetensors checkout"]
fn gemma4_safetensors_preflight_records_provenance_and_summary() {
    let Some(model_dir) = std::env::var_os("LARQL_GEMMA4_ST_DIR").map(std::path::PathBuf::from)
    else {
        eprintln!("skipped: LARQL_GEMMA4_ST_DIR is not set");
        return;
    };
    let revision = std::env::var("LARQL_GEMMA4_ST_REVISION")
        .expect("LARQL_GEMMA4_ST_REVISION is required for a pinned source audit");
    assert!(!revision.trim().is_empty());
    for required in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ] {
        assert!(model_dir.join(required).is_file(), "missing {required}");
    }
    let mut shards = std::fs::read_dir(&model_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect::<Vec<_>>();
    shards.sort();
    assert!(!shards.is_empty(), "no safetensors shards");
    let hashes = shards
        .iter()
        .map(|path| {
            let mut file = std::fs::File::open(path).unwrap();
            let mut hash = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let n = file.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                format!("{:x}", hash.finalize()),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!(hashes.values().all(|hash| hash.len() == 64));
    let report =
        audit_safetensors_preflight(&model_dir, SafetensorsPreflightOptions::default()).unwrap();
    assert!(report.is_valid(), "{}", report.diagnostic());
    let summary = serde_json::json!({ "revision": revision, "sha256": hashes, "report": report });
    assert!(summary["report"]["required"]
        .as_array()
        .is_some_and(|v| !v.is_empty()));
}
