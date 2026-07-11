use larql_models::loading::gguf::GgufFile;

#[test]
#[ignore = "manual real-model audit; set LARQL_GEMMA4_GGUF"]
fn audit_operator_supplied_gemma4_gguf() {
    let Some(path) = std::env::var_os("LARQL_GEMMA4_GGUF") else {
        eprintln!("skipped: LARQL_GEMMA4_GGUF is not set");
        return;
    };

    let gguf = GgufFile::open(std::path::Path::new(&path)).unwrap_or_else(|error| {
        panic!(
            "LARQL_GEMMA4_GGUF={} could not be audited: {error}",
            std::path::Path::new(&path).display()
        )
    });
    let config = gguf.to_config_json();
    let arch = larql_models::detect_from_json_validated(&config)
        .unwrap_or_else(|error| panic!("Gemma 4 GGUF architecture metadata is invalid: {error}"));

    assert_eq!(
        arch.family(),
        "gemma4",
        "GGUF architecture was misclassified"
    );
    assert_eq!(
        arch.config().num_layers,
        35,
        "unexpected decoder layer count"
    );
    assert_eq!(arch.config().sliding_window, Some(512));
    assert!(
        arch.has_per_layer_embeddings(),
        "required PLE metadata is absent"
    );
    assert_eq!(arch.config().layer_types.as_ref().map(Vec::len), Some(35));
    assert!(arch.is_sliding_window_layer(0));
    assert!(
        !arch.is_sliding_window_layer(34),
        "final layer must be global"
    );
    assert!(
        arch.rotary_fraction_for_layer(34) < 1.0,
        "global p-RoPE was lost"
    );
    assert!(
        gguf.tensor_infos
            .iter()
            .any(|tensor| tensor.name() == "per_layer_token_embd.weight"),
        "required PLE token embedding tensor is absent"
    );
}
