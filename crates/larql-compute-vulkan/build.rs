// build.rs — recompiles `shaders/*.comp` into `spv/*.spv` via shaderc when
// the `shader-rebuild` cargo feature is enabled, then `cargo:rerun-if-changed`
// the sources. CI compiles from the committed `spv/*.spv` (the feature is
// off by default), so the build host needs neither a Vulkan SDK nor a C++
// toolchain — matching how the CUDA compile gate works (NVRTC at runtime,
// not build time). Shader authors run `cargo build -p larql-compute-vulkan
// --features shader-rebuild` to regenerate, then commit the updated `.spv`
// (GPU-4001 §3.3).
//
// This file intentionally does nothing when the feature is off.

/// All compute shaders that ship with this crate. Add new entries here when
/// a shader is added; the rebuild loop + rerun-if-changed directives are
/// derived from this list.
const SHADERS: &[(&str, &str)] = &[
    ("shaders/q4k_matmul.comp", "spv/q4k_matmul.spv"),
    ("shaders/q4k_matmul2d.comp", "spv/q4k_matmul2d.spv"),
    ("shaders/rms_norm.comp", "spv/rms_norm.spv"),
    ("shaders/geglu_silu.comp", "spv/geglu_silu.spv"),
    ("shaders/residual_add.comp", "spv/residual_add.spv"),
];

#[cfg(feature = "shader-rebuild")]
fn main() {
    for (src, dst) in SHADERS {
        println!("cargo:rerun-if-changed={src}");
        compile_shader(src, dst);
    }
}

#[cfg(not(feature = "shader-rebuild"))]
fn main() {
    // Still emit rerun-if-changed so a rebuild after toggling the feature
    // picks up shader edits without a clean build.
    for (src, _dst) in SHADERS {
        println!("cargo:rerun-if-changed={src}");
    }
}

#[cfg(feature = "shader-rebuild")]
fn compile_shader(src: &str, dst: &str) {
    use std::fs;

    let source = fs::read_to_string(src).unwrap_or_else(|err| {
        panic!("shader-rebuild: failed to read {src}: {err}");
    });
    let mut compiler = shaderc::Compiler::new().expect("shaderc::Compiler::new");
    let options = shaderc::CompileOptions::new().expect("shaderc::CompileOptions::new");
    let artifact = compiler
        .compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            src,
            "main",
            Some(&options),
        )
        .unwrap_or_else(|err| panic!("shader-rebuild: compiling {src}: {err}"));
    fs::write(dst, artifact.as_binary_u8()).unwrap_or_else(|err| {
        panic!("shader-rebuild: writing {dst}: {err}");
    });
    println!("cargo:warning=shader-rebuild wrote {dst}");
}
