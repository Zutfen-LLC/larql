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

#[cfg(feature = "shader-rebuild")]
fn main() {
    println!("cargo:rerun-if-changed=shaders/q4k_matmul.comp");
    compile_shader("shaders/q4k_matmul.comp", "spv/q4k_matmul.spv");
}

#[cfg(not(feature = "shader-rebuild"))]
fn main() {}

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
