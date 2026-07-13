//! DeepSeek-V4 architecture — MoE + hybrid MLA/CSA attention + Hyper-Connections.
//!
//! DeepSeek-V4-Flash (284B total / 13B active MoE, 43 layers + 1 MTP) is the
//! reference model for this architecture. Key differences from V3:
//!
//! - **No `model.` prefix.** Tensors: `embed.weight`, `layers.X.attn.*`,
//!   `layers.X.ffn.*`.
//! - **`ffn` not `mlp`** for the feed-forward block.
//! - **`w1`/`w2`/`w3`** for expert weights (LLaMA-1 / OG SwiGLU naming).
//! - **FP4 expert weights** stored as I8 packed (E2M1, 2-per-byte) + F8_E8M0
//!   per-32-element scales. Each expert has `.weight` + `.scale` pairs.
//! - **FP8 attention weights** stored as E4M3 + UE8M0 block scales (128×128).
//!   Each attention projection has `.weight` + `.scale` pairs.
//! - **Hyper-Connections (HC)** replace standard residual connections. Each
//!   block maintains `hc_mult` copies of the hidden state. Per-layer HC
//!   parameters: `hc_attn_{base,fn,scale}` and `hc_ffn_{base,fn,scale}`.
//!   Global head HC: `hc_head_{base,fn,scale}`.
//! - **Hybrid MLA + CSA attention**: low-rank Q (wq_a/wq_b), fused single-head
//!   KV (wkv), attention sink tokens, sliding window (128) always on, plus
//!   per-layer KV compression with Compressor + optional Indexer submodules.
//! - **Compress layers** alternate: ratio=0 (sliding only), ratio=4 (with
//!   Indexer), ratio=128 (heavy compression, fixed indexing).
//! - **Hash-based MoE routing** for the first `n_hash_layers` layers (tid2eid
//!   lookup). Remaining layers use `sqrtsoftplus` scoring + `noaux_tc` top-k.
//! - **Grouped low-rank output projection** (wo_a/wo_b with o_groups).
//! - **MTP** (Multi-Token Prediction) head for speculative decoding.
//!
//! Currently scoped to **browse-tier extraction** — gate vectors + embeddings
//! + down_meta. Inference forward pass is the subject of the DSV4 plan
//! (`docs/dsv4-flash-implementation-plan.md`).

use crate::config::{ModelArchitecture, ModelConfig};

pub struct DeepSeekV4Arch {
    config: ModelConfig,
}

impl DeepSeekV4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for DeepSeekV4Arch {
    fn family(&self) -> &str {
        "deepseek_v4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Tensor key conventions (V4 has no `model.` prefix; uses `attn` / `ffn`) ──

    fn key_prefixes_to_strip(&self) -> &[&str] {
        // No `model.` wrapper in V4 safetensors.
        &[]
    }

    fn embed_key(&self) -> &str {
        "embed.weight"
    }

    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    // ── Attention: low-rank Q (wq_a/wq_b) + fused KV (wkv) ──
    // V4 does NOT use separate q_proj/k_proj/v_proj. The loader needs to
    // understand the low-rank structure. These keys return the wkv tensor
    // (fused KV projection) for the v_key slot since it's the closest analog.

    fn attn_q_key(&self, layer: usize) -> String {
        // Low-rank Q down-projection (compress): hidden → q_lora_rank
        format!("{}attn.wq_a.weight", self.layer_prefix(layer))
    }

    fn attn_k_key(&self, layer: usize) -> String {
        // Low-rank Q up-projection (decompress): q_lora_rank → n_heads * head_dim
        format!("{}attn.wq_b.weight", self.layer_prefix(layer))
    }

    fn attn_v_key(&self, layer: usize) -> String {
        // Fused single-head KV projection: hidden → head_dim
        format!("{}attn.wkv.weight", self.layer_prefix(layer))
    }

    fn attn_o_key(&self, layer: usize) -> String {
        // Grouped low-rank output projection (decompress).
        // wo_a: (n_heads*head_dim/groups) → (groups * o_lora_rank)
        format!("{}attn.wo_a.weight", self.layer_prefix(layer))
    }

    /// Output projection up-projection: (groups * o_lora_rank) → hidden
    fn ffn_gate_key(&self, layer: usize) -> String {
        // Override: we use this slot for wo_b since attn_o_key is wo_a.
        // The forward pass handles this via the dedicated key methods below.
        format!("{}attn.wo_b.weight", self.layer_prefix(layer))
    }

    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}ffn.shared_experts.w3.weight", self.layer_prefix(layer))
    }

    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}ffn.shared_experts.w2.weight", self.layer_prefix(layer))
    }

    // Layer norms: V4 names them `attn_norm` / `ffn_norm`
    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}attn_norm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!("{}ffn_norm.weight", self.layer_prefix(layer))
    }
    fn pre_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn post_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // ── V4-specific attention keys ──

    /// Q down-projection (compress): hidden → q_lora_rank
    fn mla_q_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wq_a.weight", self.layer_prefix(layer)))
    }

    /// Q up-projection (decompress): q_lora_rank → n_heads * head_dim
    fn mla_q_b_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wq_b.weight", self.layer_prefix(layer)))
    }

    /// Fused KV projection: hidden → head_dim (single KV head)
    fn mla_kv_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wkv.weight", self.layer_prefix(layer)))
    }

    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        // V4 fuses kv into wkv; no separate kv_b projection.
        None
    }

    // ── Q/K per-head norms ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.q_norm.weight", self.layer_prefix(layer)))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.kv_norm.weight", self.layer_prefix(layer)))
    }

    // ── MLA geometry ──

    fn uses_mla(&self) -> bool {
        // V4 uses low-rank Q + fused KV, which is MLA-shaped.
        self.config.q_lora_rank.is_some()
    }

    fn kv_lora_rank(&self) -> usize {
        // V4 doesn't have kv_lora_rank in config — the fused wkv output
        // IS the latent KV (head_dim=512). Return head_dim as the latent dim.
        self.config.head_dim
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(1024)
    }

    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        // head_dim=512 total, rope_head_dim=64 → nope = 448
        let rope = self.config.qk_rope_head_dim.unwrap_or(64);
        Some(self.config.head_dim - rope)
    }

    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        self.config.qk_rope_head_dim.or(Some(64))
    }

    fn mla_v_head_dim(&self) -> Option<usize> {
        // V4 has a single fused KV with head_dim=512 (V shares the same dim).
        Some(self.config.head_dim)
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(256)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(6)
    }

    fn num_shared_experts(&self) -> usize {
        self.config.num_shared_experts.unwrap_or(1)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn.gate.weight", self.layer_prefix(layer)))
    }

    fn moe_router_type(&self) -> &str {
        // V4-Flash uses sqrtsoftplus scoring + noaux_tc top-k
        match self.config.scoring_func.as_deref() {
            Some("sqrtsoftplus") => "sqrtsoftplus_noaux_tc",
            Some("softmax") => "top_k_softmax",
            Some("sigmoid") => "top_k_sigmoid",
            _ => "sqrtsoftplus_noaux_tc",
        }
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w2.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w2.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Sliding window (always on in V4) ──

    fn is_sliding_window_layer(&self, _layer: usize) -> bool {
        true
    }

    fn sliding_window_size(&self) -> Option<usize> {
        self.config.sliding_window.or(Some(128))
    }

    // ── FFN ──

    fn ffn_type(&self) -> crate::config::FfnType {
        crate::config::FfnType::Gated
    }

    fn activation(&self) -> crate::config::Activation {
        crate::config::Activation::Silu
    }

    fn moe_intermediate_size(&self) -> usize {
        self.config.moe_intermediate_size.unwrap_or(2048)
    }

    // ── RoPE: V4 uses partial rotary (last 64 dims of 512) ──

    fn rotary_fraction_for_layer(&self, _layer: usize) -> f64 {
        let rope_dim = self.config.qk_rope_head_dim.unwrap_or(64);
        rope_dim as f64 / self.config.head_dim as f64
    }

    // ── Norm ──

    fn norm_type(&self) -> crate::config::NormType {
        crate::config::NormType::RmsNorm
    }
}
