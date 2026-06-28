use larql_compute::async_compute_backend::{
    AsyncComputeBackend, AttentionHandle, ResidualUploadHandle,
};
use larql_compute::ffn::FfnBackend;
use larql_compute::kv_dispatch::{KvHandle, ResidualHandle};
use larql_compute::CpuBackend;
use ndarray::Array2;

use crate::VulkanBackend;

const CPU: CpuBackend = CpuBackend;

impl AsyncComputeBackend for VulkanBackend {
    fn attention_step_async(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        CPU.attention_step_async(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed_async(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        CPU.attention_step_windowed_async(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill_async(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> (AttentionHandle, KvHandle) {
        CPU.attention_prefill_async(weights, tokens_embedded, layer, window, index)
    }

    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle) {
        CPU.upload_boundary_residual_async(residual)
    }

    fn forward_from_layer_async(
        &self,
        weights: larql_models::WeightsView,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle {
        CPU.forward_from_layer_async(weights, ffn, start_layer, residuals, token_ids)
    }
}
