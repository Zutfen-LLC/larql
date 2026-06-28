use larql_compute::kv_dispatch::{CompressionCodec, KvDispatch, KvHandle, ResidualHandle};
use larql_compute::CpuBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use crate::CudaBackend;

const CPU: CpuBackend = CpuBackend;

impl KvDispatch for CudaBackend {
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        CPU.alloc_kv_buffer(layer, max_tokens, kv_dim)
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        CPU.append_kv(handle, k_row, v_row, abs_position);
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        CPU.clip_kv(handle, window_size);
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        CPU.read_kv_to_host(handle)
    }

    fn attention_step(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        CPU.attention_step(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        CPU.attention_step_windowed(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.attention_prefill(weights, tokens_embedded, layer, window, index)
    }

    fn recompute_kv_from_residuals(
        &self,
        weights: larql_models::WeightsView,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        CPU.recompute_kv_from_residuals(weights, residuals, layer)
    }

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        CPU.compressed_kv_append(handle, k, v, codec);
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        CPU.upload_boundary_residual(residual)
    }

    fn forward_from_layer(
        &self,
        weights: larql_models::WeightsView,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        CPU.forward_from_layer(weights, start_layer, residuals, token_ids)
    }

    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        CPU.residual_norm_store(x, residual, norm_weights)
    }

    fn coarse_prefill(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.coarse_prefill(weights, token_ids, index)
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.coarse_prefill_with_state(weights, token_ids, index, state)
    }

    fn coarse_decode_step(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        CPU.coarse_decode_step(weights, token_id, index, handle, abs_position)
    }

    fn coarse_decode_step_with_state(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        CPU.coarse_decode_step_with_state(weights, token_id, index, handle, abs_position, state)
    }

    fn coarse_decode_step_with_state_masked(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
        mask: larql_compute::StateDumpMask,
    ) -> Option<Array2<f32>> {
        CPU.coarse_decode_step_with_state_masked(
            weights,
            token_id,
            index,
            handle,
            abs_position,
            state,
            mask,
        )
    }
}
