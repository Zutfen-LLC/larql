//! Device-resident FFN chain launches (GPU-4003).
//!
//! These methods live on [`super::VulkanRuntime`] (a second `impl` block,
//! allowed by Rust). Each runs one elementwise/matmul op as a single
//! command buffer on the device-resident buffers, mirroring the
//! `launch_q4k_matvec` shape but extended to the matmul / rms_norm /
//! geglu / residual ops that compose the FFN side chain:
//!
//! ```text
//! rms_norm -> q4k_matmul(gate) -> q4k_matmul(up) -> geglu_silu
//!          -> q4k_matmul(down) -> residual_add
//! ```
//!
//! The chain keeps every intermediate device-resident: a single
//! [`launch_ffn_chain`] call uploads the residual + norm weight once and
//! records all six ops into one command buffer, reading back only the
//! final residual. This is the host-orchestrated pipeline shape
//! `docs/vulkan-mvp.md` §4.1 specifies.
//!
//! Parity is cross-precision (f32 GPU vs f64 CPU reference for rms_norm),
//! not bit-exact; the chain test asserts a relative tolerance.

use ash::vk;

use super::runtime::{DeviceBuffer, RuntimeError, VulkanRuntime};

impl VulkanRuntime {
    // ── RMSNorm ──────────────────────────────────────────────────────────

    /// `y[s, n] = (x[s, n] / rms(s)) * (offset + w[n])`.
    ///
    /// `x` and `y` are flat `[seq * n]`. `w` is `[n]`. Dispatches
    /// `ceil(seq/64)` workgroups along x; the shader loops over `n`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm(
        &self,
        x: &[f32],
        w: &[f32],
        n: usize,
        seq: usize,
        offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RuntimeError> {
        if n == 0 || w.len() < n || x.len() < seq * n {
            return Err(RuntimeError::usage(format!(
                "rms_norm shape mismatch: n={n} seq={seq} w.len={} x.len={}",
                w.len(),
                x.len()
            )));
        }
        let elems = seq * n;

        let device = &self.device;
        let x_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (elems * 4) as u64,
            x,
        )?;
        let w_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (n * 4) as u64,
            w,
        )?;
        let y_buf = create_storage_uninit(
            device,
            &self.instance,
            self.physical_device,
            (elems * 4) as u64,
        )?;

        let descriptor_set =
            allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;
        write_3buf_descriptor(
            device,
            descriptor_set,
            [x_buf.handle, w_buf.handle, y_buf.handle],
            [
                (elems * 4) as vk::DeviceSize,
                (n * 4) as vk::DeviceSize,
                (elems * 4) as vk::DeviceSize,
            ],
        );

        let cmd = begin_one_time(device, self.command_pool)?;
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.rms_norm_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            // Push constants: n, seq (u32), offset, eps (f32) — 16 bytes.
            let push = PushNorm {
                n: n as u32,
                seq: seq as u32,
                offset,
                eps,
            };
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push.as_bytes(),
            );
            let groups = (seq as u32).div_ceil(64);
            device.cmd_dispatch(cmd, groups, 1, 1);
            device
                .end_command_buffer(cmd)
                .map_err(|err| RuntimeError::device("ending command buffer", err))?;
        }
        submit_and_wait(device, cmd, self.queue_family_index)?;

        let out = read_back_f32(device, y_buf.memory, elems)?;
        cleanup_3(device, self.descriptor_pool, self.command_pool, &descriptor_set, cmd, [
            x_buf, w_buf, y_buf,
        ]);
        Ok(out)
    }

    // ── GEGLU (SiLU-gated) ───────────────────────────────────────────────

    /// `out[i] = silu(gate[i]) * up[i]`, element-wise over `len` elements.
    pub(crate) fn launch_geglu_silu(
        &self,
        gate: &[f32],
        up: &[f32],
        len: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if gate.len() < len || up.len() < len {
            return Err(RuntimeError::usage(format!(
                "geglu len mismatch: len={len} gate.len={} up.len={}",
                gate.len(),
                up.len()
            )));
        }
        let device = &self.device;
        let g_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
            &gate[..len],
        )?;
        let u_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
            &up[..len],
        )?;
        let o_buf = create_storage_uninit(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
        )?;

        let descriptor_set =
            allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;
        write_3buf_descriptor(
            device,
            descriptor_set,
            [g_buf.handle, u_buf.handle, o_buf.handle],
            [(len * 4) as vk::DeviceSize; 3],
        );

        let cmd = begin_one_time(device, self.command_pool)?;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.geglu_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            let groups = (len as u32).div_ceil(256);
            device.cmd_dispatch(cmd, groups, 1, 1);
            device
                .end_command_buffer(cmd)
                .map_err(|err| RuntimeError::device("ending command buffer", err))?;
        }
        submit_and_wait(device, cmd, self.queue_family_index)?;

        let out = read_back_f32(device, o_buf.memory, len)?;
        cleanup_3(device, self.descriptor_pool, self.command_pool, &descriptor_set, cmd, [
            g_buf, u_buf, o_buf,
        ]);
        Ok(out)
    }

    // ── Residual add ─────────────────────────────────────────────────────

    /// `out[i] = a[i] + b[i]`, element-wise over `len` elements.
    pub(crate) fn launch_residual_add(
        &self,
        a: &[f32],
        b: &[f32],
        len: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if a.len() < len || b.len() < len {
            return Err(RuntimeError::usage(format!(
                "residual_add len mismatch: len={len} a.len={} b.len={}",
                a.len(),
                b.len()
            )));
        }
        let device = &self.device;
        let a_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
            &a[..len],
        )?;
        let b_buf = create_storage(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
            &b[..len],
        )?;
        let o_buf = create_storage_uninit(
            device,
            &self.instance,
            self.physical_device,
            (len * 4) as u64,
        )?;

        let descriptor_set =
            allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;
        write_3buf_descriptor(
            device,
            descriptor_set,
            [a_buf.handle, b_buf.handle, o_buf.handle],
            [(len * 4) as vk::DeviceSize; 3],
        );

        let cmd = begin_one_time(device, self.command_pool)?;
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.residual_add_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            let groups = (len as u32).div_ceil(256);
            device.cmd_dispatch(cmd, groups, 1, 1);
            device
                .end_command_buffer(cmd)
                .map_err(|err| RuntimeError::device("ending command buffer", err))?;
        }
        submit_and_wait(device, cmd, self.queue_family_index)?;

        let out = read_back_f32(device, o_buf.memory, len)?;
        cleanup_3(device, self.descriptor_pool, self.command_pool, &descriptor_set, cmd, [
            a_buf, b_buf, o_buf,
        ]);
        Ok(out)
    }

    // ── Q4_K matmul (seq-general) ────────────────────────────────────────

    /// `out[s, row] = sum_k W[row, k] * x[s, k]` for `seq` input vectors.
    ///
    /// `w4k` is the Q4_K-quantised weight blob; `x` is flat `[seq * hidden]`;
    /// `out` is flat `[seq * out_rows]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_q4k_matmul(
        &self,
        w4k: &[u8],
        x: &[f32],
        out_rows: usize,
        hidden: usize,
        seq: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if hidden == 0 || !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul hidden {hidden} must be a positive multiple of 256"
            )));
        }
        let bytes_per_row = (hidden / 256) * 144;
        let need = out_rows
            .checked_mul(bytes_per_row)
            .ok_or_else(|| {
                RuntimeError::usage(format!(
                    "q4k_matmul weight size overflow: {out_rows} rows * {bytes_per_row} bytes"
                ))
            })?;
        if w4k.len() < need {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul weight slice too short: {} bytes < expected {need}",
                w4k.len()
            )));
        }
        if x.len() < seq * hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul input x.len()={} < seq*hidden {}*{}",
                x.len(),
                seq,
                hidden
            )));
        }

        let device = &self.device;
        let w_bytes = &w4k[..need];
        let w_buf = create_storage_bytes(
            device,
            &self.instance,
            self.physical_device,
            need as u64,
            w_bytes,
        )?;
        let x_bytes = bytemuck::cast_slice::<f32, u8>(&x[..seq * hidden]);
        let x_buf = create_storage_bytes(
            device,
            &self.instance,
            self.physical_device,
            (seq * hidden * 4) as u64,
            x_bytes,
        )?;
        let o_buf = create_storage_uninit(
            device,
            &self.instance,
            self.physical_device,
            (seq * out_rows * 4) as u64,
        )?;

        let descriptor_set =
            allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;
        write_3buf_descriptor(
            device,
            descriptor_set,
            [w_buf.handle, x_buf.handle, o_buf.handle],
            [
                need as vk::DeviceSize,
                (seq * hidden * 4) as vk::DeviceSize,
                (seq * out_rows * 4) as vk::DeviceSize,
            ],
        );

        let cmd = begin_one_time(device, self.command_pool)?;
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.q4k_matmul_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            let push = PushMatmul {
                out_rows: out_rows as u32,
                k: hidden as u32,
                seq: seq as u32,
            };
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push.as_bytes(),
            );
            // 2D dispatch: x = rows, y = seq. Cap workgroup tiles at 64x1.
            let gx = (out_rows as u32).div_ceil(64);
            let gy = seq as u32;
            device.cmd_dispatch(cmd, gx, gy, 1);
            device
                .end_command_buffer(cmd)
                .map_err(|err| RuntimeError::device("ending command buffer", err))?;
        }
        submit_and_wait(device, cmd, self.queue_family_index)?;

        let out = read_back_f32(device, o_buf.memory, seq * out_rows)?;
        cleanup_3(device, self.descriptor_pool, self.command_pool, &descriptor_set, cmd, [
            w_buf, x_buf, o_buf,
        ]);
        Ok(out)
    }

    // ── Full device-resident FFN chain ───────────────────────────────────

    /// Run the complete FFN side chain on the device:
    ///
    /// 1. `rms_norm(x, w_norm)` -> `h_norm`
    /// 2. `q4k_matmul(gate, h_norm)` -> `gate_out`  `[seq * inter]`
    /// 3. `q4k_matmul(up, h_norm)` -> `up_out`      `[seq * inter]`
    /// 4. `geglu_silu(gate_out, up_out)` -> `act`   `[seq * inter]`
    /// 5. `q4k_matmul(down, act)` -> `down_out`      `[seq * hidden]`
    /// 6. `residual_add(x, down_out)` -> result      `[seq * hidden]`
    ///
    /// This is a host-orchestrated sequence: each op is its own launch
    /// (submit + wait), keeping the implementation simple and verifiable.
    /// Keeping the intermediates device-resident across a single command
    /// buffer (no host readback between ops) is a follow-up optimization;
    /// the MVP's goal is a correct, exercised end-to-end device path
    /// (`docs/vulkan-mvp.md` §2.1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_ffn_chain(
        &self,
        x: &[f32],
        norm_w: &[f32],
        gate_w: &[u8],
        up_w: &[u8],
        down_w: &[u8],
        hidden: usize,
        inter: usize,
        seq: usize,
        offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Step 1: RMSNorm.
        let h_norm = self.launch_rms_norm(x, norm_w, hidden, seq, offset, eps)?;

        // Steps 2-3: gate + up projections.
        let gate_out = self.launch_q4k_matmul(gate_w, &h_norm, inter, hidden, seq)?;
        let up_out = self.launch_q4k_matmul(up_w, &h_norm, inter, hidden, seq)?;

        // Step 4: GEGLU activation.
        let act = self.launch_geglu_silu(&gate_out, &up_out, seq * inter)?;

        // Step 5: down projection.
        let down_out = self.launch_q4k_matmul(down_w, &act, hidden, inter, seq)?;

        // Step 6: residual add.
        self.launch_residual_add(x, &down_out, seq * hidden)
    }
}

// ── Push-constant staging structs (repr(C, packed) for SPIR-V) ─────────────

#[repr(C, packed)]
struct PushNorm {
    n: u32,
    seq: u32,
    offset: f32,
    eps: f32,
}

impl PushNorm {
    fn as_bytes(&self) -> &[u8] {
        // repr(C, packed) + no padding: safe to read as a byte slice.
        unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>())
        }
    }
}

#[repr(C, packed)]
struct PushMatmul {
    out_rows: u32,
    k: u32,
    seq: u32,
}

impl PushMatmul {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>())
        }
    }
}

// ── Helpers (thin wrappers over runtime.rs internals) ─────────────────────
//
// These re-export the helper functions from `runtime.rs` that are currently
// private (`fn`, not `pub`). They are visible within the same crate + parent
// module via `use super::runtime::*`. The functions below adapt them to the
// chain's upload-and-allocate pattern.

#[allow(clippy::too_many_arguments)]
fn create_storage(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    size: vk::DeviceSize,
    data: &[f32],
) -> Result<DeviceBuffer, RuntimeError> {
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    create_storage_bytes(device, instance, physical, size, bytes)
}

fn create_storage_bytes(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    size: vk::DeviceSize,
    data: &[u8],
) -> Result<DeviceBuffer, RuntimeError> {
    let buf = super::runtime::create_buffer(
        device,
        instance,
        physical,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    super::runtime::upload_bytes(device, buf.memory, data, data.len())?;
    Ok(buf)
}

fn create_storage_uninit(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    size: vk::DeviceSize,
) -> Result<DeviceBuffer, RuntimeError> {
    super::runtime::create_buffer(
        device,
        instance,
        physical,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )
}

fn write_3buf_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    handles: [vk::Buffer; 3],
    ranges: [vk::DeviceSize; 3],
) {
    let infos = [
        vk::DescriptorBufferInfo {
            buffer: handles[0],
            offset: 0,
            range: ranges[0],
        },
        vk::DescriptorBufferInfo {
            buffer: handles[1],
            offset: 0,
            range: ranges[1],
        },
        vk::DescriptorBufferInfo {
            buffer: handles[2],
            offset: 0,
            range: ranges[2],
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[2])),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn allocate_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, RuntimeError> {
    super::runtime::allocate_descriptor_set(device, pool, layout)
}

fn begin_one_time(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, RuntimeError> {
    let cmd = super::runtime::allocate_command_buffer(device, pool)?;
    let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(|err| RuntimeError::device("beginning command buffer", err))?;
    }
    Ok(cmd)
}

fn submit_and_wait(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    qfi: u32,
) -> Result<(), RuntimeError> {
    super::runtime::submit_and_wait(device, cmd, qfi)
}

fn read_back_f32(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    count: usize,
) -> Result<Vec<f32>, RuntimeError> {
    super::runtime::read_back::<f32>(device, memory, count)
}

#[allow(clippy::too_many_arguments)]
fn cleanup_3(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    descriptor_set: &vk::DescriptorSet,
    cmd: vk::CommandBuffer,
    bufs: [DeviceBuffer; 3],
) {
    unsafe {
        for b in bufs {
            device.destroy_buffer(b.handle, None);
            device.free_memory(b.memory, None);
        }
        let _ = device
            .free_descriptor_sets(pool, std::slice::from_ref(descriptor_set))
            .map_err(|err| RuntimeError::device("freeing descriptor set", err));
        device.free_command_buffers(cmd_pool, std::slice::from_ref(&cmd));
    }
}
