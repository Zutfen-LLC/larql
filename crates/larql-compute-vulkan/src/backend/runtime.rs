//! Vulkan device runtime: device selection, command-buffer recording, and
//! the Q4_K matvec kernel launch.
//!
//! Mirrors CUDA's `backend/runtime.rs` shape (init -> compile/load -> launch)
//! but with the shader compiled at build time via shaderc and the resulting
//! `.spv` embedded at compile time (see `shaders/` + `spv/` + `build.rs`).
//! There is no runtime shader-compile dependency — a deploy host needs only
//! the Vulkan loader (`libvulkan.so.1`) and a driver.
//!
//! The runtime is created lazily from [`super::VulkanBackend`] and is `None`
//! on a host with no Vulkan loader or no suitable device. In that state the
//! backend keeps running on its CPU delegate and `supports_quant` reports
//! `false` — the "compiles on a GPU-less host, runs on CPU" property is
//! preserved (GPU-0002).
//!
//! Per the GPU-4001 MVP scope, the only native kernel here is `q4k_matvec`.
//! `q4k_matmul` (seq > 1) and the elementwise ops (rms_norm / rope /
//! residual_add) land in follow-up tasks; this task proves the dispatch
//! pipeline end-to-end with the single Q4_K matvec.

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use ash::{Entry, Instance};

use crate::options::BackendOptions;

/// The Q4_K matvec SPIR-V, compiled at build time from
/// `shaders/q4k_matmul.comp` and embedded as a checked-in byte array.
const Q4K_MATVEC_SPV: &[u8] = include_bytes!("../../spv/q4k_matmul.spv");

/// A live Vulkan device + queue + descriptor pool, holding the compiled
/// q4k_matvec pipeline. Built by [`VulkanRuntime::initialize`]; stored as
/// `Option<Arc<VulkanRuntime>>` on [`super::VulkanBackend`].
pub(crate) struct VulkanRuntime {
    _entry: Arc<Entry>,
    instance: Arc<Instance>,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    /// One descriptor set layout: 3 storage buffers (w, x, out) +
    /// push-constants (n, k).
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    q4k_matvec_pipeline: vk::Pipeline,
    #[allow(dead_code)]
    properties: vk::PhysicalDeviceProperties,
    /// Pre-formatted one-line summary built at init (uses the device name +
    /// `properties.api_version`). Stored so `summary()` can borrow from
    /// `self` without allocating per call.
    summary: String,
}

impl std::fmt::Debug for VulkanRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanRuntime")
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}

impl VulkanRuntime {
    /// Initialise the Vulkan runtime. Returns `Err` if there is no loader,
    /// no instance, or no compute-capable device. The caller treats any
    /// `Err` as "no native runtime" and keeps the CPU delegate.
    pub(crate) fn initialize(_ordinal: usize) -> Result<Self, RuntimeError> {
        // Entry::load() resolves the system Vulkan loader; a missing loader
        // becomes a clean error rather than a link failure (GPU-4001 §6.1).
        let entry = unsafe { Entry::load() }
            .map_err(|err| RuntimeError::loader("loading Vulkan loader", err))?;

        let instance = create_instance(&entry)?;
        let physical_device = pick_compute_device(&instance)?;

        let (device, queue_family_index) = create_device_and_queue(&instance, physical_device)?;
        let command_pool = create_command_pool(&device, queue_family_index)?;
        let descriptor_pool = create_descriptor_pool(&device)?;
        let descriptor_set_layout = create_descriptor_set_layout(&device)?;
        let pipeline_layout = create_pipeline_layout(&device, descriptor_set_layout)?;
        let q4k_matvec_pipeline =
            create_compute_pipeline(&device, pipeline_layout, Q4K_MATVEC_SPV, "main")?;

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    properties.device_name.as_ptr() as *const u8,
                    properties.device_name.len(),
                )
            };
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        };
        let summary = format!(
            "Vulkan device {} (api {}.{}); native q4k_matvec loaded, \
             remaining ops use CPU fallback",
            device_name,
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version),
        );

        Ok(Self {
            _entry: Arc::new(entry),
            instance,
            device,
            physical_device,
            queue_family_index,
            command_pool,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            q4k_matvec_pipeline,
            properties,
            summary,
        })
    }

    pub(crate) fn summary(&self) -> &str {
        &self.summary
    }

    /// Launch the Q4_K matvec kernel: `out[n] = W[n, k] @ x[k]`.
    ///
    /// Records a single command buffer: bind pipeline, bind the w/x/out
    /// storage buffers, push (n, k), dispatch, and submit+wait. The host
    /// readback is synchronous (one launch per call, no persistent
    /// device-resident weights in the MVP).
    pub(crate) fn launch_q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        let device = &self.device;
        // Validate shapes (mirror the CPU reference's contract).
        if hidden == 0 || !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec hidden {hidden} must be a positive multiple of 256"
            )));
        }
        let bytes_per_row = (hidden / 256) * 144;
        let need = num_rows.checked_mul(bytes_per_row).ok_or_else(|| {
            RuntimeError::usage(format!(
                "q4k_matvec weight size overflow: {num_rows} rows * {bytes_per_row} bytes"
            ))
        })?;
        if q4k_data.len() < need {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec weight slice too short: {} bytes < expected {need}",
                q4k_data.len()
            )));
        }
        if x.len() < hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec input x.len()={} < hidden {hidden}",
                x.len()
            )));
        }

        let n = num_rows as u32;

        // ── Allocate device buffers ───────────────────────────────────
        let w_buffer = create_buffer(
            device,
            &self.instance,
            self.physical_device,
            need as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let x_buffer = create_buffer(
            device,
            &self.instance,
            self.physical_device,
            (hidden * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        let out_buffer = create_buffer(
            device,
            &self.instance,
            self.physical_device,
            (num_rows * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        // ── Upload w + x via host-visible mapping ─────────────────────
        upload_bytes(device, w_buffer.memory, q4k_data, need)?;
        upload_bytes(
            device,
            x_buffer.memory,
            bytemuck::cast_slice::<f32, u8>(&x[..hidden]),
            hidden * 4,
        )?;

        // ── Allocate + write the descriptor set ──────────────────────
        let descriptor_set =
            allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;
        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: w_buffer.handle,
                offset: 0,
                range: need as vk::DeviceSize,
            },
            vk::DescriptorBufferInfo {
                buffer: x_buffer.handle,
                offset: 0,
                range: (hidden * 4) as vk::DeviceSize,
            },
            vk::DescriptorBufferInfo {
                buffer: out_buffer.handle,
                offset: 0,
                range: (num_rows * 4) as vk::DeviceSize,
            },
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2])),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // ── Record + submit the command buffer ───────────────────────
        let cmd = allocate_command_buffer(device, self.command_pool)?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|err| RuntimeError::device("beginning command buffer", err))?;
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.q4k_matvec_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            let push = [n, hidden as u32];
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice::<u32, u8>(&push),
            );
            let groups = n.div_ceil(256);
            device.cmd_dispatch(cmd, groups, 1, 1);
            device
                .end_command_buffer(cmd)
                .map_err(|err| RuntimeError::device("ending command buffer", err))?;
        }

        submit_and_wait(device, cmd, self.queue_family_index)?;

        // ── Read back ────────────────────────────────────────────────
        let out = read_back::<f32>(device, out_buffer.memory, num_rows)?;

        // ── Cleanup ──────────────────────────────────────────────────
        unsafe {
            device.destroy_buffer(w_buffer.handle, None);
            device.destroy_buffer(x_buffer.handle, None);
            device.destroy_buffer(out_buffer.handle, None);
            device.free_memory(w_buffer.memory, None);
            device.free_memory(x_buffer.memory, None);
            device.free_memory(out_buffer.memory, None);
            device
                .free_descriptor_sets(self.descriptor_pool, std::slice::from_ref(&descriptor_set))
                .map_err(|err| RuntimeError::device("freeing descriptor set", err))?;
            device.free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        Ok(out)
    }
}

impl Drop for VulkanRuntime {
    fn drop(&mut self) {
        unsafe {
            let device = &self.device;
            device.device_wait_idle().ok();
            device.destroy_pipeline(self.q4k_matvec_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Host-visible buffer helper ────────────────────────────────────────────

struct DeviceBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

fn find_memory_type(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, RuntimeError> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..props.memory_type_count {
        let mt = props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0 && mt.property_flags.contains(flags) {
            return Ok(i);
        }
    }
    Err(RuntimeError::usage(format!(
        "no memory type matching bits {type_bits:#x} and flags {flags:?}"
    )))
}

fn create_buffer(
    device: &ash::Device,
    instance: &Instance,
    physical: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_flags: vk::MemoryPropertyFlags,
) -> Result<DeviceBuffer, RuntimeError> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let handle = unsafe {
        device
            .create_buffer(&info, None)
            .map_err(|err| RuntimeError::device("creating buffer", err))?
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(handle) };
    let mem_type = find_memory_type(instance, physical, reqs.memory_type_bits, mem_flags)?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|err| RuntimeError::device("allocating buffer memory", err))?
    };
    unsafe {
        device
            .bind_buffer_memory(handle, memory, 0)
            .map_err(|err| RuntimeError::device("binding buffer memory", err))?
    }
    Ok(DeviceBuffer { handle, memory })
}

fn upload_bytes(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    data: &[u8],
    len: usize,
) -> Result<(), RuntimeError> {
    let ptr = unsafe {
        device
            .map_memory(
                memory,
                0,
                len as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|err| RuntimeError::device("mapping memory", err))?
    };
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, len);
        device.unmap_memory(memory);
    }
    Ok(())
}

fn read_back<T: bytemuck::Pod>(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    count: usize,
) -> Result<Vec<T>, RuntimeError> {
    let size = count * std::mem::size_of::<T>();
    let ptr = unsafe {
        device
            .map_memory(
                memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|err| RuntimeError::device("mapping output memory", err))?
    };
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, count) };
    let out = slice.to_vec();
    unsafe { device.unmap_memory(memory) };
    Ok(out)
}

// ── Instance / device / queue ─────────────────────────────────────────────

fn create_instance(entry: &Entry) -> Result<Arc<Instance>, RuntimeError> {
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    // Vulkan 1.1 (GPU-4001 §9.2). No extensions or layers in the MVP.
    let create = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe {
        entry
            .create_instance(&create, None)
            .map_err(|err| RuntimeError::instance("creating Vulkan instance", err))?
    };
    Ok(Arc::new(instance))
}

fn pick_compute_device(instance: &Instance) -> Result<vk::PhysicalDevice, RuntimeError> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|err| RuntimeError::instance("enumerating physical devices", err))?
    };
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if vk::api_version_major(props.api_version) >= 1
            && vk::api_version_minor(props.api_version) >= 1
            && find_compute_queue_family(instance, pd).is_some()
        {
            return Ok(pd);
        }
    }
    Err(RuntimeError::usage(
        "no Vulkan 1.1+ physical device with a compute queue".to_string(),
    ))
}

fn find_compute_queue_family(instance: &Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    for (i, qf) in families.iter().enumerate() {
        if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Some(i as u32);
        }
    }
    None
}

fn create_device_and_queue(
    instance: &Instance,
    pd: vk::PhysicalDevice,
) -> Result<(ash::Device, u32), RuntimeError> {
    let qfi = find_compute_queue_family(instance, pd).ok_or_else(|| {
        RuntimeError::usage("no compute queue family on chosen device".to_string())
    })?;
    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&priorities);
    let create =
        vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    let device = unsafe {
        instance
            .create_device(pd, &create, None)
            .map_err(|err| RuntimeError::device("creating logical device", err))?
    };
    Ok((device, qfi))
}

fn create_command_pool(device: &ash::Device, qfi: u32) -> Result<vk::CommandPool, RuntimeError> {
    let info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qfi)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe {
        device
            .create_command_pool(&info, None)
            .map_err(|err| RuntimeError::device("creating command pool", err))
    }
}

fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool, RuntimeError> {
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 16,
    };
    let info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(std::slice::from_ref(&pool_size))
        .max_sets(16)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe {
        device
            .create_descriptor_pool(&info, None)
            .map_err(|err| RuntimeError::device("creating descriptor pool", err))
    }
}

fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, RuntimeError> {
    let mk = |binding: u32| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    };
    let bindings = [mk(0), mk(1), mk(2)];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&info, None)
            .map_err(|err| RuntimeError::device("creating descriptor set layout", err))
    }
}

fn create_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, RuntimeError> {
    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 8, // two u32: n, k
    };
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));
    unsafe {
        device
            .create_pipeline_layout(&info, None)
            .map_err(|err| RuntimeError::device("creating pipeline layout", err))
    }
}

fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spirv: &[u8],
    entry_point: &str,
) -> Result<vk::Pipeline, RuntimeError> {
    // SPIR-V is a stream of u32 words.
    let words: &[u32] = bytemuck::try_cast_slice(spirv).map_err(|_| {
        RuntimeError::usage("SPIR-V bytecode length not a multiple of 4".to_string())
    })?;
    let module_info = vk::ShaderModuleCreateInfo::default().code(words);
    let module = unsafe {
        device
            .create_shader_module(&module_info, None)
            .map_err(|err| RuntimeError::device("creating shader module", err))?
    };
    let entry_cstr = CString::new(entry_point)
        .map_err(|_| RuntimeError::usage("invalid entry point name".to_string()))?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_cstr);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None)
            .map(|p| p[0])
            .map_err(|(_pipelines, err)| RuntimeError::compile("creating compute pipeline", err))?
    };
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

fn allocate_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, RuntimeError> {
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&layout));
    unsafe {
        device
            .allocate_descriptor_sets(&alloc)
            .map_err(|err| RuntimeError::device("allocating descriptor set", err))
    }
    .map(|sets| sets[0])
}

fn allocate_command_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, RuntimeError> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    unsafe {
        device
            .allocate_command_buffers(&info)
            .map_err(|err| RuntimeError::device("allocating command buffer", err))
    }
    .map(|cbs| cbs[0])
}

fn submit_and_wait(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    qfi: u32,
) -> Result<(), RuntimeError> {
    let queue = unsafe { device.get_device_queue(qfi, 0) };
    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .map_err(|err| RuntimeError::device("creating fence", err))?
    };
    unsafe {
        device
            .queue_submit(queue, std::slice::from_ref(&submit), fence)
            .map_err(|err| RuntimeError::device("submitting command buffer", err))?;
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .map_err(|err| RuntimeError::device("waiting for fence", err))?;
        device.destroy_fence(fence, None);
    }
    Ok(())
}

// ── Error type ────────────────────────────────────────────────────────────

/// Errors from the Vulkan runtime init or a kernel launch. The backend
/// treats any of these as "fall back to the CPU delegate" for the offending
/// op, so the failure never reaches the caller as a wrong result.
#[derive(Debug)]
pub(crate) enum RuntimeError {
    /// Vulkan loader (libvulkan) not present / not loadable.
    Loader(String),
    Instance(String),
    Device(String),
    Compile(String),
    /// Bad input shape / no suitable memory type / no device.
    Usage(String),
}

impl RuntimeError {
    fn loader(ctx: &str, err: ash::LoadingError) -> Self {
        Self::Loader(format!("{ctx}: {err}"))
    }
    fn instance(ctx: &str, err: vk::Result) -> Self {
        Self::Instance(format!("{ctx}: {err:?}"))
    }
    fn device(ctx: &str, err: vk::Result) -> Self {
        Self::Device(format!("{ctx}: {err:?}"))
    }
    fn compile(ctx: &str, err: vk::Result) -> Self {
        Self::Compile(format!("{ctx}: {err:?}"))
    }
    fn usage(msg: String) -> Self {
        Self::Usage(msg)
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loader(m)
            | Self::Instance(m)
            | Self::Device(m)
            | Self::Compile(m)
            | Self::Usage(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Probe whether a Vulkan runtime can be created (for `supports_quant`).
/// Cached by the backend so init runs once.
pub(crate) fn try_init(_options: &BackendOptions) -> Option<VulkanRuntime> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        VulkanRuntime::initialize(0)
    })) {
        Ok(Ok(rt)) => Some(rt),
        Ok(Err(_)) | Err(_) => None,
    }
}
