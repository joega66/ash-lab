use crate::{
    AddressSlot, TypeLayout, UInt3,
    descriptor_pool::*,
    shader_module::*,
    shader_parameter::*,
    shader_permutation::*,
    shader_type::ShaderType,
    utils::Handle,
    utils::{Arena, DeviceOwner, InstanceOwner},
};
use ash::{
    Device, Entry, Instance, khr,
    vk::{self, TaggedStructure},
};
use bytemuck::{AnyBitPattern, Pod};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    ffi::CStr,
    fmt,
    io::Cursor,
    marker::PhantomData,
    ops::Index,
    rc::Rc,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
};
use vk_mem::{Alloc, AllocationCreateFlags, MemoryUsage};
use winit::window::Window;

pub struct DeviceContextCreateInfo {
    pub display_handle: Option<RawDisplayHandle>,
    pub window_handle: Option<RawWindowHandle>,
}

impl Default for DeviceContextCreateInfo {
    fn default() -> Self {
        Self {
            display_handle: None,
            window_handle: None,
        }
    }
}

pub struct DeviceContext {
    surface_loader: Option<khr::surface::Instance>,
    surface: Option<vk::SurfaceKHR>,

    physical_device: vk::PhysicalDevice,

    graphics_family: u32,
    present_family: Option<u32>,

    graphics_queue: vk::Queue,
    present_queue: Option<vk::Queue>,

    graphics_command_pool: vk::CommandPool,

    swapchain_loader: khr::swapchain::Device,

    pub(crate) graphics_timeline: QueueTimeline,

    trash_tx: Sender<Trash>,
    trash_rx: Receiver<Trash>,
    trash: VecDeque<(u64, Trash)>,

    mem_allocator: vk_mem::Allocator,

    set_allocators: HashMap<vk::DescriptorSetLayout, DescriptorSetAllocator>,

    shaders: HashMap<std::any::TypeId, ShaderModuleArray>,

    functions: HashMap<std::any::TypeId, StaticDeviceFunctionArray>,

    buffers: Arena<DeviceBufferInner>,

    rg: RgContext,

    pub device: DeviceOwner,

    #[allow(dead_code)]
    instance: InstanceOwner,

    #[allow(dead_code)]
    entry: Entry,
}

impl DeviceContext {
    /// Returns a GPU device.
    pub fn new(info: &DeviceContextCreateInfo) -> Self {
        let api_version = vk::API_VERSION_1_4;

        let entry =
            unsafe { Entry::load().expect("failed to load Vulkan library (libvulkan.dylib)") };

        let (instance, _portability_enabled) =
            unsafe { Self::create_instance(&entry, api_version, &info.display_handle) };

        let (surface_loader, surface) = (|| {
            let Some(display_handle) = info.display_handle.as_ref() else {
                return (None, None);
            };
            let Some(window_handle) = info.window_handle.as_ref() else {
                return (None, None);
            };
            let surface_loader = khr::surface::Instance::load(&entry, &instance);
            let surface = unsafe {
                ash_window::SurfaceFactory::new(&entry, &instance, *display_handle)
                    .expect("failed to load surface extension")
                    .create_surface(*window_handle, None)
                    .expect("failed to create surface")
            };
            (Some(surface_loader), Some(surface))
        })();

        let (physical_device, graphics_family, present_family, portability_subset) =
            unsafe { Self::pick_physical_device(&instance, &surface_loader, &surface) };

        let (device, graphics_queue, present_queue) = unsafe {
            Self::create_logical_device(
                &instance,
                physical_device,
                graphics_family,
                &present_family,
                portability_subset,
            )
        };

        let graphics_command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(graphics_family),
                    None,
                )
                .expect("failed to create command pool")
        };

        let shaders = unsafe { Self::create_shaders(&device) };

        let mut set_allocators = HashMap::new();

        let functions =
            unsafe { Self::create_device_functions(&device, &mut set_allocators, &shaders) };

        let swapchain_loader = khr::swapchain::Device::load(&instance, &device);

        let graphics_timeline = unsafe { Self::create_queue_timeline(&device) };

        let (trash_tx, trash_rx) = mpsc::channel();

        let trash = VecDeque::new();

        let mem_allocator = unsafe {
            let mut create_info =
                vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            create_info.vulkan_api_version = api_version;
            create_info.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            vk_mem::Allocator::new(create_info).expect("failed to initialize VMA")
        };

        Self {
            surface_loader,
            surface,
            physical_device,
            graphics_family,
            present_family,
            graphics_queue,
            present_queue,
            graphics_command_pool,
            swapchain_loader,
            graphics_timeline,
            trash_tx,
            trash_rx,
            trash,
            mem_allocator,
            set_allocators,
            shaders,
            functions,
            buffers: Arena::new(),
            rg: RgContext::new(),
            device: DeviceOwner::new(device),
            instance: InstanceOwner::new(instance),
            entry,
        }
    }

    /// Creates a buffer synchronously using the DeviceBuffer constructor.
    pub fn create_buffer<T>(&mut self, name: &str, len: usize) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::empty(),
            usage: MemoryUsage::Auto,
            ..Default::default()
        };
        self.create_buffer_inner(
            name,
            len * std::mem::size_of::<T>(),
            default_buffer_usage(),
            &memory_info,
        )
    }

    /// Enqueues the creation of a HostBuffer.
    /// This function allocates memory on the host that is accessible by the device.
    pub fn create_host_buffer<T>(&mut self, name: &str, len: usize) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::HOST_ACCESS_RANDOM,
            usage: MemoryUsage::AutoPreferHost,
            ..Default::default()
        };
        self.create_buffer_inner(
            name,
            len * std::mem::size_of::<T>(),
            default_buffer_usage(),
            &memory_info,
        )
    }

    /// Returns a new buffer for use as a ConstantBuffer descriptor.
    pub fn create_constant_buffer<T>(&mut self, name: &str) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::empty(),
            usage: MemoryUsage::Auto,
            ..Default::default()
        };
        self.create_buffer_inner(
            name,
            std::mem::size_of::<T>(),
            default_buffer_usage() | vk::BufferUsageFlags::UNIFORM_BUFFER,
            &memory_info,
        )
    }

    /// Compiles the provided function for execution on this device.
    pub fn compile_function<'a, T>(
        &self,
        permutation: &<T::Shader as ShaderModuleLike>::Permutations,
        spec_constant: Option<<T as DeviceFunctionLike>::SpecConstant>,
    ) -> DeviceFunction<T>
    where
        T: DeviceFunctionLike + 'static,
        <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
        <T as DeviceFunctionLike>::SpecConstant: ShaderType + Pod,
    {
        let function = self.get_function::<T>(&permutation);

        let shader = self.get_shader::<T::Shader>(&permutation);

        let entry_point_c_str = std::ffi::CString::new(T::new().entry_point()).unwrap();

        let map_entries: Result<_, _> = match spec_constant {
            Some(_) => {
                let spec_constant_layout = <T as DeviceFunctionLike>::SpecConstant::type_layout();
                if let TypeLayout::Struct { fields, .. } = spec_constant_layout {
                    let mut map_entries = Vec::new();
                    for (field, constant_id) in std::iter::zip(&fields, 0..fields.len()) {
                        map_entries.push(vk::SpecializationMapEntry {
                            constant_id: constant_id as u32,
                            offset: field.offset,
                            size: field.ty.size() as usize,
                        });
                    }
                    Ok(map_entries)
                } else {
                    // This situation should be impossible
                    Err(format!(
                        "expected a Struct, got a {:?}",
                        spec_constant_layout
                    ))
                }
            }
            None => Ok(Vec::new()),
        };

        let map_entries = match map_entries {
            Ok(map_entries) => map_entries,
            Err(e) => panic!("{:?}", e),
        };

        let specialization_info = vk::SpecializationInfo::default()
            .map_entries(&map_entries)
            .data(
                spec_constant
                    .as_ref()
                    .map_or(&[], |x| bytemuck::bytes_of(x)),
            );

        let pssci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader)
            .name(entry_point_c_str.as_c_str())
            .specialization_info(&specialization_info);

        let cpci = vk::ComputePipelineCreateInfo::default()
            .stage(pssci)
            .layout(function.pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None)
                .expect("failed to create compute pipelines")[0]
        };

        DeviceFunction::<T> {
            pipeline: pipeline,
            permutation: permutation.clone(),
            trash_tx: self.trash_tx.clone(),
            _marker: PhantomData {},
        }
    }

    /// Enqueues a buffer creation using the DeviceBuffer constructor.
    pub fn enqueue_create_buffer<T>(&mut self, name: &str, len: usize) -> DeviceBuffer<T> {
        self.create_buffer::<T>(name, len)
    }

    /// Enqueues an operation to fill this buffer with a specified value.
    pub fn enqueue_fill<T: U32Castable + 'static>(&mut self, input: &DeviceBuffer<T>, value: T) {
        let input = input.clone();
        let value = value.to_u32();
        self.enqueue_pass(
            "enqueue_fill",
            &[RgBufferTransition {
                buffer: input.buffer(),
                kind: RgAccessKind::Write,
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            }],
            &[],
            Box::new(move |ctx, command_buffer| {
                unsafe {
                    ctx.device.cmd_fill_buffer(
                        command_buffer,
                        input.buffer(),
                        0,
                        input.size() as vk::DeviceSize,
                        value,
                    )
                };
            }),
        );
    }

    /// Enqueues an async copy from the host to the provided device buffer. The
    /// number of bytes copied is determined by the size of the device buffer.
    pub fn enqueue_copy<T, U: EnqueueCopyable<T>>(
        &mut self,
        src_buf: U,
        dst_buf: &DeviceBuffer<T>,
    ) {
        <U as EnqueueCopyable<T>>::enqueue_copy(self, src_buf, dst_buf);
    }

    /// Enqueues an external device function for execution on this device.
    pub fn enqueue_function<T>(
        &mut self,
        permutation: &<T::Shader as ShaderModuleLike>::Permutations,
        parameters: <T as DeviceFunctionLike>::Params,
        push_constant: <T as DeviceFunctionLike>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionLike + 'static,
        <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
        <T as DeviceFunctionLike>::PushConstant: ShaderType + Pod,
    {
        self.enqueue_function_inner::<T>(None, &permutation, parameters, push_constant, grid_dim);
    }

    /// Enqueues an external device function for execution on this device.
    /// This overload accepts a [`DeviceFunction<T>`] created at runtime.
    pub fn enqueue_function_object<T>(
        &mut self,
        function: &DeviceFunction<T>,
        parameters: <T as DeviceFunctionLike>::Params,
        push_constant: <T as DeviceFunctionLike>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionLike + 'static,
        <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
        <T as DeviceFunctionLike>::PushConstant: ShaderType + Pod,
    {
        self.enqueue_function_inner::<T>(
            Some(function.pipeline),
            &function.permutation,
            parameters,
            push_constant,
            grid_dim,
        );
    }

    pub fn enqueue_pass(
        &mut self,
        name: &str,
        buffers: &[RgBufferTransition],
        images: &[RgImageTransition],
        function: Box<dyn Fn(&mut DeviceContext, vk::CommandBuffer)>,
    ) {
        let pass_id = RgPassId(self.rg.passes.len());
        let mut reads = Vec::new();
        let mut writes = Vec::new();

        for t in buffers {
            self.version_use(
                pass_id,
                RgResource::Buffer(t.buffer),
                t.kind,
                &mut reads,
                &mut writes,
            );
        }
        for t in images {
            self.version_use(
                pass_id,
                RgResource::Image(t.image),
                t.kind,
                &mut reads,
                &mut writes,
            );
        }

        self.rg.passes.push(RgPass {
            name: name.into(),
            barrier: RgPipelineBarrier {
                buffers: buffers.to_vec(),
                images: images.to_vec(),
            },
            function,
        });
    }

    fn enqueue_function_inner<T>(
        &mut self,
        pipeline: Option<vk::Pipeline>,
        permutation: &<T::Shader as ShaderModuleLike>::Permutations,
        parameters: <T as DeviceFunctionLike>::Params,
        push_constant: <T as DeviceFunctionLike>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionLike + 'static,
        <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
        <T as DeviceFunctionLike>::PushConstant: ShaderType + Pod,
    {
        // Get the ShaderParameters as a list.
        let parameters = parameters.parameters();

        // Pad push_constant block out to the same word granularity the range was created with.
        let mut push_constant_bytes = bytemuck::bytes_of(&push_constant).to_vec();
        push_constant_bytes.resize(push_constant_bytes.len().next_multiple_of(4), 0);

        // This is a compute pipe.
        let stage = vk::PipelineStageFlags2::COMPUTE_SHADER;

        // Derive buffer transitions from descriptor kinds.
        let mut transitions: Vec<RgBufferTransition> = parameters
            .iter()
            .map(|param| {
                let buffer = self.buffers.get(param.handle).unwrap().buffer;
                match param.kind {
                    DescriptorKind::ConstantBuffer => constant_buffer_read(buffer, stage),
                    DescriptorKind::StructuredBuffer => storage_buffer_read(buffer, stage),
                    DescriptorKind::RWStructuredBuffer => storage_buffer_read_write(buffer, stage),
                }
            })
            .collect();

        let function_type = std::any::TypeId::of::<T>();
        let permutation_index = permutation.flatten();
        let functions = self.functions.get(&function_type).expect(&format!(
            "missing function {}:{permutation_index}:{:?}",
            std::any::type_name::<T>(),
            permutation.defines()
        ));
        let function = functions.permutations[permutation_index]
            .as_ref()
            .expect(&format!(
                "missing function permutation {}:{permutation_index}:{:?}",
                std::any::type_name::<T>(),
                permutation.defines()
            ));
        let pipeline = match pipeline {
            Some(pipeline) => pipeline,
            None => function.pipeline,
        };
        let StaticDeviceFunction {
            set_layout,
            pipeline_layout,
            ..
        } = function.clone();

        // Insert transitions for DeviceAddress in the push constant blob.
        let address_slots = functions.address_slots.clone();
        transitions.extend(self.address_transitions(&address_slots, &push_constant_bytes, stage));

        // Enqueue the function.
        self.enqueue_pass(
            std::any::type_name::<T>(),
            &transitions,
            &[],
            Box::new(
                move |ctx: &mut DeviceContext, command_buffer: vk::CommandBuffer| {
                    let push_constant_bytes = if address_slots.is_empty() {
                        push_constant_bytes.clone()
                    } else {
                        let mut bytes = push_constant_bytes.clone();
                        for slot in address_slots.iter() {
                            let at = slot.offset as usize;
                            let raw = u64::from_ne_bytes(
                                bytes[at..at + 8].try_into().expect("slot is 8 bytes"),
                            );
                            let handle = Handle::<DeviceBufferInner>::from_u64(raw);
                            let buffer = ctx.buffers.get(handle).unwrap();
                            bytes[at..at + 8].copy_from_slice(&buffer.address().to_ne_bytes());
                        }
                        bytes
                    };

                    let set = (!parameters.is_empty()).then(|| {
                        let buffers: Vec<vk::Buffer> = parameters
                            .iter()
                            .map(|arg| match arg.kind {
                                DescriptorKind::ConstantBuffer
                                | DescriptorKind::StructuredBuffer
                                | DescriptorKind::RWStructuredBuffer => {
                                    let buffer = ctx.buffers.get(arg.handle).unwrap();
                                    buffer.buffer
                                }
                            })
                            .collect();

                        // TODO: Batch all descriptor set allocations + updates in graph preamble
                        let mut set_allocators = std::mem::take(&mut ctx.set_allocators);
                        let set_allocator = set_allocators.get_mut(&set_layout).unwrap();
                        let result =
                            set_allocator.allocate_descriptor_set(ctx, set_layout, buffers);
                        ctx.set_allocators = set_allocators;

                        match result {
                            DescriptorSetCacheLookup::Miss(set) => {
                                let mut buffer_infos = Vec::new();
                                for arg in &parameters {
                                    if arg.kind.is_buffer() {
                                        let buffer = ctx.buffers.get(arg.handle).unwrap();
                                        let buffer_info = vec![
                                            vk::DescriptorBufferInfo::default()
                                                .buffer(buffer.buffer)
                                                .offset(0)
                                                .range(buffer.size as vk::DeviceSize),
                                        ];
                                        buffer_infos.push(buffer_info);
                                    }
                                }
                                let mut writes = Vec::new();
                                let mut buffer_info_idx = 0;
                                for (i, arg) in parameters.iter().enumerate() {
                                    let mut write = vk::WriteDescriptorSet::default()
                                        .dst_set(set)
                                        .dst_binding(i as u32)
                                        .dst_array_element(0)
                                        .descriptor_count(1)
                                        .descriptor_type(arg.kind.into());
                                    if arg.kind.is_buffer() {
                                        write = write.buffer_info(&buffer_infos[buffer_info_idx]);
                                        buffer_info_idx += 1;
                                    } else {
                                        unimplemented!();
                                    }
                                    writes.push(write);
                                }
                                unsafe {
                                    ctx.device.update_descriptor_sets(&writes, &[]);
                                }
                                set
                            }
                            DescriptorSetCacheLookup::Hit(set) => set,
                        }
                    });

                    unsafe {
                        ctx.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline,
                        );
                        if let Some(set) = set {
                            ctx.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline_layout,
                                0,
                                &[set],
                                &[],
                            );
                        }
                        if !push_constant_bytes.is_empty() {
                            ctx.device.cmd_push_constants(
                                command_buffer,
                                pipeline_layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                &push_constant_bytes,
                            );
                        }
                        ctx.device.cmd_dispatch(
                            command_buffer,
                            grid_dim.x(),
                            grid_dim.y(),
                            grid_dim.z(),
                        );
                    }
                },
            ),
        );
    }

    fn version_use(
        &mut self,
        pass_id: RgPassId,
        resource: RgResource,
        kind: RgAccessKind,
        reads: &mut Vec<RgVersion>,
        writes: &mut Vec<RgVersion>,
    ) {
        let does_read = matches!(kind, RgAccessKind::Read | RgAccessKind::ReadWrite);
        let does_write = matches!(kind, RgAccessKind::Write | RgAccessKind::ReadWrite);

        if does_read {
            let rv = RgVersion {
                resource,
                version: self.rg.versions.entry(resource).or_insert(0).clone(),
            };
            if let Some(&prod) = self.rg.writers.get(&rv) {
                self.add_edge(prod, pass_id); // RAW
            }
            self.rg.readers.entry(rv).or_default().push(pass_id);
            reads.push(rv);
        }

        if does_write {
            let old = RgVersion {
                resource,
                version: self.rg.versions.entry(resource).or_insert(0).clone(),
            };
            if let Some(&prod) = self.rg.writers.get(&old) {
                self.add_edge(prod, pass_id); // WAW
            }
            if let Some(prev_readers) = self.rg.readers.get(&old).cloned() {
                for r in prev_readers {
                    self.add_edge(r, pass_id); // WAR (self-edge skipped)
                }
            }
            let new_ver = self.rg.versions[&resource] + 1;
            self.rg.versions.insert(resource, new_ver);
            let nv = RgVersion {
                resource,
                version: new_ver,
            };
            self.rg.writers.insert(nv, pass_id);
            writes.push(nv);
        }
    }

    // TODO: Return the executed graph in a pretty form
    pub fn execute(&mut self, present: Option<DeviceImage>) -> Result<(), String> {
        let result = self.execute_inner(present);

        self.garbage_collection();

        self.rg = RgContext::new();

        result
    }

    fn execute_inner(&mut self, present: Option<DeviceImage>) -> Result<(), String> {
        // Sort passes.
        let order = self.topological_sort()?;

        // Initialize state tracking.
        let mut states = RgStates {
            buffers: HashMap::new(),
            images: HashMap::new(),
        };

        // Start recording.
        let command_buffer = unsafe {
            self.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.graphics_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("failed to allocate command buffers")
                .first()
                .unwrap()
                .clone()
        };
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .expect("failed to begin recording command buffer");
        }

        // Record commands.
        let passes = std::mem::take(&mut self.rg.passes);
        for pass_id in &order {
            let pass = &passes[pass_id.0];

            let (memory_barriers, image_memory_barriers) = self.derive_barriers(&mut states, pass);

            if memory_barriers.len() > 0 || image_memory_barriers.len() > 0 {
                let dependency_info = &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .memory_barriers(&memory_barriers)
                    .image_memory_barriers(&image_memory_barriers);
                unsafe {
                    self.device
                        .cmd_pipeline_barrier2(command_buffer, &dependency_info);
                }
            }

            pass.function.as_ref()(self, command_buffer);
        }
        drop(passes);

        if let Some(present) = present {
            let prev = states
                .images
                .entry(present.image())
                .or_insert(RgImageState::initial());
            if prev.layout != vk::ImageLayout::PRESENT_SRC_KHR {
                let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(prev.stage)
                    .src_access_mask(prev.access)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(prev.layout)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(present.image())
                    .subresource_range(*present.subresource_range())];

                let dependency_info = &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .memory_barriers(&[])
                    .image_memory_barriers(&image_memory_barriers);

                unsafe {
                    self.device
                        .cmd_pipeline_barrier2(command_buffer, dependency_info);
                }
            }
        }

        // Stop recording.
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("failed to end recording command buffer");
        }

        // Submit commands.
        self.queue_submit(
            QueueType::Graphics,
            command_buffer,
            vk::PipelineStageFlags2::NONE,
        );

        // Cleanup.
        let command_pool = self.graphics_command_pool.clone();
        let _ = self
            .trash_tx
            .send(Trash::Generic(Box::new(move |device| unsafe {
                device.free_command_buffers(command_pool, &[command_buffer]);
            })));

        Ok(())
    }

    fn add_edge(&mut self, from: RgPassId, to: RgPassId) {
        if from != to {
            self.rg.edges.insert((from, to));
        }
    }

    fn topological_sort(&self) -> Result<Vec<RgPassId>, String> {
        // n passes
        let n = self.rg.passes.len();

        // Initialize per-node indegree counts to 0.
        let mut in_degree = vec![0usize; n];

        // Initialize per-node adjacency lists. (Initially empty.)
        let mut adj: Vec<Vec<RgPassId>> = vec![Vec::new(); n];

        // For each edge:
        for &(from, to) in &self.rg.edges {
            // Add dst node to src node's adjacency list.
            adj[from.0].push(to);

            // Increment dst node's indegree.
            in_degree[to.0] += 1;
        }

        // Sort each node's adjacency list by increasing node index.
        for list in adj.iter_mut() {
            list.sort_by_key(|p| p.0);
        }

        // Initialize queue with indegree 0 nodes
        let mut queue: VecDeque<RgPassId> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .map(RgPassId)
            .collect();

        // Order to execute passes
        let mut order = Vec::with_capacity(n);

        // While indegree 0 nodes remain:
        while let Some(p) = queue.pop_front() {
            // Add the node to the list of passes to execute.
            order.push(p);

            // Visit the node's adjacency list.
            for &next in &adj[p.0] {
                // Every node in the adjacency list has one fewer indegree.
                in_degree[next.0] -= 1;

                // When a node's indegree reaches 0, every one of its dependencies
                // has run before it, so we can add it to the indegree 0 queue.
                if in_degree[next.0] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            return Err("cycle detected in render graph".into());
        }

        Ok(order)
    }

    fn merged_buffer_uses(
        pass: &RgPass,
    ) -> Vec<(vk::Buffer, vk::PipelineStageFlags2, vk::AccessFlags2)> {
        let mut map: HashMap<vk::Buffer, (vk::PipelineStageFlags2, vk::AccessFlags2)> =
            HashMap::new();
        for t in &pass.barrier.buffers {
            let e = map
                .entry(t.buffer)
                .or_insert((vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE));
            e.0 |= t.stage;
            e.1 |= t.access;
        }
        let mut v: Vec<_> = map.into_iter().map(|(i, (s, a))| (i, s, a)).collect();
        v.sort_by_key(|(i, _, _)| *i);
        v
    }

    fn merged_image_uses(
        pass: &RgPass,
    ) -> Vec<(
        vk::Image,
        vk::ImageSubresourceRange,
        vk::PipelineStageFlags2,
        vk::AccessFlags2,
        vk::ImageLayout,
    )> {
        let mut map: HashMap<
            vk::Image,
            (
                vk::ImageSubresourceRange,
                vk::PipelineStageFlags2,
                vk::AccessFlags2,
                vk::ImageLayout,
            ),
        > = HashMap::new();
        for t in &pass.barrier.images {
            let e = map.entry(t.image).or_insert((
                vk::ImageSubresourceRange::default(),
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                t.layout,
            ));
            e.0 = t.subresource_range;
            e.1 |= t.stage;
            e.2 |= t.access;
            e.3 = t.layout;
        }
        let mut v: Vec<_> = map
            .into_iter()
            .map(|(i, (r, s, a, l))| (i, r, s, a, l))
            .collect();
        v.sort_by_key(|(i, _, _, _, _)| *i);
        v
    }

    fn derive_barriers(
        &self,
        states: &mut RgStates,
        pass: &RgPass,
    ) -> (
        Vec<vk::MemoryBarrier2<'_>>,
        Vec<vk::ImageMemoryBarrier2<'_>>,
    ) {
        let mut memory_barriers = Vec::new();
        let mut image_memory_barriers = Vec::new();

        let write_mask: vk::AccessFlags2 = vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE
            | vk::AccessFlags2::MEMORY_WRITE;

        for (buffer, dst_stage, dst_access) in Self::merged_buffer_uses(pass) {
            let prev = states
                .buffers
                .entry(buffer)
                .or_insert(RgBufferState::initial());
            let cur_write = dst_access & write_mask != vk::AccessFlags2::NONE;

            if prev.was_write || cur_write {
                memory_barriers.push(
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(prev.stage)
                        .src_access_mask(prev.access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access),
                );
            }

            states.buffers.insert(
                buffer,
                RgBufferState {
                    stage: dst_stage,
                    access: dst_access,
                    was_write: cur_write,
                },
            );
        }

        for (image, subresource_range, dst_stage, dst_access, new_layout) in
            Self::merged_image_uses(pass)
        {
            let prev = states
                .images
                .entry(image)
                .or_insert(RgImageState::initial());
            let cur_write = dst_access & write_mask != vk::AccessFlags2::NONE;
            let layout_change = prev.layout != new_layout;

            if layout_change || prev.was_write || cur_write {
                image_memory_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(prev.stage)
                        .src_access_mask(prev.access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(prev.layout)
                        .new_layout(new_layout)
                        .image(image)
                        .subresource_range(subresource_range),
                );
            }

            states.images.insert(
                image,
                RgImageState {
                    stage: dst_stage,
                    access: dst_access,
                    layout: new_layout,
                    was_write: cur_write,
                },
            );
        }

        (memory_barriers, image_memory_barriers)
    }

    fn address_transitions(
        &self,
        slots: &[AddressSlot],
        bytes: &[u8],
        stage: vk::PipelineStageFlags2,
    ) -> Vec<RgBufferTransition> {
        slots
            .iter()
            .map(|slot| {
                let at = slot.offset as usize;
                let raw = u64::from_ne_bytes(
                    bytes
                        .get(at..at + 8)
                        .expect("address slot lies outside the push constant block")
                        .try_into()
                        .unwrap(),
                );
                let handle = Handle::<DeviceBufferInner>::from_u64(raw);
                let buffer = self.buffers.get(handle).unwrap();
                if slot.writable {
                    storage_buffer_read_write(buffer.buffer, stage)
                } else {
                    storage_buffer_read(buffer.buffer, stage)
                }
            })
            .collect()
    }

    /// Returns a shader module compiled ahead-of-time.
    pub fn get_shader<T>(
        &self,
        permutation: &<T as ShaderModuleLike>::Permutations,
    ) -> &vk::ShaderModule
    where
        T: ShaderModuleLike + 'static,
    {
        // TypeId -> Permutation -> ShaderModuleLike
        let k = TypeId::of::<T>();
        let shader_vec = self
            .shaders
            .get(&k)
            .expect(&format!("missing shader {}", std::any::type_name::<T>()));
        let permutation_index = permutation.flatten();
        let shader = shader_vec.0.get(permutation_index).expect(&format!(
            "missing shader permutation {}:{permutation_index}:{:?}",
            std::any::type_name::<T>(),
            permutation.defines()
        ));
        shader.as_ref().expect(&format!(
            "missing shader permutation {}:{permutation_index}:{:?}",
            std::any::type_name::<T>(),
            permutation.defines()
        ))
    }

    /// Blocks until all asynchronous calls on the stream associated with this device context have completed.
    pub fn synchronize(&self) {
        let semaphores = [self.graphics_timeline.semaphore];
        let values = [self.graphics_timeline.value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            self.device
                .wait_semaphores(&wait_info, u64::MAX)
                .expect("wait semaphores failed")
        };
    }

    unsafe fn create_instance(
        entry: &Entry,
        api_version: u32,
        display_handle: &Option<raw_window_handle::RawDisplayHandle>,
    ) -> (Instance, bool) {
        let app_name = c"hello_ash";
        let engine_name = c"No Engine";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(api_version);

        let mut extension_names = Vec::new();
        if let Some(display_handle) = display_handle.as_ref() {
            extension_names.extend(
                ash_window::enumerate_required_extensions(*display_handle)
                    .expect("failed to query required surface extensions"),
            );
        }

        // Vulkan on macOS/iOS is provided through MoltenVK.
        let mut portability_enabled = false;
        let available_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .unwrap_or_default()
        };
        if available_extensions
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(khr::portability_enumeration::NAME))
        {
            extension_names.push(khr::portability_enumeration::NAME.as_ptr());
            portability_enabled = true;
        }

        let flags = if portability_enabled {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(flags);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("failed to create Vulkan instance")
        };

        (instance, portability_enabled)
    }

    unsafe fn pick_physical_device(
        instance: &Instance,
        surface_loader: &Option<khr::surface::Instance>,
        surface: &Option<vk::SurfaceKHR>,
    ) -> (vk::PhysicalDevice, u32, Option<u32>, bool) {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("failed to enumerate physical devices")
        };
        let result = (move || {
            for physical_device in physical_devices {
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };

                let mut graphics_family = None;
                let mut present_family = None;

                for (index, family) in queue_families.iter().enumerate() {
                    let index = index as u32;

                    if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_family = Some(index);
                    }

                    if let (Some(surface_loader), Some(surface)) =
                        (surface_loader.as_ref(), surface.as_ref())
                    {
                        let supports_present = unsafe {
                            surface_loader
                                .get_physical_device_surface_support(
                                    physical_device,
                                    index,
                                    *surface,
                                )
                                .unwrap_or(false)
                        };

                        if supports_present {
                            present_family = Some(index);
                        }
                    }

                    // With surface: Find graphics + present
                    // Without surface: Find graphics
                    if graphics_family.is_some() && (present_family.is_some() || surface.is_none())
                    {
                        break;
                    }
                }

                if graphics_family.is_none() || (present_family.is_none() && surface.is_some()) {
                    continue;
                }

                let extension_properties = unsafe {
                    instance
                        .enumerate_device_extension_properties(physical_device)
                        .unwrap_or_default()
                };
                let has_extension = |name: &CStr| {
                    extension_properties
                        .iter()
                        .any(|ext| ext.extension_name_as_c_str() == Ok(name))
                };

                if !has_extension(khr::swapchain::NAME) {
                    continue;
                }
                let portability_subset = has_extension(khr::portability_subset::NAME);

                return Ok((
                    physical_device,
                    graphics_family.unwrap(),
                    present_family,
                    portability_subset,
                ));
            }
            Err("no suitable Vulkan physical device found")
        })();
        let result = match result {
            Ok(result) => result,
            Err(e) => panic!("{e}"),
        };
        result
    }

    unsafe fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        graphics_family: u32,
        present_family: &Option<u32>,
        portability_subset: bool,
    ) -> (Device, vk::Queue, Option<vk::Queue>) {
        let mut unique_families = vec![graphics_family];
        if let Some(present_family) = present_family {
            if *present_family != graphics_family {
                unique_families.push(*present_family);
            }
        }

        let queue_priorities = [1.0f32];
        let queue_create_infos: Vec<_> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        let mut extension_names = Vec::new();
        if portability_subset {
            extension_names.push(khr::portability_subset::NAME.as_ptr());
        }
        if present_family.is_some() {
            extension_names.push(khr::swapchain::NAME.as_ptr());
        }
        extension_names.push(khr::timeline_semaphore::NAME.as_ptr());
        extension_names.push(khr::synchronization2::NAME.as_ptr());
        extension_names.push(khr::dynamic_rendering::NAME.as_ptr());

        let mut vulkan_11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true);
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .maintenance4(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push(&mut vulkan_11_features)
            .push(&mut vulkan_12_features)
            .push(&mut vulkan_13_features);
        features2.features = features2.features.shader_int64(true);

        let create_info = unsafe {
            vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&extension_names)
                .extend(&mut features2)
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .expect("failed to create logical device")
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
        let present_queue = present_family.map_or(None, |present_family| unsafe {
            Some(device.get_device_queue(present_family, 0))
        });

        (device, graphics_queue, present_queue)
    }

    pub unsafe fn create_swapchain(
        &mut self,
        window: &Window,
        max_frames_in_flight: usize,
    ) -> Swapchain {
        let surface_loader = self.surface_loader.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let present_family = self.present_family.as_ref().unwrap();

        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, *surface)
                .expect("failed to query surface capabilities")
        };
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(self.physical_device, *surface)
                .expect("failed to query surface formats")
        };
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, *surface)
                .expect("failed to query surface present modes")
        };

        let surface_format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(formats[0]);

        let present_mode = present_modes
            .iter()
            .copied()
            .find(|&m| m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let family_indices = [self.graphics_family, *present_family];
        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        create_info = if self.graphics_family != *present_family {
            create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&family_indices)
        } else {
            create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("failed to create swapchain")
        };
        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("failed to get swapchain images")
        };
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image_views = || -> Vec<vk::ImageView> {
            images
                .iter()
                .map(|&image| {
                    let create_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .subresource_range(subresource_range.clone());
                    unsafe {
                        self.device
                            .create_image_view(&create_info, None)
                            .expect("failed to create image view")
                    }
                })
                .collect()
        }();
        let images = images
            .iter()
            .zip(&image_views)
            .map(|(image, image_view)| {
                self.create_image_imported(
                    *image,
                    *image_view,
                    &DeviceImageCreateInfo {},
                    &subresource_range,
                )
            })
            .collect();

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let acquire_to_graphics_semaphores: Vec<_> = (0..max_frames_in_flight)
            .map(|_| unsafe {
                self.device
                    .create_semaphore(&semaphore_info, None)
                    .expect("failed to create semaphore")
            })
            .collect();

        let graphics_to_present_semaphores: Vec<_> = (0..image_views.len())
            .map(|_| unsafe {
                self.device
                    .create_semaphore(&semaphore_info, None)
                    .expect("failed to create semaphore")
            })
            .collect();

        let fences: Vec<_> = (0..max_frames_in_flight)
            .map(|_| unsafe {
                self.device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .expect("failed to create fence")
            })
            .collect();

        Swapchain {
            swapchain: swapchain,
            images: images,
            format: surface_format.format,
            extent: extent,
            acquire_to_graphics_semaphores: acquire_to_graphics_semaphores,
            graphics_to_present_semaphores: graphics_to_present_semaphores,
            fences: fences,
            current_frame: 0,
        }
    }

    unsafe fn create_shaders(device: &Device) -> HashMap<TypeId, ShaderModuleArray> {
        let mut shaders = HashMap::new();

        for (k, shader) in gpu::ShaderModuleRegistry::collect().iter() {
            let mut shader_vec = Vec::new();
            shader_vec.resize(
                shader.total_permutations(),
                Option::<vk::ShaderModule>::None,
            );

            for index in 0..shader.total_permutations() {
                if !shader.should_create(index, &device) {
                    continue;
                }

                let spirv_file_name = shader.spirv_file_name(index);
                let spirv_file_path = shader.build_dir().join(&spirv_file_name);
                let _label = spirv_file_path.file_stem().unwrap().to_str();
                let bytes = std::fs::read(&spirv_file_path).unwrap_or_else(|e| panic!("{e}"));
                let code =
                    ash::util::read_spv(&mut Cursor::new(bytes)).expect("failed to parse SPIR-V");
                let module = unsafe {
                    device
                        .create_shader_module(
                            &vk::ShaderModuleCreateInfo::default().code(&code),
                            None,
                        )
                        .expect("failed to create shader module")
                };
                shader_vec[index] = Some(module);
            }

            shaders.insert(*k, ShaderModuleArray(shader_vec));
        }

        shaders
    }

    /// Returns a new pipeline layout derived from shader reflection.
    fn create_pipeline_layout<'a>(
        device: &Device,
        parameter_types: &[ShaderParameterType],
        push_constant_range_size: u32,
    ) -> (
        Vec<vk::DescriptorSetLayoutBinding<'a>>,
        vk::DescriptorSetLayout,
        vk::PipelineLayout,
    ) {
        let mut dslbs = Vec::new();
        for (binding, parameter_ty) in parameter_types.iter().enumerate() {
            let descriptor_type = {
                match parameter_ty.kind {
                    DescriptorKind::ConstantBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    DescriptorKind::StructuredBuffer | DescriptorKind::RWStructuredBuffer => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                }
            };
            let dslb = vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(descriptor_type)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);
            dslbs.push(dslb);
        }
        let dslci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dslbs);
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dslci, None)
                .expect("failed to create descriptor set layout")
        };
        let mut push_constant_ranges = Vec::new();
        if push_constant_range_size > 0 {
            push_constant_ranges = vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_range_size),
            ]
        }
        let set_layouts = [set_layout];
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_constant_ranges),
                    None,
                )
                .expect("failed to create pipeline layout")
        };
        (dslbs, set_layout, pipeline_layout)
    }

    /// Creates all device functions at load time.
    unsafe fn create_device_functions(
        device: &Device,
        set_allocators: &mut HashMap<vk::DescriptorSetLayout, DescriptorSetAllocator>,
        shaders: &HashMap<std::any::TypeId, ShaderModuleArray>,
    ) -> HashMap<std::any::TypeId, StaticDeviceFunctionArray> {
        let mut functions = HashMap::new();

        for (k, function) in DeviceFunctionRegistry::collect().iter() {
            let entry_point_c_str = std::ffi::CString::new(function.entry_point()).unwrap();

            let shader_vec = shaders.get(&function.shader_type()).expect(&format!(
                "function {:?} is missing shader {:?}",
                k,
                function.shader_type()
            ));

            let (dslbs, set_layout, pipeline_layout) = Self::create_pipeline_layout(
                device,
                &function.parameter_types(),
                function.push_constant_range_size(),
            );

            if !set_allocators.contains_key(&set_layout) {
                set_allocators.insert(set_layout, DescriptorSetAllocator::make(&dslbs));
            }

            let address_slots = Arc::new(function.push_constant_layout().address_slots());

            let mut function_vec = Vec::new();
            function_vec.resize(shader_vec.0.len(), Option::<StaticDeviceFunction>::None);

            for (i, shader) in shader_vec.0.iter().enumerate() {
                let Some(shader) = shader.as_ref() else {
                    continue;
                };

                let pssci = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(*shader)
                    .name(entry_point_c_str.as_c_str());

                let cpci = vk::ComputePipelineCreateInfo::default()
                    .stage(pssci)
                    .layout(pipeline_layout);

                let pipeline = unsafe {
                    device
                        .create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None)
                        .expect("failed to create compute pipelines")[0]
                };

                function_vec[i] = Some(StaticDeviceFunction {
                    set_layout,
                    pipeline_layout,
                    pipeline,
                });
            }

            functions.insert(
                *k,
                StaticDeviceFunctionArray {
                    permutations: function_vec,
                    address_slots: address_slots,
                },
            );
        }

        functions
    }

    /// Returns a new QueueTimeline.
    unsafe fn create_queue_timeline(device: &Device) -> QueueTimeline {
        let initial_value: u64 = 0;

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let semaphore_create_info =
            vk::SemaphoreCreateInfo::default().push(&mut timeline_create_info);

        let timeline_semaphore = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to create timeline semaphore")
        };

        QueueTimeline {
            semaphore: timeline_semaphore,
            value: initial_value,
        }
    }

    /// Submit
    fn queue_submit(
        &mut self,
        queue_type: QueueType,
        command_buffer: vk::CommandBuffer,
        wait_stage: vk::PipelineStageFlags2,
    ) {
        let (queue, queue_timeline) = match queue_type {
            QueueType::Graphics => (self.graphics_queue, &mut self.graphics_timeline),
        };

        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(queue_timeline.semaphore)
            .value(queue_timeline.value)
            .stage_mask(wait_stage)];

        queue_timeline.value += 1;

        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(queue_timeline.semaphore)
            .value(queue_timeline.value)];

        let command_buffer_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .command_buffer_infos(&command_buffer_infos);

        unsafe {
            self.device
                .queue_submit2(queue, &[submit_info], vk::Fence::null())
                .expect("failed to submit");
        }
    }

    fn get_function<T>(
        &self,
        permutation: &<T::Shader as ShaderModuleLike>::Permutations,
    ) -> &StaticDeviceFunction
    where
        T: DeviceFunctionLike + 'static,
        <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
    {
        // TypeId -> Permutation -> DeviceFunction
        let function_type = std::any::TypeId::of::<T>();
        let permutation_index = permutation.flatten();
        let functions = self.functions.get(&function_type).expect(&format!(
            "missing function {}:{permutation_index}:{:?}",
            std::any::type_name::<T>(),
            permutation.defines()
        ));
        let function = functions.permutations[permutation_index]
            .as_ref()
            .expect(&format!(
                "missing function permutation {}:{permutation_index}:{:?}",
                std::any::type_name::<T>(),
                permutation.defines()
            ));
        function
    }
}

impl DeviceContext {
    /// Creates a buffer synchronously using the DeviceBuffer constructor.
    /// This overload accepts lower-level [`vk_mem`] allocation info.
    fn create_buffer_inner<T>(
        &mut self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
        memory_info: &vk_mem::AllocationCreateInfo,
    ) -> DeviceBuffer<T> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(usage);

        let (buffer, allocation) = unsafe {
            self.mem_allocator
                .create_buffer(&create_info, &memory_info)
                .expect("failed to create a buffer with VMA")
        };

        let address = unsafe {
            if (usage & vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                == vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            {
                self.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            } else {
                0
            }
        };

        let handle = self.buffers.insert(DeviceBufferInner {
            label: String::from(name),
            buffer,
            size,
            usage,
            memory_info: memory_info.clone(),
            allocation,
            address,
        });

        DeviceBuffer::<T> {
            shared: Rc::new(DeviceBufferShared {
                label: String::from(name),
                handle,
                buffer,
                size,
                trash_tx: self.trash_tx.clone(),
            }),
            _marker: PhantomData::default(),
        }
    }

    /// Creates a [`DeviceImage`] synchronously using the DeviceImage constructor.
    /// This overload accepts an externally created [`vk::Image`] along with its default
    /// [`vk::ImageView`].
    fn create_image_imported(
        &mut self,
        image: vk::Image,
        image_view: vk::ImageView,
        create_info: &DeviceImageCreateInfo,
        subresource_range: &vk::ImageSubresourceRange,
    ) -> DeviceImage {
        DeviceImage {
            shared: Rc::new(DeviceImageShared {
                image,
                image_view,
                create_info: create_info.clone(),
                subresource_range: subresource_range.clone(),
                allocation: None,
                trash_tx: self.trash_tx.clone(),
            }),
        }
    }

    /// Maps this device memory to host memory for CPU access.
    /// Must call [`DeviceContext::unmap_memory`] when you're done.
    fn map_memory(&self, mut allocation: vk_mem::Allocation, size: usize) -> &mut [u8] {
        unsafe {
            let raw = self
                .mem_allocator
                .map_memory(&mut allocation)
                .expect("failed to map memory with VMA");
            std::slice::from_raw_parts_mut(raw, size)
        }
    }

    /// Unmaps this device memory.
    fn unmap_memory(&self, mut allocation: vk_mem::Allocation) {
        unsafe {
            self.mem_allocator.unmap_memory(&mut allocation);
        }
    }

    /// Destroys resources that are no longer on GPU timeline.
    fn garbage_collection(&mut self) {
        let graphics_queue_time = unsafe {
            self.device
                .get_semaphore_counter_value(self.graphics_timeline.semaphore)
        }
        .unwrap();

        self.trash.extend(
            self.trash_rx
                .try_iter()
                .map(|t| (self.graphics_timeline.value, t)),
        );

        while self
            .trash
            .front()
            .is_some_and(|(v, _)| *v <= graphics_queue_time)
        {
            let trash = self.trash.pop_front().unwrap();
            Self::destroy(
                trash.1,
                &self.device,
                &mut self.buffers,
                &mut self.mem_allocator,
            );
        }

        let mut set_allocators = std::mem::take(&mut self.set_allocators);
        for (_, set_allocator) in &mut set_allocators {
            set_allocator.garbage_collection(&self, graphics_queue_time);
        }
        self.set_allocators = set_allocators;
    }

    /// Destroys a piece of Trash.
    fn destroy(
        trash: Trash,
        device: &Device,
        buffers: &mut Arena<DeviceBufferInner>,
        mem_allocator: &mut vk_mem::Allocator,
    ) {
        match trash {
            Trash::Buffer(id) => unsafe {
                let mut buffer = buffers.remove(id).unwrap();
                mem_allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation);
            },
            Trash::Image((image, image_view, allocation)) => unsafe {
                device.destroy_image_view(image_view, None);
                if let Some(mut allocation) = allocation {
                    mem_allocator.destroy_image(image, &mut allocation);
                }
            },
            Trash::Generic(function) => function(device),
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("failed to wait for device");
        }

        // --- Destroy shader modules ---
        for (_, shader_vec) in &self.shaders {
            for shader in &shader_vec.0 {
                match shader {
                    Some(shader) => unsafe {
                        self.device.destroy_shader_module(*shader, None);
                    },
                    None => {}
                }
            }
        }

        // --- Destroy device function resources ---
        for (_, function_vec) in &self.functions {
            for function in &function_vec.permutations {
                match function {
                    Some(function) => unsafe {
                        self.device
                            .destroy_descriptor_set_layout(function.set_layout, None);
                        self.device
                            .destroy_pipeline_layout(function.pipeline_layout, None);
                        self.device.destroy_pipeline(function.pipeline, None);
                    },
                    None => {}
                }
            }
        }

        // --- Destroy descriptor pools ---
        for (_, set_allocator) in &self.set_allocators {
            set_allocator.destroy(&self.device);
        }

        // --- Destroy Vulkan objects ---
        let mut trash = std::mem::take(&mut self.trash);
        for (_, trash) in trash
            .drain(..)
            .chain(self.trash_rx.try_iter().map(|t| (0, t)))
        {
            Self::destroy(
                trash,
                &self.device,
                &mut self.buffers,
                &mut self.mem_allocator,
            );
        }
        self.trash = trash;

        // --- Destroy command pools ---
        unsafe {
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
        }

        // --- Destroy timeline semaphores ---
        unsafe {
            self.device
                .destroy_semaphore(self.graphics_timeline.semaphore, None);
        };

        // --- Destroy surface (if not in headless) ---
        unsafe {
            if let (Some(surface_loader), Some(surface)) =
                (self.surface_loader.as_ref(), self.surface.as_ref())
            {
                surface_loader.destroy_surface(*surface, None);
            }
        };

        // --- Drop the memory allocator, device and instance ---
    }
}

/// Every shader's VkShaderModule permutation.
struct ShaderModuleArray(Vec<Option<vk::ShaderModule>>);

/// A device function's compute pipeline derived at load time from shader reflection.
#[derive(Clone)]
struct StaticDeviceFunction {
    set_layout: vk::DescriptorSetLayout, // Set layout #0
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// A device function's collection of compute pipelines, one for each permutation.
struct StaticDeviceFunctionArray {
    /// Every permutation of the function.
    permutations: Vec<Option<StaticDeviceFunction>>,

    /// DeviceAddress slots in the function's PushConstant blob.
    address_slots: Arc<Vec<AddressSlot>>,
}

/// A compute pipeline created at runtime.
/// Must be a specialization of a static compute pipeline, since we currently don't
/// allow runtime Slang shader compilation.
pub struct DeviceFunction<T>
where
    T: DeviceFunctionLike + 'static,
    <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
{
    pipeline: vk::Pipeline,
    permutation: <<T as DeviceFunctionLike>::Shader as ShaderModuleLike>::Permutations,
    trash_tx: Sender<Trash>,
    _marker: PhantomData<T>,
}

impl<T> Drop for DeviceFunction<T>
where
    T: DeviceFunctionLike + 'static,
    <T as DeviceFunctionLike>::Shader: ShaderModuleLike,
{
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        let _ = self
            .trash_tx
            .send(Trash::Generic(Box::new(move |device| unsafe {
                device.destroy_pipeline(pipeline, None);
            })));
    }
}

pub(crate) struct QueueTimeline {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) value: u64,
}

pub enum QueueType {
    Graphics,
}

enum Trash {
    Buffer(Handle<DeviceBufferInner>),
    Image((vk::Image, vk::ImageView, Option<vk_mem::Allocation>)),
    Generic(Box<dyn Fn(&Device)>),
}

fn default_buffer_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
}

pub struct DeviceBufferInner {
    #[allow(dead_code)]
    label: String,
    buffer: vk::Buffer,
    size: usize,
    #[allow(dead_code)]
    usage: vk::BufferUsageFlags,
    #[allow(dead_code)]
    memory_info: vk_mem::AllocationCreateInfo,
    allocation: vk_mem::Allocation,
    address: vk::DeviceAddress,
}

impl DeviceBufferInner {
    fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allocation(&self) -> vk_mem::Allocation {
        self.allocation
    }

    fn address(&self) -> vk::DeviceAddress {
        self.address
    }
}

struct DeviceBufferShared {
    label: String,
    handle: Handle<DeviceBufferInner>,
    buffer: vk::Buffer,
    size: usize,
    trash_tx: Sender<Trash>,
}

impl Drop for DeviceBufferShared {
    fn drop(&mut self) {
        let _ = self.trash_tx.send(Trash::Buffer(self.handle));
    }
}

pub struct DeviceBuffer<T> {
    shared: Rc<DeviceBufferShared>,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn name(&self) -> &str {
        &self.shared.label
    }

    pub fn handle(&self) -> Handle<DeviceBufferInner> {
        self.shared.handle
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.shared.buffer
    }

    pub fn size(&self) -> usize {
        self.shared.size
    }

    pub fn len(&self) -> usize {
        self.size() / std::mem::size_of::<T>()
    }

    pub fn as_ref(&self) -> &Self {
        self
    }
}

impl<T> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<T: AnyBitPattern> DeviceBuffer<T> {
    pub fn map_to_host<'a>(&self, ctx: &'a DeviceContext) -> HostMappedMemory<'a, T> {
        let allocation = ctx.buffers.get(self.shared.handle).unwrap().allocation;
        let raw = ctx.map_memory(allocation, self.size());
        let raw = bytemuck::cast_slice(raw);
        HostMappedMemory::<'a, T> {
            allocation,
            raw,
            ctx,
        }
    }
}

pub struct HostMappedMemory<'a, T> {
    allocation: vk_mem::Allocation,
    raw: &'a [T],
    ctx: &'a DeviceContext,
}

impl<'a, T> HostMappedMemory<'a, T> {
    pub fn len(&self) -> usize {
        self.raw.len()
    }
}

impl<T> Index<usize> for HostMappedMemory<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.raw[index]
    }
}

impl<T> Drop for HostMappedMemory<'_, T> {
    fn drop(&mut self) {
        self.ctx.unmap_memory(self.allocation);
    }
}

impl<T: fmt::Debug> fmt::Debug for HostMappedMemory<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.raw.iter()).finish()
    }
}

struct DeviceImageShared {
    image: vk::Image,
    image_view: vk::ImageView,
    create_info: DeviceImageCreateInfo,
    subresource_range: vk::ImageSubresourceRange,
    allocation: Option<vk_mem::Allocation>,
    trash_tx: Sender<Trash>,
}

impl Drop for DeviceImageShared {
    fn drop(&mut self) {
        let _ = self
            .trash_tx
            .send(Trash::Image((self.image, self.image_view, self.allocation)));
    }
}

#[derive(Clone)]
pub struct DeviceImage {
    shared: Rc<DeviceImageShared>,
}

#[derive(Clone)]
pub struct DeviceImageCreateInfo {}

impl DeviceImage {
    pub fn image(&self) -> vk::Image {
        self.shared.image
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.shared.image_view
    }

    pub fn create_info(&self) -> &DeviceImageCreateInfo {
        &self.shared.create_info
    }

    pub fn subresource_range(&self) -> &vk::ImageSubresourceRange {
        &self.shared.subresource_range
    }
}

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    images: Vec<DeviceImage>,
    format: vk::Format,
    extent: vk::Extent2D,
    acquire_to_graphics_semaphores: Vec<vk::Semaphore>,
    graphics_to_present_semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    current_frame: usize,
}

pub struct SwapchainImage {
    image_index: u32,
}

impl Swapchain {
    pub fn wait_for_fences(&self, ctx: &DeviceContext) {
        let fence = self.fences[self.current_frame];
        unsafe {
            ctx.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fences");
        }
    }

    pub fn acquire_next_image(
        &self,
        ctx: &mut DeviceContext,
    ) -> Result<(SwapchainImage, bool), vk::Result> {
        let acquire_to_graphics_semaphore = self.acquire_to_graphics_semaphores[self.current_frame];

        let result = unsafe {
            ctx.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                acquire_to_graphics_semaphore,
                vk::Fence::null(),
            )
        };

        match result {
            Ok((image_index, suboptimal)) => {
                // --- Acquire -> Graphics sync ---
                let wait_stage = vk::PipelineStageFlags2::ALL_GRAPHICS;
                let mut wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = vec![
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(acquire_to_graphics_semaphore)
                        .stage_mask(wait_stage),
                ];
                if ctx.graphics_timeline.value > 0 {
                    wait_semaphore_infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(ctx.graphics_timeline.semaphore)
                            .value(ctx.graphics_timeline.value)
                            .stage_mask(wait_stage),
                    );
                }

                ctx.graphics_timeline.value += 1;

                let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                    .semaphore(ctx.graphics_timeline.semaphore)
                    .value(ctx.graphics_timeline.value)];

                let submit_info = vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait_semaphore_infos)
                    .signal_semaphore_infos(&signal_semaphore_infos);

                unsafe {
                    ctx.device
                        .queue_submit2(ctx.graphics_queue, &[submit_info], vk::Fence::null())
                        .expect("failed to submit work on the graphics queue");
                }

                Ok((
                    SwapchainImage {
                        image_index: image_index,
                    },
                    suboptimal,
                ))
            }
            Err(result) => Err(result),
        }
    }

    pub fn queue_present(
        &mut self,
        ctx: &DeviceContext,
        frame: SwapchainImage,
    ) -> Result<bool, vk::Result> {
        // --- Graphics -> Present sync ---
        let wait_stage = vk::PipelineStageFlags2::ALL_GRAPHICS;
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(ctx.graphics_timeline.semaphore)
            .value(ctx.graphics_timeline.value)
            .stage_mask(wait_stage)];

        let signal_semaphore = self.graphics_to_present_semaphores[frame.image_index as usize];
        let signal_semaphore_infos =
            [vk::SemaphoreSubmitInfo::default().semaphore(signal_semaphore)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        let fence = self.fences[self.current_frame];

        unsafe {
            ctx.device
                .reset_fences(&[fence])
                .expect("failed to reset fences");
        }

        unsafe {
            ctx.device
                .queue_submit2(ctx.graphics_queue, &[submit_info], fence)
                .expect("failed to submit work on the graphics queue");
        }

        // --- Queue Present ---
        let swapchain_wait_semaphores = [signal_semaphore];
        let swapchains = [self.handle()];
        let image_indices = [frame.image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&swapchain_wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            ctx.swapchain_loader
                .queue_present(ctx.present_queue.unwrap(), &present_info)
        };

        self.current_frame = (self.current_frame + 1) % self.fences.len();

        present_result
    }

    pub fn image(&self, image: &SwapchainImage) -> &DeviceImage {
        &self.images[image.image_index as usize]
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent.clone()
    }

    pub unsafe fn destroy(&mut self, ctx: &DeviceContext) {
        for &semaphore in &self.acquire_to_graphics_semaphores {
            unsafe {
                ctx.device.destroy_semaphore(semaphore, None);
            }
        }
        for &semaphore in &self.graphics_to_present_semaphores {
            unsafe {
                ctx.device.destroy_semaphore(semaphore, None);
            }
        }
        for &fence in &self.fences {
            unsafe {
                ctx.device.destroy_fence(fence, None);
            }
        }
        unsafe {
            ctx.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

#[macro_export]
macro_rules! enqueue_function {
    ($ctx:expr, func: $func:expr, params: $params:expr, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function_object($func, $params, $push, $grid_dim)
    };
    ($ctx:expr, func: $func:expr, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function_object($func, (), $push, $grid_dim)
    };

    ($ctx:expr, $func:ty, permutation: $permutation:expr, params: $params:expr, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function::<$func>($permutation, $params, $push, $grid_dim)
    };
    ($ctx:expr, $func:ty, permutation: $permutation:expr, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function::<$func>($permutation, (), $push, $grid_dim)
    };
    ($ctx:expr, $func:ty, params: $params:expr, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function::<$func>(&(), $params, $push, $grid_dim)
    };
    ($ctx:expr, $func:ty, push: $push:expr, grid_dim: $grid_dim:expr $(,)?) => {
        $ctx.enqueue_function::<$func>(&(), (), $push, $grid_dim)
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RgResource {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RgVersion {
    resource: RgResource,
    version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RgPassId(usize);

#[derive(Clone, Copy)]
enum RgAccessKind {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone)]
pub struct RgBufferTransition {
    buffer: vk::Buffer,
    kind: RgAccessKind,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

#[derive(Clone)]
pub struct RgImageTransition {
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    kind: RgAccessKind,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    layout: vk::ImageLayout,
}

#[derive(Default)]
struct RgPipelineBarrier {
    buffers: Vec<RgBufferTransition>,
    images: Vec<RgImageTransition>,
}

struct RgPass {
    /// Pass name.
    #[allow(dead_code)]
    name: String,

    /// Synchronization intent (barriers).
    barrier: RgPipelineBarrier,

    /// Enqueued closure.
    function: Box<dyn Fn(&mut DeviceContext, vk::CommandBuffer)>,
}

pub struct RgContext {
    /// Registered passes.
    passes: Vec<RgPass>,

    /// Versioning.
    versions: HashMap<RgResource, u32>,

    /// Version-to-pass O(1) lookup.
    writers: HashMap<RgVersion, RgPassId>,
    readers: HashMap<RgVersion, Vec<RgPassId>>,

    /// Passes with RAW, WAW, or WAR dependencies.
    edges: HashSet<(RgPassId, RgPassId)>,
}

impl RgContext {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            versions: HashMap::new(),
            writers: HashMap::new(),
            readers: HashMap::new(),
            edges: HashSet::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct RgBufferState {
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    was_write: bool,
}

impl RgBufferState {
    fn initial() -> Self {
        Self {
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access: vk::AccessFlags2::NONE,
            was_write: false,
        }
    }
}

#[derive(Clone, Copy)]
struct RgImageState {
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    layout: vk::ImageLayout,
    was_write: bool,
}

impl RgImageState {
    fn initial() -> Self {
        Self {
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access: vk::AccessFlags2::NONE,
            layout: vk::ImageLayout::UNDEFINED,
            was_write: false,
        }
    }
}

struct RgStates {
    buffers: HashMap<vk::Buffer, RgBufferState>,
    images: HashMap<vk::Image, RgImageState>,
}

#[allow(dead_code)]
fn sampled_fragment(image: &DeviceImage) -> RgImageTransition {
    RgImageTransition {
        image: image.image(),
        subresource_range: *image.subresource_range(),
        kind: RgAccessKind::Read,
        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }
}
pub fn color_attachment(image: &DeviceImage) -> RgImageTransition {
    RgImageTransition {
        image: image.image(),
        subresource_range: *image.subresource_range(),
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }
}
pub fn depth_attachment(image: &DeviceImage) -> RgImageTransition {
    RgImageTransition {
        image: image.image(),
        subresource_range: *image.subresource_range(),
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    }
}
fn buffer_transfer_read(buffer: vk::Buffer) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer,
        kind: RgAccessKind::Read,
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
    }
}
fn buffer_transfer_write(buffer: vk::Buffer) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer,
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    }
}
pub fn constant_buffer_read(
    buffer: vk::Buffer,
    stage: vk::PipelineStageFlags2,
) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer,
        kind: RgAccessKind::Read,
        stage,
        access: vk::AccessFlags2::SHADER_READ,
    }
}
fn storage_buffer_read(buffer: vk::Buffer, stage: vk::PipelineStageFlags2) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer,
        kind: RgAccessKind::Read,
        stage,
        access: vk::AccessFlags2::SHADER_READ,
    }
}
fn storage_buffer_read_write(
    buffer: vk::Buffer,
    stage: vk::PipelineStageFlags2,
) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer,
        kind: RgAccessKind::ReadWrite,
        stage,
        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    }
}

pub trait U32Castable {
    fn to_u32(self) -> u32;
}

impl U32Castable for f32 {
    fn to_u32(self) -> u32 {
        u32::from_le_bytes(self.to_le_bytes())
    }
}

pub trait EnqueueCopyable<T> {
    fn enqueue_copy(ctx: &mut DeviceContext, src_buf: Self, dst_buf: &DeviceBuffer<T>);
}

impl<T> EnqueueCopyable<T> for &DeviceBuffer<T>
where
    T: 'static,
{
    fn enqueue_copy(ctx: &mut DeviceContext, src_buf: Self, dst_buf: &DeviceBuffer<T>) {
        let src_buf = src_buf.clone();
        let dst_buf = dst_buf.clone();
        ctx.enqueue_pass(
            "enqueue_copy_buffer",
            &[
                buffer_transfer_read(src_buf.buffer()),
                buffer_transfer_write(dst_buf.buffer()),
            ],
            &[],
            Box::new(
                move |ctx: &mut DeviceContext, command_buffer: vk::CommandBuffer| {
                    assert_eq!(src_buf.size(), dst_buf.size());

                    let regions = [vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(src_buf.size() as vk::DeviceSize)];

                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            command_buffer,
                            src_buf.buffer(),
                            dst_buf.buffer(),
                            &regions,
                        );
                    }
                },
            ),
        );
    }
}

impl<T> EnqueueCopyable<T> for &[T]
where
    T: bytemuck::Pod,
{
    fn enqueue_copy(ctx: &mut DeviceContext, src_ptr: Self, dst_buf: &DeviceBuffer<T>) {
        let dst_buf = dst_buf.clone();

        let src_ptr = bytemuck::cast_slice(src_ptr);

        let tmp_buf = ctx.create_buffer_inner::<T>(
            "staging_buffer",
            src_ptr.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            &vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                usage: vk_mem::MemoryUsage::Auto,
                ..Default::default()
            },
        );

        let src_vec = src_ptr.to_vec();

        ctx.enqueue_pass(
            "enqueue_copy",
            &[
                buffer_transfer_read(tmp_buf.buffer()),
                buffer_transfer_write(dst_buf.buffer()),
            ],
            &[],
            Box::new(
                move |ctx: &mut DeviceContext, command_buffer: vk::CommandBuffer| {
                    let tmp_buf = ctx
                        .buffers
                        .get(tmp_buf.handle())
                        .expect(&format!("missing buffer: {}", tmp_buf.name()));

                    let tmp_ptr = ctx.map_memory(tmp_buf.allocation(), tmp_buf.size());
                    tmp_ptr.copy_from_slice(&src_vec);
                    ctx.unmap_memory(tmp_buf.allocation());

                    assert_eq!(tmp_buf.size(), dst_buf.size());

                    let regions = [vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(tmp_buf.size() as vk::DeviceSize)];

                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            command_buffer,
                            tmp_buf.buffer(),
                            dst_buf.buffer(),
                            &regions,
                        );
                    }
                },
            ),
        );
    }
}

impl<T, const N: usize> EnqueueCopyable<T> for &[T; N]
where
    T: bytemuck::Pod,
{
    fn enqueue_copy(ctx: &mut DeviceContext, src_ptr: Self, dst_buf: &DeviceBuffer<T>) {
        <&[T] as EnqueueCopyable<T>>::enqueue_copy(ctx, src_ptr.as_slice(), dst_buf);
    }
}
