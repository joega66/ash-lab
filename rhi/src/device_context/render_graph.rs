use std::unimplemented;

use super::*;
use crate::UInt3;

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

impl DeviceContext {
    pub fn enqueue_create_buffer<T>(&mut self, name: &str, len: usize) -> DeviceBuffer<T> {
        // TODO: Resource aliasing
        self.create_buffer::<T>(name, len)
    }

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

    pub fn enqueue_copy<T, U: EnqueueCopyable<T>>(
        &mut self,
        src_buf: U,
        dst_buf: &DeviceBuffer<T>,
    ) {
        <U as EnqueueCopyable<T>>::enqueue_copy(self, src_buf, dst_buf);
    }

    pub fn enqueue_function<T>(
        &mut self,
        permutation: &<T::Shader as ShaderModule>::Permutations,
        parameters: <T as DeviceFunctionMeta>::Params,
        push_constant: <T as DeviceFunctionMeta>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionMeta + 'static,
        <T as DeviceFunctionMeta>::Shader: ShaderModule,
        <T as DeviceFunctionMeta>::PushConstant: ShaderType + Pod,
    {
        self.enqueue_function_inner::<T>(None, &permutation, parameters, push_constant, grid_dim);
    }

    pub fn enqueue_function_object<T>(
        &mut self,
        function: &DeviceFunction<T>,
        parameters: <T as DeviceFunctionMeta>::Params,
        push_constant: <T as DeviceFunctionMeta>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionMeta + 'static,
        <T as DeviceFunctionMeta>::Shader: ShaderModule,
        <T as DeviceFunctionMeta>::PushConstant: ShaderType + Pod,
    {
        self.enqueue_function_inner::<T>(
            Some(function.pipeline),
            &function.permutation,
            parameters,
            push_constant,
            grid_dim,
        );
    }

    fn enqueue_function_inner<T>(
        &mut self,
        pipeline: Option<vk::Pipeline>,
        permutation: &<T::Shader as ShaderModule>::Permutations,
        parameters: <T as DeviceFunctionMeta>::Params,
        push_constant: <T as DeviceFunctionMeta>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: DeviceFunctionMeta + 'static,
        <T as DeviceFunctionMeta>::Shader: ShaderModule,
        <T as DeviceFunctionMeta>::PushConstant: ShaderType + Pod,
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

#[cfg(test)]
mod tests {}
