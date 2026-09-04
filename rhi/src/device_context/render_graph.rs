use std::unimplemented;

use super::*;
use crate::UInt3;

pub use bda::{describe_tagged, tag_buffer_ref};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RgVersion {
    id: RgResourceId,
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

#[derive(Clone, Copy)]
pub struct RgBufferTransition {
    buffer: AnyRgBuffer,
    kind: RgAccessKind,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

#[derive(Clone, Copy)]
pub struct RgImageTransition {
    image: RgImage,
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
    name: String,

    /// Synchronization intent (barriers).
    barrier: RgPipelineBarrier,

    /// Enqueued closure.
    function: Box<dyn Fn(&mut DeviceContext, vk::CommandBuffer)>,
}

pub struct RgContext {
    /// Incremented every execute().
    epoch: u32,

    /// Virtual buffers/images.
    /// Only registered buffers/images outlive the graph.
    virtual_buffers: Vec<RgBufferShadow>,
    virtual_images: Vec<RgImageShadow>,

    /// Virtual handles to physical/raw buffers/images.
    physical_buffers: Vec<DeviceBufferShadow>,
    physical_images: Vec<DeviceImageShadow>,

    /// Registered passes.
    passes: Vec<RgPass>,

    /// Versioning.
    versions: HashMap<RgResourceId, u32>,

    /// Version-to-pass O(1) lookup.
    writers: HashMap<RgVersion, RgPassId>,
    readers: HashMap<RgVersion, Vec<RgPassId>>,

    /// Passes with RAW, WAW, or WAR dependencies.
    edges: HashSet<(RgPassId, RgPassId)>,
}

impl RgContext {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            virtual_buffers: Vec::new(),
            virtual_images: Vec::new(),
            physical_buffers: Vec::new(),
            physical_images: Vec::new(),
            passes: Vec::new(),
            versions: HashMap::new(),
            writers: HashMap::new(),
            readers: HashMap::new(),
            edges: HashSet::new(),
        }
    }

    fn next(&self) -> Self {
        Self {
            epoch: self.epoch + 1,
            ..Self::new()
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
    buffers: Vec<RgBufferState>,
    images: Vec<RgImageState>,
}

impl DeviceContext {
    pub fn register_buffer<T>(&mut self, name: &str, buffer: &DeviceBuffer<T>) -> RgBuffer<T> {
        let rg_buffer = RgBufferShadow {
            name: name.to_string(),
            size: buffer.size(),
            usage: buffer.details.usage,
            memory_info: buffer.details.memory_info.clone(),
            imported: Some((&buffer.details).into()),
        };
        self.push_buffer(rg_buffer)
    }

    pub fn register_image(&mut self, name: &str, image: &DeviceImage) -> RgImage {
        let rg_image = RgImageShadow {
            name: name.to_string(),
            _create_info: DeviceImageCreateInfo {},
            imported: Some(DeviceImageShadow {
                image: image.image,
                image_view: image.image_view,
                subresource_range: image.subresource_range,
            }),
        };
        self.push_image(rg_image)
    }

    pub fn enqueue_create_buffer<T>(&mut self, name: &str, len: usize) -> RgBuffer<T> {
        self.enqueue_create_buffer_inner(
            name,
            len * std::mem::size_of::<T>(),
            default_buffer_usage(),
            &vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::empty(),
                usage: vk_mem::MemoryUsage::Auto,
                ..Default::default()
            },
        )
    }

    pub fn enqueue_fill<T: BitsToU32 + 'static>(&mut self, input: RgBuffer<T>, value: T) {
        let value = value.bits_to_u32();
        self.enqueue_pass(
            "enqueue_fill",
            &[RgBufferTransition {
                buffer: (&input).into(),
                kind: RgAccessKind::Write,
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            }],
            &[],
            Box::new(move |ctx, command_buffer| {
                let input = ctx.buffer(&input);
                unsafe {
                    ctx.device.cmd_fill_buffer(
                        command_buffer,
                        input.buffer,
                        0,
                        input.size as vk::DeviceSize,
                        value,
                    )
                };
            }),
        );
    }

    fn enqueue_create_buffer_inner<T>(
        &mut self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
        memory_info: &vk_mem::AllocationCreateInfo,
    ) -> RgBuffer<T> {
        let rg_buffer = RgBufferShadow {
            name: name.to_string(),
            size: size,
            usage: usage,
            memory_info: memory_info.clone(),
            imported: None,
        };
        self.push_buffer(rg_buffer)
    }

    pub fn enqueue_create_image(
        &mut self,
        name: &str,
        create_info: &DeviceImageCreateInfo,
    ) -> RgImage {
        let rg_image = RgImageShadow {
            name: name.to_string(),
            _create_info: create_info.clone(),
            imported: None,
        };
        self.push_image(rg_image)
    }

    fn push_buffer<T>(&mut self, buffer: RgBufferShadow) -> RgBuffer<T> {
        let id = RgResourceId::buffer(self.rg.virtual_buffers.len());
        let size = buffer.size;
        self.rg.virtual_buffers.push(buffer);
        self.rg.versions.insert(id, 0); // version 0 = imported / initial
        RgBuffer::<T>::new(self.rg.epoch, id, size)
    }

    fn push_image(&mut self, image: RgImageShadow) -> RgImage {
        let id = RgResourceId::image(self.rg.virtual_images.len());
        self.rg.virtual_images.push(image);
        self.rg.versions.insert(id, 0); // version 0 = imported / initial
        RgImage::new(self.rg.epoch, id)
    }

    /// Rejects an epoch belonging to a previous execution.
    fn check_epoch(&self, epoch: u32) {
        assert_eq!(
            epoch, self.rg.epoch,
            "render graph Ref belongs to execution {epoch}, but execution {} is being built; \
             a Ref does not survive execute()",
            self.rg.epoch
        );
    }

    fn index_checked<T: RgAbstractHandle>(&self, r: &T) -> usize {
        self.check_epoch(r.epoch());
        r.id().index()
    }

    pub fn buffer<T>(&self, buffer: &RgBuffer<T>) -> &DeviceBufferShadow {
        &self.rg.physical_buffers[self.index_checked(buffer)]
    }

    fn any_buffer(&self, buffer: &AnyRgBuffer) -> &DeviceBufferShadow {
        &self.rg.physical_buffers[self.index_checked(buffer)]
    }

    fn untag(&self, raw: u64) -> AnyRgBuffer {
        bda::untag_at_epoch(raw, self.rg.epoch)
    }

    fn resolve_address(&self, raw: u64) -> vk::DeviceAddress {
        let r = self.untag(raw);
        self.any_buffer(&r).address
    }

    pub fn image(&self, image: &RgImage) -> &DeviceImageShadow {
        &self.rg.physical_images[self.index_checked(image)]
    }

    pub fn enqueue_copy<T, U: EnqueueCopyable<T>>(&mut self, src_buf: U, dst_buf: RgBuffer<T>) {
        <U as EnqueueCopyable<T>>::enqueue_copy(self, src_buf, dst_buf);
    }

    pub fn enqueue_function<T>(
        &mut self,
        permutation: <T::Shader as ShaderModule>::Permutations,
        parameters: <T as Kernel>::Params,
        push_constant: <T as Kernel>::PushConstant,
        grid_dim: UInt3,
    ) where
        T: Kernel + 'static,
        <T as Kernel>::Shader: ShaderModule,
    {
        // Get the kernel's type for hash 'n cache.
        let kernel_type = std::any::TypeId::of::<T>();

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
            .map(|param| match param.kind {
                DescriptorKind::ConstantBuffer => constant_buffer_read(param.buffer, stage),
                DescriptorKind::StructuredBuffer => storage_buffer_read(param.buffer, stage),
                DescriptorKind::RWStructuredBuffer => {
                    storage_buffer_read_write(param.buffer, stage)
                }
            })
            .collect();

        // Multi-dim permutation -> flat index.
        let permutation_idx = permutation.flatten();

        // Lookup the kernel from the flat index.
        let kernels = self.kernels.get(&kernel_type).expect(&format!(
            "{}:{} wasn't ahead-of-time compiled",
            std::any::type_name::<T>(),
            permutation_idx
        ));
        let kernel = kernels.permutations[permutation_idx]
            .as_ref()
            .expect(&format!(
                "{}:{} wasn't ahead-of-time compiled",
                std::any::type_name::<T>(),
                permutation_idx
            ))
            .clone();

        // Insert transitions for DeviceAddress in the push constant blob.
        let address_slots = kernels.address_slots.clone();
        transitions.extend(bda::address_transitions(
            &address_slots,
            &push_constant_bytes,
            self.rg.epoch,
            stage,
        ));

        // Enqueue the kernel.
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
                            bytes[at..at + 8]
                                .copy_from_slice(&ctx.resolve_address(raw).to_ne_bytes());
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
                                    ctx.rg.physical_buffers[ctx.index_checked(&arg.buffer)].buffer
                                }
                            })
                            .collect();

                        // TODO: Batch all descriptor set allocations + updates in graph preamble
                        let mut set_allocators = std::mem::take(&mut ctx.set_allocators);
                        let set_allocator = set_allocators.get_mut(&kernel.set_layout).unwrap();
                        let result =
                            set_allocator.allocate_descriptor_set(ctx, kernel.set_layout, buffers);
                        ctx.set_allocators = set_allocators;

                        match result {
                            DescriptorSetCacheLookup::Miss(set) => {
                                let mut buffer_infos = Vec::new();
                                for arg in &parameters {
                                    if arg.kind.is_buffer() {
                                        let buffer = ctx.any_buffer(&arg.buffer);
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
                            kernel.pipeline,
                        );
                        if let Some(set) = set {
                            ctx.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::COMPUTE,
                                kernel.pipeline_layout,
                                0,
                                &[set],
                                &[],
                            );
                        }
                        if !push_constant_bytes.is_empty() {
                            ctx.device.cmd_push_constants(
                                command_buffer,
                                kernel.pipeline_layout,
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
                t.buffer.epoch,
                t.buffer.id,
                t.kind,
                &mut reads,
                &mut writes,
            );
        }
        for t in images {
            self.version_use(
                pass_id,
                t.image.epoch,
                t.image.id,
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
        epoch: u32,
        id: RgResourceId,
        kind: RgAccessKind,
        reads: &mut Vec<RgVersion>,
        writes: &mut Vec<RgVersion>,
    ) {
        // Check for Ref's carried over from a previous execution.
        self.check_epoch(epoch);

        let does_read = matches!(kind, RgAccessKind::Read | RgAccessKind::ReadWrite);
        let does_write = matches!(kind, RgAccessKind::Write | RgAccessKind::ReadWrite);

        if does_read {
            let rv = RgVersion {
                id,
                version: self.rg.versions[&id],
            };
            if let Some(&prod) = self.rg.writers.get(&rv) {
                self.add_edge(prod, pass_id); // RAW
            }
            self.rg.readers.entry(rv).or_default().push(pass_id);
            reads.push(rv);
        }

        if does_write {
            let old = RgVersion {
                id,
                version: self.rg.versions[&id],
            };
            if let Some(&prod) = self.rg.writers.get(&old) {
                self.add_edge(prod, pass_id); // WAW
            }
            if let Some(prev_readers) = self.rg.readers.get(&old).cloned() {
                for r in prev_readers {
                    self.add_edge(r, pass_id); // WAR (self-edge skipped)
                }
            }
            let new_ver = self.rg.versions[&id] + 1;
            self.rg.versions.insert(id, new_ver);
            let nv = RgVersion {
                id,
                version: new_ver,
            };
            self.rg.writers.insert(nv, pass_id);
            writes.push(nv);
        }
    }

    // TODO: Return the executed graph in a pretty form
    pub fn execute(&mut self, present: Option<RgImage>) -> Result<(), String> {
        let result = self.execute_inner(present);

        self.garbage_collection();

        self.rg = self.rg.next();

        result
    }

    fn execute_inner(&mut self, present: Option<RgImage>) -> Result<(), String> {
        // Resolve virtual to physical buffers
        let mut transient_buffers = Vec::new();
        let mut physical_buffers = Vec::with_capacity(self.rg.virtual_buffers.len());
        for i in 0..self.rg.virtual_buffers.len() {
            let raw: DeviceBufferShadow = match self.rg.virtual_buffers[i].imported {
                Some(raw) => raw,
                None => {
                    let virtual_buffer = &self.rg.virtual_buffers[i];
                    let buffer = self.create_buffer_inner(
                        virtual_buffer.size,
                        virtual_buffer.usage,
                        &virtual_buffer.memory_info.clone(),
                    );
                    let raw: DeviceBufferShadow = (&buffer).into();
                    transient_buffers.push(buffer);
                    raw
                }
            };
            physical_buffers.push(raw);
        }
        self.rg.physical_buffers = physical_buffers;

        // Resolve virtual to physical images
        let mut physical_images = Vec::with_capacity(self.rg.virtual_images.len());
        for virtual_image in &self.rg.virtual_images {
            let Some(raw) = virtual_image.imported else {
                return Err("TODO: transient image creation".to_string());
            };
            physical_images.push(raw);
        }
        self.rg.physical_images = physical_images;

        // Sort passes.
        let order = self.topological_sort()?;

        // Initialize state tracking.
        let mut states = RgStates {
            buffers: vec![RgBufferState::initial(); self.rg.virtual_buffers.len()],
            images: vec![RgImageState::initial(); self.rg.virtual_images.len()],
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
            let prev = states.images[self.index_checked(&present)];
            let present = self.image(&present);
            if prev.layout != vk::ImageLayout::PRESENT_SRC_KHR {
                let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(prev.stage)
                    .src_access_mask(prev.access)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(prev.layout)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(present.image)
                    .subresource_range(present.subresource_range)];

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
    ) -> Vec<(usize, vk::PipelineStageFlags2, vk::AccessFlags2)> {
        let mut map: HashMap<usize, (vk::PipelineStageFlags2, vk::AccessFlags2)> = HashMap::new();
        for t in &pass.barrier.buffers {
            let e = map
                .entry(t.buffer.id.index())
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
        usize,
        vk::PipelineStageFlags2,
        vk::AccessFlags2,
        vk::ImageLayout,
    )> {
        let mut map: HashMap<usize, (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout)> =
            HashMap::new();
        for t in &pass.barrier.images {
            let e = map.entry(t.image.id.index()).or_insert((
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                t.layout,
            ));
            e.0 |= t.stage;
            e.1 |= t.access;
            e.2 = t.layout;
        }
        let mut v: Vec<_> = map.into_iter().map(|(i, (s, a, l))| (i, s, a, l)).collect();
        v.sort_by_key(|(i, _, _, _)| *i);
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

        for (index, dst_stage, dst_access) in Self::merged_buffer_uses(pass) {
            let prev = states.buffers[index];
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

            states.buffers[index] = RgBufferState {
                stage: dst_stage,
                access: dst_access,
                was_write: cur_write,
            };
        }

        for (index, dst_stage, dst_access, new_layout) in Self::merged_image_uses(pass) {
            let prev = states.images[index];
            let cur_write = dst_access & write_mask != vk::AccessFlags2::NONE;
            let layout_change = prev.layout != new_layout;

            if layout_change || prev.was_write || cur_write {
                let image = &self.rg.physical_images[index];
                image_memory_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(prev.stage)
                        .src_access_mask(prev.access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(prev.layout)
                        .new_layout(new_layout)
                        .image(image.image)
                        .subresource_range(image.subresource_range),
                );
            }

            states.images[index] = RgImageState {
                stage: dst_stage,
                access: dst_access,
                layout: new_layout,
                was_write: cur_write,
            };
        }

        (memory_barriers, image_memory_barriers)
    }

    #[allow(dead_code)]
    fn rg_name(&self, id: RgResourceId) -> &str {
        match id.kind() {
            RgResourceKind::Buffer => &self.rg.virtual_buffers[id.index()].name,
            RgResourceKind::Image => &self.rg.virtual_images[id.index()].name,
        }
    }

    #[allow(dead_code)]
    fn fmt_version(&self, rv: RgVersion) -> String {
        format!("{}.v{}", self.rg_name(rv.id), rv.version)
    }

    #[allow(dead_code)]
    fn pass_name(&self, p: RgPassId) -> &str {
        &self.rg.passes[p.0].name
    }
}

#[derive(Clone, Copy)]
pub struct DeviceBufferShadow {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    size: usize,
    address: vk::DeviceAddress,
}

impl From<&DeviceBufferDetails> for DeviceBufferShadow {
    fn from(value: &DeviceBufferDetails) -> Self {
        Self {
            buffer: value.buffer,
            allocation: value.allocation,
            size: value.size,
            address: value.address,
        }
    }
}

#[derive(Clone, Copy)]
pub struct DeviceImageShadow {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub subresource_range: vk::ImageSubresourceRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RgResourceKind {
    Buffer,
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RgResourceId(u32);

/// High bit of [`RgResourceId`]: clear for buffers, set for images.
const RG_KIND_BIT: u32 = 1 << 31;

impl RgResourceId {
    fn buffer(index: usize) -> Self {
        debug_assert!((index as u32) < RG_KIND_BIT);
        Self(index as u32)
    }

    fn image(index: usize) -> Self {
        debug_assert!((index as u32) < RG_KIND_BIT);
        Self(index as u32 | RG_KIND_BIT)
    }

    fn kind(self) -> RgResourceKind {
        if self.0 & RG_KIND_BIT == 0 {
            RgResourceKind::Buffer
        } else {
            RgResourceKind::Image
        }
    }

    fn index(self) -> usize {
        (self.0 & !RG_KIND_BIT) as usize
    }
}

pub struct RgBuffer<T> {
    epoch: u32,
    id: RgResourceId,
    size: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> RgBuffer<T> {
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }
}

impl<T> RgAbstractHandle for RgBuffer<T> {
    fn epoch(&self) -> u32 {
        self.epoch
    }

    fn id(&self) -> RgResourceId {
        self.id
    }
}

impl<T> Clone for RgBuffer<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for RgBuffer<T> {}
impl<T> PartialEq for RgBuffer<T> {
    fn eq(&self, o: &Self) -> bool {
        self.epoch == o.epoch && self.id == o.id
    }
}
impl<T> Eq for RgBuffer<T> {}
impl<T> Hash for RgBuffer<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.epoch.hash(h);
        self.id.hash(h);
    }
}
impl<T> std::fmt::Debug for RgBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ref({:?} @ epoch {})", self.id, self.epoch)
    }
}
impl<T> RgBuffer<T> {
    fn new(epoch: u32, id: RgResourceId, size: usize) -> Self {
        Self {
            epoch,
            id,
            size,
            _marker: PhantomData,
        }
    }
}

trait RgAbstractHandle {
    fn epoch(&self) -> u32;
    fn id(&self) -> RgResourceId;
}

macro_rules! rg_handle_impl {
    ($ty:ident) => {
        pub struct $ty {
            epoch: u32,
            id: RgResourceId,
        }
        impl $ty {
            fn new(epoch: u32, id: RgResourceId) -> Self {
                Self { epoch, id }
            }
        }
        impl RgAbstractHandle for $ty {
            fn epoch(&self) -> u32 {
                self.epoch
            }
            fn id(&self) -> RgResourceId {
                self.id
            }
        }
        impl Clone for $ty {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl Copy for $ty {}
        impl PartialEq for $ty {
            fn eq(&self, o: &Self) -> bool {
                self.epoch == o.epoch && self.id == o.id
            }
        }
        impl Eq for $ty {}
        impl Hash for $ty {
            fn hash<H: Hasher>(&self, h: &mut H) {
                self.epoch.hash(h);
                self.id.hash(h);
            }
        }
        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Ref({:?} @ epoch {})", self.id, self.epoch)
            }
        }
    };
}

rg_handle_impl!(AnyRgBuffer);
rg_handle_impl!(RgImage);

impl<T> From<RgBuffer<T>> for AnyRgBuffer {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            epoch: value.epoch,
            id: value.id,
        }
    }
}

impl<T> From<&RgBuffer<T>> for AnyRgBuffer {
    fn from(value: &RgBuffer<T>) -> Self {
        Self {
            epoch: value.epoch,
            id: value.id,
        }
    }
}

pub trait RgAbstractBufferHandle: Into<AnyRgBuffer> {}
impl<T> RgAbstractBufferHandle for RgBuffer<T> {}
impl RgAbstractBufferHandle for AnyRgBuffer {}

pub struct RgBufferShadow {
    // Label.
    name: String,

    /// Buffer size in bytes.
    size: usize,

    /// Buffer usage.
    usage: vk::BufferUsageFlags,

    // Memory info.
    memory_info: vk_mem::AllocationCreateInfo,

    /// `Some` when this was registered from an application-owned handle.
    /// `None` means transient, and `execute` allocates the backing buffer.
    imported: Option<DeviceBufferShadow>,
}

pub struct RgImageShadow {
    // Label.
    name: String,

    // Create info.
    _create_info: DeviceImageCreateInfo,

    /// `Some` when this was registered from an application-owned handle.
    /// `None` means transient, and `execute` allocates the backing image.
    imported: Option<DeviceImageShadow>,
}

#[allow(dead_code)]
fn sampled_fragment(image: RgImage) -> RgImageTransition {
    RgImageTransition {
        image,
        kind: RgAccessKind::Read,
        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }
}
pub fn color_attachment(image: RgImage) -> RgImageTransition {
    RgImageTransition {
        image,
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }
}
pub fn depth_attachment(image: RgImage) -> RgImageTransition {
    RgImageTransition {
        image,
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    }
}
fn buffer_transfer_read<T: RgAbstractBufferHandle>(buffer: T) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer.into(),
        kind: RgAccessKind::Read,
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
    }
}
fn buffer_transfer_write<T: RgAbstractBufferHandle>(buffer: T) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer.into(),
        kind: RgAccessKind::Write,
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    }
}
pub fn constant_buffer_read<T: RgAbstractBufferHandle>(
    buffer: T,
    stage: vk::PipelineStageFlags2,
) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer.into(),
        kind: RgAccessKind::Read,
        stage,
        access: vk::AccessFlags2::SHADER_READ,
    }
}
fn storage_buffer_read<T: RgAbstractBufferHandle>(
    buffer: T,
    stage: vk::PipelineStageFlags2,
) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer.into(),
        kind: RgAccessKind::Read,
        stage,
        access: vk::AccessFlags2::SHADER_READ,
    }
}
fn storage_buffer_read_write<T: RgAbstractBufferHandle>(
    buffer: T,
    stage: vk::PipelineStageFlags2,
) -> RgBufferTransition {
    RgBufferTransition {
        buffer: buffer.into(),
        kind: RgAccessKind::ReadWrite,
        stage,
        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    }
}

/// Conversion to/from Buffer Device Address and Render Graph handles.
mod bda {
    use super::{
        AddressSlot, AnyRgBuffer, RgAbstractHandle, RgBuffer, RgBufferTransition, RgResourceId,
        RgResourceKind, vk,
    };

    pub fn encode<T: RgAbstractHandle>(r: T) -> u64 {
        ((r.epoch() as u64) << 32) | r.id().0 as u64
    }

    pub fn decode(raw: u64) -> (u32, RgResourceId) {
        ((raw >> 32) as u32, RgResourceId(raw as u32))
    }

    pub fn tag_buffer_ref<T>(r: RgBuffer<T>) -> u64 {
        encode(r)
    }

    pub fn describe_tagged(raw: u64) -> String {
        let (epoch, id) = decode(raw);
        format!("{:?} @ epoch {epoch}", id)
    }

    pub fn untag_at_epoch(raw: u64, current_epoch: u32) -> AnyRgBuffer {
        let (epoch, id) = decode(raw);
        assert_eq!(
            epoch, current_epoch,
            "device address handle belongs to execution {epoch}, but execution {} is being \
             built; a Ref does not survive execute()",
            current_epoch,
        );
        assert_eq!(
            id.kind(),
            RgResourceKind::Buffer,
            "device address handle does not name a buffer",
        );
        AnyRgBuffer::new(current_epoch, id)
    }

    pub fn address_transitions(
        slots: &[AddressSlot],
        bytes: &[u8],
        epoch: u32,
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
                let buffer = untag_at_epoch(raw, epoch);
                if slot.writable {
                    super::storage_buffer_read_write(buffer, stage)
                } else {
                    super::storage_buffer_read(buffer, stage)
                }
            })
            .collect()
    }
}

pub trait BitsToU32 {
    fn bits_to_u32(self) -> u32;
}

impl BitsToU32 for f32 {
    fn bits_to_u32(self) -> u32 {
        u32::from_le_bytes(self.to_le_bytes())
    }
}

pub trait EnqueueCopyable<T> {
    fn enqueue_copy(ctx: &mut DeviceContext, src_buf: Self, dst_buf: RgBuffer<T>);
}

impl<T> EnqueueCopyable<T> for RgBuffer<T>
where
    T: 'static,
{
    fn enqueue_copy(ctx: &mut DeviceContext, src_buf: Self, dst_buf: RgBuffer<T>) {
        ctx.enqueue_pass(
            "enqueue_copy_buffer",
            &[
                buffer_transfer_read(src_buf),
                buffer_transfer_write(dst_buf),
            ],
            &[],
            Box::new(
                move |ctx: &mut DeviceContext, command_buffer: vk::CommandBuffer| {
                    let src_buf = ctx.buffer::<T>(&src_buf);
                    let dst_buf = ctx.buffer::<T>(&dst_buf);

                    assert_eq!(src_buf.size, dst_buf.size);

                    let regions = [vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(src_buf.size as vk::DeviceSize)];

                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            command_buffer,
                            src_buf.buffer,
                            dst_buf.buffer,
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
    fn enqueue_copy(ctx: &mut DeviceContext, src_ptr: Self, dst_buf: RgBuffer<T>) {
        let src_ptr = bytemuck::cast_slice(src_ptr);

        let tmp_buf = ctx.enqueue_create_buffer_inner::<T>(
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
                buffer_transfer_read(tmp_buf),
                buffer_transfer_write(dst_buf),
            ],
            &[],
            Box::new(
                move |ctx: &mut DeviceContext, command_buffer: vk::CommandBuffer| {
                    let tmp_buf = ctx.buffer(&tmp_buf);
                    let dst_buf = ctx.buffer(&dst_buf);

                    let tmp_ptr = ctx.map_memory(tmp_buf.allocation, tmp_buf.size);
                    tmp_ptr.copy_from_slice(&src_vec);
                    ctx.unmap_memory(tmp_buf.allocation);

                    assert_eq!(tmp_buf.size, dst_buf.size);

                    let regions = [vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(tmp_buf.size as vk::DeviceSize)];

                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            command_buffer,
                            tmp_buf.buffer,
                            dst_buf.buffer,
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
    fn enqueue_copy(ctx: &mut DeviceContext, src_ptr: Self, dst_buf: RgBuffer<T>) {
        <&[T] as EnqueueCopyable<T>>::enqueue_copy(ctx, src_ptr.as_slice(), dst_buf);
    }
}

#[cfg(test)]
mod tests {}
