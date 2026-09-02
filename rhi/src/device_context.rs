use crate::AddressSlot;
use crate::permutation::*;
use crate::shader_module::*;
use ash::vk::TaggedStructure;
use ash::{Device, Entry, Instance, khr, vk};
use bytemuck::AnyBitPattern;
use raw_window_handle::RawDisplayHandle;
use raw_window_handle::RawWindowHandle;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    ffi::CStr,
    hash::{Hash, Hasher},
    io::Cursor,
    sync::Arc,
    sync::mpsc::{self, Receiver, Sender},
};
use vk_mem::Alloc;
use vk_mem::{AllocationCreateFlags, MemoryUsage};
use winit::window::Window;

mod render_graph;
pub use render_graph::*;

mod descriptor_pool;
use descriptor_pool::*;

mod owned;
pub use owned::*;

use crate::shader_parameter::*;

// ____________________________________________________________________________
// DeviceContext

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

    graphics_timeline: QueueTimeline,

    trash_tx: Sender<Trash>,
    trash_rx: Receiver<Trash>,
    trash: VecDeque<(u64, Trash)>,

    mem_allocator: vk_mem::Allocator,

    set_allocators: HashMap<vk::DescriptorSetLayout, DescriptorSetAllocator>,

    shaders: HashMap<std::any::TypeId, ShaderModuleArray>,
    kernels: HashMap<std::any::TypeId, PrecompiledKernelArray>,

    rg: RgContext,

    pub device: DeviceOwner,

    #[allow(dead_code)]
    instance: InstanceOwner,

    #[allow(dead_code)]
    entry: Entry,
}

// ____________________________________________________________
// DeviceContext, public

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

        let kernels = unsafe { Self::create_kernels(&device, &mut set_allocators, &shaders) };

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
            kernels,
            rg: RgContext::new(),
            device: DeviceOwner::new(device),
            instance: InstanceOwner::new(instance),
            entry,
        }
    }

    /// Returns a new buffer.
    pub fn create_buffer<T>(&mut self, len: usize) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::empty(),
            usage: MemoryUsage::Auto,
            ..Default::default()
        };
        DeviceBuffer::<T> {
            details: self.create_buffer_inner(
                len * std::mem::size_of::<T>(),
                default_buffer_usage(),
                &memory_info,
            ),
            _marker: PhantomData,
        }
    }

    /// Returns a new buffer that can be read by the host.
    pub fn create_host_buffer<T>(&mut self, len: usize) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::HOST_ACCESS_RANDOM,
            usage: MemoryUsage::AutoPreferHost,
            ..Default::default()
        };
        DeviceBuffer::<T> {
            details: self.create_buffer_inner(
                len * std::mem::size_of::<T>(),
                default_buffer_usage(),
                &memory_info,
            ),
            _marker: PhantomData,
        }
    }

    /// Returns a new buffer suitable for using as a ConstantBuffer descriptor.
    pub fn create_constant_buffer<T>(&mut self) -> DeviceBuffer<T> {
        let memory_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::empty(),
            usage: MemoryUsage::Auto,
            ..Default::default()
        };
        DeviceBuffer::<T> {
            details: self.create_buffer_inner(
                std::mem::size_of::<T>(),
                default_buffer_usage() | vk::BufferUsageFlags::UNIFORM_BUFFER,
                &memory_info,
            ),
            _marker: PhantomData,
        }
    }

    pub fn map_memory(&self, mut allocation: vk_mem::Allocation, size: usize) -> &mut [u8] {
        unsafe {
            let raw = self
                .mem_allocator
                .map_memory(&mut allocation)
                .expect("failed to map memory with VMA");
            std::slice::from_raw_parts_mut(raw, size)
        }
    }

    pub fn unmap_memory(&self, mut allocation: vk_mem::Allocation) {
        unsafe {
            self.mem_allocator.unmap_memory(&mut allocation);
        }
    }

    pub fn get_shader<T: 'static + ShaderModule>(
        &self,
        p: &<T as ShaderModule>::Permutations,
    ) -> &vk::ShaderModule {
        let k = TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();
        let shader_vec = self
            .shaders
            .get(&k)
            .expect(&format!("missing shader module {type_name}"));
        let index = p.flatten();
        let shader = shader_vec
            .0
            .get(index)
            .expect(&format!("missing shader module {type_name}:{index}"));
        shader
            .as_ref()
            .expect(&format!("missing shader module {type_name}:{index}"))
    }

    /// Wait on every queue.
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

    /// Destroy resources that are no longer on GPU timeline.
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
            Self::destroy(trash.1, &self.device, &mut self.mem_allocator);
        }

        let mut set_allocators = std::mem::take(&mut self.set_allocators);
        for (_, set_allocator) in &mut set_allocators {
            set_allocator.garbage_collection(&self, graphics_queue_time);
        }
        self.set_allocators = set_allocators;
    }
}

// ____________________________________________________________
// DeviceContext, private

impl DeviceContext {
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

        for physical_device in physical_devices {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

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
                            .get_physical_device_surface_support(physical_device, index, *surface)
                            .unwrap_or(false)
                    };

                    if supports_present {
                        present_family = Some(index);
                    }
                }

                // With surface: Find graphics + present
                // Without surface: Find graphics
                if graphics_family.is_some() && (present_family.is_some() || surface.is_none()) {
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

            return (
                physical_device,
                graphics_family.unwrap(),
                present_family,
                portability_subset,
            );
        }

        panic!("no suitable Vulkan physical device found");
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
            .dynamic_rendering(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push(&mut vulkan_11_features)
            .push(&mut vulkan_12_features)
            .push(&mut vulkan_13_features);

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

        for (k, shader) in rhi::ShaderModuleRegistry::collect().iter() {
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

    fn create_pipeline_layout<'a>(
        device: &Device,
        parameter_types: &[ShaderParameterType],
        push_constant_size: u32,
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
        let push_constant_ranges = if push_constant_size > 0 {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_size),
            ]
        } else {
            Vec::new()
        };
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

    /// Returns all kernels designated for pre-compilation.
    unsafe fn create_kernels(
        device: &Device,
        set_allocators: &mut HashMap<vk::DescriptorSetLayout, DescriptorSetAllocator>,
        shaders: &HashMap<std::any::TypeId, ShaderModuleArray>,
    ) -> HashMap<std::any::TypeId, PrecompiledKernelArray> {
        let mut kernels = HashMap::new();

        for (k, kernel) in KernelRegistry::collect().iter() {
            let entry_point_c_str = std::ffi::CString::new(kernel.entry_point()).unwrap();

            let shader_vec = shaders.get(&kernel.shader_type()).expect(&format!(
                "kernel {:?} is missing shader {:?}",
                k,
                kernel.shader_type()
            ));

            let (dslbs, set_layout, pipeline_layout) = Self::create_pipeline_layout(
                device,
                &kernel.parameter_types(),
                kernel.push_constant_range_size(),
            );

            if !set_allocators.contains_key(&set_layout) {
                set_allocators.insert(set_layout, DescriptorSetAllocator::make(&dslbs));
            }

            let address_slots = Arc::new(kernel.push_constant_layout().address_slots());

            let mut kernel_vec = Vec::new();
            kernel_vec.resize(shader_vec.0.len(), Option::<PrecompiledKernel>::None);

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

                kernel_vec[i] = Some(PrecompiledKernel {
                    set_layout,
                    pipeline_layout,
                    pipeline,
                });
            }

            kernels.insert(
                *k,
                PrecompiledKernelArray {
                    permutations: kernel_vec,
                    address_slots: address_slots,
                },
            );
        }

        kernels
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

    /// Returns a new buffer with low-level vk_mem information.
    fn create_buffer_inner(
        &mut self,
        size: usize,
        usage: vk::BufferUsageFlags,
        memory_info: &vk_mem::AllocationCreateInfo,
    ) -> DeviceBufferDetails {
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

        DeviceBufferDetails {
            buffer,
            size: size,
            usage: usage,
            memory_info: memory_info.clone(),
            allocation,
            address,
            trash_tx: self.trash_tx.clone(),
        }
    }

    /// Wraps a VkImage created from an imported VkImage.
    fn create_image_imported(
        &mut self,
        image: vk::Image,
        image_view: vk::ImageView,
        create_info: &DeviceImageCreateInfo,
        subresource_range: &vk::ImageSubresourceRange,
    ) -> DeviceImage {
        DeviceImage {
            image,
            image_view,
            create_info: create_info.clone(),
            subresource_range: subresource_range.clone(),
            allocation: None,
            trash_tx: self.trash_tx.clone(),
        }
    }

    /// Destroy a piece of Trash.
    fn destroy(trash: Trash, device: &Device, mem_allocator: &mut vk_mem::Allocator) {
        match trash {
            Trash::Buffer((buffer, mut allocation)) => unsafe {
                mem_allocator.destroy_buffer(buffer, &mut allocation);
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

        // --- Destroy kernel resources ---
        for (_, kernel_vec) in &self.kernels {
            for kernel in &kernel_vec.permutations {
                match kernel {
                    Some(kernel) => unsafe {
                        self.device
                            .destroy_descriptor_set_layout(kernel.set_layout, None);
                        self.device
                            .destroy_pipeline_layout(kernel.pipeline_layout, None);
                        self.device.destroy_pipeline(kernel.pipeline, None);
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
            Self::destroy(trash, &self.device, &mut self.mem_allocator);
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

/// Every permutation of a shader module.
struct ShaderModuleArray(Vec<Option<vk::ShaderModule>>);

/// A kernel compiled ahead-of-time.
#[derive(Clone)]
struct PrecompiledKernel {
    set_layout: vk::DescriptorSetLayout, // Set layout #0
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct PrecompiledKernelArray {
    /// Every permutation of the kernel.
    permutations: Vec<Option<PrecompiledKernel>>,

    /// DeviceAddress slots in the kernel's PushConstant blob.
    address_slots: Arc<Vec<AddressSlot>>,
}

struct QueueTimeline {
    semaphore: vk::Semaphore,
    value: u64,
}

pub enum QueueType {
    Graphics,
}

enum Trash {
    Buffer((vk::Buffer, vk_mem::Allocation)),
    Image((vk::Image, vk::ImageView, Option<vk_mem::Allocation>)),
    Generic(Box<dyn Fn(&Device)>),
}

impl Drop for DeviceBufferDetails {
    fn drop(&mut self) {
        let _ = self
            .trash_tx
            .send(Trash::Buffer((self.buffer, self.allocation)));
    }
}

impl Drop for DeviceImage {
    fn drop(&mut self) {
        let _ = self
            .trash_tx
            .send(Trash::Image((self.image, self.image_view, self.allocation)));
    }
}

fn default_buffer_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
}

struct DeviceBufferDetails {
    /// Vulkan buffer handle.
    buffer: vk::Buffer,

    /// Buffer length in elements.
    size: usize,

    /// Buffer usage
    usage: vk::BufferUsageFlags,

    /// Memory create info.
    memory_info: vk_mem::AllocationCreateInfo,

    /// VMA handle.
    allocation: vk_mem::Allocation,

    /// GPU address.
    address: vk::DeviceAddress,

    /// Trash sender.
    trash_tx: Sender<Trash>,
}

pub struct DeviceBuffer<T> {
    /// Details.
    details: DeviceBufferDetails,

    /// Marker.
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn buffer(&self) -> vk::Buffer {
        self.details.buffer
    }

    pub fn size(&self) -> usize {
        self.details.size
    }

    pub fn address(&self) -> vk::DeviceAddress {
        self.details.address
    }
}

impl<T: AnyBitPattern> DeviceBuffer<T> {
    pub fn map_to_host<'a>(&self, ctx: &'a DeviceContext) -> HostMappedMemory<'a, T> {
        let raw = ctx.map_memory(self.details.allocation, self.size());
        HostMappedMemory::<'a, T> {
            allocation: self.details.allocation,
            raw: bytemuck::cast_slice(raw),
            ctx: ctx,
        }
    }
}

pub struct HostMappedMemory<'a, T> {
    allocation: vk_mem::Allocation,
    raw: &'a [T],
    ctx: &'a DeviceContext,
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

pub struct DeviceImage {
    /// Vulkan image handle
    image: vk::Image,

    /// Vulkan image view handle created by default
    image_view: vk::ImageView,

    /// Create info
    create_info: DeviceImageCreateInfo,

    /// Image view info
    subresource_range: vk::ImageSubresourceRange,

    /// VMA handle
    allocation: Option<vk_mem::Allocation>,

    /// Trash sender.
    trash_tx: Sender<Trash>,
}

#[derive(Clone)]
pub struct DeviceImageCreateInfo {}

impl DeviceImage {
    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }
    pub fn create_info(&self) -> &DeviceImageCreateInfo {
        &self.create_info
    }
    pub fn subresource_range(&self) -> &vk::ImageSubresourceRange {
        &self.subresource_range
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
