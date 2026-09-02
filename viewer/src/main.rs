#![allow(unsafe_op_in_unsafe_fn)]
use ash::vk;
use ash::vk::TaggedStructure;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rhi::*;
use shaders::*;
use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
mod camera;
use camera::*;

const WINDOW_TITLE: &str = "Hello, Triangle (ash + Vulkan)";
const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        let window = event_loop
            .create_window(window_attributes)
            .expect("failed to create window");

        self.renderer = Some(unsafe { Renderer::new(&window) });
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.renderer = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.keyboard_input(code, state);
                }
            }
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.mouse_input(state, button);
                }
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
            } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.mouse_wheel(delta);
                }
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resized(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(renderer)) = (&self.window, self.renderer.as_mut()) {
                    unsafe { renderer.redraw_requested(window) };
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.device_event(event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

struct Renderer {
    swapchain: Swapchain,

    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    camera_buffer: DeviceBuffer<CameraUniform>,

    camera: Camera,
    camera_uniform: CameraUniform,

    current_frame: usize,

    resize_requested: bool,

    mouse_motion: Option<(f64, f64)>,
    mouse_button_pressed: HashSet<MouseButton>,
    keys_pressed: HashSet<KeyCode>,

    ctx: DeviceContext,
}

impl Renderer {
    unsafe fn new(window: &Window) -> Self {
        let display_handle = window.display_handle().expect("no display handle").as_raw();
        let window_handle = window.window_handle().expect("no window handle").as_raw();

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo {
            display_handle: Some(display_handle),
            window_handle: Some(window_handle),
        });

        let swapchain = ctx.create_swapchain(window, MAX_FRAMES_IN_FLIGHT);

        let (set_layout, pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&ctx, swapchain.format());

        let camera_buffer = ctx.create_constant_buffer();

        let (descriptor_pool, descriptor_set) = Self::allocate_descriptor_set(&ctx, set_layout);

        Self::update_descriptor_sets(&ctx, descriptor_set, &camera_buffer);

        let camera = Camera::default(window.inner_size().width, window.inner_size().height);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        Self {
            swapchain,
            set_layout,
            pipeline_layout,
            graphics_pipeline,
            descriptor_pool,
            descriptor_set,
            camera_buffer,
            current_frame: 0,
            resize_requested: false,
            camera,
            camera_uniform,
            mouse_motion: None,
            mouse_button_pressed: HashSet::<MouseButton>::new(),
            keys_pressed: HashSet::<KeyCode>::new(),
            ctx,
        }
    }

    unsafe fn create_graphics_pipeline(
        ctx: &DeviceContext,
        color_attachment_format: vk::Format,
    ) -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
        let triangle_shader = ctx.get_shader::<TriangleShader>(&());

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(*triangle_shader)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*triangle_shader)
                .name(c"ps_main"),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend_attachments = [color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let set_layout = ctx
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .expect("create_descriptor_set_layout failed");
        let set_layouts = [set_layout];
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
            .expect("failed to create pipeline layout");

        let color_attachment_formats = [color_attachment_format];

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .subpass(0)
            .push(&mut pipeline_rendering_info);

        let pipeline = ctx
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .expect("failed to create graphics pipeline")[0];

        (set_layout, pipeline_layout, pipeline)
    }

    fn allocate_descriptor_set(
        ctx: &DeviceContext,
        set_layout: vk::DescriptorSetLayout,
    ) -> (vk::DescriptorPool, vk::DescriptorSet) {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)];
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe {
            ctx.device
                .create_descriptor_pool(&create_info, None)
                .expect("create_descriptor_pool failed")
        };
        let set_layouts = [set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe {
            ctx.device
                .allocate_descriptor_sets(&allocate_info)
                .expect("allocate_descriptor_sets failed")
        };
        (descriptor_pool, descriptor_sets[0])
    }

    fn update_descriptor_sets(
        ctx: &DeviceContext,
        set: vk::DescriptorSet,
        camera_buffer: &DeviceBuffer<CameraUniform>,
    ) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(camera_buffer.buffer())
            .offset(0)
            .range(camera_buffer.size() as vk::DeviceSize)];
        let descriptor_writes = [vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(set)
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];
        unsafe {
            ctx.device.update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    unsafe fn render(&mut self, backbuffer: &SwapchainImage) {
        let backbuffer = self
            .ctx
            .register_image("backbuffer", self.swapchain.image(backbuffer));

        let camera_buffer = self
            .ctx
            .register_buffer("camera_buffer", &self.camera_buffer);
        self.ctx.enqueue_copy(&[self.camera_uniform], camera_buffer);

        let extent = self.swapchain.extent().clone();
        let descriptor_set = self.descriptor_set.clone();
        let pipeline_layout = self.pipeline_layout.clone();
        let graphics_pipeline = self.graphics_pipeline.clone();

        self.ctx.enqueue_pass(
            "TrianglePass",
            &[constant_buffer_read(
                camera_buffer,
                vk::PipelineStageFlags2::VERTEX_SHADER,
            )],
            &[color_attachment(backbuffer)],
            Box::new(move |ctx, command_buffer| {
                let clear_value = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.01, 0.01, 0.02, 1.0],
                    },
                };

                let render_area = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: extent.clone(),
                };

                let backbuffer = ctx.image(&backbuffer);

                ctx.device.cmd_begin_rendering(
                    command_buffer,
                    &vk::RenderingInfo::default()
                        .render_area(render_area)
                        .layer_count(1)
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(backbuffer.image_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(clear_value)]),
                );

                let descriptor_sets = [descriptor_set.clone()];

                ctx.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets,
                    &[],
                );

                ctx.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                ctx.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                ctx.device
                    .cmd_set_scissor(command_buffer, 0, &[render_area]);

                ctx.device.cmd_draw(command_buffer, 3, 1, 0, 0);

                ctx.device.cmd_end_rendering(command_buffer);
            }),
        );

        self.ctx.execute(Some(backbuffer)).unwrap();
    }

    fn update(&mut self) {
        // --- WASD movement ---
        let movement_speed = 0.0167;
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.camera.position += self.camera.forward() * movement_speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.camera.position += -self.camera.right() * movement_speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.camera.position += -self.camera.forward() * movement_speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.camera.position += self.camera.right() * movement_speed;
        }

        // --- FPS camera rotation ---
        if self.mouse_button_pressed.contains(&MouseButton::Left) {
            let rotation_speed = 0.0167;
            let (mut delta_x, mut delta_y) = *self.mouse_motion.get_or_insert((0.0, 0.0));
            delta_x *= rotation_speed;
            delta_y *= rotation_speed;
            self.camera.rotate(delta_x as f32, delta_y as f32);
        }
        self.mouse_motion = None;

        // --- Update camera buffer ---
        self.camera_uniform.update_view_proj(&self.camera);
    }

    unsafe fn redraw_requested(&mut self, window: &Window) {
        self.update();

        self.swapchain.wait_for_fences(&self.ctx);

        if self.resize_requested {
            self.recreate_swapchain(window);
            self.resize_requested = false;
            return;
        }

        // --- Acquire Next Image ---
        let result = self.swapchain.acquire_next_image(&mut self.ctx);

        let frame = match result {
            Ok((frame, suboptimal)) => {
                if suboptimal {
                    self.resize_requested = true;
                }
                frame
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window);
                return;
            }
            Err(err) => panic!("failed to acquire swapchain image: {err}"),
        };

        // --- Render Scene ---
        self.render(&frame);

        // --- Present Queue Submit ---
        let present_result = self.swapchain.queue_present(&self.ctx, frame);
        match present_result {
            Ok(suboptimal) if suboptimal => self.resize_requested = true,
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.resize_requested = true,
            Err(err) => panic!("failed to present swapchain image: {err}"),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.ctx
            .device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        self.swapchain.destroy(&self.ctx);

        self.swapchain = self.ctx.create_swapchain(window, MAX_FRAMES_IN_FLIGHT);
    }

    pub fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_motion = Some(delta);
            }
            _ => {}
        }
    }

    pub fn keyboard_input(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.keys_pressed.insert(code);
            }
            ElementState::Released => {
                self.keys_pressed.remove(&code);
            }
        }
    }

    pub fn mouse_input(&mut self, state: ElementState, button: MouseButton) {
        match state {
            ElementState::Pressed => {
                self.mouse_button_pressed.insert(button);
            }
            ElementState::Released => {
                self.mouse_button_pressed.remove(&button);
            }
        }
    }

    pub fn mouse_wheel(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, _) => {}
            MouseScrollDelta::PixelDelta(_) => {}
        }
    }

    pub fn resized(&mut self, width: u32, height: u32) {
        self.resize_requested = true;
        self.camera.aspect = width as f32 / height as f32;
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();

            self.swapchain.destroy(&self.ctx);

            self.ctx
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.ctx
                .device
                .destroy_pipeline(self.graphics_pipeline, None);

            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
