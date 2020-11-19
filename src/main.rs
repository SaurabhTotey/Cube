#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, CompositeAlpha};
use vulkano::image::ImageUsage;
use vulkano::format::{Format, ClearValue};
use std::cmp::min;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use std::sync::Arc;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::{Subpass, Framebuffer};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};

fn main() {
	let instance = Instance::new(
		Some(&ApplicationInfo {
			application_name: Some(Cow::from("Cube!")),
			..ApplicationInfo::default()
		}),
		&vulkano_win::required_extensions(),
		None
	).unwrap();
	let physicalDevice = PhysicalDevice::enumerate(&instance).next().unwrap();

	let eventsLoop = EventLoop::new();
	let surface = WindowBuilder::new()
		.with_title("Cube!")
		.build_vk_surface(&eventsLoop, instance.clone()).unwrap();
	let surfaceCapabilities = surface.capabilities(physicalDevice).unwrap();

	let queueFamily = physicalDevice.queue_families()
		.find(|&family| family.supports_graphics() && surface.is_supported(family).unwrap_or(false)).unwrap();
	let (logicalDevice, mut deviceQueues) = { Device::new(
		physicalDevice,
		&Features::none(),
		&DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() },
		[(queueFamily, 1.0)].iter().cloned(),
	).unwrap() };
	let deviceQueue = deviceQueues.next().unwrap();

	let (imageFormat, colorSpace) = surfaceCapabilities.supported_formats.iter()
		.find(|formatAndColorSpace| formatAndColorSpace.0 == Format::B8G8R8A8Srgb && formatAndColorSpace.1 == ColorSpace::SrgbNonLinear)
		.unwrap_or(surfaceCapabilities.supported_formats.iter().next().unwrap());
	let (mut swapchain, swapchainImages) = {
		let supportedAlpha = surfaceCapabilities.supported_composite_alpha.iter()
			.find(|&compositeAlpha| compositeAlpha == CompositeAlpha::Opaque)
			.unwrap_or(surfaceCapabilities.supported_composite_alpha.iter().next().unwrap());
		let dimensions: [u32; 2] = surface.window().inner_size().into();
		let presentMode = if surfaceCapabilities.present_modes.mailbox { PresentMode::Mailbox } else { PresentMode::Fifo };
		let numberOfImages = min(surfaceCapabilities.min_image_count + 1, surfaceCapabilities.max_image_count.unwrap_or(surfaceCapabilities.min_image_count + 1));
		Swapchain::new(
			logicalDevice.clone(),
			surface.clone(),
			numberOfImages,
			*imageFormat,
			dimensions,
			1,
			ImageUsage::color_attachment(),
			&deviceQueue,
			SurfaceTransform::Identity,
			supportedAlpha,
			presentMode,
			FullscreenExclusive::Default,
			true,
			*colorSpace
		).unwrap()
	};

	#[derive(Default, Copy, Clone)]
	struct Vertex {
		position: [f32; 3],
		color: [f32; 4]
	}
	vulkano::impl_vertex!(Vertex, position, color);

	let vertices = [
		Vertex { position: [ 0.5,  1f32 / 3f32.sqrt() / 2f32, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
		Vertex { position: [-0.5,  1f32 / 3f32.sqrt() / 2f32, 0.0], color: [0.0, 1.0, 0.0, 1.0] },
		Vertex { position: [ 0.0, -1f32 / 3f32.sqrt()       , 0.0], color: [1.0, 0.0, 1.0, 1.0] }
	].to_vec();
	let vertexBuffer = CpuAccessibleBuffer::from_iter(
		logicalDevice.clone(),
		BufferUsage { vertex_buffer: true, ..BufferUsage::none() },
		false,
		vertices.iter().cloned()
	).unwrap();

	mod VertexShader { vulkano_shaders::shader!{ ty: "vertex", path: "./src/shader/vertex.glsl" } }
	mod FragmentShader { vulkano_shaders::shader!{ ty: "fragment", path: "./src/shader/fragment.glsl" } }
	let vertexShader = VertexShader::Shader::load(logicalDevice.clone()).unwrap();
	let fragmentShader = FragmentShader::Shader::load(logicalDevice.clone()).unwrap();

	let renderPass = Arc::new(
		vulkano::single_pass_renderpass!(
			logicalDevice.clone(),
		    attachments: {
		        color: {
		            load: Clear,
		            store: Store,
		            format: swapchain.format(),
		            samples: 1,
		        }
		    },
		    pass: {
		        color: [color],
		        depth_stencil: {}
		    }
		).unwrap()
	);

	let pipeline = Arc::new(
		GraphicsPipeline::start()
			.vertex_input_single_buffer::<Vertex>()
			.vertex_shader(vertexShader.main_entry_point(), ())
			.viewports_dynamic_scissors_irrelevant(1)
			.fragment_shader(fragmentShader.main_entry_point(), ())
			.render_pass(Subpass::from(renderPass.clone(), 0).unwrap())
			.build(logicalDevice.clone()).unwrap()
	);

	let mut dynamicState = DynamicState::none();

	let mut swapchainFramebuffers = swapchainImages.iter().map(|swapchainImage| {
		Arc::new(Framebuffer::start(renderPass.clone()).add(swapchainImage.clone()).unwrap().build().unwrap())
	});

	let commandBuffers = swapchainFramebuffers.map(|framebuffer| {
		let mut builder = AutoCommandBufferBuilder::primary(logicalDevice.clone(), queueFamily.clone()).unwrap();
		builder
			.begin_render_pass(framebuffer.clone(), false, vec![ClearValue::from([0.0, 0.0, 0.0, 1.0])]).unwrap()
			.draw(pipeline.clone(), &dynamicState, vertexBuffer.clone(), (), ()).unwrap()
			.end_render_pass().unwrap();
		builder
	}).map(|builder| builder.build().unwrap());
}
