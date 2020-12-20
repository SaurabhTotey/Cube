#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions, Queue};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, CompositeAlpha, acquire_next_image, Capabilities, Surface};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::format::Format;
use std::cmp::{min, max};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, BufferAccess};
use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{Subpass, Framebuffer, RenderPassAbstract, FramebufferAbstract};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder, AutoCommandBuffer};
use winit::event::{Event, WindowEvent};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{now, GpuFuture, SharingMode};
use vulkano::pipeline::shader::GraphicsEntryPointAbstract;
use std::collections::HashSet;

#[derive(Default, Copy, Clone)]
struct Vertex {
	position: [f32; 3],
	color: [f32; 4]
}
vulkano::impl_vertex!(Vertex, position, color);

struct Application {
	instance: Arc<Instance>,
	physicalDeviceIndex: usize,
	eventsLoop: EventLoop<()>,
	surface: Arc<Surface<Window>>,
	logicalDevice: Arc<Device>,
	graphicsQueue: Arc<Queue>,
	presentQueue: Arc<Queue>,
	swapchain: Arc<Swapchain<Window>>,
	swapchainImages: Vec<Arc<SwapchainImage<Window>>>,
	renderPass: Arc<dyn RenderPassAbstract + Send + Sync>,
	graphicsPipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
	swapchainFramebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
	vertexBuffer: Arc<CpuAccessibleBuffer<Vertex>>,
	indexBuffer: Arc<ImmutableBuffer<u32>>,
	commandBuffers: Vec<Arc<AutoCommandBuffer>>,
	previousFrameEnd: Option<Box<dyn GpuFuture>>,
	shouldRecreateSwapchain: bool
}

impl Application {

	fn new() -> Self {
		let instance = Instance::new(
			Some(&ApplicationInfo {
				application_name: Some(Cow::from("Cube!")),
				..ApplicationInfo::default()
			}),
			&vulkano_win::required_extensions(),
			None
		).unwrap();

		let eventsLoop = EventLoop::new();
		let surface = WindowBuilder::new()
			.with_title("Cube!")
			.build_vk_surface(&eventsLoop, instance.clone()).unwrap();

		let getSurfaceCapabilities = |physicalDevice: &PhysicalDevice| surface.capabilities(*physicalDevice).unwrap();
		let getGraphicsFamilyIndex = |physicalDevice: &PhysicalDevice| physicalDevice.queue_families().position(|family| family.supports_graphics());
		let getPresentFamilyIndex = |physicalDevice: &PhysicalDevice, surface: &Arc<Surface<Window>>| physicalDevice.queue_families().position(|family| surface.is_supported(family).unwrap_or(false));

		let requiredExtensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
		let physicalDeviceIndex = PhysicalDevice::enumerate(&instance).position(|physicalDevice| {
			let hasExtensions = DeviceExtensions::supported_by_device(physicalDevice).intersection(&requiredExtensions) == requiredExtensions;
			let surfaceCapabilities = getSurfaceCapabilities(&physicalDevice);
			return !surfaceCapabilities.supported_formats.is_empty() && surfaceCapabilities.present_modes.iter().next().is_some() && hasExtensions && getGraphicsFamilyIndex(&physicalDevice).is_some() && getPresentFamilyIndex(&physicalDevice, &surface).is_some();
		}).unwrap();

		let physicalDevice = PhysicalDevice::from_index(&instance, physicalDeviceIndex).unwrap();
		let queueFamilies = [getGraphicsFamilyIndex(&physicalDevice), getPresentFamilyIndex(&physicalDevice, &surface)].iter()
			.filter(|indexOption| indexOption.is_some())
			.collect::<HashSet<usize>>().iter()
			.map(|i| (physicalDevice.queue_families().nth(*i).unwrap(), 1.0));
		let (logicalDevice, mut queues) = Device::new(physicalDevice, &Features::none(), requiredExtensions, queueFamilies).unwrap();

		let graphicsQueue = queues.next().unwrap();
		let presentQueue = queues.next().unwrap_or(graphicsQueue.clone());

		let (swapchain, swapchainImages) = Self::createSwapchain(&getSurfaceCapabilities(&physicalDevice), &surface, &logicalDevice, &graphicsQueue, &presentQueue, None);
		let renderPass = Self::createRenderPass(&logicalDevice, swapchain.format());

		return Self {
			instance,
			physicalDeviceIndex,
			eventsLoop,
			surface,
			logicalDevice,
			graphicsQueue,
			presentQueue,
			swapchain,
			swapchainImages
		}
	}

	fn createSwapchain(surfaceCapabilities: &Capabilities, surface: &Arc<Surface<Window>>, logicalDevice: &Arc<Device>, graphicsQueue: &Arc<Queue>, presentQueue: &Arc<Queue>, oldSwapchain: Option<Arc<Swapchain<Window>>>) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
		let (imageFormat, colorSpace) = surfaceCapabilities.supported_formats.iter()
			.find(|formatAndColorSpace| formatAndColorSpace.0 == Format::B8G8R8A8Srgb && formatAndColorSpace.1 == ColorSpace::SrgbNonLinear)
			.unwrap_or(surfaceCapabilities.supported_formats.iter().next().unwrap());
		let presentMode = if surfaceCapabilities.present_modes.mailbox { PresentMode::Mailbox } else if surfaceCapabilities.present_modes.immediate { PresentMode::Immediate } else { PresentMode::Fifo };

		let imageCount = min(surfaceCapabilities.min_image_count + 1, surfaceCapabilities.max_image_count.unwrap_or(u32::MAX));
		let imageUsage = ImageUsage::color_attachment();
		let sharingMode: SharingMode = if *graphicsQueue != *presentQueue { vec![graphicsQueue, presentQueue].as_slice().into() } else { graphicsQueue.into() };
		let supportedAlpha = surfaceCapabilities.supported_composite_alpha.iter()
			.find(|&compositeAlpha| compositeAlpha == CompositeAlpha::Opaque)
			.unwrap_or(surfaceCapabilities.supported_composite_alpha.iter().next().unwrap());
		let dimensions: [u32; 2] = surface.window().inner_size().into();

		return if oldSwapchain.is_none() {
			Swapchain::new(
				logicalDevice.clone(),
				surface.clone(),
				imageCount,
				*imageFormat,
				dimensions,
				1,
				imageUsage,
				&sharingMode,
				surfaceCapabilities.current_transform,
				supportedAlpha,
				presentMode,
				FullscreenExclusive::Default,
				true,
				*colorSpace,
			).unwrap()
		} else {
			Swapchain::with_old_swapchain(
				logicalDevice.clone(),
				surface.clone(),
				imageCount,
				*imageFormat,
				dimensions,
				1,
				imageUsage,
				&sharingMode,
				surfaceCapabilities.current_transform,
				supportedAlpha,
				presentMode,
				FullscreenExclusive::Default,
				true,
				*colorSpace,
				oldSwapchain.unwrap()
			).unwrap()
		}
	}

	fn createRenderPass(logicalDevice: &Arc<Device>, colorFormat: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
		return Arc::new(
			vulkano::single_pass_renderpass!(
				logicalDevice.clone(),
				attachments: {
					color: {
						load: Clear,
						store: Store,
						format: colorFormat,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {}
				}
			).unwrap()
		);
	}

}

fn main() {

	let vertices = [
		Vertex { position: [ 0.5, 0.5, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
		Vertex { position: [-0.5, 0.5, 0.0], color: [0.0, 1.0, 0.0, 1.0] },
		Vertex { position: [-0.5,-0.5, 0.0], color: [0.0, 0.0, 1.0, 1.0] },
		Vertex { position: [ 0.5,-0.5, 0.0], color: [1.0, 1.0, 1.0, 1.0] }
	].to_vec();
	let vertexBuffer = CpuAccessibleBuffer::from_iter(
		logicalDevice.clone(),
		BufferUsage { vertex_buffer: true, ..BufferUsage::none() },
		false,
		vertices.iter().cloned()
	).unwrap();
	let indices = [0u32, 2, 1, 0, 2, 3];
	let indexBuffer = ImmutableBuffer::from_iter(
		indices.iter().cloned(),
		BufferUsage { index_buffer: true, ..BufferUsage::none() },
		deviceQueue.clone()
	).unwrap().0;

	// let uniformBuffers = swapchainImages.iter().map(|swapchainImage| {
	// 	CpuAccessibleBuffer::from_iter(
	// 		logicalDevice.clone(),
	// 		BufferUsage { uniform_buffer: true, ..BufferUsage::none() },
	// 		false,
	//
	// 	)
	// });

	mod VertexShader { vulkano_shaders::shader!{ ty: "vertex", path: "./src/shader/vertex.glsl" } }
	mod FragmentShader { vulkano_shaders::shader!{ ty: "fragment", path: "./src/shader/fragment.glsl" } }
	let vertexShader = VertexShader::Shader::load(logicalDevice.clone()).unwrap();
	let fragmentShader = FragmentShader::Shader::load(logicalDevice.clone()).unwrap();

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

	let mut swapchainFrameBuffers = frameBuffersForWindowSize(&swapchainImages.clone(), renderPass.clone(), &mut dynamicState);

	let mut shouldRecreateSwapchain = false;
	let mut previousFrameEnding = Some(now(logicalDevice.clone()).boxed());

	eventsLoop.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
				*control_flow = ControlFlow::Exit;
			},
			Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
				shouldRecreateSwapchain = true;
			},
			Event::RedrawEventsCleared => {
				previousFrameEnding.as_mut().unwrap().cleanup_finished();
				if shouldRecreateSwapchain {
					shouldRecreateSwapchain = false;
					let newDimensions = Into::<[u32; 2]>::into(surface.window().inner_size());
					let (newSwapchain, newImages) = match swapchain.recreate_with_dimensions(newDimensions) {
						Ok(r) => r,
						Err(_) => {
							return;
						}
					};
					swapchain = newSwapchain;
					swapchainFrameBuffers = frameBuffersForWindowSize(&newImages, renderPass.clone(), &mut dynamicState);
				}
				let (frameBufferIndex, isSuboptimal, acquireFuture) = match acquire_next_image(swapchain.clone(), None) {
					Ok(r) => r,
					Err(_) => {
						shouldRecreateSwapchain = true;
						return;
					}
				};
				if isSuboptimal {
					shouldRecreateSwapchain = true;
					return;
				}

				let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(logicalDevice.clone(), deviceQueue.family()).unwrap();
				builder
					.begin_render_pass(swapchainFrameBuffers[frameBufferIndex].clone(), false, vec![[0.0, 0.0, 0.0, 1.0].into()]).unwrap()
					.draw_indexed(pipeline.clone(), &dynamicState, vertexBuffer.clone(), indexBuffer.clone(), (), ()).unwrap()
					.end_render_pass().unwrap();
				let commandBuffer = builder.build().unwrap();

				let future = previousFrameEnding
					.take().unwrap()
					.join(acquireFuture)
					.then_execute(deviceQueue.clone(), commandBuffer).unwrap()
					.then_swapchain_present(deviceQueue.clone(), swapchain.clone(), frameBufferIndex)
					.then_signal_fence_and_flush();
				previousFrameEnding = match future {
					Ok(f) => Some(f.boxed()),
					Err(_) => {
						shouldRecreateSwapchain = true;
						Some(now(logicalDevice.clone()).boxed())
					}
				}
			},
			_ => ()
		}
	});
}

fn frameBuffersForWindowSize(images: &[Arc<SwapchainImage<Window>>], renderPass: Arc<dyn RenderPassAbstract + Send + Sync>, dynamicState: &mut DynamicState) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
	let dimensions = images[0].dimensions();
	let viewport = Viewport {
		origin: [0.0, 0.0],
		dimensions: [dimensions[0] as f32, dimensions[1] as f32],
		depth_range: 0.0..1.0,
	};
	dynamicState.viewports = Some(vec![viewport]);
	return images.iter()
		.map(|image| Arc::new(
			Framebuffer::start(renderPass.clone()).add(image.clone()).unwrap().build().unwrap()) as Arc<dyn FramebufferAbstract + Send + Sync>
		).collect::<Vec<_>>()
}
