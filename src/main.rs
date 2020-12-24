#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions, Queue};
use vulkano::swapchain::{Swapchain, PresentMode, FullscreenExclusive, ColorSpace, CompositeAlpha, Capabilities, Surface, acquire_next_image};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::format::Format;
use std::cmp::min;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, CpuBufferPool};
use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{Subpass, Framebuffer, RenderPassAbstract, FramebufferAbstract};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder, AutoCommandBuffer};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{now, GpuFuture, SharingMode};
use std::collections::HashSet;
use winit::dpi::LogicalSize;
use cgmath::{Matrix4, SquareMatrix, Rad, Deg, Point3, Vector3};
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use vulkano::descriptor::PipelineLayoutAbstract;
use std::time::Instant;

#[derive(Default, Copy, Clone)]
struct Vertex {
	position: [f32; 3],
	color: [f32; 4]
}
vulkano::impl_vertex!(Vertex, position, color);

#[derive(Copy, Clone)]
struct ModelViewProjectionTransformation {
	modelTransformation: Matrix4<f32>,
	viewTransformation: Matrix4<f32>,
	projectionTransformation: Matrix4<f32>
}
impl Default for ModelViewProjectionTransformation {
	fn default() -> Self {
		ModelViewProjectionTransformation { modelTransformation: Matrix4::identity(), viewTransformation: Matrix4::identity(), projectionTransformation: Matrix4::identity() }
	}
}

struct Application {
	instance: Arc<Instance>,
	physicalDeviceIndex: usize,
	surface: Arc<Surface<Window>>,
	logicalDevice: Arc<Device>,
	graphicsQueue: Arc<Queue>,
	presentQueue: Arc<Queue>,
	swapchain: Arc<Swapchain<Window>>,
	swapchainImages: Vec<Arc<SwapchainImage<Window>>>,
	renderPass: Arc<dyn RenderPassAbstract + Send + Sync>,
	graphicsPipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
	swapchainFramebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
	vertexBuffer: Arc<ImmutableBuffer<[Vertex]>>,
	indexBuffer: Arc<ImmutableBuffer<[u32]>>,
	descriptorSetsPool: Arc<FixedSizeDescriptorSetsPool>,
	uniformBufferPool: CpuBufferPool<ModelViewProjectionTransformation>,
	commandBuffers: Vec<Arc<AutoCommandBuffer>>,
	previousFrameEnd: Option<Box<dyn GpuFuture>>,
	shouldRecreateSwapchain: bool,
	startTime: Instant
}

impl Application {

	pub fn new() -> (Self, EventLoop<()>) {
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
			.with_inner_size(LogicalSize { width: 800.0, height: 600.0 })
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
			.map(|indexOption| indexOption.unwrap())
			.collect::<HashSet<usize>>().iter()
			.map(|i| (physicalDevice.queue_families().nth(*i).unwrap(), 1.0))
			.collect::<Vec<_>>();
		let (logicalDevice, mut queues) = Device::new(physicalDevice, &Features::none(), &requiredExtensions, queueFamilies).unwrap();

		let graphicsQueue = queues.next().unwrap();
		let presentQueue = queues.next().unwrap_or(graphicsQueue.clone());

		let (swapchain, swapchainImages) = Self::createSwapchain(&getSurfaceCapabilities(&physicalDevice), &surface, &logicalDevice, &graphicsQueue, &presentQueue, None);
		let renderPass = Self::createRenderPass(&logicalDevice, swapchain.format());
		let graphicsPipeline = Self::createGraphicsPipeline(&logicalDevice, swapchain.dimensions(), &renderPass);
		let swapchainFramebuffers = Self::createSwapchainFramebuffers(&swapchainImages, &renderPass);

		let vertices = [
			Vertex { position: [ 0.5, 0.5, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
			Vertex { position: [-0.5, 0.5, 0.0], color: [0.0, 1.0, 0.0, 1.0] },
			Vertex { position: [-0.5,-0.5, 0.0], color: [0.0, 0.0, 1.0, 1.0] },
			Vertex { position: [ 0.5,-0.5, 0.0], color: [1.0, 1.0, 1.0, 1.0] }
		].to_vec();
		let vertexBuffer = ImmutableBuffer::from_iter(
			vertices.iter().cloned(),
			BufferUsage::vertex_buffer(),
			graphicsQueue.clone()
		).unwrap().0;
		let indices = [0u32, 2, 1, 0, 2, 3];
		let indexBuffer = ImmutableBuffer::from_iter(
			indices.iter().cloned(),
			BufferUsage::index_buffer(),
			graphicsQueue.clone()
		).unwrap().0;

		let layout = graphicsPipeline.descriptor_set_layout(0).unwrap();
		let descriptorSetsPool = Arc::new(FixedSizeDescriptorSetsPool::new(layout.clone()));
		let uniformBufferPool = CpuBufferPool::<ModelViewProjectionTransformation>::new(logicalDevice.clone(), BufferUsage::uniform_buffer_transfer_destination());

		let commandBuffers = Self::createCommandBuffers(&graphicsQueue, &swapchainFramebuffers, &logicalDevice, &graphicsPipeline, &vertexBuffer, &indexBuffer);

		let previousFrameEnd = Some(Box::new(now(logicalDevice.clone())) as Box<dyn GpuFuture>);

		return (Self {
			instance,
			physicalDeviceIndex,
			surface,
			logicalDevice,
			graphicsQueue,
			presentQueue,
			swapchain,
			swapchainImages,
			renderPass,
			graphicsPipeline,
			swapchainFramebuffers,
			vertexBuffer,
			indexBuffer,
			descriptorSetsPool,
			uniformBufferPool,
			commandBuffers,
			previousFrameEnd,
			shouldRecreateSwapchain: false,
			startTime: Instant::now()
		}, eventsLoop)
	}

	pub fn run(mut self, eventsLoop: EventLoop<()>) {
		eventsLoop.run(move |event, _, controlFlow| {
			match event {
				winit::event::Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => { *controlFlow = ControlFlow::Exit },
				winit::event::Event::WindowEvent { event: winit::event::WindowEvent::Resized(_), .. } => { self.shouldRecreateSwapchain = true; }
				winit::event::Event::RedrawEventsCleared => {
					self.previousFrameEnd.as_mut().unwrap().cleanup_finished();

					if self.shouldRecreateSwapchain {
						let surfaceCapabilities = self.surface.capabilities(PhysicalDevice::from_index(&self.instance, self.physicalDeviceIndex).unwrap()).unwrap();
						let (newSwapchain, newSwapchainImages) = Self::createSwapchain(&surfaceCapabilities, &self.surface, &self.logicalDevice, &self.graphicsQueue, &self.presentQueue, Some(self.swapchain.clone()));
						self.swapchain = newSwapchain;
						self.swapchainImages = newSwapchainImages;
						self.renderPass = Self::createRenderPass(&self.logicalDevice, self.swapchain.format());
						self.graphicsPipeline = Self::createGraphicsPipeline(&self.logicalDevice, self.swapchain.dimensions(), &self.renderPass);
						self.swapchainFramebuffers = Self::createSwapchainFramebuffers(&self.swapchainImages, &self.renderPass);
						self.commandBuffers = Self::createCommandBuffers(&self.graphicsQueue, &self.swapchainFramebuffers, &self.logicalDevice, &self.graphicsPipeline, &self.vertexBuffer, &self.indexBuffer);
						self.shouldRecreateSwapchain = false;
					}

					let (imageIndex, isSuboptimal, swapchainAcquireFuture) = match acquire_next_image(self.swapchain.clone(), None) {
						Ok(value) => value,
						Err(_) => {
							self.shouldRecreateSwapchain = true;
							return;
						}
					};
					if isSuboptimal {
						self.shouldRecreateSwapchain = true;
					}

					let timePassed = Instant::now() - self.startTime;
					let transformation = ModelViewProjectionTransformation {
						modelTransformation: Matrix4::from_angle_z(Rad::from(Deg(timePassed.as_secs_f32() * 0.18))),
						viewTransformation: Matrix4::look_at(Point3::new(2f32, 2f32, 2f32), Point3::new(0f32, 0f32, 0f32), Vector3::new(0f32, 0f32, 1f32)),
						projectionTransformation: cgmath::perspective(Rad::from(Deg(45f32)), self.swapchain.dimensions()[0] as f32 / self.swapchain.dimensions()[1] as f32, 0.1, 10f32)
					};

					let commandBuffer = self.commandBuffers[imageIndex].clone();

					let future = self.previousFrameEnd
						.take().unwrap()
						.join(swapchainAcquireFuture)
						.then_execute(self.graphicsQueue.clone(), commandBuffer).unwrap()
						.then_swapchain_present(self.presentQueue.clone(), self.swapchain.clone(), imageIndex)
						.then_signal_fence_and_flush();
					match future {
						Ok(f) => { self.previousFrameEnd = Some(Box::new(f)); },
						Err(_) => {
							self.shouldRecreateSwapchain = true;
							self.previousFrameEnd = Some(Box::new(now(self.logicalDevice.clone())));
						}
					}
				},
				_ => ()
			}
		});
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
				sharingMode,
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
				sharingMode,
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

	fn createGraphicsPipeline(logicalDevice: &Arc<Device>, swapchainExtent: [u32; 2], renderPass: &Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
		mod VertexShader { vulkano_shaders::shader!{ ty: "vertex", path: "./src/shader/vertex.glsl" } }
		mod FragmentShader { vulkano_shaders::shader!{ ty: "fragment", path: "./src/shader/fragment.glsl" } }
		let vertexShader = VertexShader::Shader::load(logicalDevice.clone()).unwrap();
		let fragmentShader = FragmentShader::Shader::load(logicalDevice.clone()).unwrap();

		let viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [swapchainExtent[0] as f32, swapchainExtent[1] as f32],
			depth_range: 0.0 .. 1.0
		};

		return Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<Vertex>()
				.vertex_shader(vertexShader.main_entry_point(), ())
				.viewports(vec![viewport])
				.fragment_shader(fragmentShader.main_entry_point(), ())
				.render_pass(Subpass::from(renderPass.clone(), 0).unwrap())
				.build(logicalDevice.clone()).unwrap()
		);
	}

	fn createSwapchainFramebuffers(swapchainImages: &[Arc<SwapchainImage<Window>>], renderPass: &Arc<dyn RenderPassAbstract + Send + Sync>) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
		return swapchainImages.iter().map(|image| {
			let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(Framebuffer::start(renderPass.clone()).add(image.clone()).unwrap().build().unwrap());
			return framebuffer;
		}).collect();
	}

	fn createCommandBuffers(graphicsQueue: &Arc<Queue>, swapchainFramebuffers: &Vec<Arc<dyn FramebufferAbstract + Send + Sync>>, logicalDevice: &Arc<Device>, graphicsPipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>, vertexBuffer: &Arc<ImmutableBuffer<[Vertex]>>, indexBuffer: &Arc<ImmutableBuffer<[u32]>>) -> Vec<Arc<AutoCommandBuffer>> {
		let queueFamily = graphicsQueue.family();
		return swapchainFramebuffers.iter().map(|framebuffer| {
			let mut builder = AutoCommandBufferBuilder::primary_simultaneous_use(logicalDevice.clone(), queueFamily).unwrap();
			builder
				.begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 0.0, 1.0].into()]).unwrap()
				.draw_indexed(graphicsPipeline.clone(), &DynamicState::none(), vec![vertexBuffer.clone()], indexBuffer.clone(), (), ()).unwrap()
				.end_render_pass().unwrap();
			return Arc::new(builder.build().unwrap());
		}).collect();
	}

}

fn main() {
	let (application, eventsLoop) = Application::new();
	application.run(eventsLoop);
}
