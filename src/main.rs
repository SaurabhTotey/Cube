#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window, Fullscreen};
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions, Queue};
use vulkano::swapchain::{Swapchain, PresentMode, FullscreenExclusive, ColorSpace, CompositeAlpha, Capabilities, Surface, acquire_next_image};
use vulkano::image::{ImageUsage, SwapchainImage, AttachmentImage, ImmutableImage, Dimensions};
use vulkano::format::Format;
use std::cmp::min;
use vulkano::buffer::{BufferUsage, ImmutableBuffer, CpuBufferPool};
use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{Subpass, Framebuffer, RenderPassAbstract, FramebufferAbstract};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{now, GpuFuture, SharingMode};
use std::collections::{HashSet, HashMap};
use cgmath::{Matrix4, Rad, Point3, Vector3, Matrix3, InnerSpace};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSetsPool, PersistentDescriptorSet};
use vulkano::descriptor::{PipelineLayoutAbstract, DescriptorSet};
use std::f32::consts::PI;
use winit::event::{KeyboardInput, ElementState, VirtualKeyCode};
use std::time::Instant;
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode};
use std::io::Cursor;

mod ColoredVertexShader { vulkano_shaders::shader!{ ty: "vertex", path: "./src/shader/colored/vertex.glsl" } }
mod ColoredFragmentShader { vulkano_shaders::shader!{ ty: "fragment", path: "./src/shader/colored/fragment.glsl" } }
mod TexturedVertexShader { vulkano_shaders::shader!{ ty: "vertex", path: "./src/shader/textured/vertex.glsl" } }
mod TexturedFragmentShader { vulkano_shaders::shader!{ ty: "fragment", path: "./src/shader/textured/fragment.glsl" } }

#[derive(Default, Copy, Clone)]
struct ColoredVertex {
	position: [f32; 3],
	color: [f32; 4]
}
vulkano::impl_vertex!(ColoredVertex, position, color);

#[derive(Default, Copy, Clone)]
struct TexturedVertex {
	position: [f32; 3],
	faceId: i32,
	cornerId: i32
}
vulkano::impl_vertex!(TexturedVertex, position, faceId, cornerId);

#[derive(Copy, Clone)]
struct CameraTransformation {
	position: Point3<f32>,
	backwardsDirection: Vector3<f32>,
	rightDirection: Vector3<f32>,
	upDirection: Vector3<f32>
}
impl CameraTransformation {
	fn new() -> Self {
		CameraTransformation {
			position: Point3::new(-2.0, 0.0, 0.0),
			backwardsDirection: Vector3::new(-1.0, 0.0, 0.0),
			rightDirection: Vector3::new(0.0, -1.0, 0.0),
			upDirection: Vector3::new(0.0, 0.0, 1.0)
		}
	}

	fn rotated(&self, yaw: &f32, pitch: &f32) -> Self {
		let pitchTransformation = Matrix3::<f32>::from_axis_angle(self.rightDirection, Rad(-*pitch));
		let yawTransformation = Matrix3::<f32>::from_axis_angle(self.upDirection, Rad(-*yaw));
		let combinedTransformation = pitchTransformation * yawTransformation;
		return CameraTransformation {
			position: self.position,
			backwardsDirection: combinedTransformation * self.backwardsDirection,
			rightDirection: combinedTransformation * self.rightDirection,
			upDirection: combinedTransformation * self.upDirection
		};
	}

	fn getTransformation(&self, aspectRatio: f32) -> Matrix4<f32> {
		let viewTransformation = Matrix4::look_at_dir(self.position, -self.backwardsDirection, self.upDirection);
		let mut projectionTransformation = cgmath::perspective(Rad(PI / 4.0), aspectRatio, 0.001, 100.0);
		projectionTransformation.y *= -1.0;
		return projectionTransformation * viewTransformation;
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
	coloredGraphicsPipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
	texturedGraphicsPipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
	depthBuffer: Arc<AttachmentImage>,
	swapchainFramebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
	coloredVertexBuffer: Arc<ImmutableBuffer<[ColoredVertex]>>,
	coloredIndexBuffer: Arc<ImmutableBuffer<[u32]>>,
	texturedVertexBuffer: Arc<ImmutableBuffer<[TexturedVertex]>>,
	texturedIndexBuffer: Arc<ImmutableBuffer<[u32]>>,
	descriptorSetsPool: FixedSizeDescriptorSetsPool,
	uniformBufferPool: CpuBufferPool<Matrix4<f32>>,
	textureDescriptorSet: Arc<dyn DescriptorSet + Send + Sync>,
	previousFrameEnd: Option<Box<dyn GpuFuture>>,
	shouldRecreateSwapchain: bool,
	cameraTransformation: CameraTransformation,
	previousFrameEndInstant: Instant,
	startTime: Instant,
	keyToIsPressed: HashMap<VirtualKeyCode, bool>
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
			.build_vk_surface(&eventsLoop, instance.clone()).unwrap();
		surface.window().set_cursor_grab(true).unwrap_or(());
		surface.window().set_cursor_visible(false);
		surface.window().set_fullscreen(Some(Fullscreen::Borderless(surface.window().current_monitor())));

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
		let (coloredGraphicsPipeline, texturedGraphicsPipeline) = Self::createGraphicsPipelines(&logicalDevice, swapchain.dimensions(), &renderPass);

		let depthBuffer = AttachmentImage::transient(logicalDevice.clone(), swapchain.dimensions(), Format::D32Sfloat).unwrap();
		let swapchainFramebuffers = Self::createSwapchainFramebuffers(&swapchainImages, &renderPass, &depthBuffer);

		let coloredVertices = [
			ColoredVertex { position: [ 0.5, 0.5,-0.5], color: [1.0, 0.0, 0.0, 1.0] },
			ColoredVertex { position: [-0.5, 0.5,-0.5], color: [0.0, 1.0, 0.0, 1.0] },
			ColoredVertex { position: [-0.5,-0.5,-0.5], color: [0.0, 0.0, 1.0, 1.0] },
			ColoredVertex { position: [ 0.5,-0.5,-0.5], color: [1.0, 1.0, 1.0, 1.0] },
			ColoredVertex { position: [ 0.5, 0.5, 0.5], color: [1.0, 0.0, 0.0, 1.0] },
			ColoredVertex { position: [-0.5, 0.5, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
			ColoredVertex { position: [-0.5,-0.5, 0.5], color: [0.0, 0.0, 1.0, 1.0] },
			ColoredVertex { position: [ 0.5,-0.5, 0.5], color: [1.0, 1.0, 1.0, 1.0] }
		].to_vec();
		let coloredVertexBuffer = ImmutableBuffer::from_iter(
			coloredVertices.iter().cloned(),
			BufferUsage::vertex_buffer(),
			graphicsQueue.clone()
		).unwrap().0;
		let coloredIndices = [
			0u32, 2, 1, 0, 2, 3, //bottom face
			4   , 6, 5, 4, 6, 7, //top face
			0   , 1, 4, 1, 4, 5, //red-green side face
			1   , 2, 6, 1, 6, 5, //green-blue side face
			2   , 3, 6, 3, 6, 7, //blue-white side face
			3   , 0, 7, 0, 7, 4  //white-red side face
		];
		let coloredIndexBuffer = ImmutableBuffer::from_iter(
			coloredIndices.iter().cloned(),
			BufferUsage::index_buffer(),
			graphicsQueue.clone()
		).unwrap().0;

		let texturedVertices = [
			TexturedVertex { position: [-0.5,-0.5,-0.5], faceId: 0, cornerId: 0 },
			TexturedVertex { position: [-0.5, 0.5,-0.5], faceId: 0, cornerId: 1 },
			TexturedVertex { position: [ 0.5,-0.5,-0.5], faceId: 0, cornerId: 2 },
			TexturedVertex { position: [ 0.5, 0.5,-0.5], faceId: 0, cornerId: 3 },
			TexturedVertex { position: [-0.5,-0.5, 0.5], faceId: 1, cornerId: 0 },
			TexturedVertex { position: [-0.5, 0.5, 0.5], faceId: 1, cornerId: 1 },
			TexturedVertex { position: [ 0.5,-0.5, 0.5], faceId: 1, cornerId: 2 },
			TexturedVertex { position: [ 0.5, 0.5, 0.5], faceId: 1, cornerId: 3 },
			TexturedVertex { position: [-0.5,-0.5,-0.5], faceId: 2, cornerId: 0 },
			TexturedVertex { position: [-0.5,-0.5, 0.5], faceId: 2, cornerId: 1 },
			TexturedVertex { position: [-0.5, 0.5,-0.5], faceId: 2, cornerId: 2 },
			TexturedVertex { position: [-0.5, 0.5, 0.5], faceId: 2, cornerId: 3 },
			TexturedVertex { position: [ 0.5,-0.5,-0.5], faceId: 3, cornerId: 0 },
			TexturedVertex { position: [ 0.5,-0.5, 0.5], faceId: 3, cornerId: 1 },
			TexturedVertex { position: [ 0.5, 0.5,-0.5], faceId: 3, cornerId: 2 },
			TexturedVertex { position: [ 0.5, 0.5, 0.5], faceId: 3, cornerId: 3 },
			TexturedVertex { position: [-0.5,-0.5,-0.5], faceId: 4, cornerId: 0 },
			TexturedVertex { position: [-0.5,-0.5, 0.5], faceId: 4, cornerId: 1 },
			TexturedVertex { position: [ 0.5,-0.5,-0.5], faceId: 4, cornerId: 2 },
			TexturedVertex { position: [ 0.5,-0.5, 0.5], faceId: 4, cornerId: 3 },
			TexturedVertex { position: [-0.5, 0.5,-0.5], faceId: 5, cornerId: 0 },
			TexturedVertex { position: [-0.5, 0.5, 0.5], faceId: 5, cornerId: 1 },
			TexturedVertex { position: [ 0.5, 0.5,-0.5], faceId: 5, cornerId: 2 },
			TexturedVertex { position: [ 0.5, 0.5, 0.5], faceId: 5, cornerId: 3 }
		].to_vec();
		let texturedVertexBuffer = ImmutableBuffer::from_iter(
			texturedVertices.iter().cloned(),
			BufferUsage::vertex_buffer(),
			graphicsQueue.clone()
		).unwrap().0;
		let texturedIndices = [
			0u32,  1,  2,  3,  1,  2, // bottom face which should show the 1 face of the die
			4   ,  5,  6,  7,  5,  6, // top face which should show the 2 face of the die
			8   ,  9, 10, 11,  9, 10, // back face which should show the 3 face of the die
			12  , 13, 14, 15, 13, 14, // front face which should show the 4 face of the die
			16  , 17, 18, 19, 17, 18, // left face which should show the 5 face of the die
			20  , 21, 22, 23, 21, 22  // right face which should show the 6 face of the die
		];
		let texturedIndexBuffer = ImmutableBuffer::from_iter(
			texturedIndices.iter().cloned(),
			BufferUsage::index_buffer(),
			graphicsQueue.clone()
		).unwrap().0;

		let layout = coloredGraphicsPipeline.descriptor_set_layout(0).unwrap(); // should be the same between both pipelines
		let descriptorSetsPool = FixedSizeDescriptorSetsPool::new(layout.clone());
		let uniformBufferPool = CpuBufferPool::<Matrix4<f32>>::uniform_buffer(logicalDevice.clone());

		let imageBytes = include_bytes!("../assets/DieFaces.png").to_vec();
		let cursor = Cursor::new(imageBytes);
		let decoder = png::Decoder::new(cursor);
		let (info, mut reader) = decoder.read_info().unwrap();
		let mut imageData = Vec::new();
		imageData.resize(7 * 7 * 6 * 4 as usize, 0); // each face is 7x7, there are 6 faces, and there are 4 color values
		reader.next_frame(&mut imageData).unwrap();
		let (texture, textureFuture) = ImmutableImage::from_iter(
			imageData.iter().cloned(),
			Dimensions::Dim2d { width: info.width, height: info.height },
			Format::R8G8B8A8Srgb,
			graphicsQueue.clone()
		).unwrap();

		let sampler = Sampler::new(
			logicalDevice.clone(),
			Filter::Nearest,
			Filter::Nearest,
			MipmapMode::Nearest,
			SamplerAddressMode::Repeat,
			SamplerAddressMode::Repeat,
			SamplerAddressMode::Repeat,
			0.0,
			1.0,
			0.0,
			0.0
		).unwrap();
		let textureDescriptorLayout = texturedGraphicsPipeline.descriptor_set_layout(1).unwrap();
		let textureDescriptorSet = Arc::new(
			PersistentDescriptorSet::start(textureDescriptorLayout.clone())
				.add_sampled_image(texture.clone(), sampler.clone()).unwrap()
				.build().unwrap()
		);

		let previousFrameEnd = Some(textureFuture.boxed());

		let mut keyToIsPressed = HashMap::new();
		[VirtualKeyCode::W, VirtualKeyCode::A, VirtualKeyCode::S, VirtualKeyCode::D, VirtualKeyCode::Space, VirtualKeyCode::LShift].iter().for_each(|keyCode| {
			keyToIsPressed.insert(*keyCode, false);
		});

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
			coloredGraphicsPipeline,
			texturedGraphicsPipeline,
			depthBuffer,
			swapchainFramebuffers,
			coloredVertexBuffer,
			coloredIndexBuffer,
			texturedVertexBuffer,
			texturedIndexBuffer,
			descriptorSetsPool,
			uniformBufferPool,
			textureDescriptorSet,
			previousFrameEnd,
			shouldRecreateSwapchain: false,
			cameraTransformation: CameraTransformation::new(),
			previousFrameEndInstant: Instant::now(),
			startTime: Instant::now(),
			keyToIsPressed
		}, eventsLoop)
	}

	pub fn run(mut self, eventsLoop: EventLoop<()>) {
		eventsLoop.run(move |event, _, controlFlow| {
			match event {
				winit::event::Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => { *controlFlow = ControlFlow::Exit },
				winit::event::Event::WindowEvent { event: winit::event::WindowEvent::Resized(_), .. } => { self.shouldRecreateSwapchain = true; },
				winit::event::Event::WindowEvent { event: winit::event::WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(keycode), state: pressedState, .. }, .. }, .. } => {
					if self.keyToIsPressed.contains_key(&keycode) {
						self.keyToIsPressed.insert(keycode, pressedState == ElementState::Pressed);
					}
					if keycode == VirtualKeyCode::Escape {
						*controlFlow = ControlFlow::Exit;
					}
				},
				winit::event::Event::DeviceEvent { event: winit::event::DeviceEvent::MouseMotion { delta }, .. } => {
					let sensitivity = 0.005;
					self.cameraTransformation = self.cameraTransformation.rotated(&(delta.0 as f32 * sensitivity), &(delta.1 as f32 * sensitivity));
				},
				winit::event::Event::RedrawEventsCleared => {
					self.previousFrameEnd.as_mut().unwrap().cleanup_finished();

					if self.shouldRecreateSwapchain {
						let surfaceCapabilities = self.surface.capabilities(PhysicalDevice::from_index(&self.instance, self.physicalDeviceIndex).unwrap()).unwrap();
						let (newSwapchain, newSwapchainImages) = Self::createSwapchain(&surfaceCapabilities, &self.surface, &self.logicalDevice, &self.graphicsQueue, &self.presentQueue, Some(self.swapchain.clone()));
						self.swapchain = newSwapchain;
						self.swapchainImages = newSwapchainImages;
						self.renderPass = Self::createRenderPass(&self.logicalDevice, self.swapchain.format());
						let pipelines = Self::createGraphicsPipelines(&self.logicalDevice, self.swapchain.dimensions(), &self.renderPass);
						self.coloredGraphicsPipeline = pipelines.0;
						self.texturedGraphicsPipeline = pipelines.1;
						self.depthBuffer = AttachmentImage::transient(self.logicalDevice.clone(), self.swapchain.dimensions(), Format::D32Sfloat).unwrap();
						self.swapchainFramebuffers = Self::createSwapchainFramebuffers(&self.swapchainImages, &self.renderPass, &self.depthBuffer);
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

					let deltaTime = Instant::now() - self.previousFrameEndInstant;
					let speed = 20.0;
					let directionMultiplier = deltaTime.as_secs_f32() * speed;
					if *self.keyToIsPressed.get(&VirtualKeyCode::W).unwrap() {
						self.cameraTransformation.position -= directionMultiplier * self.cameraTransformation.backwardsDirection;
					}
					if *self.keyToIsPressed.get(&VirtualKeyCode::A).unwrap() {
						self.cameraTransformation.position -= directionMultiplier * self.cameraTransformation.rightDirection;
					}
					if *self.keyToIsPressed.get(&VirtualKeyCode::S).unwrap() {
						self.cameraTransformation.position += directionMultiplier * self.cameraTransformation.backwardsDirection;
					}
					if *self.keyToIsPressed.get(&VirtualKeyCode::D).unwrap() {
						self.cameraTransformation.position += directionMultiplier * self.cameraTransformation.rightDirection;
					}
					if *self.keyToIsPressed.get(&VirtualKeyCode::Space).unwrap() {
						self.cameraTransformation.position += directionMultiplier * self.cameraTransformation.upDirection;
					}
					if *self.keyToIsPressed.get(&VirtualKeyCode::LShift).unwrap() {
						self.cameraTransformation.position -= directionMultiplier * self.cameraTransformation.upDirection;
					}

					let uniformBuffer = self.uniformBufferPool.next(self.cameraTransformation.getTransformation(self.swapchain.dimensions()[0] as f32 / self.swapchain.dimensions()[1] as f32)).unwrap();
					let descriptorSet = Arc::new(self.descriptorSetsPool.clone().next().add_buffer(uniformBuffer.clone()).unwrap().build().unwrap());

					let angle = Rad((Instant::now() - self.startTime).as_secs_f32());
					let pushConstantsCubeOne = ColoredVertexShader::ty::ModelTransformation { transformation: Matrix4::from_angle_z(angle).into() };
					let pushConstantsCubeTwo = ColoredVertexShader::ty::ModelTransformation { transformation: Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0)).into() };
					let pushConstantsCubeThree = TexturedVertexShader::ty::ModelTransformation{ transformation: (Matrix4::from_translation(Vector3::new(-3.0, -3.0, -3.0)) * Matrix4::from_axis_angle(Vector3::new(1.0, 1.0, 1.0).normalize(), angle)).into() };

					let mut commandBufferBuilder = AutoCommandBufferBuilder::new(self.logicalDevice.clone(), self.graphicsQueue.family()).unwrap();
					commandBufferBuilder
						.begin_render_pass(self.swapchainFramebuffers[imageIndex].clone(), false, vec![[0.2, 0.2, 0.2, 1.0].into(), 1f32.into()]).unwrap()
						.draw_indexed(self.coloredGraphicsPipeline.clone(), &DynamicState::none(), vec![self.coloredVertexBuffer.clone()], self.coloredIndexBuffer.clone(), descriptorSet.clone(), pushConstantsCubeOne).unwrap()
						.draw_indexed(self.coloredGraphicsPipeline.clone(), &DynamicState::none(), vec![self.coloredVertexBuffer.clone()], self.coloredIndexBuffer.clone(), descriptorSet.clone(), pushConstantsCubeTwo).unwrap()
						.draw_indexed(self.texturedGraphicsPipeline.clone(), &DynamicState::none(), vec![self.texturedVertexBuffer.clone()], self.texturedIndexBuffer.clone(), [descriptorSet.clone(), self.textureDescriptorSet.clone()].to_vec(), pushConstantsCubeThree).unwrap()
						.end_render_pass().unwrap();
					let commandBuffer = commandBufferBuilder.build().unwrap();

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
			self.previousFrameEndInstant = Instant::now();
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
					},
					depth: {
						load: Clear,
						store: DontCare,
						format: Format::D32Sfloat,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {depth}
				}
			).unwrap()
		);
	}

	fn createGraphicsPipelines(logicalDevice: &Arc<Device>, swapchainExtent: [u32; 2], renderPass: &Arc<dyn RenderPassAbstract + Send + Sync>) -> (Arc<dyn GraphicsPipelineAbstract + Send + Sync>, Arc<dyn GraphicsPipelineAbstract + Send + Sync>) {
		let coloredVertexShader = ColoredVertexShader::Shader::load(logicalDevice.clone()).unwrap();
		let coloredFragmentShader = ColoredFragmentShader::Shader::load(logicalDevice.clone()).unwrap();
		let texturedVertexShader = TexturedVertexShader::Shader::load(logicalDevice.clone()).unwrap();
		let texturedFragmentShader = TexturedFragmentShader::Shader::load(logicalDevice.clone()).unwrap();

		let viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [swapchainExtent[0] as f32, swapchainExtent[1] as f32],
			depth_range: 0.0 .. 1.0
		};

		return (
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input_single_buffer::<ColoredVertex>()
					.vertex_shader(coloredVertexShader.main_entry_point(), ())
					.viewports(vec![viewport.clone()])
					.fragment_shader(coloredFragmentShader.main_entry_point(), ())
					.depth_stencil_simple_depth()
					.render_pass(Subpass::from(renderPass.clone(), 0).unwrap())
					.build(logicalDevice.clone()).unwrap()
			),
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input_single_buffer::<TexturedVertex>()
					.vertex_shader(texturedVertexShader.main_entry_point(), ())
					.viewports(vec![viewport.clone()])
					.fragment_shader(texturedFragmentShader.main_entry_point(), ())
					.depth_stencil_simple_depth()
					.render_pass(Subpass::from(renderPass.clone(), 0).unwrap())
					.build(logicalDevice.clone()).unwrap()
			)
		);
	}

	fn createSwapchainFramebuffers(swapchainImages: &[Arc<SwapchainImage<Window>>], renderPass: &Arc<dyn RenderPassAbstract + Send + Sync>, depthBuffer: &Arc<AttachmentImage>) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
		return swapchainImages.iter().map(|image| {
			return Arc::new(Framebuffer::start(renderPass.clone())
				.add(image.clone()).unwrap()
				.add(depthBuffer.clone()).unwrap()
				.build().unwrap()
			) as Arc<dyn FramebufferAbstract + Send + Sync>;
		}).collect();
	}

}

fn main() {
	let (application, eventsLoop) = Application::new();
	application.run(eventsLoop);
}
