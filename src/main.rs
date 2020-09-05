#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::dpi::PhysicalSize;
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, CompositeAlpha};
use vulkano::image::ImageUsage;
use vulkano::format::Format;
use std::cmp::min;

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
	let (mut swapchain, images) = {
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


}
