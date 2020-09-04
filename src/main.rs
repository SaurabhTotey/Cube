#![allow(non_snake_case)]

use vulkano::instance::{Instance, ApplicationInfo, PhysicalDevice};
use std::borrow::Cow;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::dpi::PhysicalSize;
use vulkano_win::VkSurfaceBuild;
use vulkano::device::{Device, Features, DeviceExtensions};

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
		.with_inner_size(PhysicalSize::new(1024, 1024))
		.build_vk_surface(&eventsLoop, instance.clone()).unwrap();
	let surfaceCapabilities = surface.capabilities(physicalDevice).unwrap();

	let queueFamily = physicalDevice.queue_families()
		.find(|&family| family.supports_graphics() && surface.is_supported(family).unwrap_or(false)).unwrap();
	let (logicalDevice, mut queues) = { Device::new(
		physicalDevice,
		&Features::none(),
		&DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() },
		[(queueFamily, 1.0)].iter().cloned(),
	).unwrap() };
	let queue = queues.next().unwrap();
}
