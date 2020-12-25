#version 450

layout(binding = 0) uniform CameraTransformation {
	mat4 viewTransformation;
	mat4 projectionTransformation;
} camera;

layout(push_constant) uniform ModelTransformation {
	mat4 transformation;
} model;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 fragmentColor;

void main() {
	gl_Position = camera.projectionTransformation * camera.viewTransformation * model.transformation * vec4(position, 1.0);
	fragmentColor = color;
}
