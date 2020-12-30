#version 450

layout(set = 0, binding = 0) uniform CameraTransformation {
	mat4 transformation;
} camera;

layout(push_constant) uniform ModelTransformation {
	mat4 transformation;
} model;

layout(location = 0) in vec3 position;
layout(location = 1) in int faceId;
layout(location = 2) in int cornerId;
layout(location = 0) out vec2 textureCoordinates;

void main() {
	gl_Position = camera.transformation * model.transformation * vec4(position, 1.0);
	textureCoordinates = vec2(0.0); // TODO: determine this based off of faceId and cornerId
}
