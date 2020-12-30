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

const int IMAGE_SIZE = 7;
const vec2 IMAGE_DIMENSIONS = vec2(42.0, 7.0);

void main() {
	gl_Position = camera.transformation * model.transformation * vec4(position, 1.0);
	vec2 cornerOffset = vec2(0.0);
	if (cornerId == 1) {
		cornerOffset = vec2(float(IMAGE_SIZE), 0.0);
	}
	else if (cornerId == 2) {
		cornerOffset = vec2(0.0, float(IMAGE_SIZE));
	}
	else if (cornerId == 3) {
		cornerOffset = vec2(float(IMAGE_SIZE));
	}
	vec2 faceOffset = vec2(float(faceId * IMAGE_SIZE), 0.0);
	textureCoordinates = (cornerOffset + faceOffset) / IMAGE_DIMENSIONS;
}
