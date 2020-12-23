#version 450

layout(binding = 0) uniform ModelViewProjectionTransformation {
	mat4 modelTransformation;
	mat4 viewTransformation;
	mat4 projectionTransformation;
} mvp;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 fragmentColor;

void main() {
	gl_Position = mvp.projectionTransformation * mvp.viewTransformation * mvp.modelTransformation * vec4(position, 1.0);
	fragmentColor = color;
}
