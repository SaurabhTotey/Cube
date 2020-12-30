#version 450

layout(location = 0) in vec2 textureCoordinates;
layout(location = 0) out vec4 outputColor;

//layout(set = 1, binding = 0) uniform sampler2D textureSampler;

void main() {
//	outputColor = texture(textureSampler, textureCoordinates);
	outputColor = vec4(textureCoordinates / 7.0, 0.0, 0.0);
}
