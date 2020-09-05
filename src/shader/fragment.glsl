#version 450

layout(location = 0) in vec4 fragmentColor;
layout(location = 0) out vec4 outputColor;

void main() {
	outputColor = fragmentColor;
}
