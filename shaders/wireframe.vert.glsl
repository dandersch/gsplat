#version 450

layout(location = 0) in vec3 in_position;

// SDL_GPU SPIR-V: vertex uniforms at set 1
layout(std140, set = 1, binding = 0) uniform Uniforms {
    mat4 mvp;
    vec4 color;
};

layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = color;
}
