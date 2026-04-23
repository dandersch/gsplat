#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

// SDL_GPU SPIR-V: vertex uniforms at set 1
layout(std140, set = 1, binding = 0) uniform Uniforms {
    mat4 mvp;
    vec4 color;
    float use_texture;
};

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out float v_use_texture;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = color;
    v_uv = in_uv;
    v_use_texture = use_texture;
}
