#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in float v_use_texture;

layout(location = 0) out vec4 out_color;

// SDL_GPU SPIR-V: fragment samplers at set 2
layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
    if (v_use_texture > 0.5) {
        out_color = texture(tex, v_uv);
    } else {
        out_color = v_color;
    }
}
