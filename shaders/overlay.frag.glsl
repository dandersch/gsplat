#version 450

layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 out_color;

// SDL_GPU SPIR-V: fragment samplers at set 2, uniforms at set 3
layout(set = 2, binding = 0) uniform sampler2D panorama;

layout(std140, set = 3, binding = 0) uniform OverlayUniforms {
    mat4 inv_view_proj;
    mat4 ref_rotation;   // 3x3 world-to-refcam rotation in upper-left, Y-flip baked in
    float alpha;
};

const float PI = 3.14159265358979;

void main() {
    // Reconstruct world-space ray direction from NDC
    vec4 near_world = inv_view_proj * vec4(v_ndc, 0.0, 1.0);
    vec4 far_world  = inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    near_world /= near_world.w;
    far_world  /= far_world.w;

    vec3 dir = normalize(far_world.xyz - near_world.xyz);

    // Transform to reference camera space (colmap: X-right, Y-down, Z-forward)
    vec3 ref_dir = normalize(mat3(ref_rotation) * dir);

    // Equirectangular UV mapping
    float u = atan(-ref_dir.x, ref_dir.z) / (2.0 * PI) + 0.5;
    float v = -asin(clamp(ref_dir.y, -1.0, 1.0)) / PI + 0.5;

    vec4 tex_color = texture(panorama, vec2(u, v));

    // Premultiplied alpha output (matches existing blend mode: ONE, ONE_MINUS_SRC_ALPHA)
    out_color = vec4(tex_color.rgb * alpha, alpha);
}
