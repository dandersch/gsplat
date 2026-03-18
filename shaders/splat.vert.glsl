#version 450

// SDL_GPU SPIR-V layout for vertex shaders:
//   set 0: storage textures, storage buffers  
//   set 1: uniform buffers
layout(std430, set = 0, binding = 0) readonly buffer IndexBuffer {
    uint sorted_indices[];
};

layout(std430, set = 0, binding = 1) readonly buffer GaussianBuffer {
    vec4 data[];
} gaussians;

layout(set = 1, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec2 viewport;
};

layout(location = 0) out vec3 frag_color;
layout(location = 1) out float frag_opacity;
layout(location = 2) out vec2 frag_center;
layout(location = 3) out vec3 frag_conic;

void main() {
    // 1. Determine quad corner from vertex ID
    int quad_verts[6] = int[6](0, 1, 2, 0, 2, 3);
    vec2 corners[4] = vec2[4](
        vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1)
    );
    vec2 corner = corners[quad_verts[gl_VertexIndex % 6]];

    // 2. Fetch Gaussian data
    uint idx = sorted_indices[gl_InstanceIndex];
    vec3 position = gaussians.data[idx * 4u + 0u].xyz;
    float opacity = gaussians.data[idx * 4u + 0u].w;
    vec3 scale    = gaussians.data[idx * 4u + 1u].xyz;
    vec4 rot      = gaussians.data[idx * 4u + 2u];    // (w, x, y, z)
    vec3 color    = gaussians.data[idx * 4u + 3u].xyz;

    // 3. Build rotation matrix from quaternion
    mat3 R = mat3(
        1.0 - 2.0*(rot.z*rot.z + rot.w*rot.w),
        2.0*(rot.y*rot.z + rot.x*rot.w),
        2.0*(rot.y*rot.w - rot.x*rot.z),

        2.0*(rot.y*rot.z - rot.x*rot.w),
        1.0 - 2.0*(rot.y*rot.y + rot.w*rot.w),
        2.0*(rot.z*rot.w + rot.x*rot.y),

        2.0*(rot.y*rot.w + rot.x*rot.z),
        2.0*(rot.z*rot.w - rot.x*rot.y),
        1.0 - 2.0*(rot.y*rot.y + rot.z*rot.z)
    );

    // 4. Build 3D covariance: Σ = R·S·Sᵀ·Rᵀ = M·Mᵀ where M = R·S
    mat3 S = mat3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z
    );
    mat3 M = R * S;
    mat3 cov3d = M * transpose(M);

    // 5. Transform center to view space
    vec4 p_view4 = view * vec4(position, 1.0);
    vec3 t = p_view4.xyz;

    // 6. Compute focal lengths from projection matrix
    // Note: proj[1][1] is negative for Vulkan Y-flip
    float fx = proj[0][0] * viewport.x * 0.5;
    float fy = abs(proj[1][1]) * viewport.y * 0.5;

    // 7. Jacobian of perspective projection at t
    float J00 = fx / t.z;
    float J11 = fy / t.z;
    float J02 = -fx * t.x / (t.z * t.z);
    float J12 = -fy * t.y / (t.z * t.z);

    // 8. View rotation (upper-left 3x3 of view matrix)
    mat3 W = mat3(view);

    // 9. Project 3D covariance to 2D: Σ' = J · W · Σ · Wᵀ · Jᵀ
    // First compute W · Σ · Wᵀ
    mat3 WcovW = W * cov3d * transpose(W);

    // Then apply J (2x3 matrix) to get 2x2 covariance
    // J = | J00   0   J02 |
    //     |  0   J11  J12 |
    float a = J00*J00*WcovW[0][0] + 2.0*J00*J02*WcovW[0][2] + J02*J02*WcovW[2][2];
    float b = J00*J11*WcovW[0][1] + J00*J12*WcovW[0][2] + J02*J11*WcovW[1][2] + J02*J12*WcovW[2][2];
    float c = J11*J11*WcovW[1][1] + 2.0*J11*J12*WcovW[1][2] + J12*J12*WcovW[2][2];

    // 10. Add low-pass filter
    a += 0.3;
    c += 0.3;

    // 11. Compute inverse (conic) for fragment shader
    float det = a * c - b * b;
    if (det < 1e-6) det = 1e-6;  // Guard against degenerate case
    vec3 conic = vec3(c / det, -b / det, a / det);

    // 12. Compute screen-space center
    // In Vulkan/SDL_GPU, Y=0 is at top, so we need to flip Y
    vec2 center_px = vec2(
        fx * t.x / t.z + viewport.x * 0.5,
        viewport.y * 0.5 - fy * t.y / t.z  // Flip Y for screen coords
    );

    // 13. Compute quad radius (3 sigma)
    float radius_x = ceil(3.0 * sqrt(a));
    float radius_y = ceil(3.0 * sqrt(c));

    // 14. Position this vertex
    vec2 pos_px = center_px + corner * vec2(radius_x, radius_y);
    
    // Convert pixel position to NDC
    // In SDL_GPU/Vulkan: NDC Y goes from -1 (bottom) to +1 (top)
    // But we computed pos_px with Y=0 at top, so flip again
    vec2 ndc = vec2(
        2.0 * pos_px.x / viewport.x - 1.0,
        1.0 - 2.0 * pos_px.y / viewport.y  // Flip Y for NDC
    );
    gl_Position = vec4(ndc, 0.0, 1.0);

    // 15. Pass outputs to fragment shader
    frag_color = color;
    frag_opacity = opacity;
    frag_center = center_px;
    frag_conic = conic;
}
