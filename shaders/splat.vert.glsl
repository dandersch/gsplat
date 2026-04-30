#version 450

// SDL_GPU SPIR-V layout for vertex shaders:
//   set 0: storage textures, storage buffers  
//   set 1: uniform buffers
layout(std430, set = 0, binding = 0) readonly buffer IndexBuffer {
    uint sorted_indices[];
};

// Each gaussian occupies 64 floats (256 bytes). See GPU layout in gaussian.h.
//   [0..3]   pos.xyz, opacity
//   [4..7]   scale.xyz, pad
//   [8..11]  rotation w,x,y,z
//   [12..15] color.rgb (raw f_dc), pad
//   [16..60] sh_rest: 15 coefficients × RGB triples
//   [61..63] pad
layout(std430, set = 0, binding = 1) readonly buffer GaussianBuffer {
    float data[];
} gaussians;
const uint G_STRIDE = 64u;

layout(set = 1, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec2 viewport;
    float orthographic; // ORTHO: 1.0 = orthographic, 0.0 = perspective
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
    uint idx  = sorted_indices[gl_InstanceIndex];
    uint base = idx * G_STRIDE;
    vec3 position = vec3(gaussians.data[base + 0u], gaussians.data[base + 1u], gaussians.data[base + 2u]);
    float opacity = gaussians.data[base + 3u];
    vec3 scale    = vec3(gaussians.data[base + 4u], gaussians.data[base + 5u], gaussians.data[base + 6u]);
    vec4 rot      = vec4(gaussians.data[base + 8u], gaussians.data[base + 9u],
                         gaussians.data[base + 10u], gaussians.data[base + 11u]); // (w, x, y, z)
    vec3 dc       = vec3(gaussians.data[base + 12u], gaussians.data[base + 13u], gaussians.data[base + 14u]);

    // 3. Build rotation matrix from quaternion
    // rot = (w, x, y, z) stored as rot.x=w, rot.y=x, rot.z=y, rot.w=z
    // GLSL mat3 constructor takes COLUMNS, so we provide column0, column1, column2
    float qw = rot.x, qx = rot.y, qy = rot.z, qz = rot.w;
    mat3 R = mat3(
        // Column 0
        1.0 - 2.0*(qy*qy + qz*qz),
        2.0*(qx*qy + qw*qz),
        2.0*(qx*qz - qw*qy),
        // Column 1
        2.0*(qx*qy - qw*qz),
        1.0 - 2.0*(qx*qx + qz*qz),
        2.0*(qy*qz + qw*qx),
        // Column 2
        2.0*(qx*qz + qw*qy),
        2.0*(qy*qz - qw*qx),
        1.0 - 2.0*(qx*qx + qy*qy)
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

    // 7. Jacobian of the screen-space projection at t
    // Lerp between perspective and orthographic Jacobians
    float J00 = mix(fx / t.z,                     -fx, orthographic);
    float J11 = mix(-fy / t.z,                     fy, orthographic);
    float J02 = mix(-fx * t.x / (t.z * t.z),     0.0, orthographic);
    float J12 = mix( fy * t.y / (t.z * t.z),     0.0, orthographic);

    // 8. View rotation (upper-left 3x3 of view matrix)
    mat3 W = mat3(view);

    // 9. Project 3D covariance to 2D: Σ' = J · W · Σ · Wᵀ · Jᵀ
    // First compute W · Σ · Wᵀ
    mat3 WcovW = W * cov3d * transpose(W);

    // Then apply J (2x3 matrix) to get 2x2 covariance
    // J = | J00   0   J02 |
    //     |  0   J11  J12 |
    // cov2d = J * WcovW * J^T
    // WcovW is symmetric so [i][j] == [j][i]
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

    // 12. Compute screen-space center (lerp between perspective and orthographic)
    vec2 persp_center = vec2(
        fx * t.x / t.z + viewport.x * 0.5,
        viewport.y * 0.5 - fy * t.y / t.z
    );
    vec2 ortho_center = vec2(
        -fx * t.x + viewport.x * 0.5,
        viewport.y * 0.5 + fy * t.y
    );
    vec2 center_px = mix(persp_center, ortho_center, orthographic);

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
    // Compute proper depth from projection matrix (proj[2][2]*z + proj[3][2]) / (-z)
    float ndc_z = (proj[2][2] * t.z + proj[3][2]) / (-t.z);
    gl_Position = vec4(ndc, ndc_z, 1.0);

    // 15. Evaluate spherical harmonics (degree 3) for view-dependent color.
    //     dir = normalized vector from camera to gaussian, in world space.
    //     cam_pos_world = -W^T * view[3].xyz   (W = mat3(view))
    vec3 cam_pos_world = -transpose(W) * view[3].xyz;
    vec3 dir = normalize(position - cam_pos_world);

    const float SH_C0 = 0.28209479177387814;
    const float SH_C1 = 0.4886025119029199;
    const float SH_C2_0 =  1.0925484305920792;
    const float SH_C2_1 = -1.0925484305920792;
    const float SH_C2_2 =  0.31539156525252005;
    const float SH_C2_3 = -1.0925484305920792;
    const float SH_C2_4 =  0.5462742152960396;
    const float SH_C3_0 = -0.5900435899266435;
    const float SH_C3_1 =  2.890611442640554;
    const float SH_C3_2 = -0.4570457994644658;
    const float SH_C3_3 =  0.3731763325901154;
    const float SH_C3_4 = -0.4570457994644658;
    const float SH_C3_5 =  1.445305721320277;
    const float SH_C3_6 = -0.5900435899266435;

    // Helper: fetch the k-th rest coefficient (RGB triple) at sh_rest base.
    // sh_rest occupies floats [16 .. 60] within the gaussian (15 coeffs * 3 ch).
    uint sh_base = base + 16u;
    #define SH(k) vec3( \
        gaussians.data[sh_base + uint(k) * 3u + 0u], \
        gaussians.data[sh_base + uint(k) * 3u + 1u], \
        gaussians.data[sh_base + uint(k) * 3u + 2u])

    vec3 result = SH_C0 * dc;

    float x = dir.x, y = dir.y, z = dir.z;
    // Degree 1
    result += -SH_C1 * y * SH(0);
    result +=  SH_C1 * z * SH(1);
    result += -SH_C1 * x * SH(2);
    // Degree 2
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, yz = y*z, xz = x*z;
    result += SH_C2_0 * xy            * SH(3);
    result += SH_C2_1 * yz            * SH(4);
    result += SH_C2_2 * (2.0*zz - xx - yy) * SH(5);
    result += SH_C2_3 * xz            * SH(6);
    result += SH_C2_4 * (xx - yy)     * SH(7);
    // Degree 3
    result += SH_C3_0 * y * (3.0*xx - yy)               * SH(8);
    result += SH_C3_1 * xy * z                          * SH(9);
    result += SH_C3_2 * y * (4.0*zz - xx - yy)          * SH(10);
    result += SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy)  * SH(11);
    result += SH_C3_4 * x * (4.0*zz - xx - yy)          * SH(12);
    result += SH_C3_5 * z * (xx - yy)                   * SH(13);
    result += SH_C3_6 * x * (xx - 3.0*yy)               * SH(14);

    result += 0.5;
    vec3 color = max(result, vec3(0.0));

    // 16. Pass outputs to fragment shader
    frag_color = color;
    frag_opacity = opacity;
    frag_center = center_px;
    frag_conic = conic;
}
