# Gaussian Splat Renderer — Technical Design Spec

## 1. Overview

A real-time 3D Gaussian splatting renderer targeting 60 fps on small scenes (~100k
Gaussians). Loads standard `.ply` files from the 3DGS reference implementation.
Renders via per-Gaussian screen-space quads with alpha blending in back-to-front
order. CPU radix sort, designed for future GPU sort drop-in.

**Language:** C-style C++ (structs, free functions, compiled with a C++ compiler).  
**Framework:** SDL3 + SDL_GPU for rendering.  
**UI:** Dear ImGui (SDL3 + SDL_GPU backend).  
**Shaders:** GLSL → SPIR-V (via `glslc`).  
**Build:** `build.sh`.

---

## 2. File Organization

```
gsplat/
├── build.sh
├── spec.md
├── src/
│   ├── main.cpp          # Init, main loop, shutdown, ImGui setup
│   ├── gaussian.h        # Gaussian struct, PLY loader, sort interface
│   ├── gaussian.cpp       
│   ├── renderer.h        # SDL_GPU pipeline, per-frame draw
│   ├── renderer.cpp
│   ├── camera.h          # Fly camera
│   └── camera.cpp
└── shaders/
    ├── splat.vert.glsl
    └── splat.frag.glsl
```

ImGui source files (`imgui.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`,
`imgui_widgets.cpp`, backend files `imgui_impl_sdl3.cpp`,
`imgui_impl_sdlgpu3.cpp`) are assumed to be available in a `third_party/imgui/`
directory and compiled directly into the executable.

---

## 3. Data Structures

### 3.1 CPU-Side Gaussian Storage

```cpp
struct Gaussian {
    float position[3];     // World-space center (x, y, z)
    float scale[3];        // Axis scales (exp() already applied at load time)
    float rotation[4];     // Unit quaternion (w, x, y, z)
    float color[3];        // RGB in [0, 1] (SH DC decoded at load time)
    float opacity;         // Opacity in [0, 1] (sigmoid() applied at load time)
};
```

The CPU keeps a flat array `Gaussian* gaussians` of length `gaussian_count`,
allocated once during PLY load. This array is **never reordered**; sorting
operates on an index array.

Auxiliary per-frame scratch buffers (allocated once, reused each frame):

```cpp
uint32_t* visible_indices;   // Indices of Gaussians that pass frustum cull
float*    visible_depths;    // Corresponding depths (for sort input)
uint32_t  visible_count;     // Number of visible Gaussians this frame
uint32_t* sorted_indices;    // Output of radix sort (back-to-front order)
```

All scratch buffers are sized to `gaussian_count` (worst case: everything visible).

### 3.2 GPU Buffer Layout

**Storage buffer 0 — Gaussian data (static, uploaded once):**

```glsl
struct GpuGaussian {
    vec4 pos_opacity;   // (x, y, z, opacity)
    vec4 scale_pad;     // (sx, sy, sz, 0)
    vec4 rotation;      // (w, x, y, z)
    vec4 color_pad;     // (r, g, b, 0)
};
```

Packed to 64 bytes per Gaussian (4 × `vec4`). Padding fields exist for
`std430` alignment. At 100k Gaussians this buffer is ~6.1 MB.

**Storage buffer 1 — Sorted indices (dynamic, re-uploaded each frame):**

A flat `uint32_t` array of length `visible_count`. ~400 KB worst case at 100k.
Updated via `SDL_UploadToGPUBuffer` each frame using a transfer buffer.

**Uniform data — Camera (per-frame, push uniform):**

```cpp
struct CameraUniforms {
    float view[16];          // mat4 view matrix (column-major)
    float proj[16];          // mat4 projection matrix (column-major)
    float viewport[2];       // (width, height) in pixels
    float pad[2];            // Alignment to 16 bytes
};
```

Pushed via `SDL_PushGPUVertexUniformData` each frame.

---

## 4. PLY Parsing

### 4.1 Strategy

Write a minimal, purpose-built PLY parser. Only support the binary little-endian
format (which is what the 3DGS reference implementation outputs). ASCII PLY
support is out of scope but trivial to add later.

### 4.2 Expected Properties

The parser reads the PLY header to discover property names and offsets. Required
properties (names from the reference implementation):

| Property       | PLY type | Decode             |
|----------------|----------|--------------------|
| `x, y, z`      | float    | Direct copy        |
| `scale_0/1/2`  | float    | Apply `exp()`      |
| `rot_0/1/2/3`  | float    | Normalize quaternion; order is (w,x,y,z) |
| `opacity`       | float    | Apply `sigmoid(x) = 1 / (1 + exp(-x))` |
| `f_dc_0/1/2`   | float    | `color = 0.2820947917738781 * f_dc + 0.5` |

Higher-degree SH properties (`f_rest_*`) are recognized but ignored.

### 4.3 Parsing Flow

1. Open file, read header line-by-line to extract `element vertex <count>`,
   property names, types, and byte offsets. Confirm `format binary_little_endian 1.0`.
2. Allocate `Gaussian` array of `vertex_count` elements.
3. Seek past `end_header\n`. Read vertex data in bulk (`vertex_count * stride`
   bytes, where `stride` is the sum of all property sizes).
4. Iterate over raw vertex data, extract fields by computed byte offset, apply
   decode transforms, write to `Gaussian` array.
5. Simultaneously upload raw (decoded) data to the static GPU storage buffer in
   `GpuGaussian` layout.

### 4.4 SH Decode (Degree 0)

The SH basis function for degree 0 (DC component) has the constant:

```
SH_C0 = 0.2820947917738781   (= 1 / (2√π))
```

Conversion from SH DC coefficient to linear RGB:

```
R = SH_C0 * f_dc_0 + 0.5
G = SH_C0 * f_dc_1 + 0.5
B = SH_C0 * f_dc_2 + 0.5
```

Clamp each channel to `[0, 1]`.

**Future SH upgrade path:** Store raw SH coefficients (all degrees) in a
separate CPU array and a second GPU storage buffer. The fragment shader would
evaluate the full SH function using view direction per-fragment. The data
structures and rendering pipeline need no restructuring — only the GPU buffer
contents and fragment shader change.

---

## 5. Projection Math

This section derives the 3D Gaussian → 2D screen ellipse projection, following
the EWA splatting formulation used by Kerbl et al. (2023), *3D Gaussian Splatting
for Real-Time Radiance Field Rendering*.

### 5.1 Constructing the 3D Covariance Matrix

A 3D Gaussian is parameterized by center **μ** ∈ ℝ³, rotation quaternion
**q** = (w, x, y, z), and scale **s** = (s₀, s₁, s₂).

Build rotation matrix **R** from quaternion (standard formula):

```
        ⎡ 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  ⎤
    R = ⎢  2(xy+wz)   1-2(x²+z²)   2(yz-wx)  ⎥
        ⎣  2(xz-wy)    2(yz+wx)   1-2(x²+y²) ⎦
```

Build scale matrix **S** = diag(s₀, s₁, s₂). Then:

```
    M = R · S          (3×3)
    Σ = M · Mᵀ         (3×3 symmetric positive semi-definite)
```

This is computed in the **vertex shader** per Gaussian instance.

### 5.2 Projecting to 2D: The Jacobian Approximation

Transform the Gaussian center to camera (view) space:

```
    t = (W · μ) + t_cam     where W = 3×3 rotation of the view matrix
                                   t_cam = translation component
    (equivalently: t = (V · [μ, 1])_xyz  where V = 4×4 view matrix)
```

Let `t = (tₓ, tᵧ, t_z)` be the Gaussian center in camera space. The
perspective projection maps a camera-space point to pixel coordinates:

```
    π(x, y, z) = ( fₓ · x/z + cₓ ,  fᵧ · y/z + cᵧ )
```

where `fₓ`, `fᵧ` are focal lengths in pixels, `cₓ`, `cᵧ` is the principal
point (viewport center).

The **Jacobian** of π evaluated at t is:

```
        ⎡ fₓ/t_z      0       -fₓ·tₓ/t_z²  ⎤
    J = ⎢                                     ⎥       (2×3)
        ⎣    0      fᵧ/t_z    -fᵧ·tᵧ/t_z²  ⎦
```

This linear approximation is valid when the Gaussian is small relative to its
distance from the camera (which holds for all but extreme close-up cases; those
Gaussians are culled by the near-plane threshold).

### 5.3 Computing 2D Covariance and Conic

Combine the view rotation and projection Jacobian:

```
    T = J · W          (2×3)
```

Project the 3D covariance to a 2D covariance in pixel space:

```
    Σ' = T · Σ · Tᵀ    (2×2 symmetric)
```

Add a low-pass filter for numerical stability (prevents sub-pixel Gaussians from
aliasing). Following the original implementation:

```
    Σ'[0][0] += 0.3
    Σ'[1][1] += 0.3
```

Write the 2D covariance as:

```
         ⎡ a  b ⎤
    Σ' = ⎢      ⎥
         ⎣ b  c ⎦
```

Compute the **conic** (inverse of Σ'), used by the fragment shader to evaluate
the Gaussian:

```
    det = a·c - b²
    conic = ( c/det,  -b/det,  a/det )    →  stored as vec3
```

If `det ≤ 0` (degenerate Gaussian), skip this Gaussian (emit degenerate quad
with zero area).

### 5.4 Quad Extent (Bounding the Ellipse)

The quad must cover the visible extent of the projected Gaussian. Use the **3σ
axis-aligned bounding box** of the 2D Gaussian, derived from the marginal
standard deviations:

```
    radius_x = ceil(3.0 * sqrt(a))     (a = Σ'[0][0])
    radius_y = ceil(3.0 * sqrt(c))     (c = Σ'[1][1])
```

This captures ~99.7% of the Gaussian's energy. The quad spans
`[center - radius, center + radius]` in pixel coordinates.

### 5.5 Fragment Evaluation

Given a fragment at pixel position **p** and the Gaussian center in screen space
**μ'**:

```
    d = p - μ'
    power = -0.5 · (conic.x · d.x² + 2.0 · conic.y · d.x · d.y + conic.z · d.y²)
```

This is the exponent of the 2D Gaussian: `G(d) = exp(power)`.

```
    if power > 0:   discard    (numerical guard)
    alpha = min(opacity · exp(power), 0.99)
    if alpha < 1.0/255.0:   discard    (invisible fragment)
```

The `0.99` clamp prevents fully opaque splats from blocking the background in
unexpected ways.

---

## 6. Per-Frame Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                        FRAME START                             │
├─────────────┬──────────────────────────────────────────────────┤
│  Stage 1    │  Cull + Depth (CPU)                              │
│  Stage 2    │  Sort (CPU radix sort)                           │
│  Stage 3    │  Upload sorted indices to GPU                    │
│  Stage 4    │  Render splats (instanced draw)                  │
│  Stage 5    │  Render ImGui overlay                            │
│  Stage 6    │  Submit + Present                                │
├─────────────┴──────────────────────────────────────────────────┤
│                        FRAME END                               │
└────────────────────────────────────────────────────────────────┘
```

### Stage 1: Cull + Depth (CPU)

**Input:** `gaussians[]`, camera position, view matrix, projection parameters.  
**Output:** `visible_indices[]`, `visible_depths[]`, `visible_count`.

For each Gaussian `i`:

1. Transform center to view space: `p_view = V · position[i]`.
2. **Near-plane cull:** If `p_view.z > -0.2` (too close or behind camera),
   skip.  
   Note: using the convention where the camera looks along **-Z** in view
   space; visible objects have `p_view.z < 0`.
3. **Frustum cull:** Test whether the Gaussian center (with conservative margin)
   falls within the view frustum. A fast test: project `p_view` to NDC and
   check `|ndc.x| < 1.3` and `|ndc.y| < 1.3` (margin accounts for Gaussian
   extent).
4. If visible, append `i` to `visible_indices[]` and store
   `depth = -p_view.z` (positive, larger = farther) in `visible_depths[]`.
   Increment `visible_count`.

This loop is the hot path for CPU work. Consider SIMD or multithreading as a
later optimization; at 100k it should be fine single-threaded.

### Stage 2: Sort (CPU Radix Sort)

**Input:** `visible_indices[]`, `visible_depths[]`, `visible_count`.  
**Output:** `sorted_indices[]` — indices in **back-to-front** order (descending
depth: farthest first).

**Interface boundary (sort abstraction):**

```cpp
struct SortContext {
    // Input
    const float*    depths;          // Depth values to sort by
    const uint32_t* input_indices;   // Corresponding Gaussian indices
    uint32_t        count;           // Number of elements

    // Output (caller-allocated)
    uint32_t*       sorted_indices;  // Result: indices in back-to-front order

    // Scratch (caller-allocated, reused across frames)
    uint32_t*       scratch_indices; // Temp buffer, same size as count
    uint32_t*       scratch_keys;    // Temp buffer for radix sort keys
};

void sort_gaussians(SortContext* ctx);
```

**Radix sort implementation:**

1. **Float-to-sortable-uint32 conversion:** Reinterpret float depth bits as
   `uint32_t`. If the sign bit is set, flip all bits; otherwise flip only the
   sign bit. This produces a monotonically increasing integer encoding of
   IEEE 754 floats.
2. **8-bit radix sort, 4 passes** (LSB to MSB), operating on the converted
   keys. Each pass does a counting sort with 256 buckets. The index array
   is permuted alongside the keys.
3. The result is ascending depth order. **Reverse** the output array (or
   iterate backwards when copying to `sorted_indices`) to get back-to-front
   (descending) order.

At 100k elements, this completes in well under 1 ms on modern CPUs.

**Future GPU sort upgrade path:** Replace `sort_gaussians()` with a function
that dispatches a GPU radix sort compute shader. The compute shader writes
directly to storage buffer 1 (sorted indices), eliminating the CPU→GPU upload
in Stage 3. The rest of the pipeline is unchanged. The only interface contract
is: "after sort, GPU storage buffer 1 contains `visible_count` uint32 indices
in back-to-front order."

### Stage 3: Upload Sorted Indices to GPU

**Input:** `sorted_indices[]`, `visible_count`.  
**Output:** GPU storage buffer 1 updated.

```cpp
// Map transfer buffer, memcpy sorted_indices, upload to GPU buffer
void* map = SDL_MapGPUTransferBuffer(device, transfer_buf, false);
memcpy(map, sorted_indices, visible_count * sizeof(uint32_t));
SDL_UnmapGPUTransferBuffer(device, transfer_buf);

// In a copy pass:
SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
SDL_UploadToGPUBuffer(copy, &src_region, &dst_region, false);
SDL_EndGPUCopyPass(copy);
```

### Stage 4: Render Splats (Instanced Draw)

**Input:** GPU storage buffers (Gaussian data + sorted indices), camera
uniforms, visible_count.  
**Output:** Splats rendered to the swapchain texture.

1. Begin render pass with a color target (swapchain texture), clear to
   background color (e.g., black or dark gray).
2. Bind the splat graphics pipeline.
3. Bind storage buffers 0 (Gaussian data) and 1 (sorted indices) to the
   vertex shader.
4. Push camera uniforms via `SDL_PushGPUVertexUniformData`.
5. Draw: `SDL_DrawGPUPrimitives(render_pass, 6, visible_count, 0, 0)`.  
   6 vertices per instance (2 triangles, no index buffer needed), one instance
   per visible Gaussian.
6. **Do not** end the render pass yet — ImGui renders in the same pass.

### Stage 5: Render ImGui Overlay

1. Call ImGui render functions (`ImGui_ImplSDLGPU3_RenderDrawData` or
   equivalent) within the same render pass.
2. End the render pass.

### Stage 6: Submit + Present

```cpp
SDL_SubmitGPUCommandBuffer(cmd);
```

SDL_GPU handles presentation internally when you acquired a swapchain texture.

---

## 7. Shader Specifications

### 7.1 Vertex Shader (`splat.vert.glsl`)

```glsl
#version 450

// Gaussian data (static)
layout(std430, set = 1, binding = 0) readonly buffer GaussianBuffer {
    // Each Gaussian: 4 x vec4 = 64 bytes
    //   [0]: (pos.x, pos.y, pos.z, opacity)
    //   [1]: (scale.x, scale.y, scale.z, pad)
    //   [2]: (rot.w, rot.x, rot.y, rot.z)
    //   [3]: (color.r, color.g, color.b, pad)
    vec4 data[];
} gaussians;

// Sorted index buffer (per-frame)
layout(std430, set = 1, binding = 1) readonly buffer IndexBuffer {
    uint sorted_indices[];
};

// Camera uniforms
layout(set = 2, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec2 viewport;       // (width, height) in pixels
};

// Outputs to fragment shader
layout(location = 0) out vec3 frag_color;
layout(location = 1) out float frag_opacity;
layout(location = 2) out vec2 frag_center;     // Gaussian center in pixels
layout(location = 3) out vec3 frag_conic;      // Inverse 2D covariance

void main() {
    // 1. Determine quad corner from vertex ID
    //    6 vertices per quad: two triangles (0,1,2) and (0,2,3)
    int quad_verts[6] = int[6](0, 1, 2, 0, 2, 3);
    vec2 corners[4] = vec2[4](
        vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1)
    );
    vec2 corner = corners[quad_verts[gl_VertexIndex % 6]];

    // 2. Fetch Gaussian data
    uint idx = sorted_indices[gl_InstanceIndex];
    vec3 position = gaussians.data[idx * 4 + 0].xyz;
    float opacity = gaussians.data[idx * 4 + 0].w;
    vec3 scale    = gaussians.data[idx * 4 + 1].xyz;
    vec4 rot      = gaussians.data[idx * 4 + 2];    // (w, x, y, z)
    vec3 color    = gaussians.data[idx * 4 + 3].xyz;

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
    float fx = proj[0][0] * viewport.x * 0.5;
    float fy = proj[1][1] * viewport.y * 0.5;

    // 7. Jacobian of perspective projection at t
    mat3x2 J = mat3x2(
        fx / t.z,  0.0,
        0.0,       fy / t.z,
        -fx * t.x / (t.z * t.z), -fy * t.y / (t.z * t.z)
    );

    // 8. View rotation (upper-left 3x3 of view matrix)
    mat3 W = mat3(view);

    // 9. Project 3D covariance to 2D
    //    T = J · W  (2x3),  Σ' = T · Σ · Tᵀ  (2x2)
    mat3x2 T = J * W;     // Note: mat3x2 * mat3 requires care with GLSL
                           // T is 2 rows × 3 cols
    // Actually compute as explicit 2x2:
    // Σ' = T · cov3d · Tᵀ
    // For GLSL, expand:
    //   tmp = T * cov3d   (2x3)
    //   cov2d = tmp * transpose(T)  (2x2)
    // GLSL doesn't have mat2x3 * mat3x2 directly, so compute element-wise.

    // ... (see implementation note below)

    // 10. Add low-pass filter
    // cov2d[0][0] += 0.3;  cov2d[1][1] += 0.3;

    // 11. Compute conic
    // float det = a*c - b*b;
    // vec3 conic = vec3(c/det, -b/det, a/det);

    // 12. Compute screen-space center
    // vec2 center_px = vec2(fx * t.x / t.z + viewport.x*0.5,
    //                       fy * t.y / t.z + viewport.y*0.5);

    // 13. Compute quad radius
    // float radius_x = ceil(3.0 * sqrt(a));
    // float radius_y = ceil(3.0 * sqrt(c));

    // 14. Position this vertex
    // vec2 pos_px = center_px + corner * vec2(radius_x, radius_y);
    // Convert pixel position to NDC:
    // gl_Position = vec4(2.0 * pos_px / viewport - 1.0, 0.0, 1.0);

    // 15. Pass outputs to fragment shader
    // frag_color = color;
    // frag_opacity = opacity;
    // frag_center = center_px;
    // frag_conic = conic;
}
```

**Implementation note on the 2D covariance computation:** GLSL matrix
multiplication for non-square matrices can be awkward. In practice, compute the
2×2 result with explicit dot products:

```glsl
// T is stored as two vec3 rows: T_row0, T_row1
vec3 T0 = vec3(fx/t.z, 0.0, -fx*t.x/(t.z*t.z)) * W;  // conceptually
vec3 T1 = vec3(0.0, fy/t.z, -fy*t.y/(t.z*t.z)) * W;

// But actually: T_row_i = J_row_i · W
// where J_row0 = (fx/tz, 0, -fx*tx/tz²) and J_row1 = (0, fy/tz, -fy*ty/tz²)
// T_row_i[j] = dot(J_row_i, W_col_j)

// Σ' elements:
// a = dot(T0, cov3d * T0)
// b = dot(T0, cov3d * T1)
// c = dot(T1, cov3d * T1)
```

### 7.2 Fragment Shader (`splat.frag.glsl`)

```glsl
#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in float frag_opacity;
layout(location = 2) in vec2 frag_center;     // Gaussian center in pixels
layout(location = 3) in vec3 frag_conic;      // (c/det, -b/det, a/det)

layout(location = 0) out vec4 out_color;

void main() {
    vec2 d = gl_FragCoord.xy - frag_center;

    float power = -0.5 * (
        frag_conic.x * d.x * d.x +
        2.0 * frag_conic.y * d.x * d.y +
        frag_conic.z * d.y * d.y
    );

    // Numerical guard: power should always be ≤ 0
    if (power > 0.0) discard;

    float alpha = min(frag_opacity * exp(power), 0.99);

    // Skip near-invisible fragments
    if (alpha < 1.0 / 255.0) discard;

    out_color = vec4(frag_color * alpha, alpha);
}
```

**Blend state** (configured on the pipeline, not in the shader):

```
color_blend:  src_factor = ONE,  dst_factor = ONE_MINUS_SRC_ALPHA
alpha_blend:  src_factor = ONE,  dst_factor = ONE_MINUS_SRC_ALPHA
```

Note: the fragment shader pre-multiplies color by alpha (`color * alpha`), and
the blend equation uses `ONE` for the source factor. This is **premultiplied
alpha** blending, which is the correct formulation for back-to-front Gaussian
compositing and is equivalent to the standard over operator:

```
C_dst = α_src · C_src + (1 - α_src) · C_dst
```

---

## 8. Camera Controller

### 8.1 State

```cpp
struct Camera {
    float position[3];
    float yaw;               // Radians, rotation around world Y axis
    float pitch;             // Radians, clamped to (-π/2, π/2)
    float fov_y;             // Vertical field of view in radians (default: 60°)
    float near_plane;        // Default: 0.1
    float far_plane;         // Default: 100.0
    float move_speed;        // Units/second (default: 2.0)
    float look_sensitivity;  // Radians/pixel (default: 0.003)
};
```

### 8.2 Forward / Right / Up Vectors

Derived each frame from `yaw` and `pitch`:

```
forward = (cos(pitch)*sin(yaw),  sin(pitch),  cos(pitch)*cos(yaw))
right   = normalize(cross(forward, world_up))     // world_up = (0,1,0)
up      = cross(right, forward)
```

Note: the sign conventions and axis assignments here assume a right-handed
coordinate system with Y-up, matching the typical 3DGS convention. Adjust
if the loaded `.ply` data uses a different convention (some use Z-up; detect
and convert at load time if needed).

### 8.3 Input Handling

- **Right mouse button held:** Capture mouse, apply relative motion to
  `yaw`/`pitch`. While held, ImGui does not receive mouse events.
- **WASD:** Move along `forward`/`right` vectors, scaled by `delta_time * move_speed`.
- **Space / Left Ctrl:** Move along world Y (up/down).
- **Left Shift:** Multiply `move_speed` by 3× while held.
- **Scroll wheel:** Adjust `move_speed` (optional, nice for exploring).

### 8.4 Matrix Construction

```
view_matrix = lookAt(position, position + forward, up)
proj_matrix = perspective(fov_y, aspect_ratio, near_plane, far_plane)
```

Use standard formulas or a small inline math helper (no library dependency).
Matrices are column-major for GLSL compatibility.

---

## 9. SDL_GPU Resources

### 9.1 Initialization

```cpp
SDL_GPUDevice* device = SDL_CreateGPUDevice(
    SDL_GPU_SHADERFORMAT_SPIRV,     // preferred format
    true,                            // debug mode
    NULL                             // no specific backend preference
);
SDL_Window* window = SDL_CreateWindow("gsplat", 1280, 720, SDL_WINDOW_RESIZABLE);
SDL_ClaimWindowForGPUDevice(device, window);
```

### 9.2 Shader Objects

Load SPIR-V bytecode from files compiled at build time:

```cpp
SDL_GPUShader* vert_shader = SDL_CreateGPUShader(device, &(SDL_GPUShaderCreateInfo){
    .code = vert_spirv_data,
    .code_size = vert_spirv_size,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_VERTEX,
    .num_storage_buffers = 2,      // Gaussian data + sorted indices
    .num_uniform_buffers = 1,      // Camera uniforms
});

SDL_GPUShader* frag_shader = SDL_CreateGPUShader(device, &(SDL_GPUShaderCreateInfo){
    .code = frag_spirv_data,
    .code_size = frag_spirv_size,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
    .num_storage_buffers = 0,
    .num_uniform_buffers = 0,
});
```

### 9.3 Graphics Pipeline

```cpp
SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {
    .vertex_shader = vert_shader,
    .fragment_shader = frag_shader,
    .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
    .rasterizer_state = {
        .fill_mode = SDL_GPU_FILLMODE_FILL,
        .cull_mode = SDL_GPU_CULLMODE_NONE,      // Quads are camera-facing
    },
    .depth_stencil_state = {
        .enable_depth_test = false,               // Sorted, no depth test
        .enable_depth_write = false,
    },
    .target_info = {
        .num_color_targets = 1,
        .color_target_descriptions = &(SDL_GPUColorTargetDescription){
            .format = swapchain_format,
            .blend_state = {
                .enable_blend = true,
                .src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
                .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .color_blend_op = SDL_GPU_BLENDOP_ADD,
                .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
                .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
                .color_write_mask = SDL_GPU_COLORCOMPONENT_ALL,
            },
        },
    },
    // No vertex input state — all data comes from storage buffers
    .vertex_input_state = { .num_vertex_buffers = 0, .num_vertex_attributes = 0 },
};

SDL_GPUGraphicsPipeline* splat_pipeline = SDL_CreateGPUGraphicsPipeline(device, &pipeline_info);
```

### 9.4 Buffers

```cpp
// Static Gaussian data (storage buffer, uploaded once)
SDL_GPUBuffer* gaussian_buffer = SDL_CreateGPUBuffer(device, &(SDL_GPUBufferCreateInfo){
    .usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
    .size = gaussian_count * 64,  // 64 bytes per GpuGaussian
});

// Sorted index buffer (storage buffer, re-uploaded each frame)
SDL_GPUBuffer* index_buffer = SDL_CreateGPUBuffer(device, &(SDL_GPUBufferCreateInfo){
    .usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
    .size = gaussian_count * sizeof(uint32_t),
});

// Transfer buffer for uploading index data each frame
SDL_GPUTransferBuffer* transfer_buf = SDL_CreateGPUTransferBuffer(device, &(SDL_GPUTransferBufferCreateInfo){
    .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
    .size = gaussian_count * sizeof(uint32_t),
});
```

### 9.5 Per-Frame Render Pass

```cpp
SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);

// --- Copy pass: upload sorted indices ---
SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
// (upload sorted_indices via transfer buffer)
SDL_EndGPUCopyPass(copy);

// --- Render pass ---
SDL_GPUTexture* swapchain_tex;
SDL_AcquireGPUSwapchainTexture(cmd, window, &swapchain_tex, NULL, NULL);

SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &(SDL_GPUColorTargetInfo){
    .texture = swapchain_tex,
    .load_op = SDL_GPU_LOADOP_CLEAR,
    .store_op = SDL_GPU_STOREOP_STORE,
    .clear_color = { 0.1, 0.1, 0.1, 1.0 },
}, 1, NULL);

// Bind splat pipeline and storage buffers
SDL_BindGPUGraphicsPipeline(pass, splat_pipeline);
SDL_BindGPUVertexStorageBuffers(pass, 0,
    (SDL_GPUBuffer*[]){ gaussian_buffer, index_buffer }, 2);

// Push camera uniforms
SDL_PushGPUVertexUniformData(cmd, 0, &camera_uniforms, sizeof(CameraUniforms));

// Draw all visible Gaussians
SDL_DrawGPUPrimitives(pass, 6, visible_count, 0, 0);

// Render ImGui in the same pass
ImGui_ImplSDLGPU3_RenderDrawData(ImGui::GetDrawData(), cmd, pass);

SDL_EndGPURenderPass(pass);
SDL_SubmitGPUCommandBuffer(cmd);
```

---

## 10. Sort Abstraction

The sort is isolated behind a single function with a plain data interface:

```cpp
// gaussian.h

struct SortContext {
    const float*    depths;
    const uint32_t* input_indices;
    uint32_t        count;
    uint32_t*       sorted_indices;    // Output

    // Scratch space (allocated by caller, reused across frames)
    uint32_t*       scratch_a;         // size: count
    uint32_t*       scratch_b;         // size: count
};

// CPU radix sort, back-to-front order. Fully self-contained.
void sort_gaussians(SortContext* ctx);
```

**Swap contract:** Any replacement sort implementation must write `count`
indices into `sorted_indices` such that `depths[sorted_indices[0]] >=
depths[sorted_indices[1]] >= ...` (descending depth = back-to-front).

For a GPU sort upgrade, the replacement would:
1. Upload `depths[]` and `input_indices[]` to GPU buffers.
2. Dispatch a compute shader that performs radix sort and writes results
   directly to the GPU index storage buffer.
3. Skip the Stage 3 upload entirely (data is already on GPU).

The caller code in the main loop would branch on a sort mode flag:

```cpp
if (gpu_sort_enabled) {
    gpu_sort_dispatch(cmd, ...);
} else {
    sort_gaussians(&sort_ctx);
    upload_indices_to_gpu(...);
}
```

---

## 11. ImGui Integration

### 11.1 Files Required

From the Dear ImGui repository:
- Core: `imgui.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp`
- Platform backend: `imgui_impl_sdl3.cpp`
- Renderer backend: `imgui_impl_sdlgpu3.cpp` (from SDL_gpu_examples or community)

### 11.2 Initialization

```cpp
IMGUI_CHECKVERSION();
ImGui::CreateContext();
ImGui_ImplSDL3_InitForOther(window);
ImGui_ImplSDLGPU3_Init(device, swapchain_format);   // Backend-specific
```

### 11.3 Per-Frame Flow

```cpp
ImGui_ImplSDLGPU3_NewFrame();
ImGui_ImplSDL3_NewFrame();
ImGui::NewFrame();

// --- Application UI ---
ImGui::Begin("Info");
ImGui::Text("FPS: %.1f", 1.0f / delta_time);
ImGui::Text("Visible: %u / %u", visible_count, gaussian_count);
ImGui::Text("Camera: %.1f, %.1f, %.1f", cam.position[0], cam.position[1], cam.position[2]);
ImGui::End();

ImGui::Render();
// (draw data consumed in Stage 5 of the render pass)
```

### 11.4 Input Routing

SDL events are dispatched to both ImGui and the camera controller. When ImGui
wants mouse input (`ImGui::GetIO().WantCaptureMouse`), the camera controller
ignores mouse events. When right mouse button is held for camera look, ImGui
mouse input is suppressed.

---

## 12. Build System

`build.sh`:

```bash
#!/bin/bash
set -e

# --- Config ---
CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++17 -Wall -Wextra"
LDFLAGS="-lSDL3 -lm"
OUT="gsplat"

IMGUI_DIR="third_party/imgui"

# --- Compile shaders ---
echo "Compiling shaders..."
glslc shaders/splat.vert.glsl -o shaders/splat.vert.spv
glslc shaders/splat.frag.glsl -o shaders/splat.frag.spv

# --- Source files ---
SRCS=(
    src/main.cpp
    src/gaussian.cpp
    src/renderer.cpp
    src/camera.cpp
    "$IMGUI_DIR/imgui.cpp"
    "$IMGUI_DIR/imgui_draw.cpp"
    "$IMGUI_DIR/imgui_tables.cpp"
    "$IMGUI_DIR/imgui_widgets.cpp"
    "$IMGUI_DIR/backends/imgui_impl_sdl3.cpp"
    "$IMGUI_DIR/backends/imgui_impl_sdlgpu3.cpp"
)

# --- Compile & link ---
echo "Building $OUT..."
$CXX $CXXFLAGS -I"$IMGUI_DIR" -I"$IMGUI_DIR/backends" "${SRCS[@]}" -o "$OUT" $LDFLAGS

echo "Done: ./$OUT"
```

---

## 13. Future Upgrade Paths

### GPU Compute Sort
Replace the CPU radix sort with a GPU compute shader (e.g., Onesweep or
device-level radix sort). The sort writes directly to the GPU index buffer,
eliminating the per-frame CPU→GPU transfer. The rendering pipeline is unchanged.

### SH Degrees 1–3 (View-Dependent Color)
- Add a second GPU storage buffer containing full SH coefficients (48 floats
  per Gaussian for degree 3).
- Pass view direction to the fragment shader.
- Evaluate full SH in the fragment shader:
  ```
  color = SH_C0 * sh[0]
        + SH_C1 * (y*sh[1] + z*sh[2] + x*sh[3])
        + ...  (degree 2 and 3 terms)
  ```
- The vertex shader, sort pipeline, and CPU pipeline are unchanged.

### WebGPU / Web Export
Blocked on SDL_GPU gaining a WebGPU backend. When available:
- SPIR-V → WGSL cross-compilation via Naga/Tint.
- Emscripten build of the C++ code.
- No architectural changes required.

### Performance Scaling (>1M Gaussians)
- GPU compute sort (above).
- Tile-based rasterization to reduce overdraw.
- Level-of-detail culling (skip tiny Gaussians).
- Multithreaded CPU culling with work-stealing.
