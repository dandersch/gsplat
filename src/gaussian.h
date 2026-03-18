#pragma once
#include <cstdint>

struct Gaussian {
    float position[3];
    float scale[3];
    float rotation[4]; // w, x, y, z
    float color[3];
    float opacity;
};

struct GpuGaussian {
    float pos_opacity[4];  // x, y, z, opacity
    float scale_pad[4];    // sx, sy, sz, 0
    float rotation[4];     // w, x, y, z
    float color_pad[4];    // r, g, b, 0
};

struct SortContext {
    const float*    depths;
    const uint32_t* input_indices;
    uint32_t        count;
    uint32_t*       sorted_indices;
    uint32_t*       scratch_indices;
    uint32_t*       scratch_keys;
    uint32_t*       scratch_keys2;
};

struct GaussianScene {
    Gaussian*  gaussians;
    uint32_t   gaussian_count;

    // Per-frame scratch (all sized to gaussian_count)
    uint32_t*  visible_indices;
    float*     visible_depths;
    uint32_t   visible_count;
    uint32_t*  sorted_indices;
    uint32_t*  scratch_indices;
    uint32_t*  scratch_keys;
    uint32_t*  scratch_keys2;
};

// Load PLY, populate scene. Returns false on failure.
bool load_ply(const char* path, GaussianScene* scene);

// Free scene data
void free_scene(GaussianScene* scene);

// Pack gaussians into GpuGaussian layout. Caller frees returned pointer.
GpuGaussian* pack_gpu_gaussians(const GaussianScene* scene);

// Cull and record depth. Fills visible_indices/depths/count.
void cull_gaussians(GaussianScene* scene, const float* view_matrix, const float* proj_matrix);

// Radix sort, back-to-front.
void sort_gaussians(SortContext* ctx);
