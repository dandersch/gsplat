#pragma once
#include <SDL3/SDL.h>
#include "gaussian.h"
#include "camera.h"

// Number of frames that can be in flight simultaneously
#define MAX_FRAMES_IN_FLIGHT 3

struct OverlayParams {
    SDL_GPUTexture* texture;       // panorama texture to render (NULL = no overlay)
    float           camera_ray_basis[16];
    float           camera_tan_half_fov[2];
    float           ref_rotation[16]; // mat4, 3x3 in upper-left, Y-flip baked in
    float           alpha;
};

struct NodeRenderParams {
    const float* positions;   // float[3] per node (world-space centers)
    uint32_t     count;
    float        half_size;   // AABB half-extent for the wireframe cubes
};

struct Renderer {
    SDL_GPUDevice*          device;
    SDL_Window*             window;
    SDL_GPUGraphicsPipeline* splat_pipeline;
    SDL_GPUGraphicsPipeline* overlay_pipeline;
    SDL_GPUGraphicsPipeline* wireframe_pipeline;
    SDL_GPUGraphicsPipeline* mesh_pipeline;
    SDL_GPUSampler*         overlay_sampler;
    SDL_GPUBuffer*          gaussian_buffer;
    SDL_GPUBuffer*          cube_vertex_buffer;
    SDL_GPUBuffer*          cube_index_buffer;
    SDL_GPUBuffer*          mesh_vertex_buffer;
    SDL_GPUBuffer*          mesh_index_buffer;
    SDL_GPUBuffer*          index_buffer;
    SDL_GPUTexture*         depth_texture;
    uint32_t                depth_w, depth_h;
    SDL_GPUTransferBuffer*  transfer_bufs[MAX_FRAMES_IN_FLIGHT];
    SDL_GPUFence*           frame_fences[MAX_FRAMES_IN_FLIGHT];
    uint32_t                current_frame;
    SDL_GPUTextureFormat    swapchain_format;
    uint32_t                gaussian_count;
};

bool renderer_init(Renderer* r, SDL_GPUDevice* device, SDL_Window* window);
void renderer_upload_gaussians(Renderer* r, const GaussianScene* scene);
void renderer_draw_frame(Renderer* r, const GaussianScene* scene, const CameraUniforms* cam, const OverlayParams* overlay, const NodeRenderParams* nodes, float wireframe_occlusion = 1.0f);
void renderer_destroy(Renderer* r);
