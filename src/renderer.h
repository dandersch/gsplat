#pragma once
#include <SDL3/SDL.h>
#include "gaussian.h"
#include "camera.h"

// Number of frames that can be in flight simultaneously
#define MAX_FRAMES_IN_FLIGHT 3

struct OverlayParams {
    SDL_GPUTexture* texture;       // panorama texture to render (NULL = no overlay)
    float           inv_view_proj[16];
    float           ref_rotation[16]; // mat4, 3x3 in upper-left, Y-flip baked in
    float           alpha;
};

struct Renderer {
    SDL_GPUDevice*          device;
    SDL_Window*             window;
    SDL_GPUGraphicsPipeline* splat_pipeline;
    SDL_GPUGraphicsPipeline* overlay_pipeline;
    SDL_GPUSampler*         overlay_sampler;
    SDL_GPUBuffer*          gaussian_buffer;
    SDL_GPUBuffer*          index_buffer;
    SDL_GPUTransferBuffer*  transfer_bufs[MAX_FRAMES_IN_FLIGHT];
    SDL_GPUFence*           frame_fences[MAX_FRAMES_IN_FLIGHT];
    uint32_t                current_frame;
    SDL_GPUTextureFormat    swapchain_format;
    uint32_t                gaussian_count;
};

bool renderer_init(Renderer* r, SDL_GPUDevice* device, SDL_Window* window);
void renderer_upload_gaussians(Renderer* r, const GaussianScene* scene);
void renderer_draw_frame(Renderer* r, const GaussianScene* scene, const CameraUniforms* cam, const OverlayParams* overlay);
void renderer_destroy(Renderer* r);
