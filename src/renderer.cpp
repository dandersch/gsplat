#include "renderer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

static uint8_t* load_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(sz);
    fread(data, 1, sz, f);
    fclose(f);
    *out_size = (size_t)sz;
    return data;
}

bool renderer_init(Renderer* r, SDL_GPUDevice* device, SDL_Window* window) {
    r->device = device;
    r->window = window;
    r->gaussian_buffer = NULL;
    r->index_buffer = NULL;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        r->transfer_bufs[i] = NULL;
        r->frame_fences[i] = NULL;
    }
    r->current_frame = 0;
    r->splat_pipeline = NULL;
    r->gaussian_count = 0;

    r->swapchain_format = SDL_GetGPUSwapchainTextureFormat(device, window);
    fprintf(stderr, "Swapchain format: %d\n", (int)r->swapchain_format);

    // Load shaders
    size_t vert_size, frag_size;
    uint8_t* vert_code = load_file("shaders/splat.vert.spv", &vert_size);
    uint8_t* frag_code = load_file("shaders/splat.frag.spv", &frag_size);
    if (!vert_code || !frag_code) return false;

    SDL_GPUShaderCreateInfo vert_info = {};
    vert_info.code = vert_code;
    vert_info.code_size = vert_size;
    vert_info.entrypoint = "main";
    vert_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    vert_info.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    vert_info.num_storage_buffers = 2;
    vert_info.num_uniform_buffers = 1;

    SDL_GPUShader* vert_shader = SDL_CreateGPUShader(device, &vert_info);
    free(vert_code);
    if (!vert_shader) { fprintf(stderr, "FAIL vertex shader: %s\n", SDL_GetError()); free(frag_code); return false; }

    SDL_GPUShaderCreateInfo frag_info = {};
    frag_info.code = frag_code;
    frag_info.code_size = frag_size;
    frag_info.entrypoint = "main";
    frag_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    frag_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    frag_info.num_storage_buffers = 0;
    frag_info.num_uniform_buffers = 0;

    SDL_GPUShader* frag_shader = SDL_CreateGPUShader(device, &frag_info);
    free(frag_code);
    if (!frag_shader) { fprintf(stderr, "FAIL fragment shader: %s\n", SDL_GetError()); SDL_ReleaseGPUShader(device, vert_shader); return false; }

    // Pipeline
    SDL_GPUColorTargetDescription color_target = {};
    color_target.format = r->swapchain_format;
    color_target.blend_state.enable_blend = true;
    color_target.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    color_target.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    color_target.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
    color_target.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    color_target.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    color_target.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B | SDL_GPU_COLORCOMPONENT_A;

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.vertex_shader = vert_shader;
    pipeline_info.fragment_shader = frag_shader;
    pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    pipeline_info.target_info.num_color_targets = 1;
    pipeline_info.target_info.color_target_descriptions = &color_target;

    r->splat_pipeline = SDL_CreateGPUGraphicsPipeline(device, &pipeline_info);
    SDL_ReleaseGPUShader(device, vert_shader);
    SDL_ReleaseGPUShader(device, frag_shader);

    if (!r->splat_pipeline) {
        fprintf(stderr, "FAIL pipeline: %s\n", SDL_GetError());
        return false;
    }

    fprintf(stderr, "Renderer init OK\n");
    return true;
}

void renderer_upload_gaussians(Renderer* r, const GaussianScene* scene) {
    r->gaussian_count = scene->gaussian_count;
    uint32_t buf_size = scene->gaussian_count * 64;

    // Gaussian data buffer (static)
    SDL_GPUBufferCreateInfo buf_info = {};
    buf_info.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
    buf_info.size = buf_size;
    r->gaussian_buffer = SDL_CreateGPUBuffer(r->device, &buf_info);

    // Index buffer (dynamic, updated each frame with sorted indices)
    SDL_GPUBufferCreateInfo idx_info = {};
    idx_info.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
    idx_info.size = scene->gaussian_count * sizeof(uint32_t);
    r->index_buffer = SDL_CreateGPUBuffer(r->device, &idx_info);

    // Transfer buffers for per-frame index upload (one per frame in flight)
    SDL_GPUTransferBufferCreateInfo xfer_info = {};
    xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    xfer_info.size = scene->gaussian_count * sizeof(uint32_t);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        r->transfer_bufs[i] = SDL_CreateGPUTransferBuffer(r->device, &xfer_info);
    }

    // Upload gaussian data
    GpuGaussian* gpu_data = pack_gpu_gaussians(scene);

    SDL_GPUTransferBufferCreateInfo upload_xfer_info = {};
    upload_xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    upload_xfer_info.size = buf_size;
    SDL_GPUTransferBuffer* upload_xfer = SDL_CreateGPUTransferBuffer(r->device, &upload_xfer_info);

    if (!r->gaussian_buffer || !r->index_buffer || !r->transfer_bufs[0] || !upload_xfer) {
        fprintf(stderr, "FAIL: buffer creation failed\n");
        free(gpu_data);
        return;
    }

    void* map = SDL_MapGPUTransferBuffer(r->device, upload_xfer, false);
    if (!map) { free(gpu_data); return; }

    memcpy(map, gpu_data, buf_size);
    SDL_UnmapGPUTransferBuffer(r->device, upload_xfer);
    free(gpu_data);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return;

    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTransferBufferLocation src = {};
    src.transfer_buffer = upload_xfer;
    src.offset = 0;

    SDL_GPUBufferRegion dst = {};
    dst.buffer = r->gaussian_buffer;
    dst.offset = 0;
    dst.size = buf_size;

    SDL_UploadToGPUBuffer(copy, &src, &dst, false);
    SDL_EndGPUCopyPass(copy);

    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_WaitForGPUIdle(r->device);
    SDL_ReleaseGPUTransferBuffer(r->device, upload_xfer);

    fprintf(stderr, "Uploaded %u gaussians\n", scene->gaussian_count);
}

void renderer_draw_frame(Renderer* r, const GaussianScene* scene, const CameraUniforms* cam) {
    // Get current frame's transfer buffer and fence
    uint32_t buf_idx = r->current_frame % MAX_FRAMES_IN_FLIGHT;
    SDL_GPUTransferBuffer* transfer_buf = r->transfer_bufs[buf_idx];

    // Wait for this frame slot's previous work to complete before reusing its transfer buffer
    if (r->frame_fences[buf_idx] != NULL) {
        SDL_WaitForGPUFences(r->device, true, &r->frame_fences[buf_idx], 1);
        SDL_ReleaseGPUFence(r->device, r->frame_fences[buf_idx]);
        r->frame_fences[buf_idx] = NULL;
    }

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return;

    // Upload sorted indices via transfer buffer
    if (scene->visible_count > 0 && transfer_buf) {
        void* map = SDL_MapGPUTransferBuffer(r->device, transfer_buf, false);
        if (map) {
            memcpy(map, scene->sorted_indices, scene->visible_count * sizeof(uint32_t));
            SDL_UnmapGPUTransferBuffer(r->device, transfer_buf);

            SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);

            SDL_GPUTransferBufferLocation src = {};
            src.transfer_buffer = transfer_buf;
            src.offset = 0;

            SDL_GPUBufferRegion dst = {};
            dst.buffer = r->index_buffer;
            dst.offset = 0;
            dst.size = scene->visible_count * sizeof(uint32_t);

            SDL_UploadToGPUBuffer(copy, &src, &dst, false);
            SDL_EndGPUCopyPass(copy);
        }
    }

    // Prepare ImGui draw data (must be before render pass per backend requirement)
    ImGui_ImplSDLGPU3_PrepareDrawData(ImGui::GetDrawData(), cmd);

    // Acquire swapchain
    SDL_GPUTexture* swapchain_tex = NULL;
    uint32_t sw_w = 0, sw_h = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, r->window, &swapchain_tex, &sw_w, &sw_h)) {
        SDL_SubmitGPUCommandBuffer(cmd);
        return;
    }
    if (!swapchain_tex) {
        SDL_SubmitGPUCommandBuffer(cmd);
        return;
    }

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = swapchain_tex;
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.clear_color.r = 0.1f;
    color_target.clear_color.g = 0.1f;
    color_target.clear_color.b = 0.1f;
    color_target.clear_color.a = 1.0f;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, NULL);
    if (!pass) {
        SDL_SubmitGPUCommandBuffer(cmd);
        return;
    }

    // Draw gaussians
    if (scene->visible_count > 0 && r->splat_pipeline && r->gaussian_buffer && r->index_buffer) {
        SDL_BindGPUGraphicsPipeline(pass, r->splat_pipeline);

        // Bind storage buffers - index buffer at slot 0, gaussian buffer at slot 1
        // (matches shader bindings in set 0: IndexBuffer at binding 0, GaussianBuffer at binding 1)
        SDL_GPUBuffer* storage_bufs[2] = { r->index_buffer, r->gaussian_buffer };
        SDL_BindGPUVertexStorageBuffers(pass, 0, storage_bufs, 2);

        SDL_PushGPUVertexUniformData(cmd, 0, cam, sizeof(CameraUniforms));

        SDL_DrawGPUPrimitives(pass, 6, scene->visible_count, 0, 0);
    }

    // ImGui
    ImGui_ImplSDLGPU3_RenderDrawData(ImGui::GetDrawData(), cmd, pass);

    SDL_EndGPURenderPass(pass);

    // Submit and acquire fence for this frame
    r->frame_fences[buf_idx] = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    r->current_frame++;
}

void renderer_destroy(Renderer* r) {
    // Wait for all in-flight frames to complete
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (r->frame_fences[i] != NULL) {
            SDL_WaitForGPUFences(r->device, true, &r->frame_fences[i], 1);
            SDL_ReleaseGPUFence(r->device, r->frame_fences[i]);
        }
    }
    
    if (r->splat_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->splat_pipeline);
    if (r->gaussian_buffer) SDL_ReleaseGPUBuffer(r->device, r->gaussian_buffer);
    if (r->index_buffer) SDL_ReleaseGPUBuffer(r->device, r->index_buffer);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (r->transfer_bufs[i]) SDL_ReleaseGPUTransferBuffer(r->device, r->transfer_bufs[i]);
    }
}
