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

    // --- Overlay pipeline ---
    size_t ov_vert_size, ov_frag_size;
    uint8_t* ov_vert_code = load_file("shaders/overlay.vert.spv", &ov_vert_size);
    uint8_t* ov_frag_code = load_file("shaders/overlay.frag.spv", &ov_frag_size);
    if (!ov_vert_code || !ov_frag_code) {
        fprintf(stderr, "Failed to load overlay shaders\n");
        free(ov_vert_code); free(ov_frag_code);
        return false;
    }

    SDL_GPUShaderCreateInfo ov_vert_info = {};
    ov_vert_info.code = ov_vert_code;
    ov_vert_info.code_size = ov_vert_size;
    ov_vert_info.entrypoint = "main";
    ov_vert_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    ov_vert_info.stage = SDL_GPU_SHADERSTAGE_VERTEX;

    SDL_GPUShader* ov_vert = SDL_CreateGPUShader(device, &ov_vert_info);
    free(ov_vert_code);

    SDL_GPUShaderCreateInfo ov_frag_info = {};
    ov_frag_info.code = ov_frag_code;
    ov_frag_info.code_size = ov_frag_size;
    ov_frag_info.entrypoint = "main";
    ov_frag_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    ov_frag_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    ov_frag_info.num_samplers = 1;
    ov_frag_info.num_uniform_buffers = 1;

    SDL_GPUShader* ov_frag = SDL_CreateGPUShader(device, &ov_frag_info);
    free(ov_frag_code);

    if (!ov_vert || !ov_frag) {
        fprintf(stderr, "FAIL overlay shaders: %s\n", SDL_GetError());
        if (ov_vert) SDL_ReleaseGPUShader(device, ov_vert);
        if (ov_frag) SDL_ReleaseGPUShader(device, ov_frag);
        return false;
    }

    SDL_GPUColorTargetDescription ov_color_target = {};
    ov_color_target.format = r->swapchain_format;
    ov_color_target.blend_state.enable_blend = true;
    ov_color_target.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    ov_color_target.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    ov_color_target.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
    ov_color_target.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    ov_color_target.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    ov_color_target.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    ov_color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B | SDL_GPU_COLORCOMPONENT_A;

    SDL_GPUGraphicsPipelineCreateInfo ov_pipeline_info = {};
    ov_pipeline_info.vertex_shader = ov_vert;
    ov_pipeline_info.fragment_shader = ov_frag;
    ov_pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    ov_pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    ov_pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    ov_pipeline_info.target_info.num_color_targets = 1;
    ov_pipeline_info.target_info.color_target_descriptions = &ov_color_target;

    r->overlay_pipeline = SDL_CreateGPUGraphicsPipeline(device, &ov_pipeline_info);
    SDL_ReleaseGPUShader(device, ov_vert);
    SDL_ReleaseGPUShader(device, ov_frag);

    if (!r->overlay_pipeline) {
        fprintf(stderr, "FAIL overlay pipeline: %s\n", SDL_GetError());
        return false;
    }

    // Overlay sampler (linear filtering for smooth panorama sampling)
    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    r->overlay_sampler = SDL_CreateGPUSampler(device, &sampler_info);

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

void renderer_draw_frame(Renderer* r, const GaussianScene* scene, const CameraUniforms* cam, const OverlayParams* overlay) {
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

    // Overlay (equirectangular panorama)
    if (overlay && overlay->texture && overlay->alpha > 0.0f && r->overlay_pipeline) {
        SDL_BindGPUGraphicsPipeline(pass, r->overlay_pipeline);

        SDL_GPUTextureSamplerBinding sampler_binding = {};
        sampler_binding.texture = overlay->texture;
        sampler_binding.sampler = r->overlay_sampler;
        SDL_BindGPUFragmentSamplers(pass, 0, &sampler_binding, 1);

        // Pack uniforms: camera_ray_basis (64) + tan_half_fov (8) + pad (8)
        //              + ref_rotation (64) + alpha (4) + pad (12) = 160 bytes
        struct {
            float camera_ray_basis[16];
            float camera_tan_half_fov[2];
            float camera_pad[2];
            float ref_rotation[16];
            float alpha;
            float alpha_pad[3];
        } ov_uniforms;
        memcpy(ov_uniforms.camera_ray_basis, overlay->camera_ray_basis, sizeof(float) * 16);
        memcpy(ov_uniforms.camera_tan_half_fov, overlay->camera_tan_half_fov, sizeof(float) * 2);
        ov_uniforms.camera_pad[0] = ov_uniforms.camera_pad[1] = 0.0f;
        memcpy(ov_uniforms.ref_rotation, overlay->ref_rotation, sizeof(float) * 16);
        ov_uniforms.alpha = overlay->alpha;
        ov_uniforms.alpha_pad[0] = ov_uniforms.alpha_pad[1] = ov_uniforms.alpha_pad[2] = 0.0f;

        SDL_PushGPUFragmentUniformData(cmd, 0, &ov_uniforms, sizeof(ov_uniforms));
        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
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
    if (r->overlay_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->overlay_pipeline);
    if (r->overlay_sampler) SDL_ReleaseGPUSampler(r->device, r->overlay_sampler);
    if (r->gaussian_buffer) SDL_ReleaseGPUBuffer(r->device, r->gaussian_buffer);
    if (r->index_buffer) SDL_ReleaseGPUBuffer(r->device, r->index_buffer);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (r->transfer_bufs[i]) SDL_ReleaseGPUTransferBuffer(r->device, r->transfer_bufs[i]);
    }
}
