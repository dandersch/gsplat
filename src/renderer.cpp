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
    r->mesh_pipeline = NULL;
    r->mesh_vertex_buffer = NULL;
    r->mesh_index_buffer = NULL;
    r->mesh_texture = NULL;
    r->mesh_sampler = NULL;
    r->depth_texture = NULL;
    r->depth_w = 0;
    r->depth_h = 0;
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
    color_target.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_CONSTANT_COLOR;
    color_target.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    color_target.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B | SDL_GPU_COLORCOMPONENT_A;

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.vertex_shader = vert_shader;
    pipeline_info.fragment_shader = frag_shader;
    pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    pipeline_info.depth_stencil_state.enable_depth_test = true;
    pipeline_info.depth_stencil_state.enable_depth_write = false;
    pipeline_info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
    pipeline_info.target_info.num_color_targets = 1;
    pipeline_info.target_info.color_target_descriptions = &color_target;
    pipeline_info.target_info.has_depth_stencil_target = true;
    pipeline_info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

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
    ov_color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B;

    SDL_GPUGraphicsPipelineCreateInfo ov_pipeline_info = {};
    ov_pipeline_info.vertex_shader = ov_vert;
    ov_pipeline_info.fragment_shader = ov_frag;
    ov_pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    ov_pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    ov_pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    ov_pipeline_info.target_info.num_color_targets = 1;
    ov_pipeline_info.target_info.color_target_descriptions = &ov_color_target;
    ov_pipeline_info.target_info.has_depth_stencil_target = true;
    ov_pipeline_info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

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

    // --- Wireframe pipeline ---
    size_t wf_vert_size, wf_frag_size;
    uint8_t* wf_vert_code = load_file("shaders/wireframe.vert.spv", &wf_vert_size);
    uint8_t* wf_frag_code = load_file("shaders/wireframe.frag.spv", &wf_frag_size);
    if (!wf_vert_code || !wf_frag_code) {
        fprintf(stderr, "Failed to load wireframe shaders\n");
        free(wf_vert_code); free(wf_frag_code);
        return false;
    }

    SDL_GPUShaderCreateInfo wf_vert_info = {};
    wf_vert_info.code = wf_vert_code;
    wf_vert_info.code_size = wf_vert_size;
    wf_vert_info.entrypoint = "main";
    wf_vert_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    wf_vert_info.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    wf_vert_info.num_uniform_buffers = 1;

    SDL_GPUShader* wf_vert = SDL_CreateGPUShader(device, &wf_vert_info);
    free(wf_vert_code);

    SDL_GPUShaderCreateInfo wf_frag_info = {};
    wf_frag_info.code = wf_frag_code;
    wf_frag_info.code_size = wf_frag_size;
    wf_frag_info.entrypoint = "main";
    wf_frag_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    wf_frag_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;

    SDL_GPUShader* wf_frag = SDL_CreateGPUShader(device, &wf_frag_info);
    free(wf_frag_code);

    if (!wf_vert || !wf_frag) {
        fprintf(stderr, "FAIL wireframe shaders: %s\n", SDL_GetError());
        if (wf_vert) SDL_ReleaseGPUShader(device, wf_vert);
        if (wf_frag) SDL_ReleaseGPUShader(device, wf_frag);
        return false;
    }

    SDL_GPUColorTargetDescription wf_color_target = {};
    wf_color_target.format = r->swapchain_format;
    wf_color_target.blend_state.enable_blend = true;
    wf_color_target.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA;
    wf_color_target.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    wf_color_target.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
    wf_color_target.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ZERO;
    wf_color_target.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    wf_color_target.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    wf_color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B;

    SDL_GPUVertexBufferDescription wf_vb_desc = {};
    wf_vb_desc.slot = 0;
    wf_vb_desc.pitch = 3 * sizeof(float);
    wf_vb_desc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;

    SDL_GPUVertexAttribute wf_attr = {};
    wf_attr.location = 0;
    wf_attr.buffer_slot = 0;
    wf_attr.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    wf_attr.offset = 0;

    SDL_GPUGraphicsPipelineCreateInfo wf_pipeline_info = {};
    wf_pipeline_info.vertex_shader = wf_vert;
    wf_pipeline_info.fragment_shader = wf_frag;
    wf_pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST;
    wf_pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    wf_pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    wf_pipeline_info.vertex_input_state.num_vertex_buffers = 1;
    wf_pipeline_info.vertex_input_state.vertex_buffer_descriptions = &wf_vb_desc;
    wf_pipeline_info.vertex_input_state.num_vertex_attributes = 1;
    wf_pipeline_info.vertex_input_state.vertex_attributes = &wf_attr;
    wf_pipeline_info.target_info.num_color_targets = 1;
    wf_pipeline_info.target_info.color_target_descriptions = &wf_color_target;
    wf_pipeline_info.target_info.has_depth_stencil_target = true;
    wf_pipeline_info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

    r->wireframe_pipeline = SDL_CreateGPUGraphicsPipeline(device, &wf_pipeline_info);

    if (!r->wireframe_pipeline) {
        fprintf(stderr, "FAIL wireframe pipeline: %s\n", SDL_GetError());
        SDL_ReleaseGPUShader(device, wf_vert);
        SDL_ReleaseGPUShader(device, wf_frag);
        return false;
    }

    SDL_ReleaseGPUShader(device, wf_vert);
    SDL_ReleaseGPUShader(device, wf_frag);

    // --- Mesh pipeline (dedicated shaders with UV + texture support) ---
    size_t mesh_vert_size, mesh_frag_size;
    uint8_t* mesh_vert_code = load_file("shaders/mesh.vert.spv", &mesh_vert_size);
    uint8_t* mesh_frag_code = load_file("shaders/mesh.frag.spv", &mesh_frag_size);
    if (!mesh_vert_code || !mesh_frag_code) {
        fprintf(stderr, "Failed to load mesh shaders\n");
        free(mesh_vert_code); free(mesh_frag_code);
        return false;
    }

    SDL_GPUShaderCreateInfo mesh_vert_info = {};
    mesh_vert_info.code = mesh_vert_code;
    mesh_vert_info.code_size = mesh_vert_size;
    mesh_vert_info.entrypoint = "main";
    mesh_vert_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    mesh_vert_info.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    mesh_vert_info.num_uniform_buffers = 1;

    SDL_GPUShader* mesh_vs = SDL_CreateGPUShader(device, &mesh_vert_info);
    free(mesh_vert_code);

    SDL_GPUShaderCreateInfo mesh_frag_info = {};
    mesh_frag_info.code = mesh_frag_code;
    mesh_frag_info.code_size = mesh_frag_size;
    mesh_frag_info.entrypoint = "main";
    mesh_frag_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
    mesh_frag_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    mesh_frag_info.num_samplers = 1;

    SDL_GPUShader* mesh_fs = SDL_CreateGPUShader(device, &mesh_frag_info);
    free(mesh_frag_code);

    if (!mesh_vs || !mesh_fs) {
        fprintf(stderr, "FAIL mesh shaders: %s\n", SDL_GetError());
        if (mesh_vs) SDL_ReleaseGPUShader(device, mesh_vs);
        if (mesh_fs) SDL_ReleaseGPUShader(device, mesh_fs);
        return false;
    }

    SDL_GPUColorTargetDescription mesh_color_target = {};
    mesh_color_target.format = r->swapchain_format;
    mesh_color_target.blend_state.enable_blend = false;
    mesh_color_target.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R | SDL_GPU_COLORCOMPONENT_G | SDL_GPU_COLORCOMPONENT_B | SDL_GPU_COLORCOMPONENT_A;

    SDL_GPUVertexBufferDescription mesh_vb_desc = {};
    mesh_vb_desc.slot = 0;
    mesh_vb_desc.pitch = 5 * sizeof(float); // vec3 pos + vec2 uv
    mesh_vb_desc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;

    SDL_GPUVertexAttribute mesh_attrs[2] = {};
    mesh_attrs[0].location = 0;
    mesh_attrs[0].buffer_slot = 0;
    mesh_attrs[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    mesh_attrs[0].offset = 0;
    mesh_attrs[1].location = 1;
    mesh_attrs[1].buffer_slot = 0;
    mesh_attrs[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
    mesh_attrs[1].offset = 3 * sizeof(float);

    SDL_GPUGraphicsPipelineCreateInfo mesh_pipeline_info = {};
    mesh_pipeline_info.vertex_shader = mesh_vs;
    mesh_pipeline_info.fragment_shader = mesh_fs;
    mesh_pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    mesh_pipeline_info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    mesh_pipeline_info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
    mesh_pipeline_info.vertex_input_state.num_vertex_buffers = 1;
    mesh_pipeline_info.vertex_input_state.vertex_buffer_descriptions = &mesh_vb_desc;
    mesh_pipeline_info.vertex_input_state.num_vertex_attributes = 2;
    mesh_pipeline_info.vertex_input_state.vertex_attributes = mesh_attrs;
    mesh_pipeline_info.depth_stencil_state.enable_depth_test = true;
    mesh_pipeline_info.depth_stencil_state.enable_depth_write = true;
    mesh_pipeline_info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
    mesh_pipeline_info.target_info.num_color_targets = 1;
    mesh_pipeline_info.target_info.color_target_descriptions = &mesh_color_target;
    mesh_pipeline_info.target_info.has_depth_stencil_target = true;
    mesh_pipeline_info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

    r->mesh_pipeline = SDL_CreateGPUGraphicsPipeline(device, &mesh_pipeline_info);
    SDL_ReleaseGPUShader(device, mesh_vs);
    SDL_ReleaseGPUShader(device, mesh_fs);

    if (!r->mesh_pipeline) {
        fprintf(stderr, "FAIL mesh pipeline: %s\n", SDL_GetError());
        return false;
    }

    // Upload cube geometry (unit cube centered at origin, ±0.5)
    float cube_verts[8 * 3] = {
        -0.5f, -0.5f, -0.5f,  // 0
         0.5f, -0.5f, -0.5f,  // 1
         0.5f,  0.5f, -0.5f,  // 2
        -0.5f,  0.5f, -0.5f,  // 3
        -0.5f, -0.5f,  0.5f,  // 4
         0.5f, -0.5f,  0.5f,  // 5
         0.5f,  0.5f,  0.5f,  // 6
        -0.5f,  0.5f,  0.5f,  // 7
    };
    uint16_t cube_indices[24] = {
        0,1, 1,2, 2,3, 3,0,  // back face
        4,5, 5,6, 6,7, 7,4,  // front face
        0,4, 1,5, 2,6, 3,7,  // connecting edges
    };

    uint32_t vb_size = sizeof(cube_verts);
    uint32_t ib_size = sizeof(cube_indices);

    SDL_GPUBufferCreateInfo cube_vb_info = {};
    cube_vb_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    cube_vb_info.size = vb_size;
    r->cube_vertex_buffer = SDL_CreateGPUBuffer(device, &cube_vb_info);

    SDL_GPUBufferCreateInfo cube_ib_info = {};
    cube_ib_info.usage = SDL_GPU_BUFFERUSAGE_INDEX;
    cube_ib_info.size = ib_size;
    r->cube_index_buffer = SDL_CreateGPUBuffer(device, &cube_ib_info);

    SDL_GPUTransferBufferCreateInfo cube_xfer_info = {};
    cube_xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    cube_xfer_info.size = vb_size + ib_size;
    SDL_GPUTransferBuffer* cube_xfer = SDL_CreateGPUTransferBuffer(device, &cube_xfer_info);

    void* cube_map = SDL_MapGPUTransferBuffer(device, cube_xfer, false);
    memcpy(cube_map, cube_verts, vb_size);
    memcpy((uint8_t*)cube_map + vb_size, cube_indices, ib_size);
    SDL_UnmapGPUTransferBuffer(device, cube_xfer);

    SDL_GPUCommandBuffer* cube_cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* cube_copy = SDL_BeginGPUCopyPass(cube_cmd);

    SDL_GPUTransferBufferLocation cube_src = {};
    cube_src.transfer_buffer = cube_xfer;
    cube_src.offset = 0;
    SDL_GPUBufferRegion cube_vb_dst = {};
    cube_vb_dst.buffer = r->cube_vertex_buffer;
    cube_vb_dst.size = vb_size;
    SDL_UploadToGPUBuffer(cube_copy, &cube_src, &cube_vb_dst, false);

    cube_src.offset = vb_size;
    SDL_GPUBufferRegion cube_ib_dst = {};
    cube_ib_dst.buffer = r->cube_index_buffer;
    cube_ib_dst.size = ib_size;
    SDL_UploadToGPUBuffer(cube_copy, &cube_src, &cube_ib_dst, false);

    SDL_EndGPUCopyPass(cube_copy);
    SDL_SubmitGPUCommandBuffer(cube_cmd);
    SDL_WaitForGPUIdle(device);
    SDL_ReleaseGPUTransferBuffer(device, cube_xfer);

    // Upload mesh cube geometry (per-face vertices with UVs, 4 verts per face × 6 faces = 24)
    // Each vertex: vec3 position, vec2 uv
    float mesh_verts[24 * 5] = {
        // back face (-Z)
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        // front face (+Z)
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        // right face (+X)
         0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        // left face (-X)
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        // top face (+Y)
        -0.5f,  0.5f, -0.5f,  0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        // bottom face (-Y)
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    };
    uint16_t mesh_indices[36] = {
         0, 2, 1,   0, 3, 2,  // back
         4, 5, 6,   4, 6, 7,  // front
         8,10, 9,   8,11,10,  // right
        12,14,13,  12,15,14,  // left
        16,18,17,  16,19,18,  // top
        20,22,21,  20,23,22,  // bottom
    };

    uint32_t mesh_vb_size = sizeof(mesh_verts);
    uint32_t mesh_ib_size = sizeof(mesh_indices);

    SDL_GPUBufferCreateInfo mesh_vb_info = {};
    mesh_vb_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    mesh_vb_info.size = mesh_vb_size;
    r->mesh_vertex_buffer = SDL_CreateGPUBuffer(device, &mesh_vb_info);

    SDL_GPUBufferCreateInfo mesh_ib_info = {};
    mesh_ib_info.usage = SDL_GPU_BUFFERUSAGE_INDEX;
    mesh_ib_info.size = mesh_ib_size;
    r->mesh_index_buffer = SDL_CreateGPUBuffer(device, &mesh_ib_info);

    SDL_GPUTransferBufferCreateInfo mesh_xfer_info = {};
    mesh_xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    mesh_xfer_info.size = mesh_vb_size + mesh_ib_size;
    SDL_GPUTransferBuffer* mesh_xfer = SDL_CreateGPUTransferBuffer(device, &mesh_xfer_info);

    void* mesh_map = SDL_MapGPUTransferBuffer(device, mesh_xfer, false);
    memcpy(mesh_map, mesh_verts, mesh_vb_size);
    memcpy((uint8_t*)mesh_map + mesh_vb_size, mesh_indices, mesh_ib_size);
    SDL_UnmapGPUTransferBuffer(device, mesh_xfer);

    SDL_GPUCommandBuffer* mesh_cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* mesh_copy = SDL_BeginGPUCopyPass(mesh_cmd);

    SDL_GPUTransferBufferLocation mesh_src = {};
    mesh_src.transfer_buffer = mesh_xfer;
    mesh_src.offset = 0;
    SDL_GPUBufferRegion mesh_vb_dst = {};
    mesh_vb_dst.buffer = r->mesh_vertex_buffer;
    mesh_vb_dst.size = mesh_vb_size;
    SDL_UploadToGPUBuffer(mesh_copy, &mesh_src, &mesh_vb_dst, false);

    mesh_src.offset = mesh_vb_size;
    SDL_GPUBufferRegion mesh_ib_dst = {};
    mesh_ib_dst.buffer = r->mesh_index_buffer;
    mesh_ib_dst.size = mesh_ib_size;
    SDL_UploadToGPUBuffer(mesh_copy, &mesh_src, &mesh_ib_dst, false);

    SDL_EndGPUCopyPass(mesh_copy);
    SDL_SubmitGPUCommandBuffer(mesh_cmd);
    SDL_WaitForGPUIdle(device);
    SDL_ReleaseGPUTransferBuffer(device, mesh_xfer);

    // Create 8×8 checkerboard test texture (RGBA8)
    {
        const int TEX_SIZE = 8;
        uint8_t tex_pixels[TEX_SIZE * TEX_SIZE * 4];
        for (int y = 0; y < TEX_SIZE; y++) {
            for (int x = 0; x < TEX_SIZE; x++) {
                bool white = ((x + y) % 2) == 0;
                int idx = (y * TEX_SIZE + x) * 4;
                tex_pixels[idx + 0] = white ? 255 : 50;
                tex_pixels[idx + 1] = white ? 255 : 50;
                tex_pixels[idx + 2] = white ? 255 : 50;
                tex_pixels[idx + 3] = 255;
            }
        }

        SDL_GPUTextureCreateInfo tex_info = {};
        tex_info.type = SDL_GPU_TEXTURETYPE_2D;
        tex_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        tex_info.width = TEX_SIZE;
        tex_info.height = TEX_SIZE;
        tex_info.layer_count_or_depth = 1;
        tex_info.num_levels = 1;
        tex_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
        r->mesh_texture = SDL_CreateGPUTexture(device, &tex_info);

        SDL_GPUTransferBufferCreateInfo tex_xfer_info = {};
        tex_xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        tex_xfer_info.size = sizeof(tex_pixels);
        SDL_GPUTransferBuffer* tex_xfer = SDL_CreateGPUTransferBuffer(device, &tex_xfer_info);

        void* tex_map = SDL_MapGPUTransferBuffer(device, tex_xfer, false);
        memcpy(tex_map, tex_pixels, sizeof(tex_pixels));
        SDL_UnmapGPUTransferBuffer(device, tex_xfer);

        SDL_GPUCommandBuffer* tex_cmd = SDL_AcquireGPUCommandBuffer(device);
        SDL_GPUCopyPass* tex_copy = SDL_BeginGPUCopyPass(tex_cmd);

        SDL_GPUTextureTransferInfo tex_src = {};
        tex_src.transfer_buffer = tex_xfer;
        tex_src.offset = 0;

        SDL_GPUTextureRegion tex_dst = {};
        tex_dst.texture = r->mesh_texture;
        tex_dst.w = TEX_SIZE;
        tex_dst.h = TEX_SIZE;
        tex_dst.d = 1;

        SDL_UploadToGPUTexture(tex_copy, &tex_src, &tex_dst, false);
        SDL_EndGPUCopyPass(tex_copy);
        SDL_SubmitGPUCommandBuffer(tex_cmd);
        SDL_WaitForGPUIdle(device);
        SDL_ReleaseGPUTransferBuffer(device, tex_xfer);

        SDL_GPUSamplerCreateInfo mesh_sampler_info = {};
        mesh_sampler_info.min_filter = SDL_GPU_FILTER_NEAREST;
        mesh_sampler_info.mag_filter = SDL_GPU_FILTER_NEAREST;
        mesh_sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
        mesh_sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        mesh_sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        mesh_sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        r->mesh_sampler = SDL_CreateGPUSampler(device, &mesh_sampler_info);
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

// Multiply two column-major 4x4 matrices: out = a * b
static void mat4_mul(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
        }
    }
}

// Build a column-major translation+scale model matrix
static void mat4_translate_scale(float tx, float ty, float tz, float s, float* out) {
    memset(out, 0, 16 * sizeof(float));
    out[0]  = s;
    out[5]  = s;
    out[10] = s;
    out[12] = tx;
    out[13] = ty;
    out[14] = tz;
    out[15] = 1.0f;
}

void renderer_draw_frame(Renderer* r, const GaussianScene* scene, const CameraUniforms* cam, const OverlayParams* overlay, const NodeRenderParams* nodes, float wireframe_occlusion) {
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

    // Recreate depth texture if swapchain dimensions changed
    if (sw_w != r->depth_w || sw_h != r->depth_h) {
        if (r->depth_texture) SDL_ReleaseGPUTexture(r->device, r->depth_texture);
        SDL_GPUTextureCreateInfo depth_tex_info = {};
        depth_tex_info.type = SDL_GPU_TEXTURETYPE_2D;
        depth_tex_info.format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
        depth_tex_info.width = sw_w;
        depth_tex_info.height = sw_h;
        depth_tex_info.layer_count_or_depth = 1;
        depth_tex_info.num_levels = 1;
        depth_tex_info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
        r->depth_texture = SDL_CreateGPUTexture(r->device, &depth_tex_info);
        r->depth_w = sw_w;
        r->depth_h = sw_h;
    }

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = swapchain_tex;
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.clear_color.r = 0.1f;
    color_target.clear_color.g = 0.1f;
    color_target.clear_color.b = 0.1f;
    color_target.clear_color.a = 0.0f;

    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = r->depth_texture;
    depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_DONT_CARE;
    depth_target.clear_depth = 1.0f;
    depth_target.cycle = true;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, &depth_target);
    if (!pass) {
        SDL_SubmitGPUCommandBuffer(cmd);
        return;
    }

    // Mesh test cubes (drawn before splats, writes depth so splats depth-test against it)
    if (r->mesh_pipeline && r->mesh_vertex_buffer && r->mesh_index_buffer) {
        SDL_BindGPUGraphicsPipeline(pass, r->mesh_pipeline);

        SDL_GPUBufferBinding mesh_vb_bind = {};
        mesh_vb_bind.buffer = r->mesh_vertex_buffer;
        SDL_BindGPUVertexBuffers(pass, 0, &mesh_vb_bind, 1);

        SDL_GPUBufferBinding mesh_ib_bind = {};
        mesh_ib_bind.buffer = r->mesh_index_buffer;
        SDL_BindGPUIndexBuffer(pass, &mesh_ib_bind, SDL_GPU_INDEXELEMENTSIZE_16BIT);

        // Bind checkerboard texture for textured meshes
        SDL_GPUTextureSamplerBinding mesh_tex_bind = {};
        mesh_tex_bind.texture = r->mesh_texture;
        mesh_tex_bind.sampler = r->mesh_sampler;
        SDL_BindGPUFragmentSamplers(pass, 0, &mesh_tex_bind, 1);

        float view_corrected[16];
        memcpy(view_corrected, cam->view, sizeof(view_corrected));
        view_corrected[0]  = -view_corrected[0];
        view_corrected[4]  = -view_corrected[4];
        view_corrected[8]  = -view_corrected[8];
        view_corrected[12] = -view_corrected[12];
        float vp[16];
        mat4_mul(cam->proj, view_corrected, vp);

        float model[16];
        mat4_translate_scale(0.0f, 0.0f, 0.0f, 1.0f, model);

        float mvp[16];
        mat4_mul(vp, model, mvp);

        struct { float mvp[16]; float color[4]; float use_texture; float _pad[3]; } mesh_uniforms;
        memcpy(mesh_uniforms.mvp, mvp, sizeof(mvp));
        mesh_uniforms.color[0] = 1.0f;
        mesh_uniforms.color[1] = 0.3f;
        mesh_uniforms.color[2] = 0.1f;
        mesh_uniforms.color[3] = 1.0f;
        mesh_uniforms.use_texture = 1.0f;
        mesh_uniforms._pad[0] = mesh_uniforms._pad[1] = mesh_uniforms._pad[2] = 0.0f;

        SDL_PushGPUVertexUniformData(cmd, 0, &mesh_uniforms, sizeof(mesh_uniforms));
        SDL_DrawGPUIndexedPrimitives(pass, 36, 1, 0, 0, 0);

        // Second test cube outside the scene (flat colored, no texture)
        mat4_translate_scale(-4.0f, 0.0f, -18.0f, 2.0f, model);
        mat4_mul(vp, model, mvp);
        memcpy(mesh_uniforms.mvp, mvp, sizeof(mvp));
        mesh_uniforms.color[0] = 0.1f;
        mesh_uniforms.color[1] = 1.0f;
        mesh_uniforms.color[2] = 0.3f;
        mesh_uniforms.use_texture = 0.0f;
        SDL_PushGPUVertexUniformData(cmd, 0, &mesh_uniforms, sizeof(mesh_uniforms));
        SDL_DrawGPUIndexedPrimitives(pass, 36, 1, 0, 0, 0);
    }

    // Draw gaussians (alpha accumulation scaled by wireframe_occlusion blend constant)
    if (scene->visible_count > 0 && r->splat_pipeline && r->gaussian_buffer && r->index_buffer) {
        SDL_BindGPUGraphicsPipeline(pass, r->splat_pipeline);
        SDL_FColor blend_const = { 0, 0, 0, wireframe_occlusion };
        SDL_SetGPUBlendConstants(pass, blend_const);

        SDL_GPUBuffer* storage_bufs[2] = { r->index_buffer, r->gaussian_buffer };
        SDL_BindGPUVertexStorageBuffers(pass, 0, storage_bufs, 2);

        SDL_PushGPUVertexUniformData(cmd, 0, cam, sizeof(CameraUniforms));

        SDL_DrawGPUPrimitives(pass, 6, scene->visible_count, 0, 0);
    }

    // Overlay (equirectangular panorama, no depth test/write — drawn over splats)
    if (overlay && overlay->texture && overlay->alpha > 0.0f && r->overlay_pipeline) {
        SDL_BindGPUGraphicsPipeline(pass, r->overlay_pipeline);

        SDL_GPUTextureSamplerBinding sampler_binding = {};
        sampler_binding.texture = overlay->texture;
        sampler_binding.sampler = r->overlay_sampler;
        SDL_BindGPUFragmentSamplers(pass, 0, &sampler_binding, 1);

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

    // Wireframe node cubes (depth-tested against splats, drawn over overlay)
    if (nodes && nodes->count > 0 && r->wireframe_pipeline) {
        SDL_BindGPUGraphicsPipeline(pass, r->wireframe_pipeline);

        SDL_GPUBufferBinding vb_bind = {};
        vb_bind.buffer = r->cube_vertex_buffer;
        SDL_BindGPUVertexBuffers(pass, 0, &vb_bind, 1);

        SDL_GPUBufferBinding ib_bind = {};
        ib_bind.buffer = r->cube_index_buffer;
        SDL_BindGPUIndexBuffer(pass, &ib_bind, SDL_GPU_INDEXELEMENTSIZE_16BIT);

        // Precompute view-projection.
        // The camera's view matrix has X-axis flipped (right = forward × up instead of
        // up × forward). The splat shader compensates via manual projection, but for
        // standard MVP we need to undo the flip by negating VP column 0.
        // Correct the view matrix by negating row 0 (the flipped right vector).
        // Row 0 in column-major is at indices [0], [4], [8], [12].
        float view_corrected[16];
        memcpy(view_corrected, cam->view, sizeof(view_corrected));
        view_corrected[0]  = -view_corrected[0];
        view_corrected[4]  = -view_corrected[4];
        view_corrected[8]  = -view_corrected[8];
        view_corrected[12] = -view_corrected[12];
        float vp[16];
        mat4_mul(cam->proj, view_corrected, vp);

        float scale = nodes->half_size * 2.0f; // cube verts are ±0.5, so scale by full size

        for (uint32_t i = 0; i < nodes->count; i++) {
            const float* p = &nodes->positions[i * 3];

            float model[16];
            mat4_translate_scale(p[0], p[1], p[2], scale, model);

            float mvp[16];
            mat4_mul(vp, model, mvp);

            struct { float mvp[16]; float color[4]; } uniforms;
            memcpy(uniforms.mvp, mvp, sizeof(mvp));
            uniforms.color[0] = 0.0f;
            uniforms.color[1] = 1.0f;
            uniforms.color[2] = 1.0f;
            uniforms.color[3] = 1.0f;

            SDL_PushGPUVertexUniformData(cmd, 0, &uniforms, sizeof(uniforms));
            SDL_DrawGPUIndexedPrimitives(pass, 24, 1, 0, 0, 0);
        }
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
    if (r->wireframe_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->wireframe_pipeline);
    if (r->mesh_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->mesh_pipeline);
    if (r->overlay_sampler) SDL_ReleaseGPUSampler(r->device, r->overlay_sampler);
    if (r->gaussian_buffer) SDL_ReleaseGPUBuffer(r->device, r->gaussian_buffer);
    if (r->cube_vertex_buffer) SDL_ReleaseGPUBuffer(r->device, r->cube_vertex_buffer);
    if (r->cube_index_buffer) SDL_ReleaseGPUBuffer(r->device, r->cube_index_buffer);
    if (r->mesh_vertex_buffer) SDL_ReleaseGPUBuffer(r->device, r->mesh_vertex_buffer);
    if (r->mesh_index_buffer) SDL_ReleaseGPUBuffer(r->device, r->mesh_index_buffer);
    if (r->mesh_texture) SDL_ReleaseGPUTexture(r->device, r->mesh_texture);
    if (r->mesh_sampler) SDL_ReleaseGPUSampler(r->device, r->mesh_sampler);
    if (r->index_buffer) SDL_ReleaseGPUBuffer(r->device, r->index_buffer);
    if (r->depth_texture) SDL_ReleaseGPUTexture(r->device, r->depth_texture);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (r->transfer_bufs[i]) SDL_ReleaseGPUTransferBuffer(r->device, r->transfer_bufs[i]);
    }
}
