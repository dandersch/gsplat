#include "refview.h"
#include <SDL3/SDL.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Skip the rest of the current line (handles arbitrarily long POINTS2D lines)
static void skip_line(FILE* f) {
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {}
}

// Convert colmap quaternion (world-to-camera, colmap convention: X-right Y-down Z-forward)
// to world-space camera center and yaw/pitch matching our fly camera convention (Y-up).
static void colmap_to_camera(float qw, float qx, float qy, float qz,
                             float tx, float ty, float tz,
                             float* out_pos, float* out_yaw, float* out_pitch)
{
    // Build rotation matrix R from quaternion (world-to-camera rotation)
    float R[3][3];
    R[0][0] = 1 - 2*(qy*qy + qz*qz);
    R[0][1] = 2*(qx*qy - qw*qz);
    R[0][2] = 2*(qx*qz + qw*qy);
    R[1][0] = 2*(qx*qy + qw*qz);
    R[1][1] = 1 - 2*(qx*qx + qz*qz);
    R[1][2] = 2*(qy*qz - qw*qx);
    R[2][0] = 2*(qx*qz - qw*qy);
    R[2][1] = 2*(qy*qz + qw*qx);
    R[2][2] = 1 - 2*(qx*qx + qy*qy);

    // Camera center in world space: -R^T * T
    out_pos[0] = -(R[0][0]*tx + R[1][0]*ty + R[2][0]*tz);
    out_pos[1] = -(R[0][1]*tx + R[1][1]*ty + R[2][1]*tz);
    out_pos[2] = -(R[0][2]*tx + R[1][2]*ty + R[2][2]*tz);

    // Camera forward in world space: R^T * [0,0,1] (colmap Z-forward)
    // = third column of R^T = third row of R
    float fwd_colmap[3] = { R[2][0], R[2][1], R[2][2] };

    // Colmap: Y-down. Our renderer: Y-up. Negate Y component.
    float fwd[3] = { fwd_colmap[0], -fwd_colmap[1], fwd_colmap[2] };
    out_pos[1] = -out_pos[1]; // also flip camera position Y

    // Derive yaw/pitch from forward vector
    // forward = (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))
    *out_pitch = asinf(fwd[1]);
    *out_yaw   = atan2f(fwd[0], fwd[2]);
}

bool refview_load(RefViewSet* set, const char* colmap_dir) {
    memset(set, 0, sizeof(*set));
    set->selected = -1;
    set->lerp_duration = 0.5f;

    // Build path to images.txt
    char images_txt[512];
    snprintf(images_txt, sizeof(images_txt), "%s/images.txt", colmap_dir);

    FILE* f = fopen(images_txt, "r");
    if (!f) {
        SDL_Log("RefView: Could not open %s", images_txt);
        SDL_Log("RefView: If you only have images.bin, convert with:");
        SDL_Log("  colmap model_converter --input_path %s --output_path %s --output_type TXT", colmap_dir, colmap_dir);
        return false;
    }

    // Derive image directory: colmap_dir/../../images/
    snprintf(set->image_dir, sizeof(set->image_dir), "%s/../../images", colmap_dir);
    SDL_Log("RefView: image directory: %s", set->image_dir);

    // First pass: count images
    uint32_t count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        count++;
        skip_line(f); // skip POINTS2D line (can be very long)
    }

    if (count == 0) {
        SDL_Log("RefView: No images found in %s", images_txt);
        fclose(f);
        return false;
    }

    set->views = (RefView*)calloc(count, sizeof(RefView));
    set->count = count;

    // Second pass: parse
    rewind(f);
    uint32_t idx = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        int image_id, camera_id;
        float qw, qx, qy, qz, tx, ty, tz;
        char name[256] = {};

        int parsed = sscanf(line, "%d %f %f %f %f %f %f %f %d %255s",
                            &image_id, &qw, &qx, &qy, &qz, &tx, &ty, &tz, &camera_id, name);
        if (parsed < 10) {
            SDL_Log("RefView: Failed to parse line: %s", line);
            skip_line(f);
            continue;
        }

        RefView* v = &set->views[idx];
        strncpy(v->image_name, name, sizeof(v->image_name) - 1);
        v->rotation[0] = qw;
        v->rotation[1] = qx;
        v->rotation[2] = qy;
        v->rotation[3] = qz;

        colmap_to_camera(qw, qx, qy, qz, tx, ty, tz,
                         v->position, &v->yaw, &v->pitch);

        idx++;
        skip_line(f);
    }

    set->count = idx; // actual parsed count
    fclose(f);

    SDL_Log("RefView: Loaded %u camera nodes from %s", set->count, images_txt);
    return true;
}

struct ImageLoadTask {
    char      path[768];
    SDL_Surface* result;  // NULL on failure
};

static int image_load_thread(void* data) {
    ImageLoadTask* task = (ImageLoadTask*)data;
    SDL_Surface* surface = SDL_LoadBMP(task->path);
    if (!surface) surface = SDL_LoadPNG(task->path);
    if (!surface) {
        task->result = NULL;
        return 0;
    }
    task->result = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ABGR8888);
    SDL_DestroySurface(surface);
    return 0;
}

void refview_load_images(RefViewSet* set, SDL_GPUDevice* device) {
    // Decode all images in parallel on separate threads
    ImageLoadTask* tasks = (ImageLoadTask*)calloc(set->count, sizeof(ImageLoadTask));
    SDL_Thread** threads = (SDL_Thread**)calloc(set->count, sizeof(SDL_Thread*));

    for (uint32_t i = 0; i < set->count; i++) {
        snprintf(tasks[i].path, sizeof(tasks[i].path), "%s/%s", set->image_dir, set->views[i].image_name);
        char name[32];
        snprintf(name, sizeof(name), "img_%u", i);
        threads[i] = SDL_CreateThread(image_load_thread, name, &tasks[i]);
    }

    // Wait for all threads and upload to GPU
    SDL_GPUTransferBuffer** xfer_bufs = (SDL_GPUTransferBuffer**)calloc(set->count, sizeof(SDL_GPUTransferBuffer*));
    uint32_t loaded = 0;

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);

    for (uint32_t i = 0; i < set->count; i++) {
        SDL_WaitThread(threads[i], NULL);
        RefView* v = &set->views[i];
        SDL_Surface* rgba = tasks[i].result;

        if (!rgba) {
            SDL_Log("RefView: Could not load image %s", tasks[i].path);
            continue;
        }

        v->width = rgba->w;
        v->height = rgba->h;

        SDL_GPUTextureCreateInfo tex_info = {};
        tex_info.type = SDL_GPU_TEXTURETYPE_2D;
        tex_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        tex_info.width = rgba->w;
        tex_info.height = rgba->h;
        tex_info.layer_count_or_depth = 1;
        tex_info.num_levels = 1;
        tex_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;

        v->texture = SDL_CreateGPUTexture(device, &tex_info);
        if (!v->texture) {
            SDL_Log("RefView: Failed to create GPU texture for %s: %s", tasks[i].path, SDL_GetError());
            SDL_DestroySurface(rgba);
            continue;
        }

        uint32_t data_size = rgba->w * rgba->h * 4;

        SDL_GPUTransferBufferCreateInfo xfer_info = {};
        xfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        xfer_info.size = data_size;
        SDL_GPUTransferBuffer* xfer = SDL_CreateGPUTransferBuffer(device, &xfer_info);
        if (!xfer) {
            SDL_Log("RefView: Failed to create transfer buffer for %s", tasks[i].path);
            SDL_ReleaseGPUTexture(device, v->texture);
            v->texture = NULL;
            SDL_DestroySurface(rgba);
            continue;
        }

        void* map = SDL_MapGPUTransferBuffer(device, xfer, false);
        memcpy(map, rgba->pixels, data_size);
        SDL_UnmapGPUTransferBuffer(device, xfer);
        SDL_DestroySurface(rgba);

        SDL_GPUTextureTransferInfo src = {};
        src.transfer_buffer = xfer;
        src.offset = 0;

        SDL_GPUTextureRegion dst = {};
        dst.texture = v->texture;
        dst.w = v->width;
        dst.h = v->height;
        dst.d = 1;

        SDL_UploadToGPUTexture(copy, &src, &dst, false);
        xfer_bufs[loaded] = xfer;
        loaded++;
    }

    SDL_EndGPUCopyPass(copy);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_WaitForGPUIdle(device);

    for (uint32_t i = 0; i < loaded; i++) {
        SDL_ReleaseGPUTransferBuffer(device, xfer_bufs[i]);
    }
    free(xfer_bufs);
    free(threads);
    free(tasks);

    SDL_Log("RefView: Loaded %u / %u images as GPU textures", loaded, set->count);
}

void refview_release_images(RefViewSet* set, SDL_GPUDevice* device) {
    for (uint32_t i = 0; i < set->count; i++) {
        if (set->views[i].texture) {
            SDL_ReleaseGPUTexture(device, set->views[i].texture);
            set->views[i].texture = NULL;
        }
    }
}

static float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// Smooth step for nicer interpolation
static float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

bool refview_update(RefViewSet* set, Camera* cam, float dt) {
    if (!set->lerping || set->selected < 0) return false;

    set->lerp_t += dt / set->lerp_duration;
    if (set->lerp_t >= 1.0f) {
        set->lerp_t = 1.0f;
        set->lerping = false;
    }

    float t = smoothstep(set->lerp_t);
    RefView* target = &set->views[set->selected];

    cam->position[0] = lerpf(set->start_pos[0], target->position[0], t);
    cam->position[1] = lerpf(set->start_pos[1], target->position[1], t);
    cam->position[2] = lerpf(set->start_pos[2], target->position[2], t);
    cam->yaw   = lerpf(set->start_yaw,   target->yaw,   t);
    cam->pitch = lerpf(set->start_pitch,  target->pitch,  t);

    return true;
}

void refview_get_rotation_matrix(const RefView* v, float* m) {
    // Build world-to-camera rotation from colmap quaternion
    float qw = v->rotation[0], qx = v->rotation[1];
    float qy = v->rotation[2], qz = v->rotation[3];

    // R = colmap world-to-camera rotation (colmap world has Y-down)
    // We need R_adjusted = R * diag(1, -1, 1) to account for our Y-up world
    // This negates column 1 of R
    // Output is column-major mat4

    float R[3][3];
    R[0][0] = 1 - 2*(qy*qy + qz*qz);
    R[0][1] = 2*(qx*qy - qw*qz);
    R[0][2] = 2*(qx*qz + qw*qy);
    R[1][0] = 2*(qx*qy + qw*qz);
    R[1][1] = 1 - 2*(qx*qx + qz*qz);
    R[1][2] = 2*(qy*qz - qw*qx);
    R[2][0] = 2*(qx*qz - qw*qy);
    R[2][1] = 2*(qy*qz + qw*qx);
    R[2][2] = 1 - 2*(qx*qx + qy*qy);

    // Column-major mat4, with Y-flip (negate column 1)
    // Column 0
    m[0]  = R[0][0];  m[1]  = R[1][0];  m[2]  = R[2][0];  m[3]  = 0;
    // Column 1 (negated for Y-flip)
    m[4]  = -R[0][1]; m[5]  = -R[1][1]; m[6]  = -R[2][1]; m[7]  = 0;
    // Column 2
    m[8]  = R[0][2];  m[9]  = R[1][2];  m[10] = R[2][2];  m[11] = 0;
    // Column 3
    m[12] = 0;         m[13] = 0;         m[14] = 0;         m[15] = 1;
}

void refview_free(RefViewSet* set) {
    free(set->views);
    memset(set, 0, sizeof(*set));
    set->selected = -1;
}
