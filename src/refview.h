#pragma once
#include <SDL3/SDL.h>
#include "camera.h"
#include <cstdint>

struct RefView {
    char    image_name[256];
    float   position[3];       // world-space camera center (-R^T * T)
    float   rotation[4];       // quaternion (w,x,y,z) from colmap
    float   yaw, pitch;        // derived from rotation for lerp target
    SDL_GPUTexture* texture;   // NULL until image loaded
    int     width, height;
};

struct RefViewSet {
    RefView* views;
    uint32_t count;
    int32_t  selected;         // -1 = none
    char     image_dir[512];   // resolved path to images/

    // lerp state
    bool     lerping;
    float    lerp_t;
    float    lerp_duration;
    float    start_pos[3];
    float    start_yaw, start_pitch;
};

// Parse colmap images.txt from colmap_dir. Derives image_dir as ../../images/ relative to colmap_dir.
bool refview_load(RefViewSet* set, const char* colmap_dir);

// Load images as GPU textures. Call after refview_load.
void refview_load_images(RefViewSet* set, SDL_GPUDevice* device);

// Release GPU textures.
void refview_release_images(RefViewSet* set, SDL_GPUDevice* device);

// Advance interpolation, write into cam. Returns true while lerping (camera locked).
bool refview_update(RefViewSet* set, Camera* cam, float dt);

// Build rotation matrix (with Y-flip) from colmap quaternion into a column-major mat4.
void refview_get_rotation_matrix(const RefView* v, float* out_mat4);

void refview_free(RefViewSet* set);
