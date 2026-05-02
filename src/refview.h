#pragma once
#include <SDL3/SDL.h>
#include "camera.h"
#include "hotspot.h"
#include <cstdint>

struct RefView {
    char    image_name[256];
    int     colmap_id;         // original IMAGE_ID from colmap
    float   position[3];       // world-space camera center (-R^T * T)
    float   rotation[4];       // quaternion (w,x,y,z) from colmap
    float   yaw, pitch;        // derived from rotation for lerp target
    SDL_GPUTexture* texture;   // NULL until image loaded
    int     width, height;

    // Authored clickable regions on this view's panorama. NULL until a
    // sidecar is loaded via hotspot_load_for_set.
    Hotspot* hotspots;
    uint32_t hotspot_count;
};

struct CovisEdge {
    uint32_t idx_a, idx_b;     // internal RefView indices
    uint32_t inliers;          // geometrically verified inlier matches
};

struct RefViewSet {
    RefView* views;
    uint32_t count;
    int32_t  selected;         // -1 = none
    int32_t  current_node;     // nearest node to camera (updated each frame externally)
    char     image_dir[512];   // resolved path to images/

    // lerp state
    bool     lerping;
    float    lerp_t;
    float    lerp_duration;    // computed per-transition from distance / speed
    float    lerp_speed;       // world units per second
    float    start_pos[3];
    float    start_yaw, start_pitch;

    // neighbor discovery
    float    neighbor_radius;  // only show nodes within this distance of current_node

    // covisibility graph (from colmap database.db)
    CovisEdge* covis_edges;
    uint32_t   covis_edge_count;
    int        min_inliers;        // threshold: minimum inlier count to consider connected
    bool       use_covisibility;   // true = covis graph, false = distance-based
};

// Parse colmap images.txt from colmap_dir. Derives image_dir as ../../images/ relative to colmap_dir.
bool refview_load(RefViewSet* set, const char* colmap_dir);

// Load covisibility graph from colmap database.db. Call after refview_load.
// Falls back to distance-based neighbors if db not found.
void refview_load_covisibility(RefViewSet* set, const char* colmap_dir);

// Load images as GPU textures. Call after refview_load.
void refview_load_images(RefViewSet* set, SDL_GPUDevice* device);

// Release GPU textures.
void refview_release_images(RefViewSet* set, SDL_GPUDevice* device);

// Advance interpolation, write into cam. Returns true while lerping (camera locked).
bool refview_update(RefViewSet* set, Camera* cam, float dt);

// Collect neighbor node positions (within neighbor_radius of current_node).
// Writes up to max_count positions (float[3] each) and refview indices into out arrays.
// Returns actual count written.
uint32_t refview_get_neighbors(const RefViewSet* set, float* out_positions, uint32_t* out_indices, uint32_t max_count);

// Build rotation matrix (with Y-flip) from colmap quaternion into a column-major mat4.
void refview_get_rotation_matrix(const RefView* v, float* out_mat4);

void refview_free(RefViewSet* set);
