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

void refview_free(RefViewSet* set) {
    free(set->views);
    memset(set, 0, sizeof(*set));
    set->selected = -1;
}
