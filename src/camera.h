#pragma once
#include <cmath>
#include <cstring>

struct Camera {
    float position[3];
    float yaw;
    float pitch;
    float fov_y;
    float near_plane;
    float far_plane;
    float move_speed;
    float look_sensitivity;
    bool  camera_mode;
    bool  orthographic;
    float ortho_size;
    float ortho_blend; // 0.0 = perspective, 1.0 = orthographic, intermediate = transitioning
};

struct CameraUniforms {
    float view[16];
    float proj[16];
    float viewport[2];
    float orthographic; // ORTHO: 1.0 = orthographic, 0.0 = perspective
    float pad[1];
};

void camera_init(Camera* cam);
void camera_update(Camera* cam, const bool* keys, float dx, float dy, float dt);
void camera_get_view_matrix(const Camera* cam, float* out);
void camera_get_proj_matrix(const Camera* cam, float aspect, float* out);
void camera_get_forward(const Camera* cam, float* out);
void camera_get_overlay_ray_basis(const Camera* cam, float aspect, float* out_mat4, float* out_tan_half_fov);
