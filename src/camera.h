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
    bool  right_mouse_held;
};

struct CameraUniforms {
    float view[16];
    float proj[16];
    float viewport[2];
    float pad[2];
};

void camera_init(Camera* cam);
void camera_update(Camera* cam, const bool* keys, float dx, float dy, float dt);
void camera_get_view_matrix(const Camera* cam, float* out);
void camera_get_proj_matrix(const Camera* cam, float aspect, float* out);
void camera_get_forward(const Camera* cam, float* out);
