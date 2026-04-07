#include "camera.h"

static void cross(const float* a, const float* b, float* out);
static void normalize3(float* v);

void camera_init(Camera* cam) {
    cam->position[0] = 0.0f;
    cam->position[1] = 0.0f;
    cam->position[2] = 0.0f;
    cam->yaw = 0.0f;
    cam->pitch = 0.0f;
    cam->fov_y = 60.0f * (3.14159265358979f / 180.0f);
    cam->near_plane = 0.1f;
    cam->far_plane = 100.0f;
    cam->move_speed = 2.0f;
    cam->look_sensitivity = 0.003f;
    cam->right_mouse_held = false;
}

void camera_get_forward(const Camera* cam, float* out) {
    out[0] = cosf(cam->pitch) * sinf(cam->yaw);
    out[1] = sinf(cam->pitch);
    out[2] = cosf(cam->pitch) * cosf(cam->yaw);
}

void camera_get_overlay_ray_basis(const Camera* cam, float aspect, float* out_mat4, float* out_tan_half_fov) {
    float forward[3], right[3], up[3];
    float world_up[3] = {0, 1, 0};
    camera_get_forward(cam, forward);

    // Build a right-handed horizontal basis for the panorama ray while keeping
    // the existing vertical flip in shader space.
    cross(world_up, forward, right);
    normalize3(right);
    cross(forward, right, up);
    normalize3(up);

    memset(out_mat4, 0, 16 * sizeof(float));
    out_mat4[0]  = right[0];
    out_mat4[1]  = right[1];
    out_mat4[2]  = right[2];
    out_mat4[4]  = up[0];
    out_mat4[5]  = up[1];
    out_mat4[6]  = up[2];
    out_mat4[8]  = forward[0];
    out_mat4[9]  = forward[1];
    out_mat4[10] = forward[2];
    out_mat4[15] = 1.0f;

    float tan_half_fov_y = tanf(cam->fov_y * 0.5f);
    out_tan_half_fov[0] = tan_half_fov_y * aspect;
    out_tan_half_fov[1] = tan_half_fov_y;
}

static void cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void normalize3(float* v) {
    float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-8f) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

void camera_update(Camera* cam, const bool* keys, float dx, float dy, float dt) {
    // keys: 0=W, 1=A, 2=S, 3=D, 4=Space, 5=LCtrl, 6=LShift
    if (cam->right_mouse_held) {
        cam->yaw   += dx * cam->look_sensitivity;
        cam->pitch += dy * cam->look_sensitivity;
        float limit = 3.14159265358979f * 0.5f - 0.01f;
        if (cam->pitch >  limit) cam->pitch =  limit;
        if (cam->pitch < -limit) cam->pitch = -limit;
    }

    float forward[3], right[3];
    camera_get_forward(cam, forward);
    float world_up[3] = {0, 1, 0};
    cross(forward, world_up, right);
    normalize3(right);

    float speed = cam->move_speed * dt;
    if (keys[6]) speed *= 3.0f; // shift

    if (keys[0]) { // W
        cam->position[0] += forward[0] * speed;
        cam->position[1] += forward[1] * speed;
        cam->position[2] += forward[2] * speed;
    }
    if (keys[2]) { // S
        cam->position[0] -= forward[0] * speed;
        cam->position[1] -= forward[1] * speed;
        cam->position[2] -= forward[2] * speed;
    }
    if (keys[1]) { // A
        cam->position[0] += right[0] * speed;
        cam->position[1] += right[1] * speed;
        cam->position[2] += right[2] * speed;
    }
    if (keys[3]) { // D
        cam->position[0] -= right[0] * speed;
        cam->position[1] -= right[1] * speed;
        cam->position[2] -= right[2] * speed;
    }
    if (keys[4]) { // Space - up
        cam->position[1] += speed;
    }
    if (keys[5]) { // LCtrl - down
        cam->position[1] -= speed;
    }
}

// lookAt: column-major output
void camera_get_view_matrix(const Camera* cam, float* m) {
    float forward[3], right[3], up[3];
    camera_get_forward(cam, forward);
    float world_up[3] = {0, 1, 0};
    cross(forward, world_up, right);
    normalize3(right);
    cross(right, forward, up);

    // View matrix = transpose(R) with translation
    // R columns = right, up, -forward
    // Column-major storage
    m[0]  = right[0];
    m[1]  = up[0];
    m[2]  = -forward[0];
    m[3]  = 0;

    m[4]  = right[1];
    m[5]  = up[1];
    m[6]  = -forward[1];
    m[7]  = 0;

    m[8]  = right[2];
    m[9]  = up[2];
    m[10] = -forward[2];
    m[11] = 0;

    m[12] = -(right[0]*cam->position[0] + right[1]*cam->position[1] + right[2]*cam->position[2]);
    m[13] = -(up[0]*cam->position[0] + up[1]*cam->position[1] + up[2]*cam->position[2]);
    m[14] = (forward[0]*cam->position[0] + forward[1]*cam->position[1] + forward[2]*cam->position[2]);
    m[15] = 1;
}

// Perspective projection: column-major, Vulkan clip space (Y-flip, Z [0,1])
void camera_get_proj_matrix(const Camera* cam, float aspect, float* m) {
    float f = 1.0f / tanf(cam->fov_y * 0.5f);
    float n = cam->near_plane;
    float fa = cam->far_plane;

    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = -f;  // Vulkan Y-flip
    m[10] = fa / (n - fa);
    m[11] = -1.0f;
    m[14] = (n * fa) / (n - fa);
}
