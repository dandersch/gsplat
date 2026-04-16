#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cstdio>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

#include "camera.cpp"
#include "gaussian.cpp"
#include "renderer.cpp"
#include "refview.cpp"

int main(int argc, char* argv[]) {
    const char* ply_path = NULL;
    const char* colmap_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--colmap") == 0 && i + 1 < argc) {
            colmap_dir = argv[++i];
        } else if (!ply_path) {
            ply_path = argv[i];
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, NULL);
    if (!device) {
        fprintf(stderr, "SDL_CreateGPUDevice failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("gsplat", 1280, 720, SDL_WINDOW_RESIZABLE);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    if (!SDL_ClaimWindowForGPUDevice(device, window)) {
        fprintf(stderr, "SDL_ClaimWindowForGPUDevice failed: %s\n", SDL_GetError());
        return 1;
    }

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForOther(window);

    SDL_GPUTextureFormat swapchain_format = SDL_GetGPUSwapchainTextureFormat(device, window);
    ImGui_ImplSDLGPU3_InitInfo imgui_init = {};
    imgui_init.Device = device;
    imgui_init.ColorTargetFormat = swapchain_format;
    ImGui_ImplSDLGPU3_Init(&imgui_init);

    // Renderer
    Renderer renderer = {};
    if (!renderer_init(&renderer, device, window)) {
        fprintf(stderr, "Renderer init failed\n");
        return 1;
    }

    // Scene
    GaussianScene scene = {};
    bool scene_loaded = false;
    if (ply_path) {
        scene_loaded = load_ply(ply_path, &scene);
        if (scene_loaded) {
            renderer_upload_gaussians(&renderer, &scene);
        }
    }

    // Reference views
    RefViewSet refviews = {};
    refviews.selected = -1;
    bool refviews_loaded = false;
    if (colmap_dir) {
        refviews_loaded = refview_load(&refviews, colmap_dir);
        if (refviews_loaded) {
            refview_load_covisibility(&refviews, colmap_dir);
            refview_load_images(&refviews, device);
        }
    }

    // Camera
    Camera cam;
    camera_init(&cam);
    SDL_SetWindowRelativeMouseMode(window, true); // start in camera mode

    bool keys[7] = {}; // W A S D Space LCtrl LShift
    uint64_t last_time = SDL_GetPerformanceCounter();
    uint64_t freq = SDL_GetPerformanceFrequency();
    bool running = true;
    float refview_max_alpha = 0.5f;
    float node_half_size = 0.5f;
    int frame_num = 0;

    // Neighbor scratch buffers
    const uint32_t max_neighbors = 64;
    float neighbor_positions[64 * 3];
    uint32_t neighbor_indices[64];
    uint32_t neighbor_count = 0;

    while (running) {
        uint64_t now = SDL_GetPerformanceCounter();
        float dt = (float)(now - last_time) / (float)freq;
        last_time = now;

        SDL_Event ev;
        float mouse_dx = 0, mouse_dy = 0;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);

            switch (ev.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                bool down = (ev.type == SDL_EVENT_KEY_DOWN);
                switch (ev.key.scancode) {
                    case SDL_SCANCODE_W: keys[0] = down; break;
                    case SDL_SCANCODE_A: keys[1] = down; break;
                    case SDL_SCANCODE_S: keys[2] = down; break;
                    case SDL_SCANCODE_D: keys[3] = down; break;
                    case SDL_SCANCODE_SPACE: keys[4] = down; break;
                    case SDL_SCANCODE_LCTRL: keys[5] = down; break;
                    case SDL_SCANCODE_LSHIFT: keys[6] = down; break;
                    case SDL_SCANCODE_ESCAPE: if (down) running = false; break;
                    default: break;
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (ev.button.button == SDL_BUTTON_RIGHT) {
                    cam.camera_mode = !cam.camera_mode;
                    SDL_SetWindowRelativeMouseMode(window, cam.camera_mode);
                }
                if (ev.button.button == SDL_BUTTON_LEFT && cam.camera_mode &&
                    refviews_loaded && !refviews.lerping) {
                    // Ray from screen center (crosshair) into scene
                    float forward[3];
                    camera_get_forward(&cam, forward);

                    // Test against all neighbor AABBs, pick closest hit
                    float best_t = 1e30f;
                    int best_hit = -1;
                    for (uint32_t ni = 0; ni < neighbor_count; ni++) {
                        const float* center = &neighbor_positions[ni*3];
                        float hs = node_half_size;
                        // Slab method
                        float tmin = -1e30f, tmax = 1e30f;
                        for (int axis = 0; axis < 3; axis++) {
                            float o = cam.position[axis];
                            float d = forward[axis];
                            float bmin = center[axis] - hs;
                            float bmax = center[axis] + hs;
                            if (fabsf(d) < 1e-8f) {
                                if (o < bmin || o > bmax) { tmin = 1e30f; break; }
                            } else {
                                float t1 = (bmin - o) / d;
                                float t2 = (bmax - o) / d;
                                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                                if (t1 > tmin) tmin = t1;
                                if (t2 < tmax) tmax = t2;
                                if (tmin > tmax) { tmin = 1e30f; break; }
                            }
                        }
                        if (tmin < best_t && tmax > 0.0f) {
                            best_t = tmin;
                            best_hit = (int)ni;
                        }
                    }

                    if (best_hit >= 0) {
                        uint32_t view_idx = neighbor_indices[best_hit];
                        RefView* tv = &refviews.views[view_idx];
                        float dx = tv->position[0] - cam.position[0];
                        float dy = tv->position[1] - cam.position[1];
                        float dz = tv->position[2] - cam.position[2];
                        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                        refviews.selected = (int32_t)view_idx;
                        refviews.lerping = true;
                        refviews.lerp_t = 0.0f;
                        refviews.lerp_duration = (dist > 1e-6f) ? dist / refviews.lerp_speed : 0.1f;
                        refviews.start_pos[0] = cam.position[0];
                        refviews.start_pos[1] = cam.position[1];
                        refviews.start_pos[2] = cam.position[2];
                        refviews.start_yaw = cam.yaw;
                        refviews.start_pitch = cam.pitch;
                    }
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (cam.camera_mode) {
                    mouse_dx += ev.motion.xrel;
                    mouse_dy += ev.motion.yrel;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (!ImGui::GetIO().WantCaptureMouse) {
                    cam.move_speed *= (ev.wheel.y > 0) ? 1.2f : (1.0f / 1.2f);
                    if (cam.move_speed < 0.1f) cam.move_speed = 0.1f;
                    if (cam.move_speed > 100.0f) cam.move_speed = 100.0f;
                }
                break;
            }
        }

        // Update reference view interpolation (locks camera input while active)
        bool camera_locked = refview_update(&refviews, &cam, dt);

        // Update camera (allow mouse look during lerp, but block WASD movement)
        if (camera_locked) {
            camera_update(&cam, keys, mouse_dx, mouse_dy, 0);
        } else if (cam.camera_mode || !ImGui::GetIO().WantCaptureKeyboard) {
            camera_update(&cam, keys, mouse_dx, mouse_dy, dt);
        } else {
            camera_update(&cam, keys, mouse_dx, mouse_dy, 0);
        }

        // Get window size
        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);
        float aspect = (float)win_w / (float)win_h;

        // Build camera uniforms
        CameraUniforms cam_uniforms = {};
        camera_get_view_matrix(&cam, cam_uniforms.view);
        camera_get_proj_matrix(&cam, aspect, cam_uniforms.proj);
        cam_uniforms.viewport[0] = (float)win_w;
        cam_uniforms.viewport[1] = (float)win_h;

        // Cull + sort
        if (scene_loaded) {
            cull_gaussians(&scene, cam_uniforms.view, cam_uniforms.proj);

            if (scene.visible_count > 0) {
                SortContext sort_ctx = {};
                sort_ctx.depths = scene.visible_depths;
                sort_ctx.input_indices = scene.visible_indices;
                sort_ctx.count = scene.visible_count;
                sort_ctx.sorted_indices = scene.sorted_indices;
                sort_ctx.scratch_indices = scene.scratch_indices;
                sort_ctx.scratch_keys = scene.scratch_keys;
                sort_ctx.scratch_keys2 = scene.scratch_keys2;
                sort_gaussians(&sort_ctx);
            }
        }

        // ImGui frame
        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Info");
        ImGui::Text("FPS: %.1f", dt > 0 ? 1.0f / dt : 0.0f);
        if (scene_loaded) {
            ImGui::Text("Visible: %u / %u", scene.visible_count, scene.gaussian_count);
        }
        ImGui::Text("Camera: %.1f, %.1f, %.1f", cam.position[0], cam.position[1], cam.position[2]);
        ImGui::Text("Speed: %.1f", cam.move_speed);
        float fov_deg = cam.fov_y * (180.0f / 3.14159265358979f);
        if (ImGui::SliderFloat("FOV", &fov_deg, 10.0f, 170.0f, "%.0f°")) {
            cam.fov_y = fov_deg * (3.14159265358979f / 180.0f);
        }
        if (refviews_loaded) {
            ImGui::SliderFloat("Ref View Opacity", &refview_max_alpha, 0.0f, 1.0f);
            ImGui::Checkbox("Use Covisibility", &refviews.use_covisibility);
            if (refviews.use_covisibility) {
                ImGui::SliderInt("Min Inliers", &refviews.min_inliers, 0, 500);
            } else {
                ImGui::SliderFloat("Neighbor Radius", &refviews.neighbor_radius, 0.5f, 10.0f);
            }
            ImGui::SliderFloat("Node Box Size", &node_half_size, 0.1f, 1.0f);
            ImGui::SliderFloat("Transition Speed", &refviews.lerp_speed, 1.0f, 10.0f);
            if (refviews.current_node >= 0) {
                ImGui::Text("Current Node: %d", refviews.current_node);
                ImGui::Text("Neighbors: %u", neighbor_count);
            }
        }
        ImGui::End();

        if (refviews_loaded) {
            ImGui::Begin("Reference Views");
            for (uint32_t i = 0; i < refviews.count; i++) {
                char label[32];
                snprintf(label, sizeof(label), "%u", i);
                bool is_selected = ((int32_t)i == refviews.selected);
                if (ImGui::Selectable(label, is_selected)) {
                    RefView* tv = &refviews.views[i];
                    float dx = tv->position[0] - cam.position[0];
                    float dy = tv->position[1] - cam.position[1];
                    float dz = tv->position[2] - cam.position[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    refviews.selected = i;
                    refviews.lerping = true;
                    refviews.lerp_t = 0.0f;
                    refviews.lerp_duration = (dist > 1e-6f) ? dist / refviews.lerp_speed : 0.1f;
                    refviews.start_pos[0] = cam.position[0];
                    refviews.start_pos[1] = cam.position[1];
                    refviews.start_pos[2] = cam.position[2];
                    refviews.start_yaw = cam.yaw;
                    refviews.start_pitch = cam.pitch;
                }
            }
            ImGui::End();
        }

        // Draw crosshair in camera mode (highlight when aiming at a node)
        if (cam.camera_mode) {
            bool crosshair_hover = false;
            if (refviews_loaded && !refviews.lerping && neighbor_count > 0) {
                float forward[3];
                camera_get_forward(&cam, forward);
                for (uint32_t ni = 0; ni < neighbor_count; ni++) {
                    const float* c = &neighbor_positions[ni*3];
                    float hs = node_half_size;
                    float tmin = -1e30f, tmax = 1e30f;
                    for (int axis = 0; axis < 3; axis++) {
                        float o = cam.position[axis];
                        float d = forward[axis];
                        float bmin = c[axis] - hs, bmax = c[axis] + hs;
                        if (fabsf(d) < 1e-8f) {
                            if (o < bmin || o > bmax) { tmin = 1e30f; break; }
                        } else {
                            float t1 = (bmin - o) / d, t2 = (bmax - o) / d;
                            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                            if (t1 > tmin) tmin = t1;
                            if (t2 < tmax) tmax = t2;
                            if (tmin > tmax) { tmin = 1e30f; break; }
                        }
                    }
                    if (tmax > 0.0f && tmin < 1e30f) { crosshair_hover = true; break; }
                }
            }

            ImDrawList* dl = ImGui::GetForegroundDrawList();
            ImVec2 center(win_w * 0.5f, win_h * 0.5f);
            if (crosshair_hover) {
                dl->AddCircleFilled(center, 5.0f, IM_COL32(0, 200, 255, 240));
                dl->AddCircle(center, 8.0f, IM_COL32(0, 200, 255, 120), 0, 1.5f);
            } else {
                dl->AddCircleFilled(center, 3.0f, IM_COL32(255, 255, 255, 200));
            }
        }

        ImGui::Render();

        // Find closest refview node to camera (used for overlay + current_node tracking)
        OverlayParams overlay = {};
        OverlayParams* overlay_ptr = NULL;
        if (refviews_loaded) {
            float best_dist2 = 1e30f;
            int best_idx = -1;
            for (uint32_t i = 0; i < refviews.count; i++) {
                if (!refviews.views[i].texture) continue;
                float dx = cam.position[0] - refviews.views[i].position[0];
                float dy = cam.position[1] - refviews.views[i].position[1];
                float dz = cam.position[2] - refviews.views[i].position[2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best_dist2) { best_dist2 = d2; best_idx = (int)i; }
            }

            refviews.current_node = best_idx;

            if (best_idx >= 0 && refview_max_alpha > 0.0f) {
                RefView* rv = &refviews.views[best_idx];
                float dist = sqrtf(best_dist2);
                float fade_dist = 0.5f;
                float alpha = 1.0f - dist / fade_dist;
                if (alpha < 0.0f) alpha = 0.0f;
                if (alpha > refview_max_alpha) alpha = refview_max_alpha;

                if (alpha > 0.0f) {
                    overlay.texture = rv->texture;
                    overlay.alpha = alpha;

                    camera_get_overlay_ray_basis(&cam, (float)win_w / (float)win_h,
                                                 overlay.camera_ray_basis,
                                                 overlay.camera_tan_half_fov);

                    refview_get_rotation_matrix(rv, overlay.ref_rotation);
                    overlay_ptr = &overlay;
                }
            }

            // Collect neighbor nodes for wireframe rendering + click targets
            neighbor_count = refview_get_neighbors(&refviews, neighbor_positions, neighbor_indices, max_neighbors);
        }

        // Build node render params
        NodeRenderParams node_params = {};
        NodeRenderParams* node_ptr = NULL;
        if (refviews_loaded && neighbor_count > 0) {
            node_params.positions = neighbor_positions;
            node_params.count = neighbor_count;
            node_params.half_size = node_half_size;
            node_ptr = &node_params;
        }

        // Render
        renderer_draw_frame(&renderer, &scene, &cam_uniforms, overlay_ptr, node_ptr);
        frame_num++;
    }

    SDL_WaitForGPUIdle(device);

    if (scene_loaded) free_scene(&scene);
    if (refviews_loaded) {
        refview_release_images(&refviews, device);
        refview_free(&refviews);
    }
    renderer_destroy(&renderer);

    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
    SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}
