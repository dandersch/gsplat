#include "mesh.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <SDL3/SDL.h>

bool mesh_load_obj(const char* obj_path, Mesh* mesh) {
    memset(mesh, 0, sizeof(*mesh));

    // Extract directory from obj_path for MTL/texture lookup
    std::string obj_str(obj_path);
    std::string base_dir;
    size_t last_slash = obj_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        base_dir = obj_str.substr(0, last_slash + 1);
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                               obj_path, base_dir.empty() ? NULL : base_dir.c_str());

    if (!warn.empty()) fprintf(stderr, "OBJ warning: %s\n", warn.c_str());
    if (!err.empty())  fprintf(stderr, "OBJ error: %s\n", err.c_str());
    if (!ok) return false;

    // Deduplicate vertices: OBJ has separate indices for pos/uv/normal,
    // but the GPU wants a single index buffer into interleaved vertices.
    struct VertexKey {
        int v, vt;
        bool operator<(const VertexKey& o) const {
            if (v != o.v) return v < o.v;
            return vt < o.vt;
        }
    };

    std::vector<float> verts;       // interleaved pos3 + uv2
    std::vector<uint32_t> indices;
    std::map<VertexKey, uint32_t> vertex_map;

    for (auto& shape : shapes) {
        for (auto& idx : shape.mesh.indices) {
            VertexKey key = { idx.vertex_index, idx.texcoord_index };
            auto it = vertex_map.find(key);
            if (it != vertex_map.end()) {
                indices.push_back(it->second);
            } else {
                uint32_t new_idx = (uint32_t)(verts.size() / 5);
                vertex_map[key] = new_idx;

                // Position (negate Y: OBJ is Y-up, renderer is Y-down)
                verts.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                verts.push_back(-attrib.vertices[3 * idx.vertex_index + 1]);
                verts.push_back(attrib.vertices[3 * idx.vertex_index + 2]);

                // UV (default to 0,0 if missing)
                if (idx.texcoord_index >= 0) {
                    verts.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                    verts.push_back(1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]); // flip V
                } else {
                    verts.push_back(0.0f);
                    verts.push_back(0.0f);
                }

                indices.push_back(new_idx);
            }
        }
    }

    mesh->vertex_count = (uint32_t)(verts.size() / 5);
    mesh->index_count  = (uint32_t)indices.size();

    mesh->vertices = (float*)malloc(verts.size() * sizeof(float));
    memcpy(mesh->vertices, verts.data(), verts.size() * sizeof(float));

    mesh->indices = (uint32_t*)malloc(indices.size() * sizeof(uint32_t));
    memcpy(mesh->indices, indices.data(), indices.size() * sizeof(uint32_t));

    // Load texture from first material that has a diffuse map
    for (auto& mat : materials) {
        if (mat.diffuse_texname.empty()) continue;

        std::string tex_path = base_dir + mat.diffuse_texname;
        SDL_Surface* surface = SDL_LoadBMP(tex_path.c_str());
        if (!surface) surface = SDL_LoadPNG(tex_path.c_str());
        if (!surface) {
            fprintf(stderr, "Could not load texture: %s\n", tex_path.c_str());
            continue;
        }

        SDL_Surface* rgba = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ABGR8888);
        SDL_DestroySurface(surface);
        if (!rgba) continue;

        mesh->tex_w = rgba->w;
        mesh->tex_h = rgba->h;
        uint32_t tex_size = rgba->w * rgba->h * 4;
        mesh->tex_rgba = (uint8_t*)malloc(tex_size);
        memcpy(mesh->tex_rgba, rgba->pixels, tex_size);
        SDL_DestroySurface(rgba);
        break;
    }

    fprintf(stderr, "Loaded OBJ: %u verts, %u indices, %zu materials\n",
            mesh->vertex_count, mesh->index_count, materials.size());

    return true;
}

void mesh_free(Mesh* mesh) {
    free(mesh->vertices);
    free(mesh->indices);
    free(mesh->tex_rgba);
    memset(mesh, 0, sizeof(*mesh));
}
