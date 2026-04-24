#pragma once
#include <cstdint>

struct Mesh {
    float*    vertices;     // interleaved: vec3 pos + vec2 uv per vertex
    uint32_t* indices;
    uint32_t  vertex_count;
    uint32_t  index_count;
    uint8_t*  tex_rgba;     // RGBA8 texture pixels (NULL if no texture)
    uint32_t  tex_w, tex_h;
};

bool mesh_load_obj(const char* obj_path, Mesh* mesh);
void mesh_free(Mesh* mesh);
