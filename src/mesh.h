#pragma once
#include <cstdint>

struct MeshTexture {
    uint8_t* rgba;     // RGBA8 pixels
    uint32_t w, h;
};

struct MeshSubmesh {
    uint32_t index_offset;
    uint32_t index_count;
    int32_t  texture_id;   // index into Mesh::textures, -1 if no texture
};

struct Mesh {
    float*       vertices;     // interleaved: vec3 pos + vec2 uv per vertex
    uint32_t*    indices;
    uint32_t     vertex_count;
    uint32_t     index_count;
    MeshSubmesh* submeshes;
    uint32_t     submesh_count;
    MeshTexture* textures;
    uint32_t     texture_count;
};

bool mesh_load_obj(const char* obj_path, Mesh* mesh);
void mesh_free(Mesh* mesh);
