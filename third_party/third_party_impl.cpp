// Single translation unit that instantiates the implementations of the
// single-header libraries in third_party/. Built once into libthirdparty.a
// and cached so we don't pay the parse cost on every incremental build.

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
