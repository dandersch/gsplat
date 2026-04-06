#!/bin/bash
set -e

CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers"
LDFLAGS="-lSDL3 -lm"
OUT="gsplat"

IMGUI_DIR="third_party/imgui"

echo "Compiling shaders..."
glslc -fshader-stage=vertex shaders/splat.vert.glsl -o shaders/splat.vert.spv
glslc -fshader-stage=fragment shaders/splat.frag.glsl -o shaders/splat.frag.spv
glslc -fshader-stage=vertex shaders/overlay.vert.glsl -o shaders/overlay.vert.spv
glslc -fshader-stage=fragment shaders/overlay.frag.glsl -o shaders/overlay.frag.spv

SRCS=(
    src/main.cpp
    src/gaussian.cpp
    src/renderer.cpp
    src/camera.cpp
    src/refview.cpp
    "$IMGUI_DIR/imgui.cpp"
    "$IMGUI_DIR/imgui_draw.cpp"
    "$IMGUI_DIR/imgui_tables.cpp"
    "$IMGUI_DIR/imgui_widgets.cpp"
    "$IMGUI_DIR/backends/imgui_impl_sdl3.cpp"
    "$IMGUI_DIR/backends/imgui_impl_sdlgpu3.cpp"
)

echo "Building $OUT..."
$CXX $CXXFLAGS -I"$IMGUI_DIR" -I"$IMGUI_DIR/backends" "${SRCS[@]}" -o "$OUT" $LDFLAGS

echo "Done: ./$OUT"
