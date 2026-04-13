#!/bin/bash
set -e

CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers"
LDFLAGS="-lSDL3 -lm"
OUT="gsplat"

IMGUI_DIR="third_party/imgui"
IMGUI_LIB="third_party/libimgui.a"

echo "Compiling shaders..."
glslc -fshader-stage=vertex shaders/splat.vert.glsl -o shaders/splat.vert.spv
glslc -fshader-stage=fragment shaders/splat.frag.glsl -o shaders/splat.frag.spv
glslc -fshader-stage=vertex shaders/overlay.vert.glsl -o shaders/overlay.vert.spv
glslc -fshader-stage=fragment shaders/overlay.frag.glsl -o shaders/overlay.frag.spv

# Build imgui static lib if missing
if [ ! -f "$IMGUI_LIB" ]; then
    echo "Building imgui..."
    IMGUI_OBJS=()
    for src in "$IMGUI_DIR"/imgui.cpp "$IMGUI_DIR"/imgui_draw.cpp \
               "$IMGUI_DIR"/imgui_tables.cpp "$IMGUI_DIR"/imgui_widgets.cpp \
               "$IMGUI_DIR"/backends/imgui_impl_sdl3.cpp \
               "$IMGUI_DIR"/backends/imgui_impl_sdlgpu3.cpp; do
        obj="${src%.cpp}.o"
        $CXX $CXXFLAGS -I"$IMGUI_DIR" -I"$IMGUI_DIR/backends" -c "$src" -o "$obj"
        IMGUI_OBJS+=("$obj")
    done
    ar rcs "$IMGUI_LIB" "${IMGUI_OBJS[@]}"
    rm "${IMGUI_OBJS[@]}"
fi

echo "Building $OUT..."
$CXX $CXXFLAGS -I"$IMGUI_DIR" -I"$IMGUI_DIR/backends" src/main.cpp -o "$OUT" "$IMGUI_LIB" $LDFLAGS

echo "Done: ./$OUT"
