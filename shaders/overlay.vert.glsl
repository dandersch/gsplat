#version 450

layout(location = 0) out vec2 v_ndc;

void main() {
    // Fullscreen triangle from vertex index (3 vertices cover entire screen)
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_ndc = pos * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
