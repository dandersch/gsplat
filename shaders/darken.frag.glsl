#version 450

layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 out_color;

// Constant dark overlay used to dim the FPS view behind the top-down map.
// Uses the same premultiplied-alpha blend as the panorama overlay
// (src=ONE, dst=ONE_MINUS_SRC_ALPHA), which yields:
//   final.rgb = bg.rgb * (1 - alpha)
// i.e. simply scales the underlying FPS pixels toward black.
void main() {
    const float alpha = 0.6; // 0 = no darken, 1 = fully black
    out_color = vec4(0.0, 0.0, 0.0, alpha);
}
