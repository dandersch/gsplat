#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in float frag_opacity;
layout(location = 2) in vec2 frag_center;
layout(location = 3) in vec3 frag_conic;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 d = gl_FragCoord.xy - frag_center;

    float power = -0.5 * (
        frag_conic.x * d.x * d.x +
        2.0 * frag_conic.y * d.x * d.y +
        frag_conic.z * d.y * d.y
    );

    // Numerical guard: power should always be <= 0
    if (power > 0.0) discard;

    float alpha = min(frag_opacity * exp(power), 0.99);

    // Skip near-invisible fragments
    if (alpha < 1.0 / 255.0) discard;

    // Premultiplied alpha output
    out_color = vec4(frag_color * alpha, alpha);
}
