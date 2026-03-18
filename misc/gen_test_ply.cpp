#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>

struct TestGaussian {
    float x, y, z;
    float nx, ny, nz;
    float f_dc_0, f_dc_1, f_dc_2;
    float opacity;
    float scale_0, scale_1, scale_2;
    float rot_0, rot_1, rot_2, rot_3;
};

int main() {
    const int N = 5;
    TestGaussian gs[N];
    memset(gs, 0, sizeof(gs));

    // Place 5 gaussians in a cross pattern at z=3 (in front of default camera looking +Z)
    // Red center
    gs[0] = { 0, 0, 3,  0,0,1,  1.77f, -1.77f, -1.77f,  2.0f,  -3.0f, -3.0f, -3.0f,  1,0,0,0 };
    // Green right
    gs[1] = { 1, 0, 3,  0,0,1,  -1.77f, 1.77f, -1.77f,  2.0f,  -3.0f, -3.0f, -3.0f,  1,0,0,0 };
    // Blue left
    gs[2] = { -1, 0, 3,  0,0,1,  -1.77f, -1.77f, 1.77f,  2.0f,  -3.0f, -3.0f, -3.0f,  1,0,0,0 };
    // Yellow up
    gs[3] = { 0, 1, 3,  0,0,1,  1.77f, 1.77f, -1.77f,  2.0f,  -3.0f, -3.0f, -3.0f,  1,0,0,0 };
    // Cyan down
    gs[4] = { 0, -1, 3,  0,0,1,  -1.77f, 1.77f, 1.77f,  2.0f,  -3.0f, -3.0f, -3.0f,  1,0,0,0 };
    // Note: f_dc values are raw SH coefficients. color = 0.282 * f_dc + 0.5
    //   1.77  -> 0.282*1.77+0.5 = 1.0 (red)
    //  -1.77  -> 0.282*(-1.77)+0.5 = 0.0
    // opacity 2.0 -> sigmoid(2.0) = 0.88
    // scale -3.0 -> exp(-3.0) = 0.05

    FILE* f = fopen("test.ply", "wb");
    fprintf(f, "ply\n");
    fprintf(f, "format binary_little_endian 1.0\n");
    fprintf(f, "element vertex %d\n", N);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "property float nx\n");
    fprintf(f, "property float ny\n");
    fprintf(f, "property float nz\n");
    fprintf(f, "property float f_dc_0\n");
    fprintf(f, "property float f_dc_1\n");
    fprintf(f, "property float f_dc_2\n");
    fprintf(f, "property float opacity\n");
    fprintf(f, "property float scale_0\n");
    fprintf(f, "property float scale_1\n");
    fprintf(f, "property float scale_2\n");
    fprintf(f, "property float rot_0\n");
    fprintf(f, "property float rot_1\n");
    fprintf(f, "property float rot_2\n");
    fprintf(f, "property float rot_3\n");
    fprintf(f, "end_header\n");
    fwrite(gs, sizeof(TestGaussian), N, f);
    fclose(f);

    printf("Wrote test.ply with %d gaussians\n", N);
    return 0;
}
