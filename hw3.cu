// 優化後的 CUDA 程式
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#define pi 3.1415926535897932384626433832795f // 使用 float

// 使用 float 取代 double
typedef glm::vec2 vec2; 
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat3 mat3;

// Device constants (切換為 float)
__constant__ float d_power = 8.0f;
__constant__ float d_md_iter = 24.0f;
__constant__ float d_step_limiter = 0.2f;
__constant__ float d_ray_multiplier = 0.1f;
__constant__ float d_bailout = 2.0f;
__constant__ float d_eps = 0.0005f;
__constant__ float d_FOV = 1.5f;
__constant__ float d_far_plane = 100.0f;
__constant__ int d_ray_step = 10000;
__constant__ int d_shadow_step = 1500;
__constant__ int d_AA = 3;

__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ vec2 d_iResolution;

// Device function: Mandelbulb distance estimation (使用 float)
__device__ float md(vec3 p, float& trap) {
    vec3 v = p;
    float dr = 1.0f;
    float r = glm::length(v);
    trap = r;

    for (int i = 0; i < d_md_iter; ++i) {
        // glm::atan, asin 等函數會自動使用 float 版本
        float theta = glm::atan(v.y, v.x) * d_power;
        float phi = glm::asin(v.z / r) * d_power;
        dr = d_power * glm::pow(r, d_power - 1.0f) * dr + 1.0f;
        v = p + glm::pow(r, d_power) *
                vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));

        trap = glm::min(trap, r);
        r = glm::length(v);
        if (r > d_bailout) break;
    }
    return 0.5f * log(r) * r / dr;
}

// Device function: Scene mapping (使用 float)
__device__ float map(vec3 p, float& trap, int& ID) {
    vec2 rt = vec2(cos(pi / 2.0f), sin(pi / 2.0f));
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * p;
    ID = 1;
    return md(rp, trap);
}

__device__ float map(vec3 p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Device function: Palette (使用 float)
__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2.0f * pi * (c * t + d));
}

// Device function: Soft shadow (使用 float)
__device__ float softshadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0f;
    float t = 0.0f;
    for (int i = 0; i < d_shadow_step; ++i) {
        float h = map(ro + rd * t);
        res = glm::min(res, k * h / t);
        if (res < 0.02f) return 0.02f;
        t += glm::clamp(h, 0.001f, d_step_limiter);
    }
    return glm::clamp(res, 0.02f, 1.0f);
}

// Device function: Calculate normal (使用 float)
__device__ vec3 calcNor(vec3 p) {
    vec2 e = vec2(d_eps, 0.0f);
    return normalize(vec3(
        map(p + e.xyy()) - map(p - e.xyy()),
        map(p + e.yxy()) - map(p - e.yxy()),
        map(p + e.yyx()) - map(p - e.yyx())
    ));
}

// Device function: Ray tracing (使用 float)
__device__ float trace(vec3 ro, vec3 rd, float& trap, int& ID) {
    float t = 0.0f;
    float len = 0.0f;

    for (int i = 0; i < d_ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (glm::abs(len) < d_eps || t > d_far_plane) break;
        t += len * d_ray_multiplier;
    }
    return t < d_far_plane ? t : -1.0f;
}

// CUDA Kernel: Render each pixel
__global__ void renderKernel(unsigned char* image, int width, int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x coordinate (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y coordinate (height)

    if (i >= height || j >= width) return;

    // 使用 float
    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;

    // Anti-aliasing loop
    for (int m = 0; m < d_AA; ++m) {
        for (int n = 0; n < d_AA; ++n) {
            vec2 p = vec2(j, i) + vec2(m, n) / (float)d_AA;

            // Screen space to normalized coordinates
            vec2 uv = (-d_iResolution.xy() + 2.0f * p) / d_iResolution.y;
            uv.y *= -1.0f;

            // Create camera
            vec3 ro = d_camera_pos;
            vec3 ta = d_target_pos;
            vec3 cf = glm::normalize(ta - ro);
            vec3 cs = glm::normalize(glm::cross(cf, vec3(0.f, 1.f, 0.f)));
            vec3 cu = glm::normalize(glm::cross(cs, cf));
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + d_FOV * cf);

            // Ray marching
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);

            // Lighting
            vec3 col(0.0f);
            vec3 sd = glm::normalize(d_camera_pos);
            vec3 sc = vec3(1.f, 0.9f, 0.717f);

            // Coloring
            if (d < 0.0f) {
                col = vec3(0.0f);  // Sky
            } else {
                vec3 pos = ro + rd * d;
                vec3 nr = calcNor(pos);
                vec3 hal = glm::normalize(sd - rd);

                col = pal(trap - 0.4f, vec3(0.5f), vec3(0.5f), vec3(1.0f), vec3(0.0f, 0.1f, 0.2f));
                vec3 ambc = vec3(0.3f);
                float gloss = 32.0f;

                float amb = (0.7f + 0.3f * nr.y) *
                             (0.2f + 0.8f * glm::clamp(0.05f * log(trap), 0.0f, 1.0f));
                float sdw = softshadow(pos + 0.001f * nr, sd, 16.0f);
                float dif = glm::clamp(glm::dot(sd, nr), 0.0f, 1.0f) * sdw;
                float spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0.0f, 1.0f), gloss) * dif;

                vec3 lin(0.0f);
                lin += ambc * (0.05f + 0.95f * amb);
                lin += sc * dif * 0.8f;
                col *= lin;

                col = glm::pow(col, vec3(0.7f, 0.9f, 1.0f));
                col += spe * 0.8f;
            }

            col = glm::clamp(glm::pow(col, vec3(0.4545f)), 0.0f, 1.0f);
            fcol_r += col.r;
            fcol_g += col.g;
            fcol_b += col.b;
        }
    }

    // Average and convert to 0-255
    float f_AA = (float)(d_AA * d_AA);
    unsigned char r = (unsigned char)(fcol_r / f_AA * 255.0f);
    unsigned char g = (unsigned char)(fcol_g / f_AA * 255.0f);
    unsigned char b = (unsigned char)(fcol_b / f_AA * 255.0f);
    unsigned char a = 255;

    // *** 優化點 2: 合併寫入 ***
    // 將 RGBA 打包成一個 32-bit 整數
    // lodepng_encode32_file 期待的記憶體順序是 R, G, B, A。
    // 在 Little-Endian 系統 (x86, NVIDIA GPU) 上，這會正確寫入記憶體。
    uint32_t rgba_packed = (a << 24) | (b << 16) | (g << 8) | r;
    
    // 計算 1D 索引
    int pixel_idx = i * width + j;

    // 將 unsigned char* 轉型為 uint32_t* 並執行一次 4-byte 寫入
    // 這是 100% Coalesced Store
    ((uint32_t*)image)[pixel_idx] = rgba_packed;
}

// Save image to PNG (不變)
void write_png(const char* filename, unsigned char* image, int width, int height) {
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    assert(argc == 10);

    // Parse arguments (使用 float)
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int width = atoi(argv[7]);
    int height = atoi(argv[8]);
    vec2 iResolution = vec2(width, height);

    // Copy constants to device (自動處理 float)
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));

    // Allocate host memory
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    unsigned char* h_image = new unsigned char[image_size];

    // Allocate device memory
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    printf("Rendering with %dx%d blocks of %dx%d threads\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    renderKernel<<<gridDim, blockDim>>>(d_image, width, height);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    // Save image
    write_png(argv[9], h_image, width, height);

    // Cleanup
    cudaFree(d_image);
    delete[] h_image;

    return 0;
}