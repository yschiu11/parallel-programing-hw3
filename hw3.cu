// 優化後的 CUDA 程式 - 動態負載平衡 + 數學運算優化
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#define pi 3.1415926535897932384626433832795f

typedef glm::vec2 vec2; 
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat3 mat3;

// Device constants
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

// ===== 優化 1: 動態 Tile 分配 =====
#define TILE_SIZE 16
__device__ unsigned int g_tileCounter = 0;

// ===== 優化 2: 數學運算優化的 Mandelbulb =====
__device__ float md(vec3 p, float& trap) {
    vec3 v = p;
    float dr = 1.0f;
    float r = glm::length(v);
    trap = r;

    for (int i = 0; i < (int)d_md_iter; ++i) {
        if (r == 0.0f) break;
        
        // 優化: 使用 sincos 同時計算 sin 和 cos
        float theta = glm::atan(v.y, v.x) * d_power;
        float phi = glm::asin(glm::clamp(v.z / r, -1.0f, 1.0f)) * d_power;
        
        float sinTheta, cosTheta;
        float sinPhi, cosPhi;
        sincosf(theta, &sinTheta, &cosTheta);
        sincosf(phi, &sinPhi, &cosPhi);
        
        // 優化: 避免 pow，改用乘法計算 r^7 和 r^8
        float r2 = r * r;
        float r4 = r2 * r2;
        float r8 = r4 * r4;
        float r7 = (r > 0.0f) ? (r8 / r) : 0.0f;
        
        // 使用預計算的值
        dr = d_power * r7 * dr + 1.0f;
        v = p + r8 * vec3(cosTheta * cosPhi, cosPhi * sinTheta, -sinPhi);
        
        trap = glm::min(trap, r);
        r = glm::length(v);
        if (r > d_bailout) break;
    }
    return 0.5f * logf(r) * r / dr;
}

// Device function: Scene mapping
__device__ float map(vec3 p, float& trap, int& ID) {
    vec2 rt = vec2(cosf(pi / 2.0f), sinf(pi / 2.0f));
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * p;
    ID = 1;
    return md(rp, trap);
}

__device__ float map(vec3 p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Device function: Palette
__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2.0f * pi * (c * t + d));
}

// Device function: Soft shadow
__device__ float softshadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0f;
    float t = 0.0f;
    for (int i = 0; i < d_shadow_step; ++i) {
        float h = map(ro + rd * t);
        res = glm::min(res, k * h / glm::max(t, 1e-6f));
        if (res < 0.02f) return 0.02f;
        t += glm::clamp(h, 0.001f, d_step_limiter);
    }
    return glm::clamp(res, 0.02f, 1.0f);
}

// Device function: Calculate normal
__device__ vec3 calcNor(vec3 p) {
    vec2 e = vec2(d_eps, 0.0f);
    return normalize(vec3(
        map(p + e.xyy()) - map(p - e.xyy()),
        map(p + e.yxy()) - map(p - e.yxy()),
        map(p + e.yyx()) - map(p - e.yyx())
    ));
}

// Device function: Ray tracing
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

// ===== 優化後的 Kernel: 動態 Tile-based 渲染 =====
__global__ void renderKernel_tiled(unsigned char* image, int width, int height) {
    // 計算總 tile 數量
    const unsigned tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    const unsigned tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    const unsigned totalTiles = tilesX * tilesY;
    
    // Shared memory 用於廣播 tile ID
    __shared__ unsigned tileId;
    __shared__ int shouldQuit;
    
    // 預計算相機參數
    vec3 ro = d_camera_pos;
    vec3 ta = d_target_pos;
    vec3 cf = glm::normalize(ta - ro);
    vec3 cs = glm::normalize(glm::cross(cf, vec3(0.f, 1.f, 0.f)));
    vec3 cu = glm::normalize(glm::cross(cs, cf));
    
    // 動態抓取 tiles 直到全部完成
    while (true) {
        // 只用一個 thread 抓取下一個 tile
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            tileId = atomicAdd(&g_tileCounter, 1u);
        }
        __syncthreads();
        
        // 檢查是否所有 tile 都完成了
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            shouldQuit = (tileId >= totalTiles);
        }
        __syncthreads();
        
        if (shouldQuit) break;
        
        // 計算當前 tile 的起始座標
        unsigned tx = (tileId % tilesX) * TILE_SIZE;
        unsigned ty = (tileId / tilesX) * TILE_SIZE;
        
        // 計算當前 thread 處理的像素座標
        unsigned j = tx + threadIdx.x;  // x coordinate (width)
        unsigned i = ty + threadIdx.y;  // y coordinate (height)
        
        // 邊界檢查
        if (j >= width || i >= height) {
            __syncthreads();
            continue;
        }
        
        // 渲染單個像素
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
                
                // Ray direction
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
                                 (0.2f + 0.8f * glm::clamp(0.05f * logf(glm::max(trap, 1e-8f)), 0.0f, 1.0f));
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
        
        // 合併寫入 (保持原有的優化)
        uint32_t rgba_packed = (a << 24) | (b << 16) | (g << 8) | r;
        int pixel_idx = i * width + j;
        ((uint32_t*)image)[pixel_idx] = rgba_packed;
        
        __syncthreads();
    }
}

// Save image to PNG
void write_png(const char* filename, unsigned char* image, int width, int height) {
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    assert(argc == 10);
    
    // Parse arguments
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int width = atoi(argv[7]);
    int height = atoi(argv[8]);
    vec2 iResolution = vec2(width, height);
    
    // Copy constants to device
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));
    
    // Allocate host memory
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    unsigned char* h_image = new unsigned char[image_size];
    
    // Allocate device memory
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size);
    cudaMemset(d_image, 0, image_size);  // 預先清零
    
    // Reset tile counter
    unsigned zero = 0;
    cudaMemcpyToSymbol(g_tileCounter, &zero, sizeof(unsigned));
    
    // Launch kernel with dynamic tiling
    // 使用多個 blocks，每個 block 是 TILE_SIZE x TILE_SIZE
    int device = 0;
    int numSMs = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    int numBlocks = numSMs * 12;  // 每個 SM 12 個 blocks (可調整 8/12/16)
    dim3 gridDim(numBlocks, 1);
    
    printf("Rendering with %d blocks of %dx%d threads (Dynamic Tiling)\n",
           numBlocks, blockDim.x, blockDim.y);
    printf("Total tiles: %d x %d = %d\n",
           (width + TILE_SIZE - 1) / TILE_SIZE,
           (height + TILE_SIZE - 1) / TILE_SIZE,
           ((width + TILE_SIZE - 1) / TILE_SIZE) * ((height + TILE_SIZE - 1) / TILE_SIZE));
    
    // Measure kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    renderKernel_tiled<<<gridDim, blockDim>>>(d_image, width, height);
    cudaEventRecord(stop, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double mpix_per_s = (milliseconds > 0.0f) ? 
                        (double(width) * double(height) / 1e6) / (milliseconds / 1000.0f) : 0.0;
    
    printf("Kernel time: %.3f ms\n", milliseconds);
    printf("Throughput: %.3f Mpix/s\n", mpix_per_s);
    
    // Copy result back
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    
    // Save image
    write_png(argv[9], h_image, width, height);
    printf("Image saved to %s\n", argv[9]);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    delete[] h_image;
    
    return 0;
}