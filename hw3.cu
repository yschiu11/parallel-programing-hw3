// Your cuda program :)
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;
typedef glm::dvec3 vec3;
typedef glm::dvec4 vec4;
typedef glm::dmat3 mat3;

// Device constants (read-only, cached in constant memory)
__constant__ double d_power = 8.0;
__constant__ double d_md_iter = 24;
__constant__ double d_step_limiter = 0.2;
__constant__ double d_ray_multiplier = 0.1;
__constant__ double d_bailout = 2.0;
__constant__ double d_eps = 0.0005;
__constant__ double d_FOV = 1.5;
__constant__ double d_far_plane = 100.0;
__constant__ int d_ray_step = 10000;
__constant__ int d_shadow_step = 1500;
__constant__ int d_AA = 3;

__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ vec2 d_iResolution;

// Device function: Mandelbulb distance estimation
__device__ double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.0;
    double r = glm::length(v);
    trap = r;

    for (int i = 0; i < d_md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * d_power;
        double phi = glm::asin(v.z / r) * d_power;
        dr = d_power * glm::pow(r, d_power - 1.0) * dr + 1.0;
        v = p + glm::pow(r, d_power) *
                vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));

        trap = glm::min(trap, r);
        r = glm::length(v);
        if (r > d_bailout) break;
    }
    return 0.5 * log(r) * r / dr;
}

// Device function: Scene mapping
__device__ double map(vec3 p, double& trap, int& ID) {
    vec2 rt = vec2(cos(pi / 2.0), sin(pi / 2.0));
    vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) * p;
    ID = 1;
    return md(rp, trap);
}

__device__ double map(vec3 p) {
    double dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Device function: Palette
__device__ vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2.0 * pi * (c * t + d));
}

// Device function: Soft shadow
__device__ double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.0;
    for (int i = 0; i < d_shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(res, k * h / t);
        if (res < 0.02) return 0.02;
        t += glm::clamp(h, 0.001, d_step_limiter);
    }
    return glm::clamp(res, 0.02, 1.0);
}

// Device function: Calculate normal
__device__ vec3 calcNor(vec3 p) {
    vec2 e = vec2(d_eps, 0.0);
    return normalize(vec3(
        map(p + e.xyy()) - map(p - e.xyy()),
        map(p + e.yxy()) - map(p - e.yxy()),
        map(p + e.yyx()) - map(p - e.yyx())
    ));
}

// Device function: Ray tracing
__device__ double trace(vec3 ro, vec3 rd, double& trap, int& ID) {
    double t = 0.0;
    double len = 0.0;

    for (int i = 0; i < d_ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (glm::abs(len) < d_eps || t > d_far_plane) break;
        t += len * d_ray_multiplier;
    }
    return t < d_far_plane ? t : -1.0;
}

// CUDA Kernel: Render each pixel
__global__ void renderKernel(unsigned char* image, int width, int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x coordinate (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y coordinate (height)

    if (i >= height || j >= width) return;

    double fcol_r = 0.0;
    double fcol_g = 0.0;
    double fcol_b = 0.0;

    // Anti-aliasing loop
    for (int m = 0; m < d_AA; ++m) {
        for (int n = 0; n < d_AA; ++n) {
            vec2 p = vec2(j, i) + vec2(m, n) / (double)d_AA;

            // Screen space to normalized coordinates
            vec2 uv = (-d_iResolution.xy() + 2.0 * p) / d_iResolution.y;
            uv.y *= -1.0;

            // Create camera
            vec3 ro = d_camera_pos;
            vec3 ta = d_target_pos;
            vec3 cf = glm::normalize(ta - ro);
            vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));
            vec3 cu = glm::normalize(glm::cross(cs, cf));
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + d_FOV * cf);

            // Ray marching
            double trap;
            int objID;
            double d = trace(ro, rd, trap, objID);

            // Lighting
            vec3 col(0.0);
            vec3 sd = glm::normalize(d_camera_pos);
            vec3 sc = vec3(1., 0.9, 0.717);

            // Coloring
            if (d < 0.0) {
                col = vec3(0.0);  // Sky
            } else {
                vec3 pos = ro + rd * d;
                vec3 nr = calcNor(pos);
                vec3 hal = glm::normalize(sd - rd);

                col = pal(trap - 0.4, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.1, 0.2));
                vec3 ambc = vec3(0.3);
                double gloss = 32.0;

                double amb = (0.7 + 0.3 * nr.y) *
                             (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));
                double sdw = softshadow(pos + 0.001 * nr, sd, 16.0);
                double dif = glm::clamp(glm::dot(sd, nr), 0.0, 1.0) * sdw;
                double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0.0, 1.0), gloss) * dif;

                vec3 lin(0.0);
                lin += ambc * (0.05 + 0.95 * amb);
                lin += sc * dif * 0.8;
                col *= lin;

                col = glm::pow(col, vec3(0.7, 0.9, 1.0));
                col += spe * 0.8;
            }

            col = glm::clamp(glm::pow(col, vec3(0.4545)), 0.0, 1.0);
            fcol_r += col.r;
            fcol_g += col.g;
            fcol_b += col.b;
        }
    }

    // Average and convert to 0-255
    fcol_r = fcol_r / (double)(d_AA * d_AA) * 255.0;
    fcol_g = fcol_g / (double)(d_AA * d_AA) * 255.0;
    fcol_b = fcol_b / (double)(d_AA * d_AA) * 255.0;

    // Write to image
    int idx = (i * width + j) * 4;
    image[idx + 0] = (unsigned char)fcol_r;
    image[idx + 1] = (unsigned char)fcol_g;
    image[idx + 2] = (unsigned char)fcol_b;
    image[idx + 3] = 255;
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
