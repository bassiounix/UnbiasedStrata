// pcie-probe: host->GPU bandwidth with the GPU busy (P0), the state in which the link should run at Gen 4.
// Prints GB/s on one line; ~13 = healthy Gen 4 x8, ~1-2 = stuck at Gen 1.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
__global__ void spin(float* x, long iters) { float v = threadIdx.x; for (long i = 0; i < iters; ++i) v = v * 1.0000001f + 0.5f; if (v == 1234.5f) x[0] = v; }
int main() {
    const size_t n = 64ull << 20;
    void *h = nullptr, *d = nullptr; float* x = nullptr;
    if (cudaMallocHost(&h, n) != cudaSuccess || cudaMalloc(&d, n) != cudaSuccess || cudaMalloc(&x, 4) != cudaSuccess) { std::printf("0\n"); return 1; }
    cudaStream_t sc, sm; cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sm, cudaStreamNonBlocking);
    spin<<<20, 256, 0, sc>>>(x, 200000000L);
    double best = 0;
    for (int rep = 0; rep < 12; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; ++i) cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, sm);
        cudaStreamSynchronize(sm);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (4.0 * n / s / 1e9 > best) best = 4.0 * n / s / 1e9;
    }
    std::printf("%.1f\n", best);
    cudaDeviceReset();
    return 0;
}
