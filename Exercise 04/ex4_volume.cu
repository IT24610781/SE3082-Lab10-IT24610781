#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NX 500
#define NY 500
#define NZ 500

#define CHECK(call) do {                                           \
    const cudaError_t error = (call);                              \
    if (error != cudaSuccess) {                                   \
        fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n",      \
                __FILE__, __LINE__, (int)error,                    \
                cudaGetErrorString(error));                       \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

__global__ void mulElem3D(const int *a, const int *b, int *c,
                          int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t idx = ((size_t)z * ny + y) * nx + x;
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *array, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        array[i] = rand() % 100;
}

int main()
{
    size_t count = (size_t)NX * NY * NZ;
    size_t bytes = count * sizeof(int);

    int *a = (int *)malloc(bytes);
    int *b = (int *)malloc(bytes);
    int *c = (int *)malloc(bytes);

    if (!a || !b || !c) {
        fprintf(stderr, "Host allocation failed.\n");
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    random_ints(a, count);
    random_ints(b, count);

    int *d_a, *d_b, *d_c;

    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));

    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    printf("Grid: %u x %u x %u blocks\n",
           numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Threads per block: %u x %u x %u = %u\n",
           threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.z,
           threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z);

    double start = cpuSecond();

    mulElem3D<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_c, NX, NY, NZ
    );
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double elapsed = cpuSecond() - start;

    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (size_t idx = count - 1000; idx < count; ++idx) {
        size_t x = idx % NX;
        size_t y = (idx / NX) % NY;
        size_t z = idx / ((size_t)NX * NY);

        printf("C[%zu][%zu][%zu] = %d x %d = %d\n",
               z, y, x, a[idx], b[idx], c[idx]);
    }

    size_t errors = 0;
    for (size_t i = 0; i < count; ++i) {
        if (c[i] != a[i] * b[i])
            ++errors;
    }

    printf("Verification: %zu mismatches out of %zu\n",
           errors, count);
    printf("Time taken - %.7f seconds\n", elapsed);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
