#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ROWS 10000
#define COLS 10000

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at line %d: %s\n",             \
                __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

__global__ void mulElem(const int *a, const int *b, int *c,
                        int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        size_t idx = (size_t)row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *x, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        x[i] = rand() % 100;
}

int main()
{
    size_t count = (size_t)ROWS * COLS;
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

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Grid: %u x %u blocks\n", numBlocks.x, numBlocks.y);
    printf("Threads per block: %u x %u = %u\n",
           threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.x * threadsPerBlock.y);

    mulElem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c,
                                            ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    int row = ROWS - 1;
    for (int col = COLS - 1000; col < COLS; ++col) {
        size_t idx = (size_t)row * COLS + col;
        printf("C[%d][%d]: %d x %d = %d\n",
               row, col, a[idx], b[idx], c[idx]);
    }

    size_t errors = 0;
    for (size_t i = 0; i < count; ++i) {
        if (c[i] != a[i] * b[i])
            ++errors;
    }

    printf("Verification: %zu mismatches out of %zu\n",
           errors, count);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
