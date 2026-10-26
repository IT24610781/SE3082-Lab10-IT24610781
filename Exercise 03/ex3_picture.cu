#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define WIDTH 800
#define HEIGHT 600

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

__global__ void PictureKernel(const unsigned char *input,
                              unsigned char *output,
                              int height, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        int value = 2 * input[idx];
        output[idx] = (value > 255) ? 255 : value;
    }
}

int main()
{
    size_t count = (size_t)WIDTH * HEIGHT;
    size_t bytes = count * sizeof(unsigned char);

    unsigned char *input = (unsigned char *)malloc(bytes);
    unsigned char *output = (unsigned char *)malloc(bytes);

    if (!input || !output) {
        fprintf(stderr, "Host allocation failed.\n");
        free(input);
        free(output);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < count; ++i)
        input[i] = rand() % 256;

    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc((void **)&d_input, bytes));
    CHECK(cudaMalloc((void **)&d_output, bytes));

    CHECK(cudaMemcpy(d_input, input, bytes,
                     cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Grid: %u x %u blocks of %u x %u threads\n",
           numBlocks.x, numBlocks.y,
           threadsPerBlock.x, threadsPerBlock.y);

    double start = cpuSecond();

    PictureKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, HEIGHT, WIDTH
    );
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double elapsed = cpuSecond() - start;

    CHECK(cudaMemcpy(output, d_output, bytes,
                     cudaMemcpyDeviceToHost));

    printf("Original top-left 50 x 50:\n");
    for (int row = 0; row < 50; ++row) {
        for (int col = 0; col < 50; ++col)
            printf("%3d ", (int)input[row * WIDTH + col]);
        printf("\n");
    }

    printf("Updated top-left 50 x 50:\n");
    for (int row = 0; row < 50; ++row) {
        for (int col = 0; col < 50; ++col)
            printf("%3d ", (int)output[row * WIDTH + col]);
        printf("\n");
    }

    size_t errors = 0;
    size_t capped = 0;

    for (size_t i = 0; i < count; ++i) {
        int expected = 2 * input[i];

        if (expected > 255) {
            expected = 255;
            ++capped;
        }

        if (output[i] != expected)
            ++errors;
    }

    printf("Verification: %zu mismatches out of %zu\n",
           errors, count);
    printf("Pixels capped at 255: %zu\n", capped);
    printf("Time taken - %.7f seconds\n", elapsed);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    free(input);
    free(output);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
