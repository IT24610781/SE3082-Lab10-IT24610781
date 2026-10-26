#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS_PER_BLOCK 512

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at line %d: %s\n",             \
                __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

__global__ void vecMul(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        c[i] = a[i] * b[i];
}

void random_ints(int *x, int count)
{
    for (int i = 0; i < count; ++i)
        x[i] = rand() % 100;
}


double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main()
{
    size_t bytes = (size_t)N * sizeof(int);

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

    random_ints(a, N);
    random_ints(b, N);

    int *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    double totalStart = cpuSecond();

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;



    
    double start = cpuSecond();

    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double elapsed = cpuSecond() - start;


    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    double totalElapsed = cpuSecond() - totalStart;

    printf("Kernel time = %.9f seconds\n", elapsed);
    printf("Whole-sequence time = %.9f seconds\n", totalElapsed);

    for (int i = N - 1000; i < N; ++i)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] * b[i])
            ++errors;
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
