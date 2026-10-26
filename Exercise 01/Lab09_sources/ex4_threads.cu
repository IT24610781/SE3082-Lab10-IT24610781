#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 512

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at line %d: %s\n",             \
                __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

__global__ void add(const int *a, const int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void random_ints(int *x, int count)
{
    for (int i = 0; i < count; ++i)
        x[i] = rand() % 100;
}

int main()
{
    size_t bytes = N * sizeof(int);

    int *a = (int *)malloc(bytes);
    int *b = (int *)malloc(bytes);
    int *c = (int *)malloc(bytes);

    if (!a || !b || !c) {
        fprintf(stderr, "Host memory allocation failed.\n");
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

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    add<<<1, N>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;

    for (int i = 0; i < N; ++i) {
        printf("%d) %d + %d = %d\n", i, a[i], b[i], c[i]);

        if (c[i] != a[i] + b[i])
            ++errors;
    }

    printf("Launch: 1 block x %d threads = %d threads\n", N, N);
    printf("Verification: %d mismatches out of %d\n", errors, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
