#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>


#define CHECK(call) do {                                           \
    const cudaError_t error = (call);                              \
    if (error != cudaSuccess) {                                   \
        fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n",      \
                __FILE__, __LINE__, (int)error,                    \
                cudaGetErrorString(error));                       \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

#define N 512

__global__ void add(const int *a, const int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
    size_t bytes = N * sizeof(int);

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
    CHECK(cudaMalloc((void **)&d_a, (size_t)1 << 50));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));

    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    double start = cpuSecond();

    add<<<N, 1>>>(d_a, d_b, d_c);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double elapsed = cpuSecond() - start;

    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; ++i)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] + b[i])
            ++errors;
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Time taken - %.7f seconds\n", elapsed);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
