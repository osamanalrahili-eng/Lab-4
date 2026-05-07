__kernel void gemm_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}