#include <catch2/catch.hpp>
#include <random>
#include <iostream>

#include <iomanip>
#include <string>

extern "C"
{
    void v1_matmul_64_48_64_16( float const *a,
                                float const *b,
                                float *c,
                                int64_t lda,
                                int64_t ldb,
                                int64_t ldc,
                                int64_t br_stride_a,
                                int64_t br_stride_b );

    void v2_matmul_64_48_64_16( float const *a,
                                float const *b,
                                float *c,
                                int64_t lda,
                                int64_t ldb,
                                int64_t ldc,
                                int64_t br_stride_a,
                                int64_t br_stride_b );
}

// void print_matrix(const float* mat, int rows, int cols, const std::string& name) {
//     std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):\n";
//     for (int r = 0; r < rows; ++r) {
//         for (int c = 0; c < cols; ++c) {
//             std::cout << std::setw(8) << std::setprecision(4) << mat[r + c * rows] << " ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << std::endl;
// }

TEST_CASE("Tests the v1_matmul_64_48_64_16 microkernel function with random matrices", "[v1_matmul_64_48_64_16]")
{
    const int M = 64;
    const int N = 48;
    const int K = 64;
    const int BATCH_SIZE = 16;

    float A[M * K * BATCH_SIZE];
    float B[K * N * BATCH_SIZE];
    float C[M * N];
    float C_expected[M * N];

    const int64_t br_stride_a = M * K;
    const int64_t br_stride_b = K * N;


    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 10.0f);

    for (int i = 0; i < M * K * BATCH_SIZE; ++i)
    {
        A[i] = dist(gen);  // Initialize A
        B[i] = dist(gen);  // Initialize B
    }

    for (int i = 0; i < M * N; ++i)
    {
        float val = dist(gen);
        C[i] = val;
        C_expected[i] = val;
    }

    for (int b = 0; b < BATCH_SIZE; ++b)
    {
        float const* A_b = A + b * br_stride_a;
        float const* B_b = B + b * br_stride_b;

        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A_b[row + k * M] * B_b[k + col * K];
                }
                C_expected[row + col * M] += sum;
            }
        }
    }

    v1_matmul_64_48_64_16(A, B, C, M, K, M, br_stride_a, br_stride_b);

    // print_matrix(C, M, N, "C after all batches");
    // print_matrix(C_expected, M, N, "C_expected");

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
    }
}

TEST_CASE("Tests the v2_matmul_64_48_64_16 microkernel function with random matrices", "[v2_matmul_64_48_64_16]")
{
    const int M = 64;
    const int N = 48;
    const int K = 64;
    const int BATCH_SIZE = 16;

    float A[M * K * BATCH_SIZE];
    float B[K * N * BATCH_SIZE];
    float C[M * N];
    float C_expected[M * N];

    const int64_t br_stride_a = M * K;
    const int64_t br_stride_b = K * N;


    // Initialize matrices A and B with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 10.0f);

    for (int i = 0; i < M * K * BATCH_SIZE; ++i)
    {
        A[i] = dist(gen);  // Initialize A
        B[i] = dist(gen);  // Initialize B
    }

    for (int i = 0; i < M * N; ++i)
    {
        float val = dist(gen);
        C[i] = val;
        C_expected[i] = val;
    }

    for (int b = 0; b < BATCH_SIZE; ++b)
    {
        float const* A_b = A + b * br_stride_a;
        float const* B_b = B + b * br_stride_b;

        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A_b[row + k * M] * B_b[k + col * K];
                }
                C_expected[row + col * M] += sum;
            }
        }
    }

    v2_matmul_64_48_64_16(A, B, C, M, K, M, br_stride_a, br_stride_b);

    // print_matrix(C, M, N, "C after all batches");
    // print_matrix(C_expected, M, N, "C_expected");

    for (int i = 0; i < M * N; i++)
    {
        REQUIRE(C[i] == Approx(C_expected[i]).epsilon(0.01));
    }
}
