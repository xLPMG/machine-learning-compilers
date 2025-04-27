#include <cstdint>
#include <iostream>

extern "C"
{
    /**
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void matmul_16_6_1(float const *a,
                       float const *b,
                       float *c,
                       int64_t lda,
                       int64_t ldb,
                       int64_t ldc);
}

int main()
{

    const int M = 16;
    const int N = 6;
    const int K = 1;

    float A[M * K];
    float B[K * N];
    float C[M * N] = {0.0f};

    // Initialize matrices
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = static_cast<float>(i);
    }
    for (int j = 0; j < K * N; ++j)
    {
        B[j] = static_cast<float>(j);
    }

    // Print matrix A
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            std::cout << A[i * K + j] << " ";
        }
        std::cout << "\n";
    }

    // Print matrix B
    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << B[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    matmul_16_6_1(A,
                  B,
                  C,
                  16,
                  1,
                  16);

    // Print the result
    std::cout << "Result Matrix C:" << std::endl;
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            std::cout << C[col * M + row] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}