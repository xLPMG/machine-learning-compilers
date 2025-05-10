#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstring>

extern "C" {
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

/*
 * Benchmarks the throughput for the different MatMul (loop) versions.
 *
 * @param n: number of loop iterations.
 * @param instruction: a string selecting the instruction to benchmark.
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 */
 void benchmark_thr( int64_t n, 
                     std::string instruction,
                     float const *a,
                     float const *b,
                     float *c ) 
{
    std::cout << "-----------------------------------------------\n";
    double elapsedTime = 1;

    std::string v1_matmul( "v1_matmul" );
    int res_1 = v1_matmul.compare( instruction );

    std::string v2_matmul( "v2_matmul" );
    int res_2 = v2_matmul.compare( instruction );

    double opsPerMatmul = 1;

    // Time measuring
    if ( res_1 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v1_matmul_64_48_64_16( a, 
                                   b, 
                                   c, 
                                   64, 
                                   64, 
                                   64,
                                   64 * 64,
                                   48 * 64 );
        }
        std::memset( c, 0, 64 * 48 * sizeof( float ) ); 

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v1_matmul_64_48_64_16( a, 
                                   b, 
                                   c, 
                                   64, 
                                   64, 
                                   64,
                                   64 * 64,
                                   48 * 64 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 64 * 48 * 64 * 16 ) * 2;
    }
    else if ( res_2 == 0 )
    {
        // Warmup
        for ( int i = 0; i < 1000; i++ )
        {
            v2_matmul_64_48_64_16( a, 
                                   b, 
                                   c, 
                                   64, 
                                   64, 
                                   64,
                                   64 * 64,
                                   48 * 64 );
        }
        std::memset( c, 0, 64 * 48 * sizeof( float ) ); 

        auto l_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < n; i++ )
        {
            v2_matmul_64_48_64_16( a, 
                                   b, 
                                   c, 
                                   64, 
                                   64, 
                                   64,
                                   64 * 64,
                                   48 * 64 );
        }
        auto l_end_time = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>( l_end_time - l_start_time ).count() / 1e6;

        // (M x N x K) x (FMLA OPs)
        opsPerMatmul = ( 64 * 48 * 64 * 16 ) * 2;
    }

    double loopIterations = n;
    double totalFLOPs = opsPerMatmul * loopIterations;

    double flopsPerSec = totalFLOPs / elapsedTime;
    double gflops = totalFLOPs / ( elapsedTime * 1e9 );

    std::cout << "Measuring throughput for " << "Instruction\n";
    std::cout << "Total time (s):   " << elapsedTime << "\n";
    std::cout << "Instructions per Second:   " << flopsPerSec << "\n";
    std::cout << "Estimated GFLOPS:   " << gflops << " GFLOPS/sec\n";
    std::cout << "-----------------------------------------------\n";
}

int main() 
{
    const int M = 64;
    const int N = 48;
    const int K = 64;
    const int BATCH_SIZE = 16;

    float A[M * K * BATCH_SIZE];
    float B[K * N * BATCH_SIZE];
    float C[M * N];

    // Initialize matrices
    for ( int i = 0; i < M * K * BATCH_SIZE; ++i )
    {
        A[i] = static_cast<float>( i );
    }
    for ( int j = 0; j < K * N * BATCH_SIZE; ++j )
    {
        B[j] = static_cast<float>( j );
    }
    std::memset( C, 0, 64 * 48 * sizeof(float) ); 

    int64_t l_iter = 25000;
    std::string v1_matmul( "v1_matmul" );

    std::cout << "\nBenchmarking V1_Matmul_64_48_64_16 throughput ...\n";
    benchmark_thr( l_iter, v1_matmul, A, B, C );

    std::string v2_matmul( "v2_matmul" );

    std::cout << "\nBenchmarking V2_Matmul_64_48_64_16 throughput ...\n";
    benchmark_thr( l_iter, v2_matmul, A, B, C );

    return 0;
}
