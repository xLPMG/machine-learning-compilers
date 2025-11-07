#include <chrono>
#include <mlc/Brgemm.h>
#include <mlc/Kernel.h>
#include <mlc/benchmarks/Benchmark.h>
#include <mlc/benchmarks/matmul/naive_matmul_m_n_k.bench.h>
#include <random>

mini_jit::benchmarks::NaiveMatmulMNKBench::NaiveMatmulMNKBench(double run_time,
                                                               int    m,
                                                               int    n,
                                                               int    k) : Benchmark()
{
    m_M        = m;
    m_N        = n;
    m_K        = k;
    m_run_time = run_time;
}

void mini_jit::benchmarks::NaiveMatmulMNKBench::run()
{
    m_A = new float[m_M * m_K];
    m_B = new float[m_K * m_N];
    m_C = new float[m_M * m_N];

    // Initialize matrices A and B with random values
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < m_M * m_K; i++)
    {
        m_A[i] = dist(gen);
    }
    for (int i = 0; i < m_K * m_N; i++)
    {
        m_B[i] = dist(gen);
    }
    // Initialize matrix C with zeros
    for (int i = 0; i < m_M * m_N; ++i)
    {
        m_C[i] = 0.0f;
    }

    // RUN
    long   l_num_reps   = 0;
    auto   l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed    = 0.0;
    double l_runTimeMs  = m_run_time * 1e6;
    do
    {
        // Reference GEMM calculation
        for (int col = 0; col < m_N; ++col)
        {
            for (int row = 0; row < m_M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < m_K; ++k)
                {
                    sum += m_A[row + k * m_M] * m_B[k + col * m_K];
                }
                m_C[row + col * m_M] += sum;
            }
        }
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_runTimeMs);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long   l_totalOperations = 2.0 * m_M * m_N * m_K * l_num_reps;
    double l_gflops          = ((double)l_totalOperations) / (l_elapsed * 1e9);

    // Store the results
    m_benchmarkResult.numReps             = l_num_reps;
    m_benchmarkResult.elapsedSeconds      = l_elapsed;
    m_benchmarkResult.totalNumberElements = m_M * m_N * m_K * l_num_reps;
    m_benchmarkResult.totalOperations     = l_totalOperations;
    m_benchmarkResult.gflops              = l_gflops;

    delete[] m_A;
    delete[] m_B;
    delete[] m_C;
}
