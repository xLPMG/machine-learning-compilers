#ifndef NAIVE_MATMUL_BR_M_N_K_BENCH_H
#define NAIVE_MATMUL_BR_M_N_K_BENCH_H
#include <mlc/benchmarks/Benchmark.h>

namespace mini_jit
{
    namespace benchmarks
    {
        /**
         * @brief Benchmark for naive matrix multiplication using BRGEMM.
         */
        class NaiveMatmulBrMNKBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for naive matrix multiplication using BRGEMM.
             * @param run_time The time to run the benchmark in seconds.
             * @param m number of rows in A and C.
             * @param n number of columns in B and C.
             * @param k number of columns in A and rows in B.
             * @param br_size The size of the batch-reduce.
             */
            NaiveMatmulBrMNKBench(double run_time,
                                  int    m,
                                  int    n,
                                  int    k,
                                  int    br_size);
            //! Destructor
            ~NaiveMatmulBrMNKBench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            int    m_M;
            int    m_N;
            int    m_K;
            int    m_br_size;
            double m_run_time;
            float* m_A;
            float* m_B;
            float* m_C;
        };

    } // namespace benchmarks
} // namespace mini_jit

#endif // NAIVE_MATMUL_BR_M_N_K_BENCH_H