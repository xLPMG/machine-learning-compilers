#ifndef NAIVE_SIGMOID_PRIMITIVE_BENCH_H
#define NAIVE_SIGMOID_PRIMITIVE_BENCH_H
#include <cstdint>
#include <mlc/benchmarks/Benchmark.h>

namespace mini_jit
{
    namespace benchmarks
    {
        class NaiveSigmoidPrimitiveBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for the naive sigmoid primitive.
             * @param runTime The time to run the benchmark in seconds.
             * @param m number of rows in A and B.
             * @param n number of columns in A and B.
             */
            NaiveSigmoidPrimitiveBench(double   runTime,
                                      uint32_t m,
                                      uint32_t n);
            //! Destructor
            ~NaiveSigmoidPrimitiveBench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            uint32_t m_M;
            uint32_t m_N;
            double   m_runTime;
            float*   m_A;
            float*   m_B;
        };

    } // namespace benchmarks
} // namespace mini_jit

#endif // NAIVE_SIGMOID_PRIMITIVE_BENCH_H