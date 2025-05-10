# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: microbench test
	@echo "Build complete. Run the test with ./build/test"

test:
	mkdir -p build
	g++ tests/tests.cpp tests/matmul_64_48_64_16.test.cpp optimization/v1_matmul_64_48_64_16.S optimization/v2_matmul_64_48_64_16.S -I$(INCLUDE_DIR) -o build/test

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp optimization/v1_matmul_64_48_64_16.S optimization/v2_matmul_64_48_64_16.S -I$(INCLUDE_DIR) -o build/microbench

clean:
	rm -rf build