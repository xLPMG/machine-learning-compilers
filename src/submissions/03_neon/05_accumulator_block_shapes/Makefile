# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: microbench test
	@echo "Build complete. Run the test with ./build/test"

test:
	mkdir -p build
	g++ tests.cpp matmul_64_64_64.test.cpp matmul_64_64_64.s -I$(INCLUDE_DIR) -o build/test

microbench:
	mkdir -p build
	g++ microbench.cpp optimization/v1_matmul_64_64_64.s optimization/v2_matmul_64_64_64.s -I$(INCLUDE_DIR) -o build/microbench

clean:
	rm -rf build