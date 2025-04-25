# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: microkernel test
	@echo "Build complete. Run the microkernel with ./build/microkernel and the test with ./build/test"

microkernel:
	mkdir -p build
	g++ microkernel.cpp matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/microkernel

test:
	mkdir -p build
	g++ tests.cpp matmul_16_6_1.test.cpp matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/test

clean:
	rm -rf build