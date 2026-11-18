HIPCC ?= hipcc
HIP_ARCH ?= gfx936
HIP_BIN_DIR ?= bin
HIP_CPPFLAGS ?= -I./src/ -DHIP_ENABLE_WARP_SYNC_BUILTINS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP
HIP_CXXFLAGS ?= -O3 -std=c++17 -ffp-contract=off -fopenmp -Xarch_host -march=native -Xarch_host -mno-fma
CHECK_DIR ?= artifacts/check

all: serial openmp gpu

serial:
	mkdir -p bin/f32/serial/
	mkdir -p bin/f64/serial/
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/abs_compress_ser src/f32_abs_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/abs_decompress_ser src/f32_abs_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/rel_compress_ser src/f32_rel_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/rel_decompress_ser src/f32_rel_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/noa_compress_ser src/f32_noa_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/serial/noa_decompress_ser src/f32_noa_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/abs_compress_ser src/f64_abs_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/abs_decompress_ser src/f64_abs_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/rel_compress_ser src/f64_rel_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/rel_decompress_ser src/f64_rel_decomp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/noa_compress_ser src/f64_noa_comp_cpu.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/serial/noa_decompress_ser src/f64_noa_decomp_cpu.cpp

openmp:
	mkdir -p bin/f32/openmp/
	mkdir -p bin/f64/openmp/
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/abs_compress_omp src/f32_abs_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/abs_decompress_omp src/f32_abs_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/rel_compress_omp src/f32_rel_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/rel_decompress_omp src/f32_rel_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/noa_compress_omp src/f32_noa_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f32/openmp/noa_decompress_omp src/f32_noa_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/abs_compress_omp src/f64_abs_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/abs_decompress_omp src/f64_abs_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/rel_compress_omp src/f64_rel_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/rel_decompress_omp src/f64_rel_decomp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/noa_compress_omp src/f64_noa_comp_cpu.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o bin/f64/openmp/noa_decompress_omp src/f64_noa_decomp_cpu.cpp

# Keep the original GPU entry point as an alias for the HIP build.
gpu: hip

hip:
	mkdir -p "$(HIP_BIN_DIR)/f32/hip/"
	mkdir -p "$(HIP_BIN_DIR)/f64/hip/"
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_abs_compress_hip" src/f32_abs_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_abs_decompress_hip" src/f32_abs_decomp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_rel_compress_hip" src/f32_rel_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_rel_decompress_hip" src/f32_rel_decomp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_noa_compress_hip" src/f32_noa_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f32/hip/f32_noa_decompress_hip" src/f32_noa_decomp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_abs_compress_hip" src/f64_abs_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_abs_decompress_hip" src/f64_abs_decomp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_rel_compress_hip" src/f64_rel_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_rel_decompress_hip" src/f64_rel_decomp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_noa_compress_hip" src/f64_noa_comp_gpu.hip
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) -o "$(HIP_BIN_DIR)/f64/hip/f64_noa_decompress_hip" src/f64_noa_decomp_gpu.hip

hip-test-build: hip
	mkdir -p "$(HIP_BIN_DIR)/validation/"
	$(HIPCC) $(HIP_CPPFLAGS) $(HIP_CXXFLAGS) --offload-arch=$(HIP_ARCH) tools/validation/hip_runtime_probe.hip -o "$(HIP_BIN_DIR)/validation/hip_runtime_probe"

check:
	@test -n "$(HIP_TEST_DEVICE)" && test -n "$(HIP_TEST_PCI)" || { echo 'Set HIP_TEST_DEVICE and HIP_TEST_PCI.'; exit 1; }
	@test ! -e "$(CHECK_DIR)" || { echo 'Select a fresh CHECK_DIR.'; exit 1; }
	python3 -B tools/validation/build_reference.py --output "$(CHECK_DIR)/cpu-build"
	python3 -B tools/validation/reference_suite.py --smoke --manifest "$(CHECK_DIR)/cpu-build/manifest.json" --output "$(CHECK_DIR)/cpu"
	python3 -B tools/validation/build_hip_manifest.py --output "$(CHECK_DIR)/hip-build"
	python3 -B tools/validation/hip_equivalence_suite.py --smoke --reference-results "$(CHECK_DIR)/cpu/results.json" --hip-manifest "$(CHECK_DIR)/hip-build/manifest.json" --device "$(HIP_TEST_DEVICE)" --expected-pci "$(HIP_TEST_PCI)" --output "$(CHECK_DIR)/gpu"

clean:
	rm -rf bin/
.PHONY: all serial openmp gpu hip hip-test-build check clean
