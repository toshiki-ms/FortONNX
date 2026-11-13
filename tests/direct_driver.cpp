// Calls the direct entry of a kernel library as a program without ONNX Runtime does (test_kernelgen_cuda.py).
// usage: direct_driver LIBRARY PREFIX GRAPH ROWS OUT
//   ROWS: FP32 rows, row-major; OUT: their outputs, FP32, written by this program
// Prints "open: status S MESSAGE"; exit code 3 when the library refuses the graph, 1 on other failures.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  void* library = dlopen(argv[1], RTLD_NOW);
  if (!library) { std::printf("dlopen: %s\n", dlerror()); return 1; }
  const std::string prefix = argv[2];
  auto open = reinterpret_cast<int (*)(const char*, void**, char*, int)>(dlsym(library, (prefix + "_open").c_str()));
  auto shape = reinterpret_cast<void (*)(void*, int*, int*)>(dlsym(library, (prefix + "_shape").c_str()));
  auto forward = reinterpret_cast<int (*)(void*, const float*, int, float*, void*)>(dlsym(library, (prefix + "_forward").c_str()));
  auto close = reinterpret_cast<void (*)(void*)>(dlsym(library, (prefix + "_close").c_str()));
  if (!open || !shape || !forward || !close) { std::printf("no direct entry %s\n", argv[2]); return 1; }
  char message[512] = "";
  void* kernel = nullptr;
  const int status = open(argv[3], &kernel, message, int(sizeof message));
  std::printf("open: status %d %s\n", status, message);
  if (status || !kernel) return 3;
  int row_columns = 0, output_columns = 0;
  shape(kernel, &row_columns, &output_columns);
  std::FILE* file = std::fopen(argv[4], "rb");
  if (!file) return 1;
  std::vector<float> rows;
  float buffer[4096];
  for (size_t got; (got = std::fread(buffer, 4, 4096, file)) > 0;) rows.insert(rows.end(), buffer, buffer + got);
  std::fclose(file);
  const int n = int(rows.size() / size_t(row_columns));
  std::vector<float> out(size_t(n) * output_columns);
  cudaStream_t stream;
  float *device_rows = nullptr, *device_out = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaMalloc(&device_rows, 4 * rows.size()) != cudaSuccess ||
      cudaMalloc(&device_out, 4 * out.size()) != cudaSuccess ||
      cudaMemcpy(device_rows, rows.data(), 4 * rows.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  for (int repeat = 0; repeat < 2; ++repeat)   // the kernel is used again without anything being set up
    if (forward(kernel, device_rows, n, device_out, stream) != 0 || cudaStreamSynchronize(stream) != cudaSuccess) return 1;
  if (cudaMemcpy(out.data(), device_out, 4 * out.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  close(kernel);
  file = std::fopen(argv[5], "wb");
  if (!file || std::fwrite(out.data(), 4, out.size(), file) != out.size()) return 1;
  std::fclose(file);
  std::printf("rows %d, values of a row %d, of its output %d\n", n, row_columns, output_columns);
  return 0;
}
