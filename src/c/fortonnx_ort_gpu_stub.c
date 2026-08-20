#include "fortonnx_ort.h"

void fortonnx_set_global_error(const char* message);

void* fortonnx_session_create_gpu(
    const char* model_path,
    int backend,
    int device_id,
    void* user_compute_stream,
    int use_tf32,
    int tensorrt_fp16,
    int tensorrt_engine_cache,
    const char* tensorrt_cache_path) {
  (void)model_path;
  (void)backend;
  (void)device_id;
  (void)user_compute_stream;
  (void)use_tf32;
  (void)tensorrt_fp16;
  (void)tensorrt_engine_cache;
  (void)tensorrt_cache_path;
  fortonnx_set_global_error(
      "FortONNX was built with BACKEND=cpu; rebuild with cuda or tensorrt");
  return 0;
}


void* fortonnx_session_create_gpu_with_custom_ops(
    const char* model_path,
    int backend,
    int device_id,
    void* user_compute_stream,
    int use_tf32,
    int tensorrt_fp16,
    int tensorrt_engine_cache,
    const char* tensorrt_cache_path,
    const char* custom_op_library) {
  (void)custom_op_library;
  return fortonnx_session_create_gpu(
      model_path, backend, device_id, user_compute_stream, use_tf32,
      tensorrt_fp16, tensorrt_engine_cache, tensorrt_cache_path);
}
