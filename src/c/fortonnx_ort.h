#ifndef FORTONNX_ORT_H
#define FORTONNX_ORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  FORTONNX_BACKEND_CPU = 0,
  FORTONNX_BACKEND_CUDA = 1,
  FORTONNX_BACKEND_TENSORRT = 2
};

/* Values match ONNXTensorElementDataType. Bool buffers contain one byte per element. */
enum {
  FORTONNX_FLOAT32 = 1,
  FORTONNX_BOOL = 9,
  FORTONNX_FLOAT64 = 11
};

int fortonnx_session_get_tensor_type_at(
    void* session, int is_input, int64_t index, int* element_type);
/* Typed entry points check the caller type against the model, without casting.
   The original bind entry points remain float32-only for compatibility. */
int fortonnx_session_bind_input_typed(
    void* session, int64_t index, void* data, int memory_backend,
    const int64_t* shape, int64_t rank, int element_type);
int fortonnx_session_bind_output_typed(
    void* session, int64_t index, void* data, int memory_backend,
    const int64_t* shape, int64_t rank, int element_type);
int fortonnx_session_bind_tensor_typed(
    void* session, void* input, void* output, int memory_backend,
    const int64_t* input_shape, int64_t input_rank,
    const int64_t* output_shape, int64_t output_rank,
    int input_type, int output_type);

void* fortonnx_cpu_context_create(int intra_op_threads, int inter_op_threads);
void fortonnx_cpu_context_destroy(void* context);

void* fortonnx_session_create_cpu(const char* model_path, void* cpu_context);
void* fortonnx_session_create_cpu_with_custom_ops(
    const char* model_path, void* cpu_context, const char* custom_op_library);
void* fortonnx_session_create_gpu(
    const char* model_path,
    int backend,
    int device_id,
    void* user_compute_stream,
    int use_tf32,
    int tensorrt_fp16,
    int tensorrt_engine_cache,
    const char* tensorrt_cache_path);
void* fortonnx_session_create_gpu_with_custom_ops(
    const char* model_path,
    int backend,
    int device_id,
    void* user_compute_stream,
    int use_tf32,
    int tensorrt_fp16,
    int tensorrt_engine_cache,
    const char* tensorrt_cache_path,
    const char* custom_op_library);

int fortonnx_session_get_features(
    void* session, int64_t* input_features, int64_t* output_features);
int fortonnx_session_get_tensor_count(
    void* session, int is_input, int64_t* count);
int fortonnx_session_get_tensor_rank(
    void* session, int is_input, int64_t* rank);
int fortonnx_session_get_tensor_shape(
    void* session, int is_input, int64_t* shape, int64_t rank);
int fortonnx_session_get_tensor_rank_at(
    void* session, int is_input, int64_t index, int64_t* rank);
int fortonnx_session_get_tensor_shape_at(
    void* session, int is_input, int64_t index, int64_t* shape, int64_t rank);
int fortonnx_session_find_tensor(
    void* session, int is_input, const char* name, int64_t* index);
int fortonnx_session_begin_bind(void* session);
int fortonnx_session_bind_input(
    void* session,
    int64_t index,
    void* data,
    int memory_backend,
    const int64_t* shape,
    int64_t rank);
int fortonnx_session_bind_output(
    void* session,
    int64_t index,
    void* data,
    int memory_backend,
    const int64_t* shape,
    int64_t rank);
int fortonnx_session_bind(
    void* session,
    void* input,
    void* output,
    int memory_backend,
    int64_t batch_size,
    int64_t input_features,
    int64_t output_features);
int fortonnx_session_bind_tensor(
    void* session,
    void* input,
    void* output,
    int memory_backend,
    const int64_t* input_shape,
    int64_t input_rank,
    const int64_t* output_shape,
    int64_t output_rank);
int fortonnx_session_run(void* session);
void fortonnx_session_destroy(void* session);

size_t fortonnx_last_error(void* session, char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
