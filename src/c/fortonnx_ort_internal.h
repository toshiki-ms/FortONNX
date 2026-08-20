#ifndef FORTONNX_ORT_INTERNAL_H
#define FORTONNX_ORT_INTERNAL_H

#include "fortonnx_ort.h"
#include <onnxruntime_c_api.h>

typedef struct {
  OrtValue* value;
  char* name;
  int64_t* shape;
  size_t rank;
} FortonnxTensor;

typedef struct {
  const OrtApi* api;
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;
  OrtMemoryInfo* memory_info;
  OrtIoBinding* binding;
  OrtRunOptions* run_options;
  FortonnxTensor* inputs;
  FortonnxTensor* outputs;
  size_t input_count;
  size_t output_count;
  int backend;
  int owns_env;
  int asynchronous;
  char error[2048];
} FortonnxSession;

FortonnxSession* fortonnx_session_allocate(void);
int fortonnx_store_status(FortonnxSession* session, OrtStatus* status);
int fortonnx_store_message(FortonnxSession* session, const char* message);
int fortonnx_register_custom_ops_library(
    FortonnxSession* session, const char* custom_op_library);
int fortonnx_finish_session(FortonnxSession* session, const char* model_path,
                            const char* memory_name, int device_id);
void fortonnx_set_global_error(const char* message);

#endif
