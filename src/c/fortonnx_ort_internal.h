#ifndef FORTONNX_ORT_INTERNAL_H
#define FORTONNX_ORT_INTERNAL_H

#include "fortonnx_ort.h"
#include <onnxruntime_c_api.h>

typedef struct {
  OrtValue* value;
  char* name;
  int64_t* shape;
  size_t rank;
  ONNXTensorElementDataType element_type;
  void* data;           /* bound buffer */
  int64_t* bound_shape; /* bound shape (rank) */
} FortonnxTensor;

/* CUDA Graph replay: one captured graph per distinct binding (buffer addresses and shapes). */
#define FORTONNX_CUDA_GRAPH_SLOTS 16
typedef struct {
  OrtRunOptions* run_options; /* carries gpu_graph_id of this binding */
  void** data;                /* bound buffers, inputs then outputs */
  int64_t* shape;             /* bound shapes, concatenated */
} FortonnxGraphSlot;

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
  int cuda_graph;
  size_t graph_limit; /* captured graphs the executing provider keeps for this model */
  size_t graph_count;
  int graph_ids;       /* graph ids given out; an id is not used again after a failed first run */
  FortonnxGraphSlot graphs[FORTONNX_CUDA_GRAPH_SLOTS];
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
