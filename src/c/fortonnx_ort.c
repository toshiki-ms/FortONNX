#include "fortonnx_ort_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const OrtApi* api;
  OrtEnv* env;
  int inter_op_threads;
} FortonnxCpuContext;

static char global_error[2048] = "no error";

void fortonnx_set_global_error(const char* message) {
  snprintf(global_error, sizeof(global_error), "%s", message != NULL ? message : "unknown error");
}

int fortonnx_store_message(FortonnxSession* session, const char* message) {
  char* destination = session != NULL ? session->error : global_error;
  size_t capacity = session != NULL ? sizeof(session->error) : sizeof(global_error);
  snprintf(destination, capacity, "%s", message != NULL ? message : "unknown error");
  return 1;
}

int fortonnx_store_status(FortonnxSession* session, OrtStatus* status) {
  const OrtApi* api;
  if (status == NULL) return 0;
  api = session != NULL ? session->api : OrtGetApiBase()->GetApi(ORT_API_VERSION);
  fortonnx_store_message(
      session, api != NULL ? api->GetErrorMessage(status) : "unknown ONNX Runtime error");
  if (api != NULL) api->ReleaseStatus(status);
  return 1;
}

int fortonnx_register_custom_ops_library(
    FortonnxSession* session, const char* custom_op_library) {
  if (custom_op_library == NULL || custom_op_library[0] == '\0') return 0;
  return fortonnx_store_status(
      session, session->api->RegisterCustomOpsLibrary_V2(
                   session->session_options, custom_op_library));
}

FortonnxSession* fortonnx_session_allocate(void) {
  FortonnxSession* session = (FortonnxSession*)calloc(1, sizeof(*session));
  if (session == NULL) {
    fortonnx_set_global_error("failed to allocate a FortONNX session");
    return NULL;
  }
  session->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (session->api == NULL) {
    snprintf(global_error, sizeof(global_error),
             "ONNX Runtime API version %d is unavailable", ORT_API_VERSION);
    free(session);
    return NULL;
  }
  snprintf(session->error, sizeof(session->error), "no error");
  return session;
}

static int copy_io_name(
    FortonnxSession* session, int input, size_t index, char** destination) {
  OrtAllocator* allocator = NULL;
  char* runtime_name = NULL;
  OrtStatus* status;
  size_t length;

  if (fortonnx_store_status(
          session, session->api->GetAllocatorWithDefaultOptions(&allocator))) return 1;
  status = input
      ? session->api->SessionGetInputName(session->session, index, allocator, &runtime_name)
      : session->api->SessionGetOutputName(session->session, index, allocator, &runtime_name);
  if (fortonnx_store_status(session, status)) return 1;
  length = strlen(runtime_name);
  *destination = (char*)malloc(length + 1);
  if (*destination == NULL) {
    allocator->Free(allocator, runtime_name);
    return fortonnx_store_message(session, "failed to allocate an ONNX tensor name");
  }
  memcpy(*destination, runtime_name, length + 1);
  allocator->Free(allocator, runtime_name);
  return 0;
}

static int query_tensor_shape(
    FortonnxSession* session, int input, size_t index,
    int64_t** shape, size_t* shape_rank) {
  OrtTypeInfo* type_info = NULL;
  const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  ONNXTensorElementDataType element_type;
  int64_t* dimensions = NULL;
  size_t rank = 0;
  OrtStatus* status;

  status = input
      ? session->api->SessionGetInputTypeInfo(session->session, index, &type_info)
      : session->api->SessionGetOutputTypeInfo(session->session, index, &type_info);
  if (fortonnx_store_status(session, status)) return 1;
  if (fortonnx_store_status(
          session, session->api->CastTypeInfoToTensorInfo(type_info, &tensor_info))) goto fail;
  if (tensor_info == NULL) {
    fortonnx_store_message(session, "only tensor inputs and outputs are supported");
    goto fail;
  }
  if (fortonnx_store_status(
          session, session->api->GetTensorElementType(tensor_info, &element_type))) goto fail;
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    fortonnx_store_message(session, "FortONNX currently supports float32 tensors only");
    goto fail;
  }
  if (fortonnx_store_status(
          session, session->api->GetDimensionsCount(tensor_info, &rank))) goto fail;
  if (rank == 0) {
    fortonnx_store_message(session, "FortONNX currently requires tensor rank to be positive");
    goto fail;
  }
  if (rank > SIZE_MAX / sizeof(*dimensions)) {
    fortonnx_store_message(session, "ONNX tensor rank is too large");
    goto fail;
  }
  dimensions = (int64_t*)malloc(rank * sizeof(*dimensions));
  if (dimensions == NULL) {
    fortonnx_store_message(session, "failed to allocate ONNX tensor shape metadata");
    goto fail;
  }
  if (fortonnx_store_status(
          session, session->api->GetDimensions(tensor_info, dimensions, rank))) goto fail;
  *shape = dimensions;
  *shape_rank = rank;
  session->api->ReleaseTypeInfo(type_info);
  return 0;

fail:
  free(dimensions);
  session->api->ReleaseTypeInfo(type_info);
  return 1;
}

static void release_tensor_values(FortonnxSession* session) {
  size_t index;
  if (session == NULL || session->api == NULL) return;
  for (index = 0; index < session->input_count; ++index) {
    if (session->inputs[index].value != NULL) {
      session->api->ReleaseValue(session->inputs[index].value);
      session->inputs[index].value = NULL;
    }
  }
  for (index = 0; index < session->output_count; ++index) {
    if (session->outputs[index].value != NULL) {
      session->api->ReleaseValue(session->outputs[index].value);
      session->outputs[index].value = NULL;
    }
  }
}

static void release_tensor_metadata(FortonnxTensor* tensors, size_t count) {
  size_t index;
  if (tensors == NULL) return;
  for (index = 0; index < count; ++index) {
    free(tensors[index].name);
    free(tensors[index].shape);
  }
  free(tensors);
}

void fortonnx_session_destroy(void* opaque) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  if (session == NULL) return;
  if (session->api == NULL) {
    free(session);
    return;
  }
  if (session->binding != NULL) {
    session->api->ClearBoundInputs(session->binding);
    session->api->ClearBoundOutputs(session->binding);
  }
  release_tensor_values(session);
  if (session->binding != NULL) session->api->ReleaseIoBinding(session->binding);
  if (session->run_options != NULL) session->api->ReleaseRunOptions(session->run_options);
  if (session->memory_info != NULL) session->api->ReleaseMemoryInfo(session->memory_info);
  if (session->session != NULL) session->api->ReleaseSession(session->session);
  if (session->session_options != NULL) {
    session->api->ReleaseSessionOptions(session->session_options);
  }
  if (session->owns_env && session->env != NULL) session->api->ReleaseEnv(session->env);
  release_tensor_metadata(session->inputs, session->input_count);
  release_tensor_metadata(session->outputs, session->output_count);
  free(session);
}

int fortonnx_finish_session(FortonnxSession* session, const char* model_path,
                            const char* memory_name, int device_id) {
  size_t index;

  if (fortonnx_store_status(
          session, session->api->CreateSession(
                       session->env, model_path, session->session_options, &session->session))) return 1;
  if (fortonnx_store_status(
          session, session->api->SessionGetInputCount(
                       session->session, &session->input_count))) return 1;
  if (fortonnx_store_status(
          session, session->api->SessionGetOutputCount(
                       session->session, &session->output_count))) return 1;
  if (session->input_count == 0 || session->output_count == 0) {
    return fortonnx_store_message(session, "ONNX models must have inputs and outputs");
  }
  if (session->input_count > SIZE_MAX / sizeof(*session->inputs) ||
      session->output_count > SIZE_MAX / sizeof(*session->outputs)) {
    return fortonnx_store_message(session, "ONNX tensor count is too large");
  }
  session->inputs = (FortonnxTensor*)calloc(session->input_count, sizeof(*session->inputs));
  session->outputs = (FortonnxTensor*)calloc(session->output_count, sizeof(*session->outputs));
  if (session->inputs == NULL || session->outputs == NULL) {
    return fortonnx_store_message(session, "failed to allocate ONNX tensor metadata");
  }
  for (index = 0; index < session->input_count; ++index) {
    if (copy_io_name(session, 1, index, &session->inputs[index].name)) return 1;
    if (query_tensor_shape(
            session, 1, index, &session->inputs[index].shape,
            &session->inputs[index].rank)) return 1;
  }
  for (index = 0; index < session->output_count; ++index) {
    if (copy_io_name(session, 0, index, &session->outputs[index].name)) return 1;
    if (query_tensor_shape(
            session, 0, index, &session->outputs[index].shape,
            &session->outputs[index].rank)) return 1;
  }
  if (memory_name == NULL) {
    if (fortonnx_store_status(
            session, session->api->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault, &session->memory_info))) return 1;
  } else {
    if (fortonnx_store_status(
            session, session->api->CreateMemoryInfo(
                         memory_name, OrtArenaAllocator, device_id,
                         OrtMemTypeDefault, &session->memory_info))) return 1;
  }
  return fortonnx_store_status(
      session, session->api->CreateIoBinding(session->session, &session->binding));
}

void* fortonnx_cpu_context_create(int intra_op_threads, int inter_op_threads) {
  FortonnxCpuContext* context = (FortonnxCpuContext*)calloc(1, sizeof(*context));
  OrtThreadingOptions* threading_options = NULL;
  OrtStatus* status = NULL;

  if (context == NULL) {
    fortonnx_set_global_error("failed to allocate a CPU context");
    return NULL;
  }
  context->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  context->inter_op_threads = inter_op_threads;
  if (context->api == NULL) {
    snprintf(global_error, sizeof(global_error),
             "ONNX Runtime API version %d is unavailable", ORT_API_VERSION);
    goto fail;
  }
  if (intra_op_threads < 0 || inter_op_threads < 1) {
    fortonnx_set_global_error("CPU threads must satisfy intra_op >= 0 and inter_op >= 1");
    goto fail;
  }
  status = context->api->CreateThreadingOptions(&threading_options);
  if (status != NULL) goto status_fail;
  status = context->api->SetGlobalIntraOpNumThreads(threading_options, intra_op_threads);
  if (status != NULL) goto status_fail;
  status = context->api->SetGlobalInterOpNumThreads(threading_options, inter_op_threads);
  if (status != NULL) goto status_fail;
  status = context->api->CreateEnvWithGlobalThreadPools(
      ORT_LOGGING_LEVEL_WARNING, "fortonnx-cpu", threading_options, &context->env);
  if (status != NULL) goto status_fail;
  context->api->ReleaseThreadingOptions(threading_options);
  return context;

status_fail:
  fortonnx_set_global_error(context->api->GetErrorMessage(status));
  context->api->ReleaseStatus(status);
fail:
  if (threading_options != NULL && context->api != NULL) {
    context->api->ReleaseThreadingOptions(threading_options);
  }
  if (context->env != NULL && context->api != NULL) context->api->ReleaseEnv(context->env);
  free(context);
  return NULL;
}

void fortonnx_cpu_context_destroy(void* opaque) {
  FortonnxCpuContext* context = (FortonnxCpuContext*)opaque;
  if (context == NULL) return;
  if (context->env != NULL && context->api != NULL) context->api->ReleaseEnv(context->env);
  free(context);
}

void* fortonnx_session_create_cpu(const char* model_path, void* opaque_context) {
  return fortonnx_session_create_cpu_with_custom_ops(
      model_path, opaque_context, NULL);
}

void* fortonnx_session_create_cpu_with_custom_ops(
    const char* model_path, void* opaque_context,
    const char* custom_op_library) {
  FortonnxCpuContext* context = (FortonnxCpuContext*)opaque_context;
  FortonnxSession* session = fortonnx_session_allocate();
  if (session == NULL) return NULL;
  if (model_path == NULL || context == NULL || context->api == NULL || context->env == NULL) {
    fortonnx_store_message(session, "invalid CPU context or model path");
    goto fail;
  }
  session->backend = FORTONNX_BACKEND_CPU;
  session->owns_env = 0;
  session->api = context->api;
  session->env = context->env;
  if (fortonnx_store_status(
          session, session->api->CreateSessionOptions(&session->session_options))) goto fail;
  if (fortonnx_register_custom_ops_library(session, custom_op_library)) goto fail;
  if (fortonnx_store_status(
          session, session->api->CreateRunOptions(&session->run_options))) goto fail;
  if (fortonnx_store_status(
          session, session->api->SetSessionGraphOptimizationLevel(
                       session->session_options, ORT_ENABLE_ALL))) goto fail;
  if (fortonnx_store_status(
          session, session->api->DisablePerSessionThreads(session->session_options))) goto fail;
  if (fortonnx_store_status(
          session, session->api->SetSessionExecutionMode(
                       session->session_options,
                       context->inter_op_threads > 1 ? ORT_PARALLEL : ORT_SEQUENTIAL))) goto fail;
  if (fortonnx_finish_session(session, model_path, NULL, 0)) goto fail;
  return session;

fail:
  fortonnx_set_global_error(session->error);
  fortonnx_session_destroy(session);
  return NULL;
}

int fortonnx_session_get_features(
    void* opaque, int64_t* input_features, int64_t* output_features) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  if (session == NULL || input_features == NULL || output_features == NULL) {
    return fortonnx_store_message(session, "invalid argument to get_features");
  }
  if (session->input_count != 1 || session->output_count != 1 ||
      session->inputs[0].rank != 2 || session->outputs[0].rank != 2 ||
      session->inputs[0].shape[1] <= 0 || session->outputs[0].shape[1] <= 0) {
    return fortonnx_store_message(
        session, "features requires one rank-2 input and output with static feature dimensions");
  }
  *input_features = session->inputs[0].shape[1];
  *output_features = session->outputs[0].shape[1];
  return 0;
}

int fortonnx_session_get_tensor_count(
    void* opaque, int is_input, int64_t* count) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  if (session == NULL || count == NULL) {
    return fortonnx_store_message(session, "invalid argument to get_tensor_count");
  }
  *count = (int64_t)(is_input ? session->input_count : session->output_count);
  return 0;
}

static FortonnxTensor* tensor_at(
    FortonnxSession* session, int is_input, int64_t index) {
  size_t count;
  if (session == NULL || index < 0) return NULL;
  count = is_input ? session->input_count : session->output_count;
  if ((uint64_t)index >= (uint64_t)count) return NULL;
  return is_input ? &session->inputs[index] : &session->outputs[index];
}

int fortonnx_session_get_tensor_rank_at(
    void* opaque, int is_input, int64_t index, int64_t* rank) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  FortonnxTensor* tensor = tensor_at(session, is_input, index);
  if (tensor == NULL || rank == NULL) {
    return fortonnx_store_message(session, "invalid tensor index or rank destination");
  }
  *rank = (int64_t)tensor->rank;
  return 0;
}

int fortonnx_session_get_tensor_shape_at(
    void* opaque, int is_input, int64_t index, int64_t* shape, int64_t rank) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  FortonnxTensor* tensor = tensor_at(session, is_input, index);
  if (tensor == NULL || shape == NULL || rank < 1) {
    return fortonnx_store_message(session, "invalid tensor index, shape, or rank");
  }
  if ((uint64_t)rank != (uint64_t)tensor->rank) {
    return fortonnx_store_message(session, "tensor shape rank does not match the model");
  }
  memcpy(shape, tensor->shape, tensor->rank * sizeof(*shape));
  return 0;
}

int fortonnx_session_get_tensor_rank(
    void* opaque, int is_input, int64_t* rank) {
  return fortonnx_session_get_tensor_rank_at(opaque, is_input, 0, rank);
}

int fortonnx_session_get_tensor_shape(
    void* opaque, int is_input, int64_t* shape, int64_t rank) {
  return fortonnx_session_get_tensor_shape_at(opaque, is_input, 0, shape, rank);
}

int fortonnx_session_find_tensor(
    void* opaque, int is_input, const char* name, int64_t* index) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  FortonnxTensor* tensors;
  size_t count;
  size_t candidate;
  if (session == NULL || name == NULL || index == NULL) {
    return fortonnx_store_message(session, "invalid argument to find_tensor");
  }
  tensors = is_input ? session->inputs : session->outputs;
  count = is_input ? session->input_count : session->output_count;
  for (candidate = 0; candidate < count; ++candidate) {
    if (strcmp(tensors[candidate].name, name) == 0) {
      *index = (int64_t)candidate;
      return 0;
    }
  }
  return fortonnx_store_message(session, "tensor name does not exist in the ONNX model");
}

static int shape_matches(
    const int64_t* model_shape, size_t model_rank,
    const int64_t* actual_shape, int64_t actual_rank) {
  size_t axis;
  if (actual_rank < 1 || (uint64_t)actual_rank != (uint64_t)model_rank) return 0;
  for (axis = 0; axis < model_rank; ++axis) {
    if (actual_shape[axis] <= 0) return 0;
    if (model_shape[axis] > 0 && actual_shape[axis] != model_shape[axis]) return 0;
  }
  return 1;
}

static int tensor_bytes(
    FortonnxSession* session, const int64_t* shape, size_t rank, size_t* bytes) {
  size_t count = 1;
  size_t axis;
  for (axis = 0; axis < rank; ++axis) {
    size_t dimension = (size_t)shape[axis];
    if (dimension > SIZE_MAX / count) {
      return fortonnx_store_message(session, "tensor element count overflows size_t");
    }
    count *= dimension;
  }
  if (count > SIZE_MAX / sizeof(float)) {
    return fortonnx_store_message(session, "tensor byte size overflows size_t");
  }
  *bytes = count * sizeof(float);
  return 0;
}

int fortonnx_session_begin_bind(void* opaque) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  if (session == NULL || session->binding == NULL) {
    return fortonnx_store_message(session, "invalid session or I/O binding");
  }
  session->api->ClearBoundInputs(session->binding);
  session->api->ClearBoundOutputs(session->binding);
  release_tensor_values(session);
  return 0;
}

static int bind_one_tensor(
    FortonnxSession* session,
    int is_input,
    int64_t index,
    void* data,
    int memory_backend,
    const int64_t* shape,
    int64_t rank) {
  FortonnxTensor* tensor = tensor_at(session, is_input, index);
  int session_is_cpu;
  int memory_is_cpu;
  size_t bytes = 0;

  if (tensor == NULL || data == NULL || shape == NULL) {
    return fortonnx_store_message(session, "invalid tensor index, buffer pointer, or shape");
  }
  session_is_cpu = session->backend == FORTONNX_BACKEND_CPU;
  memory_is_cpu = memory_backend == FORTONNX_BACKEND_CPU;
  if (session_is_cpu != memory_is_cpu) {
    return fortonnx_store_message(session, "buffer memory does not match the session backend");
  }
  if (!shape_matches(tensor->shape, tensor->rank, shape, rank)) {
    return fortonnx_store_message(
        session, is_input ? "input tensor shape does not match the model"
                          : "output tensor shape does not match the model");
  }
  if (tensor->value != NULL) {
    return fortonnx_store_message(session, "the same model tensor was bound more than once");
  }
  if (tensor_bytes(session, shape, tensor->rank, &bytes)) return 1;
  if (fortonnx_store_status(
          session, session->api->CreateTensorWithDataAsOrtValue(
                       session->memory_info, data, bytes, shape, tensor->rank,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor->value))) return 1;
  if (is_input) {
    return fortonnx_store_status(
        session, session->api->BindInput(session->binding, tensor->name, tensor->value));
  }
  return fortonnx_store_status(
      session, session->api->BindOutput(session->binding, tensor->name, tensor->value));
}

int fortonnx_session_bind_input(
    void* opaque, int64_t index, void* data, int memory_backend,
    const int64_t* shape, int64_t rank) {
  return bind_one_tensor(
      (FortonnxSession*)opaque, 1, index, data, memory_backend, shape, rank);
}

int fortonnx_session_bind_output(
    void* opaque, int64_t index, void* data, int memory_backend,
    const int64_t* shape, int64_t rank) {
  return bind_one_tensor(
      (FortonnxSession*)opaque, 0, index, data, memory_backend, shape, rank);
}

int fortonnx_session_bind_tensor(
    void* opaque,
    void* input,
    void* output,
    int memory_backend,
    const int64_t* input_shape,
    int64_t input_rank,
    const int64_t* output_shape,
    int64_t output_rank) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  if (session == NULL || session->input_count != 1 || session->output_count != 1) {
    return fortonnx_store_message(
        session, "the scalar tensor bind overload requires one input and one output");
  }
  if (fortonnx_session_begin_bind(opaque)) return 1;
  if (fortonnx_session_bind_input(
          opaque, 0, input, memory_backend, input_shape, input_rank)) return 1;
  return fortonnx_session_bind_output(
      opaque, 0, output, memory_backend, output_shape, output_rank);
}

int fortonnx_session_bind(
    void* opaque,
    void* input,
    void* output,
    int memory_backend,
    int64_t batch_size,
    int64_t input_features,
    int64_t output_features) {
  int64_t input_shape[2];
  int64_t output_shape[2];
  input_shape[0] = batch_size;
  input_shape[1] = input_features;
  output_shape[0] = batch_size;
  output_shape[1] = output_features;
  return fortonnx_session_bind_tensor(
      opaque, input, output, memory_backend, input_shape, 2, output_shape, 2);
}

int fortonnx_session_run(void* opaque) {
  FortonnxSession* session = (FortonnxSession*)opaque;
  size_t index;
  if (session == NULL) {
    return fortonnx_store_message(session, "invalid session");
  }
  for (index = 0; index < session->input_count; ++index) {
    if (session->inputs[index].value == NULL) {
      return fortonnx_store_message(session, "not all input buffers have been bound");
    }
  }
  for (index = 0; index < session->output_count; ++index) {
    if (session->outputs[index].value == NULL) {
      return fortonnx_store_message(session, "not all output buffers have been bound");
    }
  }
  if (!session->asynchronous &&
      fortonnx_store_status(
          session, session->api->SynchronizeBoundInputs(session->binding))) return 1;
  if (fortonnx_store_status(
          session, session->api->RunWithBinding(
                       session->session, session->run_options, session->binding))) return 1;
  if (!session->asynchronous &&
      fortonnx_store_status(
          session, session->api->SynchronizeBoundOutputs(session->binding))) return 1;
  return 0;
}

size_t fortonnx_last_error(void* opaque, char* buffer, size_t buffer_size) {
  const FortonnxSession* session = (const FortonnxSession*)opaque;
  const char* message = session != NULL ? session->error : global_error;
  size_t length = strlen(message);
  if (buffer != NULL && buffer_size > 0) {
    size_t copied = length < buffer_size - 1 ? length : buffer_size - 1;
    memcpy(buffer, message, copied);
    buffer[copied] = '\0';
  }
  return length;
}
