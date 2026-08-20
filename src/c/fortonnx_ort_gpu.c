#include "fortonnx_ort_internal.h"

#include <stdio.h>

void* fortonnx_session_create_gpu(
    const char* model_path,
    int backend,
    int device_id,
    void* user_compute_stream,
    int use_tf32,
    int tensorrt_fp16,
    int tensorrt_engine_cache,
    const char* tensorrt_cache_path) {
  return fortonnx_session_create_gpu_with_custom_ops(
      model_path, backend, device_id, user_compute_stream, use_tf32,
      tensorrt_fp16, tensorrt_engine_cache, tensorrt_cache_path, NULL);
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
  FortonnxSession* session = fortonnx_session_allocate();
  OrtCUDAProviderOptionsV2* cuda_options = NULL;
  OrtTensorRTProviderOptionsV2* tensorrt_options = NULL;
  char device_text[32];
  char tf32_text[2];
  const char* cuda_keys[] = {"device_id", "do_copy_in_default_stream", "use_tf32"};
  const char* cuda_values[] = {device_text, "1", tf32_text};

  if (session == NULL) return NULL;
  if (model_path == NULL ||
      (backend != FORTONNX_BACKEND_CUDA && backend != FORTONNX_BACKEND_TENSORRT)) {
    fortonnx_store_message(session, "invalid GPU backend or model path");
    goto fail;
  }
  session->backend = backend;
  session->owns_env = 1;
  if (fortonnx_store_status(
          session, session->api->CreateEnv(
                       ORT_LOGGING_LEVEL_WARNING, "fortonnx-gpu", &session->env))) goto fail;
  if (fortonnx_store_status(
          session, session->api->CreateSessionOptions(&session->session_options))) goto fail;
  if (fortonnx_register_custom_ops_library(session, custom_op_library)) goto fail;
  if (fortonnx_store_status(
          session, session->api->CreateRunOptions(&session->run_options))) goto fail;
  if (fortonnx_store_status(
          session, session->api->SetSessionGraphOptimizationLevel(
                       session->session_options, ORT_ENABLE_ALL))) goto fail;
  if (fortonnx_store_status(
          session, session->api->AddSessionConfigEntry(
                       session->session_options,
                       "session.disable_cpu_ep_fallback", "1"))) goto fail;

  if (backend == FORTONNX_BACKEND_TENSORRT) {
    char trt_device_text[32];
    const char* trt_fp16_text = tensorrt_fp16 != 0 ? "true" : "false";
    const char* trt_cache_text = tensorrt_engine_cache != 0 ? "true" : "false";
    const char* trt_keys[7];
    const char* trt_values[7];
    size_t count = 0;

    snprintf(trt_device_text, sizeof(trt_device_text), "%d", device_id);
    trt_keys[count] = "device_id";
    trt_values[count++] = trt_device_text;
    trt_keys[count] = "trt_fp16_enable";
    trt_values[count++] = trt_fp16_text;
    trt_keys[count] = "trt_engine_cache_enable";
    trt_values[count++] = trt_cache_text;
    trt_keys[count] = "trt_timing_cache_enable";
    trt_values[count++] = trt_cache_text;
    if (tensorrt_engine_cache && tensorrt_cache_path != NULL && tensorrt_cache_path[0] != '\0') {
      trt_keys[count] = "trt_engine_cache_path";
      trt_values[count++] = tensorrt_cache_path;
      trt_keys[count] = "trt_timing_cache_path";
      trt_values[count++] = tensorrt_cache_path;
    }
    if (fortonnx_store_status(
            session, session->api->CreateTensorRTProviderOptions(&tensorrt_options))) goto fail;
    if (fortonnx_store_status(
            session, session->api->UpdateTensorRTProviderOptions(
                         tensorrt_options, trt_keys, trt_values, count))) goto fail;
    if (user_compute_stream != NULL &&
        fortonnx_store_status(
            session, session->api->UpdateTensorRTProviderOptionsWithValue(
                         tensorrt_options, "user_compute_stream", user_compute_stream))) goto fail;
    if (fortonnx_store_status(
            session, session->api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                         session->session_options, tensorrt_options))) goto fail;
    session->api->ReleaseTensorRTProviderOptions(tensorrt_options);
    tensorrt_options = NULL;
  }

  snprintf(device_text, sizeof(device_text), "%d", device_id);
  snprintf(tf32_text, sizeof(tf32_text), "%d", use_tf32 != 0 ? 1 : 0);
  if (fortonnx_store_status(
          session, session->api->CreateCUDAProviderOptions(&cuda_options))) goto fail;
  if (fortonnx_store_status(
          session, session->api->UpdateCUDAProviderOptions(
                       cuda_options, cuda_keys, cuda_values, 3))) goto fail;
  if (user_compute_stream != NULL &&
      fortonnx_store_status(
          session, session->api->UpdateCUDAProviderOptionsWithValue(
                       cuda_options, "user_compute_stream", user_compute_stream))) goto fail;
  if (fortonnx_store_status(
          session, session->api->SessionOptionsAppendExecutionProvider_CUDA_V2(
                       session->session_options, cuda_options))) goto fail;
  session->api->ReleaseCUDAProviderOptions(cuda_options);
  cuda_options = NULL;

  if (user_compute_stream != NULL) {
    session->asynchronous = 1;
    if (fortonnx_store_status(
            session, session->api->AddRunConfigEntry(
                         session->run_options,
                         "disable_synchronize_execution_providers", "1"))) goto fail;
  }
  if (fortonnx_finish_session(session, model_path, "Cuda", device_id)) goto fail;
  return session;

fail:
  if (cuda_options != NULL && session->api != NULL) {
    session->api->ReleaseCUDAProviderOptions(cuda_options);
  }
  if (tensorrt_options != NULL && session->api != NULL) {
    session->api->ReleaseTensorRTProviderOptions(tensorrt_options);
  }
  fortonnx_set_global_error(session->error);
  fortonnx_session_destroy(session);
  return NULL;
}

