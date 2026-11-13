#include "fortonnx_ort_internal.h"

#include <stdio.h>
#include <string.h>
#include "fortonnx_onnx_precision.h"
#include "fortonnx_onnx_operators.h"

/* Operators that a CUDA Graph replay does not repeat correctly (kernels of ONNX Runtime 1.30).
   Random numbers: the host draws new seeds in every run and passes them as kernel arguments, so a replayed graph
   repeats the captured numbers. Dropout and its fused com.microsoft forms draw numbers only in training mode,
   which the training_mode input selects: the third input of Dropout(data, ratio, training_mode) and
   BitmaskDropout, the fifth of BiasDropout(data, bias, residual, ratio, training_mode) and BitmaskBiasDropout. */
static const FortonnxOperatorRule fortonnx_replay_random[] = {
    {"RandomNormal", 0}, {"RandomNormalLike", 0}, {"RandomUniform", 0}, {"RandomUniformLike", 0},
    {"Bernoulli", 0}, /* no kernel of its own: ONNX Runtime runs its function body with RandomUniformLike */
    {"Dropout", 3}, {"BitmaskDropout", 3}, {"BiasDropout", 5}, {"BitmaskBiasDropout", 5}, {NULL, 0}};
/* Host temporaries: the CUDA provider's GatherND copies one to the device in every run; a replayed graph repeats
   the copy from the freed address and returns wrong values without an error. */
static const FortonnxOperatorRule fortonnx_replay_host_copy[] = {{"GatherND", 0}, {NULL, 0}};

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
      tensorrt_fp16, tensorrt_engine_cache, tensorrt_cache_path, NULL, 0);
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
    const char* custom_op_library,
    int cuda_graph) {
  FortonnxSession* session = fortonnx_session_allocate();
  OrtCUDAProviderOptionsV2* cuda_options = NULL;
  OrtTensorRTProviderOptionsV2* tensorrt_options = NULL;
  char device_text[32];
  char tf32_text[2];
  const char* cuda_keys[] = {"device_id", "do_copy_in_default_stream", "use_tf32", "enable_cuda_graph"};
  const char* cuda_values[] = {device_text, "1", tf32_text, "0"};
  int tensorrt_engine; /* TensorRT runs the graph; otherwise the CUDA provider does */

  if (session == NULL) return NULL;
  if (model_path == NULL ||
      (backend != FORTONNX_BACKEND_CUDA && backend != FORTONNX_BACKEND_TENSORRT)) {
    fortonnx_store_message(session, "invalid GPU backend or model path");
    goto fail;
  }
  /* Preserve FP64: TensorRT is bypassed, CUDA remains the device provider. */
  tensorrt_engine = backend == FORTONNX_BACKEND_TENSORRT && !fortonnx_model_needs_fp64(model_path);
  if (cuda_graph != 0) {
    const char* random = fortonnx_model_find_operator(model_path, fortonnx_replay_random);
    if (random != NULL) {
      char text[320];
      snprintf(text, sizeof(text),
               "cuda_graph: the model has a %s node%s, whose random numbers come from host-side state of each run "
               "(a replayed graph repeats the captured numbers); switch cuda_graph off",
               random, strstr(random, "Dropout") != NULL ? " with a training_mode input" : "");
      fortonnx_store_message(session, text);
      goto fail;
    }
    if (!tensorrt_engine && fortonnx_model_find_operator(model_path, fortonnx_replay_host_copy) != NULL) {
      fortonnx_store_message(
          session, "cuda_graph: the model has a GatherND node, whose CUDA kernel copies a host temporary in every "
                   "run (a replayed graph gives wrong values); rewrite it as Gather or switch cuda_graph off");
      goto fail;
    }
  }
  session->backend = backend;
  session->cuda_graph = cuda_graph != 0;
  /* One provider captures the graphs: TensorRT keeps one graph per session, the CUDA provider one per graph id. */
  session->graph_limit = tensorrt_engine ? 1 : FORTONNX_CUDA_GRAPH_SLOTS;
  if (cuda_graph != 0 && !tensorrt_engine) cuda_values[3] = "1";
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

  if (tensorrt_engine) {
    char trt_device_text[32];
    const char* trt_fp16_text = tensorrt_fp16 != 0 ? "true" : "false";
    const char* trt_cache_text = tensorrt_engine_cache != 0 ? "true" : "false";
    const char* trt_keys[8];
    const char* trt_values[8];
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
    trt_keys[count] = "trt_cuda_graph_enable";
    trt_values[count++] = cuda_graph != 0 ? "true" : "false";
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
                       cuda_options, cuda_keys, cuda_values, 4))) goto fail;
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

