#include "fortonnx_ort.h"
#include "fortonnx_onnx_precision.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void check(int code, void* session) {
  if (code) {
    char message[2048];
    fortonnx_last_error(session, message, sizeof(message));
    fprintf(stderr, "%s\n", message);
    assert(code == 0);
  }
}

int main(int argc, char** argv) {
  int64_t shape[] = {3, 2};
  int element_type = 0;
  double input[6] = {1.0 + 0x1p-40, -1.0, 0x1p-44, 2.0, -2.0, 0.0};
  struct { unsigned char pre, data[6], post; } enabled = {71, {1,0,1,0,1,0}, 72};
  struct { unsigned char pre, data[6], post; } flags = {73, {0}, 74};
  struct { double pre, data[6], post; } output = {75, {0}, 76};
  char message[2048];
  void* context;
  void* session;
  size_t i;
  assert(argc == 3);
  assert(fortonnx_model_needs_fp64(argv[1]));
  assert(!fortonnx_model_needs_fp64(argv[2]));
  assert(fortonnx_model_needs_fp64("/missing/model.onnx"));
  context = fortonnx_cpu_context_create(1, 1);
  assert(context != NULL);
  session = fortonnx_session_create_cpu(argv[1], context);
  assert(session != NULL);
  check(fortonnx_session_get_tensor_type_at(session, 1, 0, &element_type), session);
  assert(element_type == FORTONNX_FLOAT64);
  check(fortonnx_session_begin_bind(session), session);
  assert(fortonnx_session_bind_input(session, 0, input, FORTONNX_BACKEND_CPU, shape, 2));
  fortonnx_last_error(session, message, sizeof(message));
  assert(strstr(message, "model float64, caller float32") != NULL);
  check(fortonnx_session_bind_input_typed(session, 0, input, FORTONNX_BACKEND_CPU,
                                         shape, 2, FORTONNX_FLOAT64), session);
  check(fortonnx_session_bind_input_typed(session, 1, enabled.data, FORTONNX_BACKEND_CPU,
                                         shape, 2, FORTONNX_BOOL), session);
  check(fortonnx_session_bind_output_typed(session, 0, output.data, FORTONNX_BACKEND_CPU,
                                          shape, 2, FORTONNX_FLOAT64), session);
  check(fortonnx_session_bind_output_typed(session, 1, flags.data, FORTONNX_BACKEND_CPU,
                                          shape, 2, FORTONNX_BOOL), session);
  check(fortonnx_session_run(session), session);
  for (i = 0; i < 6; ++i) {
    assert(output.data[i] == input[i] + (enabled.data[i] ? 0x1p-42 : 0));
    assert(flags.data[i] == (enabled.data[i] && input[i] > 0));
  }
  assert(flags.pre == 73 && flags.post == 74);
  assert(enabled.pre == 71 && enabled.post == 72);
  assert(output.pre == 75 && output.post == 76);
  check(fortonnx_session_begin_bind(session), session);
  shape[0] = INT64_MAX;
  assert(fortonnx_session_bind_input_typed(session, 0, input, FORTONNX_BACKEND_CPU,
                                          shape, 2, FORTONNX_FLOAT64));
  fortonnx_last_error(session, message, sizeof(message));
  assert(strstr(message, "overflows size_t") != NULL);
  fortonnx_session_destroy(session);
  fortonnx_cpu_context_destroy(context);
  puts("FortONNX FP64/bool C test passed (byte sizes, canaries, type mismatch, overflow)");
  return 0;
}
