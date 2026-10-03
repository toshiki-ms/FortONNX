#ifndef FORTONNX_ONNX_PRECISION_H
#define FORTONNX_ONNX_PRECISION_H

/* Conservative, streaming ONNX protobuf inspection before registering TensorRT.
   TensorRT may downcast DOUBLE weights. Never give it a graph containing DOUBLE,
   including initializers, subgraphs and function bodies. No tensor data is loaded.
   Unreadable/malformed messages also choose CUDA; ORT supplies the load error.
   Field numbers follow onnx.proto. Integer attributes equal to DOUBLE (11) are
   conservatively excluded too, covering Cast's `to` without allocating strings. */
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

enum FortonnxProtoKind {
  FP_MODEL, FP_GRAPH, FP_NODE, FP_ATTRIBUTE, FP_TENSOR, FP_VALUE,
  FP_TYPE, FP_SEQUENCE, FP_MAP, FP_TENSOR_TYPE, FP_SPARSE, FP_FUNCTION
};

static int fortonnx_proto_varint(FILE* file, long end, uint64_t* value) {
  unsigned shift;
  *value = 0;
  for (shift = 0; shift < 64; shift += 7) {
    int byte;
    if (ftell(file) >= end || (byte = fgetc(file)) == EOF) return 1;
    if (shift == 63 && (byte & 0xfe)) return 1;
    *value |= (uint64_t)(byte & 0x7f) << shift;
    if (!(byte & 0x80)) return 0;
  }
  return 1;
}

static int fortonnx_proto_child(int kind, unsigned field) {
  switch (kind) {
    case FP_MODEL:
      if (field == 7) return FP_GRAPH;
      if (field == 25) return FP_FUNCTION;
      break;
    case FP_GRAPH:
      if (field == 1) return FP_NODE;
      if (field == 5) return FP_TENSOR;
      if (field == 11 || field == 12 || field == 13) return FP_VALUE;
      if (field == 15) return FP_SPARSE;
      break;
    case FP_NODE: if (field == 5) return FP_ATTRIBUTE; break;
    case FP_ATTRIBUTE:
      if (field == 5 || field == 10) return FP_TENSOR;
      if (field == 6 || field == 11) return FP_GRAPH;
      if (field == 14 || field == 15) return FP_TYPE;
      if (field == 22 || field == 23) return FP_SPARSE;
      break;
    case FP_VALUE: if (field == 2) return FP_TYPE; break;
    case FP_TYPE:
      if (field == 1 || field == 8) return FP_TENSOR_TYPE;
      if (field == 4 || field == 9) return FP_SEQUENCE;
      if (field == 5) return FP_MAP;
      break;
    case FP_SEQUENCE: if (field == 1) return FP_TYPE; break;
    case FP_MAP: if (field == 2) return FP_TYPE; break;
    case FP_SPARSE: if (field == 1 || field == 2) return FP_TENSOR; break;
    case FP_FUNCTION:
      if (field == 7) return FP_NODE;
      if (field == 11) return FP_ATTRIBUTE;
      if (field == 12) return FP_VALUE;
      break;
    default: break;
  }
  return -1;
}

static int fortonnx_proto_has_double(FILE* file, long end, int kind, unsigned depth) {
  if (depth > 64) return 1;
  while (ftell(file) < end) {
    uint64_t tag, value;
    unsigned field, wire;
    long position;
    if (fortonnx_proto_varint(file, end, &tag) || tag == 0 || (tag >> 3) > UINT_MAX) return 1;
    field = (unsigned)(tag >> 3);
    wire = (unsigned)(tag & 7);
    if (wire == 0) {
      if (fortonnx_proto_varint(file, end, &value)) return 1;
      if (value == 11 && ((kind == FP_TENSOR && field == 2) ||
          (kind == FP_TENSOR_TYPE && field == 1) ||
          (kind == FP_ATTRIBUTE && field == 3))) return 1;
    } else {
      int child = -1;
      if (wire == 2) {
        if (fortonnx_proto_varint(file, end, &value)) return 1;
        child = fortonnx_proto_child(kind, field);
      } else if (wire == 1) {
        value = 8;
      } else if (wire == 5) {
        value = 4;
      } else {
        return 1;
      }
      position = ftell(file);
      if (position < 0 || value > (uint64_t)(end - position)) return 1;
      if (child >= 0) {
        if (fortonnx_proto_has_double(file, position + (long)value, child, depth + 1)) return 1;
      } else if (fseek(file, (long)value, SEEK_CUR) != 0) {
        return 1;
      }
    }
  }
  return ftell(file) != end;
}

static int fortonnx_model_needs_fp64(const char* path) {
  FILE* file = fopen(path, "rb");
  long end;
  int result;
  if (file == NULL) return 1;
  if (fseek(file, 0, SEEK_END) != 0 || (end = ftell(file)) <= 0 ||
      fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 1;
  }
  result = fortonnx_proto_has_double(file, end, FP_MODEL, 0);
  fclose(file);
  return result;
}
#endif
