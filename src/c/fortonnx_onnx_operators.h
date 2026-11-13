#ifndef FORTONNX_ONNX_OPERATORS_H
#define FORTONNX_ONNX_OPERATORS_H

/* Streaming search of an ONNX model file for node operators (NodeProto of onnx.proto: input = field 1,
   op_type = field 4), including subgraphs and function bodies. No tensor data is loaded. */
#include <string.h>
#include "fortonnx_onnx_precision.h"

typedef struct {
  const char* name; /* operator type; NULL ends a list of rules */
  unsigned input;   /* 0: every node of this type; n > 0: only a node whose n-th input is given */
} FortonnxOperatorRule;

static const char* fortonnx_proto_find_operator(
    FILE* file, long end, int kind, unsigned depth, const FortonnxOperatorRule* rules) {
  char type[32];
  size_t type_length = 0;
  unsigned inputs = 0;
  uint64_t given = 0; /* bit n: input n + 1 of this node has a name */
  if (depth > 64) return NULL;
  while (ftell(file) < end) {
    uint64_t tag, value;
    unsigned wire;
    long position;
    int child = -1;
    if (fortonnx_proto_varint(file, end, &tag) || tag == 0 || (tag >> 3) > UINT_MAX) return NULL;
    wire = (unsigned)(tag & 7);
    if (wire == 0) {
      if (fortonnx_proto_varint(file, end, &value)) return NULL;
      continue;
    }
    if (wire == 2) {
      if (fortonnx_proto_varint(file, end, &value)) return NULL;
      child = fortonnx_proto_child(kind, (unsigned)(tag >> 3));
    } else if (wire == 1) {
      value = 8;
    } else if (wire == 5) {
      value = 4;
    } else {
      return NULL;
    }
    position = ftell(file);
    if (position < 0 || value > (uint64_t)(end - position)) return NULL;
    if (kind == FP_NODE && wire == 2 && (tag >> 3) == 1) {
      if (value > 0 && inputs < 64) given |= UINT64_C(1) << inputs;
      inputs += 1;
    } else if (kind == FP_NODE && wire == 2 && (tag >> 3) == 4) {
      type_length = value < sizeof type ? (size_t)value : 0; /* longer than every rule: no match */
      if (fread(type, 1, type_length, file) != type_length) return NULL;
    } else if (child == FP_GRAPH || child == FP_NODE || child == FP_ATTRIBUTE || child == FP_FUNCTION) {
      const char* found = fortonnx_proto_find_operator(file, position + (long)value, child, depth + 1, rules);
      if (found != NULL) return found;
    }
    if (fseek(file, position + (long)value, SEEK_SET) != 0) return NULL;
  }
  if (kind == FP_NODE) {
    for (; rules->name != NULL; ++rules) {
      if (strlen(rules->name) == type_length && memcmp(type, rules->name, type_length) == 0 &&
          (rules->input == 0 || (rules->input <= 64 && ((given >> (rules->input - 1)) & 1) != 0)))
        return rules->name;
    }
  }
  return NULL;
}

/* Name of the first rule that a node of the model meets; NULL: none, or the file is unreadable (ORT supplies the
   load error). */
static const char* fortonnx_model_find_operator(const char* path, const FortonnxOperatorRule* rules) {
  FILE* file = fopen(path, "rb");
  long end;
  const char* found = NULL;
  if (file == NULL) return NULL;
  if (fseek(file, 0, SEEK_END) == 0 && (end = ftell(file)) > 0 && fseek(file, 0, SEEK_SET) == 0)
    found = fortonnx_proto_find_operator(file, end, FP_MODEL, 0, rules);
  fclose(file);
  return found;
}
#endif
