SHELL := /bin/sh

CONFIG_FILE ?= config.mk
-include $(CONFIG_FILE)

ifeq ($(origin FC),default)
FC := gfortran
endif
CC ?= cc
AR ?= ar
RANLIB ?= ranlib
PYTHON ?= python3
BACKEND ?= cpu
PREFIX ?= /usr/local
BUILD_DIR ?= build
ONNXRUNTIME_INCLUDE ?=
ONNXRUNTIME_LIBDIR ?=
ONNXRUNTIME_LIBRARY ?=
ONNXRUNTIME_SONAME ?= libonnxruntime.so
RUNTIME_LIBRARY_PATH ?=

FC_NAME := $(notdir $(firstword $(FC)))
CC_NAME := $(notdir $(firstword $(CC)))
SUPPORTED_FC := gfortran flang ifx nvfortran
ifeq ($(filter $(FC_NAME),$(SUPPORTED_FC)),)
$(error unsupported Fortran compiler '$(FC_NAME)'; choose one of: $(SUPPORTED_FC))
endif

include make/compiler/$(FC_NAME).mk

ifeq ($(CC_NAME),nvc)
CFLAGS += -Mnodwarf -notraceback
endif

ifeq ($(BACKEND),cpu)
GPU_C_SOURCE := src/c/fortonnx_ort_gpu_stub.c
else ifeq ($(BACKEND),cuda)
GPU_C_SOURCE := src/c/fortonnx_ort_gpu.c
else ifeq ($(BACKEND),tensorrt)
GPU_C_SOURCE := src/c/fortonnx_ort_gpu.c
else
$(error BACKEND must be cpu, cuda, or tensorrt)
endif

ORT_OPTIONAL_GOALS := configure clean distclean release-check test-python scientific-package-files scientific-package-validate
ifneq ($(strip $(MAKECMDGOALS)),)
ifeq ($(strip $(filter-out $(ORT_OPTIONAL_GOALS),$(MAKECMDGOALS))),)
SKIP_ORT_CHECK := 1
endif
endif

ifeq ($(SKIP_ORT_CHECK),)
ifeq ($(strip $(ONNXRUNTIME_INCLUDE)),)
$(error ONNXRUNTIME_INCLUDE is unset; run './configure' or set it on the make command line)
endif
ifeq ($(strip $(ONNXRUNTIME_LIBDIR)),)
$(error ONNXRUNTIME_LIBDIR is unset; run './configure' or set it on the make command line)
endif
ifeq ($(strip $(ONNXRUNTIME_LIBRARY)),)
$(error ONNXRUNTIME_LIBRARY is unset; run './configure' or set it on the make command line)
endif
endif

CPPFLAGS += -Isrc/c -I$(ONNXRUNTIME_INCLUDE)
CFLAGS += -O2 -fPIC -std=c11 -Wall -Wextra
LDFLAGS += -L$(ONNXRUNTIME_LIBDIR)
LDLIBS += $(ONNXRUNTIME_LIBRARY)
empty :=
space := $(empty) $(empty)
RUNTIME_RPATH_DIRS = $(abspath $(LIB_DIR)) $(ONNXRUNTIME_LIBDIR) $(subst :, ,$(RUNTIME_LIBRARY_PATH))
RPATH_FLAGS = -Wl,--disable-new-dtags $(foreach directory,$(RUNTIME_RPATH_DIRS),-Wl,-rpath,$(directory))
SHARED_RPATH_FLAGS = -Wl,--disable-new-dtags -Wl,-rpath,'$$ORIGIN'

NATIVE_DIR := $(BUILD_DIR)/native/$(FC_NAME)-$(BACKEND)
MOD_DIR := $(NATIVE_DIR)/mod
OBJ_DIR := $(NATIVE_DIR)/obj
LIB_DIR := $(NATIVE_DIR)/lib
BIN_DIR := $(NATIVE_DIR)/bin
TEST_DIR := $(BUILD_DIR)/test
SCIENTIFIC_PACKAGE_SOURCE_DIR := examples/scientific_model_package
SCIENTIFIC_PACKAGE_DIR := $(BUILD_DIR)/examples/scientific_model_package
SCIENTIFIC_PACKAGE_DESCRIPTION := $(SCIENTIFIC_PACKAGE_DIR)/package-description.json
SCIENTIFIC_PACKAGE_DESCRIPTION_DIGEST := $(SCIENTIFIC_PACKAGE_DIR)/package-description.sha256
SCIENTIFIC_PACKAGE_VALIDATOR := $(SCIENTIFIC_PACKAGE_SOURCE_DIR)/validate_package.py
SCIENTIFIC_PACKAGE_VALIDATION_REPORT := $(BUILD_DIR)/examples/scientific_model_package-validation.json
SCIENTIFIC_PACKAGE_VALIDATION_SUMMARY := $(BUILD_DIR)/examples/scientific_model_package-validation.txt
SCIENTIFIC_PACKAGE_SOURCES := \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/build_package.py \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/reference_run.py \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/run.f90 \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/package-template/model-card.md.in \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/package-template/metadata/abstract-model-mapping.md \
	$(SCIENTIFIC_PACKAGE_SOURCE_DIR)/package-template/verification/nominal.csv \
	LICENSE

C_OBJECTS := \
	$(OBJ_DIR)/fortonnx_ort.o \
	$(OBJ_DIR)/$(notdir $(GPU_C_SOURCE:.c=.o))
F_OBJECTS := \
	$(OBJ_DIR)/fortonnx_c_api.o \
	$(OBJ_DIR)/fortonnx.o

ifeq ($(FC_NAME),nvfortran)
ifneq ($(BACKEND),cpu)
FPPFLAGS += -DFORTONNX_WITH_CUDA_FORTRAN
endif
endif

LIB_STATIC := $(LIB_DIR)/libfortonnx.a
LIB_SHARED := $(LIB_DIR)/libfortonnx.so
ORT_RUNTIME_LINK := $(LIB_DIR)/$(ONNXRUNTIME_SONAME)
ifneq ($(BACKEND),cpu)
ORT_PROVIDER_CANDIDATES := libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so
ifeq ($(BACKEND),tensorrt)
ORT_PROVIDER_CANDIDATES += libonnxruntime_providers_tensorrt.so
endif
ORT_PROVIDER_NAMES := $(foreach provider,$(ORT_PROVIDER_CANDIDATES),$(if $(wildcard $(ONNXRUNTIME_LIBDIR)/$(provider)),$(provider)))
ORT_PROVIDER_LINKS := $(addprefix $(LIB_DIR)/,$(ORT_PROVIDER_NAMES))
endif

.PHONY: all configure clean distclean release-check install uninstall examples test test-gpu test-gpu-build test-python info scientific-package-files scientific-package-example scientific-package-validate

all: $(LIB_STATIC) $(LIB_SHARED) $(ORT_RUNTIME_LINK) $(ORT_PROVIDER_LINKS)

configure:
	./configure --fc "$(FC)" --cc "$(CC)" --backend "$(BACKEND)" \
		--onnxruntime-root "$(ONNXRUNTIME_ROOT)" \
		--onnxruntime-include "$(ONNXRUNTIME_INCLUDE)" \
		--onnxruntime-libdir "$(ONNXRUNTIME_LIBDIR)" \
		--onnxruntime-library "$(ONNXRUNTIME_LIBRARY)" \
		--runtime-library-path "$(RUNTIME_LIBRARY_PATH)" \
		--prefix "$(PREFIX)"

$(BUILD_DIR) $(NATIVE_DIR) $(OBJ_DIR) $(MOD_DIR) $(LIB_DIR) $(BIN_DIR) $(TEST_DIR):
	mkdir -p $@

$(ORT_RUNTIME_LINK): | $(LIB_DIR)
	ln -sfn $(ONNXRUNTIME_LIBRARY) $@

$(ORT_PROVIDER_LINKS): $(LIB_DIR)/%: $(ONNXRUNTIME_LIBDIR)/% | $(LIB_DIR)
	ln -sfn $< $@

$(OBJ_DIR)/fortonnx_ort.o: src/c/fortonnx_ort.c src/c/fortonnx_ort.h src/c/fortonnx_ort_internal.h | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/fortonnx_ort_gpu.o: src/c/fortonnx_ort_gpu.c src/c/fortonnx_ort.h src/c/fortonnx_ort_internal.h src/c/fortonnx_onnx_precision.h | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/fortonnx_ort_gpu_stub.o: src/c/fortonnx_ort_gpu_stub.c src/c/fortonnx_ort.h | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/fortonnx_c_api.o: src/fortonnx_c_api.f90 | $(OBJ_DIR) $(MOD_DIR)
	$(FC) $(FFLAGS) $(FMOD_OUT)$(MOD_DIR) -c $< -o $@

$(OBJ_DIR)/fortonnx.o: src/fortonnx.F90 $(OBJ_DIR)/fortonnx_c_api.o | $(OBJ_DIR) $(MOD_DIR)
	$(FC) $(FFLAGS) $(FPPFLAGS) $(FMOD_IN)$(MOD_DIR) $(FMOD_OUT)$(MOD_DIR) -c $< -o $@

$(LIB_STATIC): $(C_OBJECTS) $(F_OBJECTS) | $(LIB_DIR)
	$(AR) rcs $@ $^
	$(RANLIB) $@

$(LIB_SHARED): $(C_OBJECTS) $(F_OBJECTS) | $(LIB_DIR)
	$(FC) $(FSHARED) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(SHARED_RPATH_FLAGS)

$(BIN_DIR)/cpu_minimal: examples/cpu_minimal.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/multi_model_cpu: examples/multi_model_cpu.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/scientific_package_cpu: $(SCIENTIFIC_PACKAGE_SOURCE_DIR)/run.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

examples: $(BIN_DIR)/cpu_minimal $(BIN_DIR)/multi_model_cpu $(BIN_DIR)/scientific_package_cpu

$(SCIENTIFIC_PACKAGE_DESCRIPTION): $(SCIENTIFIC_PACKAGE_SOURCES) | $(BUILD_DIR)
	$(PYTHON) $(SCIENTIFIC_PACKAGE_SOURCE_DIR)/build_package.py --replace $(SCIENTIFIC_PACKAGE_DIR)

$(SCIENTIFIC_PACKAGE_DESCRIPTION_DIGEST): $(SCIENTIFIC_PACKAGE_DESCRIPTION)
	$(PYTHON) $(SCIENTIFIC_PACKAGE_SOURCE_DIR)/build_package.py --replace $(SCIENTIFIC_PACKAGE_DIR)

scientific-package-files: $(SCIENTIFIC_PACKAGE_DESCRIPTION)

scientific-package-validate: $(SCIENTIFIC_PACKAGE_DESCRIPTION_DIGEST) $(SCIENTIFIC_PACKAGE_VALIDATOR)
	$(PYTHON) $(SCIENTIFIC_PACKAGE_VALIDATOR) $(SCIENTIFIC_PACKAGE_DIR) \
		--expected-description-digest-file $(SCIENTIFIC_PACKAGE_DESCRIPTION_DIGEST) \
		--run-verification \
		--report $(SCIENTIFIC_PACKAGE_VALIDATION_REPORT) \
		--summary $(SCIENTIFIC_PACKAGE_VALIDATION_SUMMARY)

scientific-package-example: $(SCIENTIFIC_PACKAGE_DESCRIPTION) $(BIN_DIR)/scientific_package_cpu
	$(BIN_DIR)/scientific_package_cpu $(SCIENTIFIC_PACKAGE_DIR)

$(TEST_DIR)/fp64.onnx: tests/make_fp64_model.py | $(TEST_DIR)
	$(PYTHON) $< $@

$(BIN_DIR)/test_fp64_cpu: tests/test_fp64_cpu.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_fp64_c: tests/test_fp64_c.c src/c/fortonnx_onnx_precision.h $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(TEST_DIR)/linear.onnx: tests/make_test_model.py | $(TEST_DIR)
	$(PYTHON) $< $@

$(TEST_DIR)/cnn.onnx: tests/make_cnn_model.py | $(TEST_DIR)
	$(PYTHON) $< $@

$(TEST_DIR)/multi_input.onnx: tests/make_multi_input_model.py | $(TEST_DIR)
	$(PYTHON) $< $@

$(BIN_DIR)/test_cpu: tests/test_cpu.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_cnn_cpu: tests/test_cnn_cpu.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_multi_input_cpu: tests/test_multi_input_cpu.f90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) | $(BIN_DIR)
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

test: $(TEST_DIR)/fp64.onnx $(BIN_DIR)/test_fp64_cpu $(BIN_DIR)/test_fp64_c \
		$(TEST_DIR)/linear.onnx $(TEST_DIR)/cnn.onnx $(TEST_DIR)/multi_input.onnx \
		$(BIN_DIR)/test_cpu $(BIN_DIR)/test_cnn_cpu $(BIN_DIR)/test_multi_input_cpu
	$(BIN_DIR)/test_fp64_cpu $(TEST_DIR)/fp64.onnx
	$(BIN_DIR)/test_fp64_c $(TEST_DIR)/fp64.onnx $(TEST_DIR)/linear.onnx
	$(BIN_DIR)/test_cpu $(TEST_DIR)/linear.onnx
	$(BIN_DIR)/test_cnn_cpu $(TEST_DIR)/cnn.onnx
	$(BIN_DIR)/test_multi_input_cpu $(TEST_DIR)/multi_input.onnx

$(BIN_DIR)/test_fp64_cuda: tests/test_fp64_cuda.F90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) $(ORT_PROVIDER_LINKS) | $(BIN_DIR)
	@if [ "$(FC_NAME)" != nvfortran ] || [ "$(BACKEND)" = cpu ]; then \
		echo 'test-gpu requires FC=nvfortran and BACKEND=cuda or tensorrt' >&2; exit 2; \
	fi
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_cuda: tests/test_cuda.F90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) $(ORT_PROVIDER_LINKS) | $(BIN_DIR)
	@if [ "$(FC_NAME)" != nvfortran ] || [ "$(BACKEND)" = cpu ]; then \
		echo 'test-gpu requires FC=nvfortran and BACKEND=cuda or tensorrt' >&2; exit 2; \
	fi
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_cnn_cuda: tests/test_cnn_cuda.F90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) $(ORT_PROVIDER_LINKS) | $(BIN_DIR)
	@if [ "$(FC_NAME)" != nvfortran ] || [ "$(BACKEND)" = cpu ]; then \
		echo 'test-gpu requires FC=nvfortran and BACKEND=cuda or tensorrt' >&2; exit 2; \
	fi
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

$(BIN_DIR)/test_multi_input_cuda: tests/test_multi_input_cuda.F90 $(LIB_STATIC) $(ORT_RUNTIME_LINK) $(ORT_PROVIDER_LINKS) | $(BIN_DIR)
	@if [ "$(FC_NAME)" != nvfortran ] || [ "$(BACKEND)" = cpu ]; then \
		echo 'test-gpu requires FC=nvfortran and BACKEND=cuda or tensorrt' >&2; exit 2; \
	fi
	$(FC) $(FFLAGS) $(FMOD_IN)$(MOD_DIR) $< $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) $(RPATH_FLAGS) -o $@

test-gpu-build: $(TEST_DIR)/fp64.onnx $(TEST_DIR)/linear.onnx $(TEST_DIR)/cnn.onnx $(TEST_DIR)/multi_input.onnx \
		$(BIN_DIR)/test_cuda $(BIN_DIR)/test_cnn_cuda $(BIN_DIR)/test_multi_input_cuda $(BIN_DIR)/test_fp64_cuda

# Build-only is safe on systems without a GPU; test-gpu still executes the tests.
test-gpu: test-gpu-build
	$(BIN_DIR)/test_fp64_cuda $(TEST_DIR)/fp64.onnx $(BACKEND)
	$(BIN_DIR)/test_cuda $(TEST_DIR)/linear.onnx $(BACKEND)
	$(BIN_DIR)/test_cnn_cuda $(TEST_DIR)/cnn.onnx $(BACKEND)
	$(BIN_DIR)/test_multi_input_cuda $(TEST_DIR)/multi_input.onnx $(BACKEND)

test-python:
	PYTHONPATH=python/src $(PYTHON) -m pytest -q tests/test_export.py tests/test_scientific_package_example.py tests/test_scientific_package_validator.py tests/test_fp64_package.py

install: all
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include/fortonnx
	install -m 0755 tools/fortonnx-config $(DESTDIR)$(PREFIX)/bin/
	install -m 0644 $(LIB_STATIC) $(LIB_SHARED) $(DESTDIR)$(PREFIX)/lib/
	install -m 0644 $(MOD_DIR)/*.mod $(DESTDIR)$(PREFIX)/include/fortonnx/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/lib/libfortonnx.a $(DESTDIR)$(PREFIX)/lib/libfortonnx.so
	rm -f $(DESTDIR)$(PREFIX)/bin/fortonnx-config
	rm -rf $(DESTDIR)$(PREFIX)/include/fortonnx

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -f $(CONFIG_FILE) *.mod *.smod *.ses
	rm -rf .pytest_cache dist python/src/*.egg-info
	find python tests -type d -name __pycache__ -prune -exec rm -rf {} +

release-check:
	./tools/check-release.sh

info:
	@echo "FC=$(FC)"
	@echo "CC=$(CC)"
	@echo "BACKEND=$(BACKEND)"
	@echo "ONNXRUNTIME_INCLUDE=$(ONNXRUNTIME_INCLUDE)"
	@echo "ONNXRUNTIME_LIBDIR=$(ONNXRUNTIME_LIBDIR)"
	@echo "ONNXRUNTIME_LIBRARY=$(ONNXRUNTIME_LIBRARY)"
	@echo "ONNXRUNTIME_SONAME=$(ONNXRUNTIME_SONAME)"
	@echo "RUNTIME_LIBRARY_PATH=$(RUNTIME_LIBRARY_PATH)"
