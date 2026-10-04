# ONNX BOOL buffers require C-compatible 0/1 logical values.
FFLAGS ?= -O2 -fPIC -Mnodwarf -notraceback -Munixlogical
ifneq ($(BACKEND),cpu)
FFLAGS += -cuda -gpu=nodebug,nolineinfo
endif
FMOD_OUT := -module 
FMOD_IN := -I
FSHARED := -shared -Mnorpath
