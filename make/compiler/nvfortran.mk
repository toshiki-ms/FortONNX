FFLAGS ?= -O2 -fPIC -Mnodwarf -notraceback
ifneq ($(BACKEND),cpu)
FFLAGS += -cuda -gpu=nodebug,nolineinfo
endif
FMOD_OUT := -module 
FMOD_IN := -I
FSHARED := -shared -Mnorpath
