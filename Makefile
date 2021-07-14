CC=g++
NVCC=nvcc
ARCH=sm_86
INC=-I/usr/local/cuda/include/
NVCCFLAGS=-Wall -Ofast -fomit-frame-pointer -march=native -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
CVFLAGS=$(shell pkg-config opencv4 --cflags --libs)
LIBS=-L/usr/local/cuda/lib64 -lcudart -lpthread -lboost_serialization
CUSOURCES=$(wildcard *.cu)
OBJECTS=$(CUSOURCES:.cu=.o)

all: %

%:%.cu
	$(NVCC) --use_fast_math -arch=$(ARCH) -O3 -ccbin $(CC) -std=c++17 $(INC) -Xcompiler "$(NVCCFLAGS)" -o $@ $^ $(LIBS) $(CVFLAGS)

.PHONY:clean

clean:
	rm -rf *.o