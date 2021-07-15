CC=g++
NVCC=nvcc
ARCH=sm_86
INC=-I/usr/local/cuda/include/
CFLAGS=-Wall -ansi -pedantic -std=c++17 
NVCCFLAGS=-Wall -Ofast -fomit-frame-pointer -march=native -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
CVFLAGS=$(shell pkg-config opencv4 --cflags --libs)
CLIBS=-lboost_serialization -lpthread
LIBS=-L/usr/local/cuda/lib64 -lcudart -lpthread -lboost_serialization
CUSOURCES=fisheye2equirect.cu
CPPSOURCES=generateLUT.cpp
CU_EXEC_FILE=$(CUSOURCES:.cu=.out)
CPP_EXEC_FILE=$(CPPSOURCES:.cpp=.out)

all: $(CU_EXEC_FILE) $(CPP_EXEC_FILE)

$(CU_EXEC_FILE):$(CUSOURCES)
	$(NVCC) --use_fast_math -arch=$(ARCH) -O3 -ccbin $(CC) -std=c++17 $(INC) -Xcompiler "$(NVCCFLAGS)" -o $@ $< $(LIBS) $(CVFLAGS)

$(CPP_EXEC_FILE):$(CPPSOURCES)
	$(CC) $(CFLAGS) -o $@ $< $(CVFLAGS) $(CLIBS)

.PHONY:clean

clean:
	rm -rf *.o