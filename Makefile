CC=g++

TARGET=%
OBJECT=%.o
HEADER=mLUT.h
CPP=%.cpp
CFLAGS=-Wall -ansi -pedantic -std=c++17 -pthread
CVFLAGS=`pkg-config opencv4 --cflags --libs`
OFLAGS=-lboost_serialization

all: %

%:%.cpp
	$(CC) $(CFLAGS) -o $@ $< $(HEADER) $(CVFLAGS) $(OFLAGS)

.PHONY:clean

clean:
	rm -f *.o
