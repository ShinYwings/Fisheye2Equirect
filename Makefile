CC=g++

CFLAGS=-std=c++17
CVFLAGS=`pkg-config opencv4 --cflags --libs`

% : %.cpp
	$(CC) $(CFLAGS) -o $@ $< $(CVFLAGS)

.PHONY:clean

clean:
	rm -f *.o