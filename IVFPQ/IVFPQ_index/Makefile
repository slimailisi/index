CXX = c++
CXXFLAGS = -std=c++0x -O3 -g -Wall -fmessage-length=0

all: multi_frame_index

IVFPQ_index.o: IVFPQ_index.cpp IVFPQ_index.h
	$(CXX) $(CXXFLAGS) -c IVFPQ_index.cpp -o IVFPQ_index.o

multi_frame_index.o: multi_frame_index.cpp
	$(CXX) $(CXXFLAGS) -c multi_frame_index.cpp -o multi_frame_index.o

multi_frame_index: multi_frame_index.o IVFPQ_index.o
	$(CXX) $(CXXFLAGS) -o multi_frame_index multi_frame_index.o IVFPQ_index.o


clean:
	rm IVFPQ_index.o multi_frame_index.o multi_frame_index