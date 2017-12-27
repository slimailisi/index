CXX = c++
CXXFLAGS = -std=c++0x -O3 -g -Wall -fmessage-length=0

all: multi_frame_insert_index

IVFPQ_insert_index.o: IVFPQ_insert_index.cpp IVFPQ_insert_index.h
	$(CXX) $(CXXFLAGS) -c IVFPQ_insert_index.cpp -o IVFPQ_insert_index.o

multi_frame_insert_index.o: multi_frame_insert_index.cpp
	$(CXX) $(CXXFLAGS) -c multi_frame_insert_index.cpp -o multi_frame_insert_index.o

multi_frame_insert_index: multi_frame_insert_index.o IVFPQ_insert_index.o
	$(CXX) $(CXXFLAGS) -o multi_frame_insert_index multi_frame_insert_index.o IVFPQ_insert_index.o


clean:
	rm IVFPQ_insert_index.o multi_frame_insert_index.o multi_frame_insert_index