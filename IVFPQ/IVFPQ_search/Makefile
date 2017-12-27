CXX = c++
CXXFLAGES = -std=c++0x -O3 -g -Wall -fmessage-length=0 -fopenmp

all: multi_frame_search

IVFPQ_search.o: IVFPQ_search.cpp IVFPQ_search.h
	$(CXX) $(CXXFLAGES) -c IVFPQ_search.cpp -o IVFPQ_search.o

multi_frame_search.o: multi_frame_search.cpp
	$(CXX) $(CXXFLAGES) -c multi_frame_search.cpp -o multi_frame_search.o

multi_frame_search: multi_frame_search.o IVFPQ_search.o
	$(CXX) $(CXXFLAGES) -o multi_frame_search multi_frame_search.o IVFPQ_search.o

clean:
	rm IVFPQ_search.o multi_frame_search.o multi_frame_search
