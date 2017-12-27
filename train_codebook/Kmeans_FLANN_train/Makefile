FLANN_DIR = /home/yuanyong/lib/flann
CXX=g++
CXXFLAGS = -std=c++0x -O3 -g -Wall -fmessage-length=0 -fopenmp -I$(FLANN_DIR)/include
LDFLAGS = -L$(FLANN_DIR)/lib -lflann_cpp -lflann

all: Kmeans-FLANN

AKM.o: AKM.h AKM.cpp
	$(CXX) $(CXXFLAGS) -c AKM.cpp -o AKM.o

Kmeans-FLANN.o: Kmeans-FLANN.cpp
	$(CXX) $(CXXFLAGS) -c Kmeans-FLANN.cpp -o Kmeans-FLANN.o

Kmeans-FLANN: Kmeans-FLANN.o AKM.o
	$(CXX) $(CXXFLAGS) -o Kmeans-FLANN Kmeans-FLANN.o AKM.o $(LDFLAGS)

clean:
	rm Kmeans-FLANN Kmeans-FLANN.o AKM.o

