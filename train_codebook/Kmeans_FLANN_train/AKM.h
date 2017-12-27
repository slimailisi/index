#ifndef AKM_H
#define AKM_H

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <flann/flann.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

using namespace std;

struct SiftgeoInfo
{
	float x;
	float y;
	float scl;
	float ori;
	float p[6];
	unsigned char feat[128];
};


class AKM
{
public:
	AKM();
	~AKM();
	void LoadFeatureSample(string srcDir, int maxTrainFeatNum);
	void Kmeans_FLANN(int k, int iterNum, int maxCoreNum);
	void SaveCodebook(string desDir);
	void RootTransform();

	template<typename T> void SortData(T *a, int num, unsigned int *m_index, bool flag);
	template<typename T> void QuickSort_ascend(T *a, int low, int high, unsigned int  *m_index);
	template<typename T> void QuickSort_descend(T *a, int low, int high, unsigned int  *m_index);
	void RandomGen(unsigned int *init_index, int featNum, int ClusterNum);

	int m_maxTrainFeatNum;
	int m_featNum;
	int m_codebookSize;
	int m_featDim;

private:
	string m_srcDir;
	string m_desDir;	
	float **m_ppFeatData;
	float **m_ppClusterCenter;
	//struct FLANNParameters m_flannPara;
	//float m_flannSpeedup;
	//flann_index_t m_flannIndex_id;
};

template<typename T> void Delete2DArray(T** &f)
{
	if (f != NULL)
	{
		if (f[0] != NULL)
		{
			delete [] f[0];
			f[0] = NULL;
		}
		delete []f;
		f = NULL;
	}
}




template<typename T> void Init2DArray(T** &f, int row, int col)
{
	T* pf = new T[row * col];
	memset(pf, 0, sizeof(T) * row * col);

	f = new T*[row];
	for (int i = 0; i < row; i ++)
	{
		f[i] = pf + i * col;
	}
}





#endif
