#ifndef IVFPQ_SEARCH_H
#define IVFPQ_SEARCH_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <algorithm>
#include <queue>

typedef unsigned char uchar;

using namespace std;
const int max_path = 260;
struct ImgNameStruct
{
	char ptr[max_path];
};

struct IVFelem
{
	int videoId;
	uchar PQindex[16];
};

class IVFPQsearch
{

public:
	IVFPQsearch();
	~IVFPQsearch();
	void LoadIndex(string srcFile);
	void PerformQuery(string featFile, vector<vector<float> >& matchScore);
	void PerformQuery(string featFile, vector<vector<float> >& matchScore, int nk = 1);
	void LoadSingleFeatFile(string srcFile, float**& m_ppFeat, int& m_frameNum);

	ImgNameStruct* m_imgLocation;

private:
	float** m_ppCoarseCluster;
	float*** m_prodQuantizer;
	int* m_ivfSize;
	IVFelem** m_ivfList;

	int m_coarseK;
	int m_pq_m;
	int m_pq_k;
	int m_pq_step;
	int m_featDim;
	int m_imgNum;

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
