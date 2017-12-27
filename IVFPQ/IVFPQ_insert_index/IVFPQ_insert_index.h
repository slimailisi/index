#ifndef IVFPQ_INSERT_INDEX_H
#define IVFPQ_INSERT_INDEX_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <algorithm>

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
	vector<uchar> PQindex;
};

class IVFPQindex{

public:
	IVFPQindex(int maxIndexNum);
	~IVFPQindex();
	void LoadTrainCode(string srcFile);
	void InsertImagePQ();
	void LoadSingleFeatFile(string srcFile);
	void IndexDatabase(vector<string> featFiles);
	void SaveIndex(string desDir);
	void LoadIndex(string srcFile);
	void IndexDatabaseInsert(vector<string> featFiles);

private:
	float** m_ppCoarseCluster;
	float*** m_prodQuantizer;
	float** m_ppFeat;
	ImgNameStruct* m_imgLocation;
	ImgNameStruct* m_imgLocationOri;

	vector<vector<IVFelem> > m_ivfList;
	int m_coarseK;
	int m_pq_m;
	int m_pq_k;
	int m_pq_step;
	int m_featDim;
	int m_frameNum;
	int m_imgNum;
	int m_maxIndexNum;

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