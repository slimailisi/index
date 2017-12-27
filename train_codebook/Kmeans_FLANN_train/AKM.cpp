
#include "AKM.h"


AKM::AKM()
{
	m_featDim = 128;
	m_ppClusterCenter = NULL;
	m_ppFeatData = NULL;
}

AKM::~AKM()
{
	Delete2DArray(m_ppClusterCenter);
	Delete2DArray(m_ppFeatData);

	//flann_free_index(m_flannIndex_id, &m_flannPara);
}


void AKM::LoadFeatureSample(string srcDir, int maxSampeNum)
{
	m_srcDir = srcDir;
	m_maxTrainFeatNum = maxSampeNum;

	FILE* pFile = fopen(m_srcDir.c_str(), "rb");
	if (pFile == NULL)
	{
		cout << "Fail to open the source file " << srcDir << endl;
		return;
	}
	else
	{
		cout << "Load training features, please wait...\n";
		fseek(pFile, 0, SEEK_END);
		long long nFileLen = ftell(pFile);
		fseek(pFile, 0, SEEK_SET);
		int f_size = m_featDim*sizeof(float);           
		int num = int(nFileLen / f_size);
		m_featNum = min(num, m_maxTrainFeatNum);
		
		
		Init2DArray(m_ppFeatData, m_featNum, m_featDim);
		int numR = 200000;
		float** ppBuffer = NULL;
		Init2DArray(ppBuffer, numR, m_featDim);
		int nRead;

		if (m_featNum < numR)
		{
			nRead = fread(ppBuffer[0], sizeof(float),m_featDim*m_featNum, pFile);
			for (int m = 0; m < m_featNum; m++)
			{
				for (int n = 0; n < m_featDim; n++)
				{
					m_ppFeatData[m][n] = float(ppBuffer[m][n]);
				}
			}

		}
		else
		{
			int readTimes = m_featNum / numR;	
			int inc = 0;
			for (int i = 0; i < readTimes; i++)
			{
				nRead = fread(ppBuffer[0], sizeof(float),m_featDim*numR, pFile);
				
				for (int m = 0; m < numR; m++)
				{
					for (int n = 0; n < m_featDim; n++)
					{
						m_ppFeatData[m + inc][n] = float(ppBuffer[m][n]);
					}
				}
				inc += numR;
			}
			int tmp = m_featNum - readTimes * numR;
			if (tmp > 0)
			{
				nRead = fread(ppBuffer[0], sizeof(float),m_featDim*tmp, pFile);
				for (int m = 0; m < tmp; m++)
				{
					for (int n = 0; n < m_featDim; n++)
					{
						m_ppFeatData[m + inc][n] = float(ppBuffer[m][n]);
					}
				}
				inc += tmp;
				assert(inc == m_featNum);
			}
		}
		
		fclose(pFile);	
		Delete2DArray(ppBuffer);
		cout << m_featNum << "  SIFT features loaded!\n";
	}
	
}


void AKM::RootTransform()
{

	for (int m = 0; m < m_featNum; m++)
	{
		float s = 0.0f;
		for (int n = 0; n < m_featDim; n++)
		{
			s += m_ppFeatData[m][n];
		}
		for (int n = 0; n < m_featDim; n++)
		{
			m_ppFeatData[m][n] = sqrt(m_ppFeatData[m][n] / s);
		}
	}
}


void AKM::SaveCodebook(string desDir)
{
	string strFeatNum, strK, strDim;
	stringstream sin_str;
	sin_str << m_featNum;
	sin_str >> strFeatNum;
	sin_str.clear();

	sin_str << m_codebookSize;
	sin_str >> strK;
	sin_str.clear();

	sin_str << m_featDim;
	sin_str >> strDim;
	sin_str.clear();
	
	m_desDir = desDir + "/AKM_db_" + strFeatNum +"_dim_" + strDim + "_k_" + strK + ".fvecs";

	ofstream outFile(m_desDir.c_str(), ios::binary);

	outFile.write((char*)&m_codebookSize, sizeof(int));
	outFile.write((char*)&m_featDim, sizeof(int));
	outFile.write((char*)m_ppClusterCenter[0], sizeof(float) * m_featDim * m_codebookSize);

	outFile.close();
}

void AKM::Kmeans_FLANN(int k, int iterNum, int maxCoreNum)
{
	m_codebookSize = k;
	int num_feature = m_featNum;
	int maxIteration = iterNum;

	Init2DArray(m_ppClusterCenter, k, m_featDim);

	// randome sample cluster center
	unsigned int *pRandIndex = new unsigned int[k];
	srand((unsigned)time(NULL));   
	RandomGen(pRandIndex, num_feature, k);
	for (int i = 0; i < k; i++)
	{
		memcpy(m_ppClusterCenter[i], m_ppFeatData[pRandIndex[i]], sizeof(float) * m_featDim);
	}

	//
	int *pAssignID = new int[num_feature];
	float *pDists = new float[num_feature];
	int *pClusSize = new int[m_codebookSize];
	double *pCostInter = new double[maxIteration];
	memset(pCostInter, 0, sizeof(double) * maxIteration);
	for (int i = 0; i < maxIteration; i++)
	{		
		struct timeval starttime, endtime;
		double timeuse;
		gettimeofday(&starttime, NULL);
		
		cout << "OMP for feature ANN search...\n";
		int step = num_feature / maxCoreNum;
		float *pCoreCost = new float[maxCoreNum]; 
		memset(pCoreCost, 0, sizeof(float) * maxCoreNum);
		memset(pDists, 0, sizeof(float) * num_feature);

#pragma omp parallel for
		for (int m = 0; m < maxCoreNum; m++)
		{
			float m_flannSpeedup;
			struct FLANNParameters m_flannPara;
			m_flannPara = DEFAULT_FLANN_PARAMETERS;
			m_flannPara.algorithm = FLANN_INDEX_KDTREE;
			m_flannPara.log_level = FLANN_LOG_INFO;
			m_flannPara.trees = 12;
			m_flannPara.checks = 128; 			
			//m_flannPara.build_weight = 0.01f;
			//m_flannPara.memory_weight = 0.0f;
			//m_flannPara.sample_fraction = 0.1f;

			//flann_index_t m_flannIndex_id = flann_build_index(m_ppClusterCenter[0], k, 128, &m_flannSpeedup, &m_flannPara);// "Computing FLANN index.\n";
			//int tmp = flann_find_nearest_neighbors_index(m_flannIndex_id, m_ppFeatData[m * step], step, pAssignID + m * step, pDists + m * step, 1, &m_flannPara);		
			//for(int j = m * step; j < (m+1) * step; ++j)                      // feature assignment
			//{	
			//	float dists = 0.0f;
			//	int tmp = flann_find_nearest_neighbors_index(m_flannIndex_id, m_ppFeatData[j], 1, pAssignID + j, &dists, 1, &m_flannPara);				
			//	pCoreCost[m] += dists;
			//}
			/*flann_free_index(m_flannIndex_id, &m_flannPara);*/
			flann_find_nearest_neighbors(m_ppClusterCenter[0], k, 128, m_ppFeatData[step * m], step, pAssignID + step * m, pDists + step * m, 1, &m_flannPara);
		}
		/*for (int m = 0; m < maxCoreNum; m++)
		{
			pCostInter[i] += pCoreCost[m];0
		}*/
		for (int m = 0; m < num_feature; m++)
		{
			pCostInter[i] += sqrt(pDists[m]);
		}
		delete []pCoreCost;
		pCoreCost = NULL;

		//=============================Linear scan to test the precision================================================================
		float validCount = 0;
		int testNum = min(50000, num_feature);
		int *pTestInd = new int[testNum];

		for(int j = 0; j < testNum; ++j)                      // feature assignment
		{	
			float minDis = 100000000.0f;
			int minInd = -1;
			for(int m = 0; m < m_codebookSize; m++)  
			{
				float tmpDis = 0.0f;
				for(int n = 0; n < m_featDim; n++)
				{
					float tv = m_ppFeatData[j][n] - m_ppClusterCenter[m][n];
					tmpDis += tv * tv;
				}
				tmpDis = sqrt(tmpDis);
				if (tmpDis < minDis)
				{
					minDis = tmpDis;
					minInd = m;
				}
			}
			pTestInd[j] = minInd;
		}

		for(int j = 0; j < testNum; ++j)   
		{
			if (pTestInd[j] == pAssignID[j])
			{
				validCount++;
			}
		}
		delete []pTestInd;
		pTestInd = NULL;
		float validPrecision = validCount / testNum;
		cout << "Precision of ANN : " << validPrecision  << endl;
		//=============================================================================================

		

		memset(m_ppClusterCenter[0], 0, sizeof(float) * k * m_featDim);
		memset(pClusSize, 0, sizeof(int) * k);		
		for(int j = 0; j < num_feature; ++j)                      // cluster size
		{	
			pClusSize[pAssignID[j]]++;
		}



		for (int j = 0; j < num_feature; ++j)                      // accumulated assigned feature
		{
			int id = pAssignID[j];
			for (int n = 0; n < m_featDim; n++)
			{
				m_ppClusterCenter[id][n] += m_ppFeatData[j][n];
			}
		}

//#pragma omp parallel for
		for (int j = 0; j < k; ++j)                      //
		{
			if (pClusSize[j] <= 0)
			{
				cout << "Warning: empty cluster 1: " << j << endl;
		
					while (1)
					{
						srand((unsigned)time(NULL));
						int rnd = rand() % m_featNum;
						if (pClusSize[pAssignID[rnd]]>1)
						{
							pClusSize[j] += 1;
							pClusSize[pAssignID[rnd]] -= 1;
							for (int n = 0; n < m_featDim; n++)
							{
								m_ppClusterCenter[j][n] += m_ppFeatData[j][n];
								m_ppClusterCenter[pAssignID[rnd]][n] -= m_ppFeatData[rnd][n];
							}
							break;
						}				
					
				}


			}
		}




		

		for (int j = 0; j < k; ++j)                      // re-generate cluster center
		{
			if (pClusSize[j] > 0)
			{
				for (int n = 0; n < m_featDim; n++)
				{
					m_ppClusterCenter[j][n] /= pClusSize[j];
				}
			}
			else
			{
				cout << "Warning: empty cluster 2" << j << endl;
				for (int n = 0; n < m_featDim; n++)
				{
					srand((unsigned)time(NULL));
					int rnd = rand() % m_featNum;
					m_ppClusterCenter[j][n] = m_ppFeatData[j][n];
					pClusSize[j] += 1;
					pClusSize[rnd] -= 1;
					m_ppClusterCenter[pAssignID[rnd]][n] -= m_ppFeatData[rnd][n];
				}


			}
		}
	
		gettimeofday(&endtime,NULL);
		timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;  
        timeuse/=1000;
		if (pCostInter[i] < 0)
		{
			cout << "Warning: cost less than 0!" << endl;
		}
		else
		{
			cout << "Iteration " << i << " - Cost : " << pCostInter[i];
			cout << "     Time cost : " << timeuse << " ms\n\n";
		}
		
	}

	delete []pRandIndex;
	pRandIndex = NULL;
	delete []pClusSize;
	pClusSize = NULL;
	delete []pAssignID;
	pAssignID = NULL;
	delete []pCostInter;
	pCostInter = NULL;
	delete []pDists;
	pDists = NULL;
}

template<typename T> void AKM::SortData(T *a, int num, unsigned int *m_index, bool flag) //  order
{
	for (int i = 0; i < num; i ++)
	{
		m_index[i] = i;
	}
	if (flag)
	{
		QuickSort_ascend(a, 0, num-1, m_index);
	}
	else
	{
		QuickSort_descend(a, 0, num-1, m_index);
	}
}

template<typename T> void AKM::QuickSort_ascend(T *a, int low, int high, unsigned int  *m_index) // ascending order
{
	int i = low;
	int j = high;
	T temp = a[low]; 
	int temp_ind = m_index[low];

	while (i < j)
	{
		while ((i < j) && (temp < a[j])) 
		{
			j--;
		}
		if(i<j)
		{
			a[i] = a[j];
			m_index[i] = m_index[j];

			i++;
		}

		while (i<j && (a[i] < temp))       
		{
			i++;
		}
		if (i<j)
		{
			a[j] = a[i];
			m_index[j] = m_index[i];

			j--;
		}
	}

	a[i] = temp;
	m_index[i] = temp_ind;

	if (low < i)
	{
		QuickSort_ascend(a, low, i-1, m_index);  // ¶Ô×ó×Ó¼¯½øÐÐµÝ¹é
	}
	if (i < high) 
	{
		QuickSort_ascend(a, j+1, high, m_index);  // ¶ÔÓÒ×Ó¼¯½øÐÐµÝ¹é
	}
}


template<typename T> void AKM::QuickSort_descend(T *a, int low, int high, unsigned int  *m_index) // descending order
{
	int i = low;
	int j = high;
	T temp = a[low]; 
	int temp_ind = m_index[low];

	while (i < j)
	{
		while ((i < j) && (temp > a[j])) 
		{
			j--;
		}
		if(i<j)
		{
			a[i] = a[j];
			m_index[i] = m_index[j];

			i ++;
		}

		while (i<j && (a[i] > temp))       
		{
			i++;
		}
		if (i<j)
		{
			a[j] = a[i];
			m_index[j] = m_index[i];

			j--;
		}
	}

	a[i] = temp;
	m_index[i] = temp_ind;

	if (low < i)
	{
		QuickSort_descend(a, low, i-1, m_index);  
	}
	if (i < high) 
	{
		QuickSort_descend(a, j+1, high, m_index);
	}
}

void AKM::RandomGen(unsigned int *init_index, int featNum, int ClusterNum)
{
	srand((unsigned)time(NULL));   
	int *pVal = new int[featNum];
	unsigned int *pIndex = new unsigned int[featNum];
	for (int i = 0; i < featNum; i++)   // random generate N unique values between 0 and sample_count-1                     
	{
		pVal[i] = rand(); 
	}
	SortData(pVal, featNum, pIndex, true);
	memcpy(init_index, pIndex, sizeof(unsigned int) * ClusterNum);

	delete []pIndex;
	pIndex = NULL;
	delete []pVal;
	pVal = NULL;
}
