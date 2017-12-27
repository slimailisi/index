#include "IVFPQ_search.h"

IVFPQsearch::IVFPQsearch()
{
	m_ppCoarseCluster = NULL;
	m_prodQuantizer = NULL;
	m_imgLocation = NULL;
//	m_ppFeat = NULL;
	m_ivfSize = NULL;
	m_ivfList = NULL;
//	m_frameNum = 0;
}

IVFPQsearch::~IVFPQsearch()
{
	Delete2DArray(m_ppCoarseCluster);
//	Delete2DArray(m_ppFeat);

	if(m_ivfSize != NULL)
	{
		delete[] m_ivfSize;
		m_ivfSize = NULL;
	}

	if(m_imgLocation != NULL)
	{
		delete[] m_imgLocation;
		m_imgLocation = NULL;
	}

	if(m_prodQuantizer != NULL)
	{
		for(int i = 0; i < m_pq_m; i++)
		{
			Delete2DArray(m_prodQuantizer[i]);
		}
		delete []m_prodQuantizer;
		m_prodQuantizer = NULL;
	}

	if(m_ivfList != NULL)
	{
		for(int i=0; i<m_coarseK;i++)
		{
			delete[] m_ivfList[i];
			m_ivfList[i] = NULL;
		}
		delete[] m_ivfList;
		m_ivfList = NULL;
	}
}

void IVFPQsearch::LoadIndex(string srcFile)
{
        cout<<"load index..."<<endl;
	ifstream fin(srcFile, ios::binary);
	if(!fin.is_open())
	{
		cout<<"Can not open the index file."<<endl;
		exit(0);
	}
	else
	{
		fin.read((char*)&m_featDim, sizeof(int));
		fin.read((char*)&m_coarseK, sizeof(int));
		fin.read((char*)&m_pq_m, sizeof(int));
		fin.read((char*)&m_pq_k, sizeof(int));
		m_pq_step = m_featDim/m_pq_m;
		fin.read((char*)&m_imgNum, sizeof(int));

		Init2DArray(m_ppCoarseCluster, m_coarseK, m_featDim);
		fin.read((char*)m_ppCoarseCluster[0], sizeof(float)*m_featDim*m_coarseK);

		m_prodQuantizer = new float**[m_pq_m];
		for(int i=0;i<m_pq_m;i++)
		{
			Init2DArray(m_prodQuantizer[i],m_pq_k,m_pq_step);
			fin.read((char*)m_prodQuantizer[i][0], sizeof(float)*m_pq_k*m_pq_step);
		}
               
              
		m_ivfSize = new int[m_coarseK];
		m_ivfList = new IVFelem*[m_coarseK];
		for(int i=0; i<m_coarseK; i++)
		{
			fin.read((char*)&m_ivfSize[i], sizeof(int));
                        if(m_ivfSize[i]>0)
                        {
			    m_ivfList[i] = new IVFelem[m_ivfSize[i]];
		            fin.read((char*)m_ivfList[i],sizeof(IVFelem)*m_ivfSize[i]);
                        }
		}

		m_imgLocation = new ImgNameStruct[m_imgNum];
		for(int i=0;i<m_imgNum;i++)
		{
			fin.read(m_imgLocation[i].ptr, sizeof(char)*max_path);
		}
		fin.close();
	}
}

void IVFPQsearch::PerformQuery(string featFile, vector<vector<float> >& matchScore)
{
	
        int m_frameNum = 0;
        float** m_ppFeat;
        LoadSingleFeatFile(featFile, m_ppFeat, m_frameNum);

	if(m_frameNum == 0)
	{
		return;
	}

	matchScore.resize(m_frameNum);

	for(int f = 0; f < m_frameNum; f++)
	{

		int vw = 0;
		float dis_min = UINT_MAX;
		float tmp;
		for(int i = 0; i < m_coarseK; i++)
		{
			float dis_tmp = 0.0f;
			for(int j = 0; j < m_featDim; j++)
			{
				tmp = (m_ppFeat[f][j] - m_ppCoarseCluster[i][j]);
				dis_tmp += tmp*tmp;
			}

			if(dis_tmp < dis_min)
			{
				dis_min = dis_tmp;
				vw = i;
			}
		}

		float* feat_res = new float[m_featDim];
		float** PQ_table;
		Init2DArray(PQ_table, m_pq_m, m_pq_k);
		for(int i = 0; i < m_featDim; i++)
		{
			feat_res[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
		}

		for(int i = 0; i < m_pq_m; i++)
		{
			for(int j = 0; j < m_pq_k; j++)
			{
				float dis = 0.0f;
				for(int k = 0; k < m_pq_step; k++)
				{
					tmp = feat_res[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
					dis += tmp*tmp;
				}
				PQ_table[i][j] = dis;
			}
		}

		matchScore.at(f).resize(m_imgNum, INT_MAX);
	
		for(int j = 0; j < m_ivfSize[vw]; j++)
		{
			float score = 0.0f;
			for(int k = 0; k < m_pq_m; k++)
			{
				score += PQ_table[k][m_ivfList[vw][j].PQindex[k]];
			}
			matchScore.at(f).at(m_ivfList[vw][j].videoId) = min(score, matchScore.at(f).at(m_ivfList[vw][j].videoId));
		}
		
		delete[] feat_res;
		Delete2DArray(PQ_table);

	}
        Delete2DArray(m_ppFeat);
}

void IVFPQsearch::PerformQuery(string featFile, vector<vector<float> >& matchScore, int nk)
{
//	Delete2DArray(m_ppFeat);
//	m_frameNum = 0;
        float** m_ppFeat;
        
        int m_frameNum = 0;
	LoadSingleFeatFile(featFile, m_ppFeat, m_frameNum);
	if(m_frameNum == 0)
	{
		return;
	}

	matchScore.resize(m_frameNum);

	priority_queue<pair<float, int> > qu;

	for(int f = 0; f < m_frameNum; f++)
	{

		
		float tmp;

		for(int i = 0; i < m_coarseK; i++)
		{
			float dis_tmp = 0.0f;
			for(int j = 0; j < m_featDim; j++)
			{
                                
				tmp = (m_ppFeat[f][j] - m_ppCoarseCluster[i][j]);
				dis_tmp += tmp*tmp;
			}

			if(i < nk)
			{
				qu.push({dis_tmp,i});
			}
			else
			{
				if(dis_tmp < qu.top().first)
				{
					qu.pop();
					qu.push({dis_tmp,i});
				}
			}
		}
		matchScore.at(f).resize(m_imgNum, INT_MAX);

		for(int nq_id = 0; nq_id < nk; nq_id++)
		{
//                        cout<<"nq_id:"<<nq_id<<endl;
			int vw = qu.top().second;
			qu.pop();

			float* feat_res = new float[m_featDim];
			float** PQ_table;
			Init2DArray(PQ_table, m_pq_m, m_pq_k);
			for(int i = 0; i < m_featDim; i++)
			{
				feat_res[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
			}
  //                      cout<<"flag 0"<<endl;
			for(int i = 0; i < m_pq_m; i++)
			{
				for(int j = 0; j < m_pq_k; j++)
				{
					float dis = 0.0f;
					for(int k = 0; k < m_pq_step; k++)
					{
						tmp = feat_res[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
						dis += tmp*tmp;
					}
					PQ_table[i][j] = dis;
				}
			}
    //                    cout<<"0"<<endl;
			for(int j = 0; j < m_ivfSize[vw]; j++)
			{
				float score = 0.0f;
				for(int k = 0; k < m_pq_m; k++)
				{
					score += PQ_table[k][m_ivfList[vw][j].PQindex[k]];
				}
                                
				matchScore.at(f).at(m_ivfList[vw][j].videoId) = min(score, matchScore.at(f).at(m_ivfList[vw][j].videoId));
			}
      //                  cout<<"1"<<endl;
			delete[] feat_res;
			Delete2DArray(PQ_table);

		}

	}
	Delete2DArray(m_ppFeat);
}

void IVFPQsearch::LoadSingleFeatFile(string srcFile, float**& m_ppFeat, int& m_frameNum)
{
	ifstream fin(srcFile, ios::binary);
	if(!fin.is_open())
	{
		cout<<"Error open the feat file: "<<srcFile<<endl;
		return;
	}
	else
	{
		fin.seekg(0,ios::end);
		int file_size = fin.tellg();
		fin.seekg(0,ios::beg);
		m_frameNum = file_size/(sizeof(float)*m_featDim);
                Init2DArray(m_ppFeat,m_frameNum,m_featDim);
		fin.read((char*)m_ppFeat[0], file_size);
		fin.close();
	}
}
