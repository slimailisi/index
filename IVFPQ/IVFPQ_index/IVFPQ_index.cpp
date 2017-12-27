#include "IVFPQ_index.h"

IVFPQindex::IVFPQindex(int maxIndexNum):m_maxIndexNum(maxIndexNum)
{
	m_ppCoarseCluster = NULL;
	m_prodQuantizer = NULL;
	m_imgLocation = NULL;
	m_ppFeat = NULL;
	m_frameNum = 0;
	m_imgNum = 0;
}

IVFPQindex::~IVFPQindex()
{
	Delete2DArray(m_ppCoarseCluster);
	Delete2DArray(m_ppFeat);
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
}

void IVFPQindex::LoadTrainCode(string srcFile)
{
	cout<<"load training file...."<<endl;
	ifstream fin(srcFile, ios::binary);
	if(!fin.is_open())
	{
		cout<<"Can not open the codebook file."<<endl;
		exit(0);
	}
	else
	{
		fin.read((char*)&m_featDim, sizeof(int));
		fin.read((char*)&m_coarseK, sizeof(int));
		fin.read((char*)&m_pq_m, sizeof(int));
		fin.read((char*)&m_pq_k, sizeof(int));
		m_pq_step = m_featDim / m_pq_m;
		cout<<m_pq_step<<endl;
    
    		Init2DArray(m_ppCoarseCluster,m_coarseK,m_featDim);
                cout << "0"<<endl;
		fin.read((char*)m_ppCoarseCluster[0], sizeof(float)*m_coarseK*m_featDim);
                cout<<"1"<<endl;
		m_prodQuantizer = new float**[m_pq_m];
		for(int i=0; i<m_pq_m; i++)
		{
			Init2DArray(m_prodQuantizer[i],m_pq_k,m_pq_step);
			fin.read((char*)m_prodQuantizer[i][0],sizeof(float)*m_pq_k*m_pq_step);
		}
                cout << "2"<<endl;
		fin.close();
	}
}

void IVFPQindex::IndexDatabase(vector<string> featFiles)
{

	cout<<"Index database...."<<endl;
	m_ivfList.resize(m_coarseK);
	
	int num = min(int(featFiles.size()),m_maxIndexNum);
	m_imgLocation = new ImgNameStruct[num];

	for(int i = 0; i < num; i++)
	{
		struct timeval starttime, endtime;
		double timeuse;
		gettimeofday(&starttime, NULL);

		Delete2DArray(m_ppFeat);
		m_frameNum = 0;
		string srcFile = featFiles.at(i);
		LoadSingleFeatFile(srcFile);

		if(m_frameNum > 0)
		{
			InsertImagePQ();

			memset(m_imgLocation[m_imgNum].ptr, 0, max_path * sizeof(char));
			strcpy(m_imgLocation[m_imgNum].ptr, srcFile.c_str());
			m_imgNum++;
		}
		gettimeofday(&endtime, NULL);
		timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec);
		timeuse /= 1000;

		cout<<"Id:" << i <<"    timeuse: "<<timeuse<<endl;
		
	}
	
}

void IVFPQindex::InsertImagePQ()
{
	
	for(int f = 0; f < m_frameNum; f++)
	{
		
		float tmp;
		int vw = -1;
		float dismin = UINT_MAX;
		
		for(int i = 0; i < m_coarseK; i++)
		{
			
			float distmp = 0.0f;
			for(int j = 0; j < m_featDim; j++)
			{
				tmp = m_ppFeat[f][j] - m_ppCoarseCluster[i][j];
				distmp += tmp*tmp;
			}

			if(distmp < dismin)
			{
				dismin = distmp;
				vw = i;
			}
		}

		IVFelem elem;
		elem.videoId = m_imgNum;
		float* feat_res = new float[m_featDim];
		for(int i = 0; i < m_featDim; i++)
		{
			feat_res[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
		}

		for(int i = 0; i < m_pq_m; i++)
		{
			float dismin1 = UINT_MAX;
			int vw1 = -1;
			for(int j = 0; j < m_pq_k; j++)
			{
				float distmp1 = 0.0f;
				float tmp;
				for(int k = 0; k < m_pq_step; k++)
				{
					tmp = feat_res[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
					distmp1 += tmp*tmp;
				}
				if(distmp1 < dismin1)
				{
					dismin1 = distmp1;
					vw1 = j;
				}
			}
			elem.PQindex.push_back(vw1);
		}

		m_ivfList.at(vw).push_back(elem);
		delete[] feat_res;
		feat_res = NULL;
	}
	
}

void IVFPQindex::LoadSingleFeatFile(string srcFile)
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



void IVFPQindex::SaveIndex(string desDir)
{
	cout<<"save index file..."<<endl;
	string strImgNum, strK, strDim, str_pq_m, str_pq_k;
	stringstream sin_str;
	sin_str << m_imgNum;
	sin_str >> strImgNum;
	sin_str.clear();

	sin_str << m_coarseK;
	sin_str >> strK;
	sin_str.clear();

	sin_str << m_featDim;
	sin_str >> strDim;
	sin_str.clear();

	sin_str << m_pq_m;
	sin_str >> str_pq_m;
	sin_str.clear();

	sin_str << m_pq_k;
	sin_str >> str_pq_k;
	sin_str.clear();
	
	string m_desDir = desDir + "/OPQ_Index_db_" + strImgNum +"_dim_" + strDim + "_k_" + strK +"_PQ_m"+str_pq_m + "_k"+str_pq_k+ ".fvecs";
	ofstream outFile(m_desDir.c_str(), ios::binary);

	outFile.write((char*)&m_featDim, sizeof(int));
	outFile.write((char*)&m_coarseK, sizeof(int));
	outFile.write((char*)&m_pq_m, sizeof(int));
	outFile.write((char*)&m_pq_k, sizeof(int));
	outFile.write((char*)&m_imgNum, sizeof(int));
	outFile.write((char*)m_ppCoarseCluster[0], sizeof(float) * m_featDim * m_coarseK);

	for(int i=0;i<m_pq_m;i++)
	{
		outFile.write((char*)m_prodQuantizer[i][0], sizeof(float) * m_pq_step * m_pq_k);
	}
        
        ofstream fout("log.txt");
	for(int i = 0; i < m_coarseK; i++)
	{
		int n = m_ivfList.at(i).size();
                fout<<n<<endl;
		outFile.write((char*)&n, sizeof(int));
		for(int j = 0; j < n; j++)
		{
			outFile.write((char*)&m_ivfList.at(i).at(j).videoId, sizeof(int));
			for(int k = 0; k < m_pq_m; k++)
			{
				outFile.write((char*)&m_ivfList.at(i).at(j).PQindex.at(k), sizeof(uchar));
			}
		}
	}
        fout.close();
	for(int i = 0; i < m_imgNum; i++)
	{
		outFile.write(m_imgLocation[i].ptr, sizeof(char)*max_path);
	}

	outFile.close();
}
