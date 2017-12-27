
#include "AKM.h"

int main(int argc, char* argv[])
{
	if (argc != 7)
	{
		cout << "Error in input parameters!\n";
		return -1;
	}


	int maxSampeNum, k, iterNum, maxCoreNum;

	string srcDir = string(argv[1]);
	string desDir = string(argv[2]);
	maxSampeNum = atoi(argv[3]);
	k = atoi(argv[4]);
	iterNum = atoi(argv[5]);
	maxCoreNum = atoi(argv[6]);

	AKM m_aKmeans;

	m_aKmeans.LoadFeatureSample(srcDir, maxSampeNum);
	m_aKmeans.Kmeans_FLANN(k, iterNum, maxCoreNum);
	m_aKmeans.SaveCodebook(desDir);

	return 0;
}



