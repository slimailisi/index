#include "train_PQ_codebook.h"

int main(int argc, char* argv[])
{
	if (argc != 8)
	{
		cout << "Error in input parameters!\n";
		return -1;
	}


	int maxSampeNum, k, featDim, pq_m, pq_k;

	string srcDir = string(argv[1]);
	string desDir = string(argv[2]);
	maxSampeNum = atoi(argv[3]);
	k = atoi(argv[4]);
	featDim = atoi(argv[5]);
	pq_m = atoi(argv[6]);
	pq_k = atoi(argv[7]);

	TrainPQ m_train_pq(maxSampeNum,featDim,k,pq_k,pq_m);

	m_train_pq.LoadFeatureSample(srcDir);
	m_train_pq.IFVPQ();
	m_train_pq.SaveCodebook(desDir);

	return 0;
}