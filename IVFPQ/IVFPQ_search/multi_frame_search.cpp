#include "IVFPQ_search.h"

void get_vector_of_strings_from_file_lines(const string file_name,
	vector<string>& out) {
	ifstream in_file(file_name.c_str());
	string line;
	out.clear();
	while (!in_file.eof()) {
		if (getline(in_file, line)) out.push_back(line);
	}
	in_file.close();
}



vector<pair<float, uint>> get_sort_results(const vector<float>& match_score, int results_per_query)
{
	int num = match_score.size();
	vector<pair<float, uint> > results(num);
	vector<pair<float, uint> > t(results_per_query);
	for (uint i = 0; i < num; i++)
	{
		results.at(i) = make_pair(match_score.at(i), i);
	}
	//partial_sort(results.begin(), results.begin() + 200, results.end(), cmp_float_uint_descend);
	partial_sort_copy(results.begin(), results.end(), t.begin(), t.end());
	return t;
}

string get_base_name(const string path)
{
	size_t pos = path.find_last_of("/\\");
        string str1 = path.substr(pos+1);
        pos = str1.find_last_of('.');
        string str2 = str1.substr(0,pos);
        return str2;
}

int main(int argc, char* argv[])
{
	if(argc != 6)
	{
		cout<<"Error in input parameters!"<<endl;
		return -1;
	}
	string srcFile = argv[1];
	string queryImgLists = argv[2];
	string desFile = argv[3];
	int num_nearest = atoi(argv[4]);
	int num_show = atoi(argv[5]);

	vector<string> queryPaths;
	get_vector_of_strings_from_file_lines(queryImgLists,queryPaths);

	IVFPQsearch ivfpq_search;
	ivfpq_search.LoadIndex(srcFile);

	int n = queryPaths.size();

	ofstream fout(desFile.c_str());

#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		struct timeval starttime, endtime;
	        vector<vector<float> > score;
		double timeuse;
		gettimeofday(&starttime, NULL);

		ivfpq_search.PerformQuery(queryPaths.at(i), score, num_nearest);
		
		gettimeofday(&endtime,NULL);
		timeuse = 1000000 *(endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec);
		timeuse /=1000;

		int frame_num = score.size();

		
		if(frame_num > 0)
		{
			int img_num = score.at(0).size();
			vector<float> score_total(img_num, 0.0f);
			for(int j = 0; j < frame_num; j++)
			{
				for(int k = 0; k < img_num; k++)
				{
					score_total.at(k) += score.at(j).at(k);
				}
			}
			vector<pair<float, uint> > result = get_sort_results(score_total, num_show);

			#pragma omp critical
			{
				fout << queryPaths.at(i) <<"  "<<i<< endl;
				for(int j = 0; j < num_show; j++)
				{
					fout << get_base_name(ivfpq_search.m_imgLocation[result.at(j).second].ptr) << " ";
				}
				fout<<endl;

				for(int j = 0; j < num_show; j++)
				{
					fout << result.at(j).first << " ";
				}
				fout<<endl<<endl;
			}
		}


		cout << "query ID: " << i <<"     frame_num:"<< frame_num <<"     timeuse:"<<timeuse<<endl;

	}
        fout.close();
	return 0;
}
