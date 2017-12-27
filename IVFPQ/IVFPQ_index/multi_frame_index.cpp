#include "IVFPQ_index.h"

void get_vector_of_strings_from_file_lines(const string file_name,
	vector<string>& out);

int main(int argc, char* argv[])
{
	if(argc!=5)
	{
		cout<<"Error in input parameters!\n";
		return -1;
	}
	string trainFile = argv[1];
	string imgLists = argv[2];
	string desDir = argv[3];
	int maxImageNum = atoi(argv[4]);

	vector<string> featFiles;
	get_vector_of_strings_from_file_lines(imgLists, featFiles);
	int num = featFiles.size();
        cout<<num<<endl;
	IVFPQindex index(maxImageNum);
	index.LoadTrainCode(trainFile);
	index.IndexDatabase(featFiles);
	index.SaveIndex(desDir);

	return 0;
}

void get_vector_of_strings_from_file_lines(const string file_name,
	vector<string>& out) {
	ifstream in_file(file_name.c_str());
	string line;
	out.clear();
	while (!in_file.eof()) {
		if (getline(in_file, line)) out.push_back(line);
	}
}
