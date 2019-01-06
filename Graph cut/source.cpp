//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <math.h>
#include <vector>
#include <time.h>
#include "GCoptimization.h"
#include <string>
#include <fstream>
#include <io.h>
using namespace std;

int readFile(string file_name, vector<int>* a){
	ifstream input_file;
	input_file.open(file_name);
	if (input_file.fail()){
		cout << "no such file: " <<file_name<< endl;
		exit(-1);
	}
	int count = 0;
	while (!input_file.eof()){
		int tmp = 0;
		input_file >> tmp;
		if (!input_file.eof()){
			a->push_back(tmp);
			count++;
		}
	}
	input_file.close();
	return count;
}

double calculateAC(string resultFile,int num_label,vector<int> label_[8], int NumOfAllMesh, ofstream &Result, string index, double *area, int *array_bool){
	int* voteResult = new int[NumOfAllMesh];
	double correct_ = 0;
	//-----
	//int correct_num_[7];
	//for (int y = 0; y < 7; y++)correct_num_[y] = 0;
	//data_labels
	ofstream data_labels(resultFile + "voteResult\\" + index + "_labels.txt");
	if (data_labels.fail()){
		cout << "no voteResult file_dir, I will make one." << endl;
		_mkdir((resultFile + "voteResult\\").c_str());
		data_labels.close();
		data_labels.open(resultFile + "voteResult\\" + index + "_labels.txt");
	}
	for (int i = 0; i < NumOfAllMesh; i++){
		int *tmp_vote = new int[num_label];
		for (int v = 0; v < num_label; v++)tmp_vote[v] = 0;
		for (int j = 0; j < 7; j++){
			if (array_bool[j]){
				if (label_[j][i] >= num_label){
					cout << "the label num is too less, please input the correct label num in config.txt" << endl;
					system("pause");
					exit(-1);
				}
				tmp_vote[label_[j][i]]++;
			}
			//----
			//if (label_[j][i] == label_[7][i]){ correct_num_[j]++; }
		}
		voteResult[i] = 0;
		int tmp = 0;
		for (int v = 0; v < num_label; v++){
			if (tmp_vote[v]>tmp){
				tmp = tmp_vote[v];
				voteResult[i] = v;
			}
			//data_label
			data_labels << tmp_vote[v] * 1.0 / 7 << " ";
		}
		data_labels << endl;
		if (voteResult[i] == label_[7][i])correct_ += area[i];
		delete[]tmp_vote;
	}
	//-----
	//for (int z = 0; z < 7;z++){
	//	Result << correct_num_[z] * 1.0 / NumOfAllMesh << endl;
	//}
	//Result << "---------------" << endl;
	//output vote Result, if graphcut can not improve the result, it need this file
	ofstream vote_labels(resultFile + "voteResult\\" + index + "_voteResult.txt");
	for (int i = 0; i < NumOfAllMesh; i++){
		vote_labels << voteResult[i] << endl;
	}
	vote_labels.close();
	delete[] voteResult;
	return correct_;
}

void getJustCurrentDir(string path, vector<string>& files)
{
	long long  hFile = 0;
	//file information
	struct _finddatai64_t fileinfo;
	string p;
	if ((hFile = _findfirsti64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				files.push_back(fileinfo.name);
			}
			else
			{
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
			}
		} while (_findnexti64(hFile, &fileinfo)==0);
		_findclose(hFile);
	}
}

void getJustCurrentFile(string path, vector<string>& files)
{
	long long  hFile = 0;
	//file information 
	struct _finddatai64_t fileinfo;
	string p;
	if ((hFile = _findfirsti64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				;
			}
			else{
				files.push_back(fileinfo.name);
			}
		} while (_findnexti64(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

double GeneralGraph_Optimize(string file_labels, string file_neighbour, string file_nbr_dis, int num_neighbour, int num_labels, int num_mesh_all, double* area, vector<int> true_label, double vote_ac){
	ifstream file_l;
	ifstream file_n;
	ifstream file_n_d;
	ofstream result;
	file_l.open(file_labels);
	file_n.open(file_neighbour);
	file_n_d.open(file_nbr_dis);
	cout << '111' << endl;
	if ((!file_l.is_open()) || !file_n.is_open() || (!file_n_d.is_open())){
		printf("error open file\n");
		cout << file_labels << endl;
		cout << file_neighbour << endl;
		cout << file_nbr_dis << endl;
		char a;
		cin >> a;
		exit(-1);
	}
	int count = 0;
	float *n_dis = new float[num_mesh_all*num_neighbour];
	int *neighbour = new int[num_mesh_all*num_neighbour];
	float *data = new float[num_mesh_all*num_labels];
	//record the distance of neighbour
	while (!file_n_d.eof()){
		if (count == num_mesh_all*num_neighbour)break;
		file_n_d >> n_dis[count];
		count++;
	}
	//printf("%d\n%f", count, n_dis[count - 1]);
	//input data probability
	count = 0;
	while (!file_l.eof()){
		if (count == num_mesh_all*num_labels)break;
		double tmp_label;
		file_l >> tmp_label;
		data[count] = -log((100 + tmp_label)*area[count / num_labels] / 100);
		//data[count] = -log((100 + tmp_label) / 100);
		count++;
	}
	count = 0;
	//input neighbour
	while (!file_n.eof()){
		if (count == num_mesh_all*num_neighbour)break;
		int tmp_neighbour;
		file_n >> tmp_neighbour;
		for (int i = 0; i < num_neighbour; i++){
			file_n >> neighbour[count];
			//printf("%f\n", data[count]);
			count++;
		}
	}
	//printf("\n%d\n%d\n", count, neighbour[count - 1]);

	//=============================Graph Cut==========================================
	//set the distance between labels and different distance makes different results sometimes, it may range from 0.001 to 10
	float final_label_distance = 0.001;
	float final_ac = vote_ac;
	bool flag = true;
	float *smooth = new float[num_labels*num_labels];
	//set the init label distance to smooth
	for (int l1 = 0; l1 < num_labels; l1++)
		for (int l2 = 0; l2 < num_labels; l2++)
			if (l1 == l2){
				smooth[l1*num_labels + l2] = 0.0f;
			}
	//output to file
	string out_label_index = file_labels.substr(0, file_labels.length() - 11);
	string out_label_path = out_label_index	 + ".seg";
	cout << out_label_path << endl;
	ofstream out_label(out_label_path);

	for (int l1 = 0; l1 < num_labels; l1++)
		for (int l2 = 0; l2 < num_labels; l2++)
			if (l1 != l2){
				smooth[l1*num_labels + l2] = final_label_distance;
			}
	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_mesh_all, num_labels);

		// set up the needed data to pass to function for the data costs
		gc->setDataCost(data);
		//use default smooth
		gc->setSmoothCost(smooth);
		for (int i = 0; i < num_mesh_all; i++){
			for (int j = 0; j < num_neighbour; j++){
				gc->setNeighbors(i, neighbour[i*num_neighbour + j], n_dis[i*num_neighbour + j]);
				//cout << neighbour[i*num_neighbour + j]<<" " <<n_dis[i*num_neighbour + j] << endl;
			}
		}
		gc->swap(2);
		count = 0;
		double correct_ = 0;
		for (; count < num_mesh_all; count++){
			//cout <<true_label[count]<<" "<< gc->whatLabel(count) << endl;
			if (true_label[count] == gc->whatLabel(count))
				correct_ += area[count];
		}
		final_ac = correct_;
		for (; count < num_mesh_all; count++){
			//cout <<true_label[count]<<" "<< gc->whatLabel(count) << endl;
			out_label << gc->whatLabel(count) << endl;
		}
		delete gc;
	}
	catch (GCException e){
		e.Report();
	}
	out_label.close();
	delete[] smooth;
	delete[] data;
	delete[] neighbour;
	delete[] n_dis;
	return final_ac;
}

void readArea(string filename, double *a, int NumOfAllMesh){
	ifstream area(filename);
	for (int i = 0; i < NumOfAllMesh; i++){
		area >> a[i];
	}
	
}

double gc(string resultFile, int num_label,int *array_bool)
{
	//ground truth
	const string truthFile = resultFile + "gt\\";
	ofstream allResult(resultFile + "allResult.txt");
	ofstream gcResult(resultFile + "gcResult.txt");
	vector<string> filename;
	getJustCurrentFile(resultFile + "2\\", filename);
	double all_ac = 0;
	double gc_ac = 0;
	for (int i = 0; i < filename.size(); i++){
		string IndexOfModel = filename[i];
		string Index = IndexOfModel.substr(0, IndexOfModel.find("."));
		string head_file = resultFile + "perm\\" + Index;
		string file_area = head_file + "_FaceArea.txt";
		int NumOfAllMesh = 0;
		vector<int> label_[8];
		//read prediction result
		for (int i = 0; i < 7; i++){
			if (array_bool[i] == 1){
				string i_s = to_string(i);
				string result = resultFile + i_s + "\\" + IndexOfModel;
				if (NumOfAllMesh == 0)NumOfAllMesh = readFile(result, &label_[i]);
				else if (NumOfAllMesh != readFile(result, &label_[i])){
					
					printf("error: the num of mesh in this file is not equal with other file\n");\
					cout <<"error file is: "<<i<<"//"<< Index <<".txt"<< endl;
					exit(-1);
				}
			}
		}
		//read groundtruth
		string groundtruth = truthFile + Index + ".seg";
		readFile(groundtruth, &label_[7]);
		double *area = new double[NumOfAllMesh];
		readArea(file_area, area, NumOfAllMesh);
		double tmp_ac = calculateAC(resultFile,num_label,label_, NumOfAllMesh, allResult, Index, area,array_bool);
		allResult << tmp_ac << endl;
		all_ac += tmp_ac;

		//--------------------------Graph Cut-----------------------------
		
		//const string FILE_OUTPUT = resultFile +"gcResult"+ Index+"_output.txt";
		const string FILE_INPUT = truthFile;
		string file_labels = resultFile + "voteResult\\" + Index + "_labels.txt";
		string file_neighbour = head_file + "_adjacentfaces.txt";
		string file_nbr_dis = head_file + "_Dist.txt";
		int num_neighbour = 3;
		
		float final_ac = GeneralGraph_Optimize(file_labels, file_neighbour, file_nbr_dis, num_neighbour, num_label, NumOfAllMesh, area, label_[7], tmp_ac);
		gcResult << final_ac << " " << Index << endl;
		gc_ac += final_ac;
		delete[] area;
	}
	allResult << "avg:" << all_ac / filename.size() << endl;
	gcResult << "avg:" << gc_ac / filename.size() << endl;

	allResult.close();
	gcResult.close();

	return gc_ac/filename.size();
}

double gc_dir(int *array_bool,int num_label){
	char array_[7];
	for (int i = 0; i < 7; i++){
		array_[i] = (char)(array_bool[i]+'0');
	}
	
	vector<string>file_dir_name;
	//current path
	char *buf_path;
	buf_path = _getcwd(NULL,0);
	getJustCurrentDir(buf_path,file_dir_name);
	string dir_path(buf_path);
	//output file of all accuracy: result_0011001.txt
	ofstream resultOfavg(dir_path + "\\" + "result_" + string(array_)+".txt");
	double result = 0;
	//cout << "result_" + string(array_) + ".txt" << endl;
	//cout << buf_path << endl;
	
	//filt . and ..
	for (int i = 2; i < file_dir_name.size(); i++){
		string resultFile;
		resultFile = dir_path+"\\"+file_dir_name[i]+"\\";
		double tmp = gc(resultFile, num_label, array_bool);
		cout << tmp << endl;
		result += tmp;
		resultOfavg << tmp << endl;
		//cout << file_dir_name[i]<<endl;
	}
	resultOfavg << result / (file_dir_name.size()-2) << endl;
	resultOfavg.close();
	printf("\n	Finished %d (%d) clock per sec", clock() / CLOCKS_PER_SEC, clock());
	return result / (file_dir_name.size() - 2);
}
/////////////////////////////////////////////////////////////////////////////////

int main(){
	int array_bool[7];
	char *buf_path;
	buf_path = _getcwd(NULL, 0);
	string dir_path(buf_path);

	ofstream result(dir_path+"\\result_127.txt");
	int num_label;
	//config.txt determined the number of label.
	ifstream config(dir_path + "\\config.txt");
	string tmp;
	getline(config, tmp);
	config >> num_label;
	//defined the used features, there are 7 kinds features. And they have various combinations. If one is used, the bool value is 1.   
	ifstream index(dir_path+"\\index.txt");
	if (index.fail()){
		cout << "no index.txt" << endl;
		exit(-1);
	}
	//index split by '\n'
	while (!index.eof()){
		string array_;
		index >> array_;
		for (int i = 0; i < 7; i++){
			array_bool[i] = array_[i] - '0';
		}
		result << gc_dir(array_bool,num_label) << " ";
		for (int x = 0; x < 7; x++){
			result << array_bool[x];
		}
		result << endl;
	}
	result.close();
	system("pause");
}