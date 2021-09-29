#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE
#endif
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "network.h"
#include <ctime>
#include <random>
#include <fstream>
#include <algorithm>


using namespace std;
using namespace Eigen;

int main() {
	cin.sync_with_stdio(0);
	cin.tie(0);
	srand(clock());
	Eigen::initParallel();
	Eigen::setNbThreads(6);
	omp_set_num_threads(12);
	std::vector<int> sizes{ 2,12,2 };
	network test(sizes);

	std::vector<Eigen::VectorXf> train_in;
	std::vector<Eigen::VectorXf> train_out;

	ifstream file; file.open("C:\\Users\\Aurko\\Documents\\training.txt");
	vector<pair<VectorXf, VectorXf>> train;

	string line;
	while(getline(file,line)) {
		stringstream sline(line);
		VectorXf input(2);
		VectorXf output(2);
		double x, y; int cat;
		sline >> x >> y >> cat;
		cat--;
		input << x, y;
		output << -1, -1;
		output[cat] += 2;
		train.push_back({ input, output });
	}
	file.close();

	random_shuffle(train.begin(), train.end());
	for (auto pair : train) {
		train_in.push_back(pair.first);
		train_out.push_back(pair.second);
	}
	cout << endl << endl;

	int user_input;
	cin >> user_input;
	while (user_input != 0) {
		clock_t start = clock();
		test.train3(train_in, train_out, user_input, 10);
		std::cout << "elapsed: " << (clock() - start) / (double)CLOCKS_PER_SEC << endl << endl;
		ifstream validfile; validfile.open("C:\\Users\\Aurko\\Documents\\validation.txt");
		int correct = 0; int one = 0;
		ofstream outputf; outputf.open("model_output.txt");
		outputf << "x,y,cat" << endl;


		while (getline(validfile, line)) {
			double x, y; int cat;
			stringstream sline(line);
			sline >> x >> y >> cat;
			//cout << cat << endl;
			VectorXf in(2);
			in << x, y;
			VectorXf out = test.compute(in);
			if (cat == 1) one++;
			if (out[0] >= out[1]) {
				if (cat == 1) correct++;
				outputf << x << ',' << y << ',' << 1 << endl;
			} else {
				if (cat == 2) correct++;

				outputf << x << ',' << y << ',' << 2 << endl;
			}
		}
		validfile.close();
		outputf.close();

		ofstream modelf; modelf.open("model.txt");
		for (int i = 0; i < test.layers.size(); i++) {
			modelf << test.layers[i].weights << endl << test.layers[i].bias << endl;
		}

		modelf.close();
		cout << correct << endl;

		cin >> user_input;
	}
	

	return 0;
}