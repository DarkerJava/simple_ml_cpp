#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "network.h"
#include <ctime>
#include <random>
using namespace std;
using namespace Eigen;

int main() {
	srand(clock());
	Eigen::setNbThreads(14);
	clock_t start = clock();
	std::vector<int> sizes{ 1,1,1 };
	network test(sizes);

	std::vector<Eigen::VectorXf> train_in;
	std::vector<Eigen::VectorXf> train_out;
	
	/*
	for (int i = 0; i < 100; i++) {
		VectorXf input = ArrayXf::LinSpaced(100,-1,1);
		input = input.array().pow(i) * ((float) rand() / (float) RAND_MAX);
		VectorXf output(2);
		output << (int) (i % 4 == 2), (int)(i % 4 != 2);
		train_in.push_back(input);
		train_out.push_back(output);
	}*/

	for (int i = 0; i < 100; i++) {
		VectorXf input(1);
		input << i % 2;
		VectorXf output(1);
		output << i + 1 % 2;
		train_in.push_back(input);
		train_out.push_back(output);
	}

	cout << endl << endl;
	for (VectorXf v : train_in) {
		cout << test.compute(v) << endl << endl;
	}

	test.train(train_in, train_out, 100, 0.01);
	for (int i = 0; i < test.layers.size(); i++) {
		cout << test.layers[i].weights << endl << endl << test.layers[i].bias << endl << endl;
	}
	std::cout << "elapsed: " << (clock() - start) / (double)CLOCKS_PER_SEC << endl << endl;
	return 0;
}