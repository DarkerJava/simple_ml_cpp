#pragma once
#include <vector>
#include <Eigen/Dense>

struct network {
public:
	std::vector<int> sizes;
	std::vector<struct layer> layers;

	network(std::vector<int> sizes);
	Eigen::VectorXf compute(Eigen::VectorXf);
	float backprop(Eigen::VectorXf x, Eigen::VectorXf truth);
	float backprop2(Eigen::VectorXf x, Eigen::VectorXf truth);
	float backprop3(Eigen::VectorXf x, Eigen::VectorXf truth);
	void train(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate);
	void train2(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate);
	void train3(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate);
	void train4(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate);
};

struct layer {
public:
	Eigen::VectorXf bias;
	Eigen::MatrixXf weights;
	layer(int input, int output);
	Eigen::VectorXf dz;
	Eigen::MatrixXf dW;
	Eigen::VectorXf y;
	Eigen::VectorXf z;
};