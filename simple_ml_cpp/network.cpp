#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include "network.h"
#include <math.h>
#include <iostream>

using namespace std;

network::network(std::vector<int> dimensions) : sizes(dimensions) {
	for (int i = 1; i < dimensions.size(); i++) {
		this->layers.push_back(layer(dimensions[i - 1], dimensions[i]));
	}
}

Eigen::VectorXf network::compute(Eigen::VectorXf x) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].z = layers[i].weights * x + layers[i].bias;
		//x = (0.0 * layers[i].z).cwiseMax(layers[i].z);
		x = layers[i].z.array().tanh();
		layers[i].y = x;
	}

	return x;
}

layer::layer(int input, int output) : weights(Eigen::MatrixXf::Random(output, input)), bias(Eigen::VectorXf::Zero(output)){
	weights.array() /= 2;
	weights.array() = (weights.array() * 2.50662827463 + weights.array().pow(3) * 2.62493499095 + weights.array().pow(5) * 5.77253353861) * sqrt(1.0 / (float)input);
}



float network::backprop(Eigen::VectorXf x, Eigen::VectorXf truth) {
	Eigen::VectorXf daL = (this->compute(x) - truth);
	if (layers.size() == 1) {
		layers[0].dz = daL.cwiseProduct((1.0 - layers[0].y.array().tanh().square()).matrix());
		layers[0].dW = layers[0].dz * x.transpose();
	} else {
		layers[layers.size() - 1].dz = daL.cwiseProduct((1 - layers[layers.size() - 1].y.array().tanh().square()).matrix());
		layers[layers.size() - 1].dW = Eigen::MatrixXf(layers[layers.size() - 1].dz * layers[layers.size() - 2].y.transpose());
		for (int i = layers.size() - 2; i >= 1; i--) {
			layers[i].dz = (layers[i + 1].dW.transpose() * layers[i + 1].dz).cwiseProduct(layers[i].y.cwiseSign());
			layers[i].dW = layers[i].dz * layers[i - 1].y.transpose();
		}
		layers[0].dz = (layers[1].dW.transpose() * layers[1].dz).cwiseProduct(layers[0].y.cwiseSign());
		layers[0].dW = layers[0].dz * x.transpose();
	}
	return daL.norm();
}

/*
void network::train2(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	for (int iters = 0; iters < iter; iters++) {

		std::vector<Eigen::VectorXf> bias_batch;
		std::vector<Eigen::MatrixXf> weight_batch;
		float avg_error = 0;
		for (int i = 0; i < layers.size(); i++) {
			bias_batch.push_back(Eigen::VectorXf::Zero(layers[i].bias.rows()));
			weight_batch.push_back(Eigen::MatrixXf::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		}

		for (int i = 0; i < in.size(); i++) {
			avg_error += backprop(in[i], out[i]) / (float) in.size();
			
			for (int i = 0; i < layers.size(); i++) {
				bias_batch[i] += layers[i].dz / (float)in.size();
				weight_batch[i] += layers[i].dW / (float)in.size();
			}
		}
		//cout << avg_error << endl;
		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= bias_batch[i] * rate;
			layers[i].weights -= weight_batch[i] * rate;
		}
	}
}*/

void network::train(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	for (int iters = 0; iters < iter; iters++) {

		float avg_error;
		int train = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
		avg_error = backprop(in[train], out[train]);

		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= layers[i].dz * rate;
			layers[i].weights -= layers[i].dW * rate;
		}
		//cout << avg_error << endl;
	}
}