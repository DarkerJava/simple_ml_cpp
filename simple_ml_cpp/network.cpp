#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include "network.h"
#include <math.h>
#include <iostream>
#include <queue>
using namespace std;

network::network(std::vector<int> dimensions) : sizes(dimensions) {
	for (int i = 1; i < dimensions.size(); i++) {
		this->layers.push_back(layer(dimensions[i - 1], dimensions[i]));
	}
}

Eigen::VectorXf network::compute(Eigen::VectorXf x) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].z = layers[i].weights * x + layers[i].bias;
		//x = (0.1 * layers[i].z).cwiseMax(layers[i].z);
		x = layers[i].z.array().tanh();
		layers[i].y = x;
	}

	return x;
}

layer::layer(int input, int output) : weights(Eigen::MatrixXf::Random(output, input)), bias(Eigen::VectorXf::Zero(output)){
	//weights.array() /= 2;
	//weights.array() = (weights.array() * 2.50662827463 + weights.array().pow(3) * 2.62493499095 + weights.array().pow(5) * 5.77253353861) * sqrt(1.0 / (float)input);
}



float network::backprop(Eigen::VectorXf x, Eigen::VectorXf truth) {
	Eigen::VectorXf daL = (this->compute(x) - truth);
	
	if (layers.size() == 1) {
		layers[0].dz = daL.cwiseProduct((1.0 - layers[0].y.array().square()).matrix());
		layers[0].dW = layers[0].dz * x.transpose();
	} else {
		layers[layers.size() - 1].dz = daL.cwiseProduct((1 - layers[layers.size() - 1].y.array().square()).matrix());
		layers[layers.size() - 1].dW = Eigen::MatrixXf(layers[layers.size() - 1].dz * layers[layers.size() - 2].y.transpose());
		for (int i = layers.size() - 2; i >= 1; i--) {
			layers[i].dz = (layers[i + 1].dW.transpose() * layers[i + 1].dz).cwiseProduct((1.0 - layers[i].y.array().square()).matrix());
			layers[i].dW = layers[i].dz * layers[i - 1].y.transpose();
		}
		layers[0].dz = (layers[1].dW.transpose() * layers[1].dz).cwiseProduct((1.0 - layers[0].y.array().square()).matrix());
		layers[0].dW = layers[0].dz * x.transpose();
	}
	return daL.norm();
}


float network::backprop2(Eigen::VectorXf x, Eigen::VectorXf truth) {
	Eigen::VectorXf daL = (this->compute(x) - truth);
	if (layers.size() == 1) {
		layers[0].dz = daL.cwiseProduct((1.0 - layers[0].y.array().square()).matrix());
		layers[0].dW = layers[0].dz * x.transpose();
	} else {
		layers[layers.size() - 1].dz = daL.cwiseProduct((1.0 - layers[layers.size() - 1].y.array().square()).matrix());
		layers[layers.size() - 1].dW = Eigen::MatrixXf(layers[layers.size() - 1].dz * layers[layers.size() - 2].y.transpose());
		for (int i = layers.size() - 2; i >= 1; i--) {
			layers[i].dz = (layers[i + 1].dz * layers[i].weights).cwiseProduct((1.0 - layers[i].y.array().square()).matrix());
			layers[i].dW = layers[i].dz * layers[i - 1].y.transpose();
		}
		layers[0].dz = (layers[1].dz * layers[0].weights).cwiseProduct((1.0 - layers[0].y.array().square()).matrix());
		layers[0].dW = layers[0].dz * x.transpose();
	}
	return daL.norm();
}

float network::backprop3(Eigen::VectorXf x, Eigen::VectorXf truth) {
	Eigen::VectorXf daL = (this->compute(x) - truth);
	
	if (layers.size() == 1) {
		layers[0].dz.array() = daL.array() * (1.0 - layers[0].y.array().square());
		layers[0].dW = layers[0].dz * x.transpose();
	} else {
		layers[layers.size() - 1].dz.array() = daL.array() * (1 - layers[layers.size() - 1].y.array().square());
		layers[layers.size() - 1].dW = layers[layers.size() - 1].dz * layers[layers.size() - 2].y.transpose();
		for (int i = layers.size() - 2; i >= 1; i--) {
			layers[i].dz = (layers[i + 1].dW.transpose() * layers[i + 1].dz).array() * (1.0 - layers[i].y.array().square());
			layers[i].dW = layers[i].dz * layers[i - 1].y.transpose();
		}
		layers[0].dz = (layers[1].dW.transpose() * layers[1].dz).array() * (1.0 - layers[0].y.array().square());
		layers[0].dW = layers[0].dz * x.transpose();
	}
	return daL.norm();
}

void network::train3(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	std::queue<float> true_avg;
	float true_error = 0;
	float target = 0.45;
	int adaptive_smoothness = 10;
	for (int iters = 0; iters < iter; iters++) {
		std::vector<network> temp_networks(12, *this);
		std::vector<vector<Eigen::VectorXf>> bias_batch(12,std::vector<Eigen::VectorXf>());
		std::vector<vector<Eigen::MatrixXf>> weight_batch(12, std::vector<Eigen::MatrixXf>());
		std::vector<float> avg_error(12,0);
		std::vector<Eigen::VectorXf> bias_batches;
		std::vector<Eigen::MatrixXf> weight_batches;
		float avg_errors = 0;

		for (int i = 0; i < layers.size(); i++) {
			bias_batches.push_back(Eigen::VectorXf::Zero(layers[i].bias.rows()));
			weight_batches.push_back(Eigen::MatrixXf::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		}
#pragma omp parallel for
		for (int thread = 0; thread < 12; thread++) {
			for (int i = 0; i < layers.size(); i++) {
				bias_batch[thread].push_back(Eigen::VectorXf::Zero(layers[i].bias.rows()));
				weight_batch[thread].push_back(Eigen::MatrixXf::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
			}
		}
		
		int batch_size = 300;
		if (true_error < 1) {
			if (true_avg.size() == adaptive_smoothness) {
				batch_size = batch_size + pow((1 - (true_error - target)), 15) * (in.size() - batch_size);
			} else {
				batch_size = in.size();
			}
		} else {
		}
		vector<int> training_batch(batch_size);

#pragma omp parallel for		
		for (int i = 0; i < batch_size; i++) {
			training_batch[i] = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
		}

#pragma omp parallel for
		for (int thread = 0; thread < 12; thread++) {
			for (int i = thread; i < batch_size; i+= 12) {
				//int train = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
				int train = training_batch[i];
				avg_error[thread] += temp_networks[thread].backprop2(in[train], out[train]) / (float)batch_size;

				for (int i = 0; i < layers.size(); i++) {
					bias_batch[thread][i] += temp_networks[thread].layers[i].dz / (float)batch_size;
					weight_batch[thread][i] += temp_networks[thread].layers[i].dW / (float)batch_size;
				}
			}
		}
		for (int thread = 0; thread < 12; thread++) {
			avg_errors += avg_error[thread];

			for (int i = 0; i < layers.size(); i++) {
				bias_batches[i] += bias_batch[thread][i];
				weight_batches[i] += weight_batch[thread][i];
			}
		}
		true_error += avg_errors / adaptive_smoothness;
		true_avg.push(avg_errors / adaptive_smoothness);
		if (true_avg.size() > adaptive_smoothness) {
			true_error -= true_avg.front();
			true_avg.pop();
			cout << true_error << " " << batch_size << endl;
		}
		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= bias_batches[i] * rate * tanh(exp(true_error - target - 1));
			layers[i].weights -= weight_batches[i] * rate * tanh(exp(true_error - target - 1));
		}
	}
}


void network::train2(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	for (int iters = 0; iters < iter; iters++) {
		
		std::vector<Eigen::VectorXf> bias_batch;
		std::vector<Eigen::MatrixXf> weight_batch;
		float avg_error = 0;
		for (int i = 0; i < layers.size(); i++) {
			bias_batch.push_back(Eigen::VectorXf::Zero(layers[i].bias.rows()));
			weight_batch.push_back(Eigen::MatrixXf::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		}
		int batch_size = in.size();
		for (int i = 0; i < batch_size; i++) {
			int train = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
			//int train = (i + (long long) iters * batch_size) % in.size();
			//int train = i;
			avg_error += backprop2(in[train], out[train]) / (float) batch_size;
			
			for (int j = 0; j < layers.size(); j++) {
				bias_batch[j] += layers[j].dz / (float) batch_size;
				weight_batch[j] += layers[j].dW / (float) batch_size;
			}
		}
		cout << avg_error << endl;
		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= bias_batch[i] * rate;
			layers[i].weights -= weight_batch[i] * rate;
		}
	}
}

void network::train4(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	for (int iters = 0; iters < iter; iters++) {

		std::vector<Eigen::VectorXf> bias_batch;
		std::vector<Eigen::MatrixXf> weight_batch;
		float avg_error = 0;
		for (int i = 0; i < layers.size(); i++) {
			bias_batch.push_back(Eigen::VectorXf::Zero(layers[i].bias.rows()));
			weight_batch.push_back(Eigen::MatrixXf::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		}
		int batch_size = in.size();
		for (int i = 0; i < batch_size; i++) {
			//int train = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
			//int train = (i + (long long)iters * batch_size) % in.size();
			int train = i;
			avg_error += backprop2(in[train], out[train]) / (float)batch_size;

			for (int j = 0; j < layers.size(); j++) {
				bias_batch[j] += layers[j].dz / (float)batch_size;
				weight_batch[j] += layers[j].dW / (float)batch_size;
			}
		}
		cout << avg_error << endl;
		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= bias_batch[i] * rate;
			layers[i].weights -= weight_batch[i] * rate;
		}
	}
}
void network::train(std::vector<Eigen::VectorXf> in, std::vector<Eigen::VectorXf> out, int iter, float rate) {
	for (int iters = 0; iters < iter; iters++) {

		float avg_error;
		//int train = ((float)rand() / (float)RAND_MAX) * (in.size() - 1);
		int train = iters % in.size();
		avg_error = backprop2(in[train], out[train]);

		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= layers[i].dz * rate;
			layers[i].weights -= layers[i].dW * rate;
		}
		cout << avg_error << endl;
	}
}