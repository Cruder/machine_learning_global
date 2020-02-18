#include "neuratron.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>

// std::srand(std::time(nullptr));

extern "C"{
    struct DeepModel* create_deep_model(int* neurons_per_layers, int size) {
        struct DeepModel* model = new DeepModel;
        model->layer_count = size;
        model->d = new int[size];
        model->deltas = new double*[size];
        model->x = new double*[size];
        model->w = new double**[size - 1];
        for(int i = 0; i < size - 1; i++) {
            const int size_layer = neurons_per_layers[i] + 1;
            model->d[i] = size_layer;
            model->deltas[i] = new double[size_layer];
            model->w[i] = new double *[size_layer];
            for (int j = 0; j < size_layer; j++) {
                model->w[i][j] = new double[neurons_per_layers[i + 1]];
                for (int k = 0; k < neurons_per_layers[i + 1]; k++)
                    model->w[i][j][k] = 0;
                model->x[i] = new double[size_layer];
            }
        }
        const int size_last_layer = neurons_per_layers[size - 1] + 1;
        model->d[size - 1] = size_last_layer;
        model->deltas[size - 1] = new double[size_last_layer];
        for (int j = 0; j < size_last_layer; j++) {
            model->x[j] = new double[size_last_layer];

        }
        return model;
    }

    bool train_deep_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size) {
        return true;
    }


    // struct DeepModel{
    //     int layer_count;
    //     double*** w;
    //     double** x;
    //     double** deltas;
    //     int* d;
    // };

    void generate_xs_model(struct DeepModel* model, double* input, std::vector<Eigen::MatrixXd>& matrices) {
        Eigen::MatrixXd x0(1, model->d[0]);

        x0(0, 0) = 1;
        for(int i = 0; i < model->d[0] - 1; ++i) {
            x0(0, i + 1) = input[i];
        }

        std::cout << "x0" << std::endl << x0 << std::endl;

        std::cout << "Map from <=> to : " << model->d[0] << " <=> " << model->d[1] << std::endl;

        // Setup w0
        Eigen::MatrixXd w0(model->d[0], model->d[1]);

        for(int i = 0; i < model->d[0]; ++i) {
            for(int j = 0; j < model->d[1]; ++j) {
                w0(i, j) = model->w[0][i][j];
            }
        }

        std::cout << "w0" << std::endl << w0 << std::endl;
        // End Setup w0

        // Setup x1
        Eigen::MatrixXd x1 = x0 * w0;
        x1(0, 0) = 1;
        std::cout << "x1" << std::endl << x1 << std::endl;
        // End Setup x1

        std::cout << "Map from <=> to : " << model->d[1] << " <=> " << model->d[2] << std::endl;


        // Setup w1
        Eigen::MatrixXd w1(model->d[1], model->d[2]);

        for(int i = 0; i < model->d[1]; ++i) {
            for(int j = 0; j < model->d[2]; ++j) {
                w1(i, j) = model->w[1][i][j];
            }
        }

        std::cout << "w1" << std::endl << w1 << std::endl;
        // End Setup w1

        // Setup x2
        Eigen::MatrixXd x2 = x1 * w1;
        x2(0, 0) = 1;
        std::cout << "x2" << std::endl << x2 << std::endl;
        // End Setup x2

        matrices.push_back(x0);
        matrices.push_back(x1);
        matrices.push_back(x2);
    }

    double* predict_deep_model_regression(struct DeepModel* model, double* input) {
        int output_size = model->d[model->layer_count - 1] - 1;

        std::vector<Eigen::MatrixXd> matrices {};
        generate_xs_model(model, input, matrices);

        double* results = new double[output_size];
        auto matrix = matrices[matrices.size() -1];

        Eigen::Map<Eigen::MatrixXd>(results, matrix.rows(), matrix.cols()) = matrix;

        return results;
    }

    double predict_deep_model_classification(struct DeepModel* model, double* input) {
        return 0.0;
    }

    int free_deep_model(DeepModel* model) {
        for (size_t i = 0; i < model->layer_count; i++) {
            const int size_layer = model->d[i];
            for (size_t j = 0; j < size_layer; j++) {
                delete[] model->w[i][j];
            }
            delete[] model->deltas[i];
            delete[] model->x[i];
            if (i != model->layer_count - 1) {
                delete[] model->w[i];
            }
        }
        delete[] model->d;
        delete[] model->deltas;
        delete[] model->x;
        delete[] model->w;
        delete model;
        return 0;
    }
}
