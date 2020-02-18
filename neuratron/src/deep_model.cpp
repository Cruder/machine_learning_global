#include "neuratron.h"
#include <stdlib.h>
#include <Eigen/Dense>

extern "C"{
    struct DeepModel* create_deep_model(int* neurons_per_layers, int size) {
        struct DeepModel* model = new DeepModel;
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
        const int size_last_layer = neurons_per_layers[size - 1];
        model->d[size - 1] = size_last_layer;
        model->deltas[size - 1] = new double[size_last_layer];
        for (int j = 0; j < size_last_layer; j++) {
            model->x[j] = new double[size_last_layer + 1];

        }
        return model;
    }

    bool train_deep_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size) {
        return true;
    }

    double predict_deep_model_regression(struct DeepModel* model, double* input) {
        return 0.0;
    }

    double predict_deep_model_classification(struct DeepModel* model, double* input) {
        return 0.0;
    }

    int free_deep_model(DeepModel* model) {
        for (size_t i = 0; i < model->layer_count; i++) {
            const int size_layer = neurons_per_layers[i];
            for (size_t j = 0; j < model->size_layer; j++) {
                delete[] model->w[i][j];
            }
            delete[] model->delta[i];
            delete[] model->x[i];
            if (i != model->layer_count - 1) {
                delete[] model->w[i];
            }
        }
        delete[] model->d;
        delete[] model->delta;
        delete[] model->x;
        delete[] model->w;
        delete model;
        return 0;
    }
}
