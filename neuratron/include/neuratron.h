#ifndef __NEURATRON_H__
#define __NEURATRON_H__

#include <vector>
#include <Eigen/Dense>

extern "C" {
    struct DeepModel{
        int layer_count;
        double*** w;
        double** x;
        double** deltas;
        int* d;
    };

    struct LinearModel{
        double* inputs;
        int sizeInput;
        int sizeOutput;
    };

    struct RadialModel{
        double gamma;
        int examples_count;
        double** w[2];
        int size_input;
        int size_output;
    };

    struct LinearModel* create_linear_model(int input_size, int output_size);
    bool train_linear_model(struct LinearModel* model, double* input, int input_size, double* output, int output_size);
    int free_linear_model(LinearModel* model);

    struct DeepModel* create_deep_model(int* neurons_per_layers, int size);
    bool train_deep_regression_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate);
    bool train_deep_classification_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate);
}

#endif // __NEURATRON_H__
