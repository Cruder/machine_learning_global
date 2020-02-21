#ifndef __NEURATRON_H__
#define __NEURATRON_H__

#include <vector>
#include <Eigen/Dense>


Eigen::MatrixXd matrix_hadamard(Eigen::MatrixXd& x, Eigen::MatrixXd& y);

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
        double* examples;
        double** w;
        int size_input;
        int size_output;
    };

    struct LinearModel* create_linear_model(int input_size, int output_size);
    bool train_linear_model(struct LinearModel* model, double* input, int input_size, double* output, int output_size);
    int free_linear_model(LinearModel* model);
    double* predict_linear_model_classification(struct LinearModel* model, double* input);

    struct DeepModel* create_deep_model(int* neurons_per_layers, int size);
    bool train_deep_regression_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate);
    bool train_deep_classification_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate);

    struct RadialModel* create_radial_model(double* examples, int count_example, int size_input, int size_output, double gamma);

    bool train_radial_regression(RadialModel* model, double* expected_outputs);
    bool train_radial_classification(RadialModel* model, double* expected_outputs, int iteration);
    double* predict_radial_classification(const RadialModel* model, double* batch_input, int size_batch);
    double* predict_radial_regression(const RadialModel* model, double *batch_input, int size_batch);
}

#endif // __NEURATRON_H__
