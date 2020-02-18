#include "neuratron.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <ctime>

// std::srand(std::time(nullptr));


void calculus_regression_delta_last_layer(struct DeepModel* model, double* output){
    int count_neuron_last_layer = model->d[model->layer_count-1];
    for(int i = 0; i < count_neuron_last_layer; i++){
        model->deltas[count_neuron_last_layer][i] = model->x[count_neuron_last_layer][i] - output[i];
    }
}

void calculus_classification_delta_last_layer(struct DeepModel* model, double* output){
    int count_neuron_last_layer = model->d[model->layer_count-1];
    for(int i = 0; i < count_neuron_last_layer; i++){
        const double output_Lj = model->x[count_neuron_last_layer][i];
        model->deltas[count_neuron_last_layer][i] = (1 - output_Lj*output_Lj) * (output_Lj - output[i]);
    }
}

void calculus_delta_layers(struct DeepModel* model){
    for(int layer_l = model->layer_count; layer_l > 0; layer_l--){
        for(int i_neuron = 1 ; i_neuron < model->d[layer_l-1]; i_neuron++){
            double sigma_w_delta = 0.;
            for(int j_neuron = 1; j_neuron < model->d[layer_l]; j_neuron++){
                const double delta_l_j = model->deltas[layer_l][j_neuron];
                const double weight_l_ij = model->w[layer_l-1][i_neuron][j_neuron];
                sigma_w_delta+= delta_l_j * weight_l_ij;
            }
            double output_layer_of_i_neuron = model->x[layer_l-1][i_neuron];
            double delta_layer_of_i_neuron = (1 - output_layer_of_i_neuron * output_layer_of_i_neuron) * sigma_w_delta;
            model->deltas[layer_l][i_neuron] = delta_layer_of_i_neuron;
        }
    }
}

void print_a(const double* array, size_t size){
    for(size_t i = 0 ; i < size; i++)
        std::cout << array[i] << " ";
    std::cout << std::endl;
}


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

    bool train_deep_regression_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size) {
        int size_input_layer = model->d[0] - 1;
        int size_output_layer = model->d[model->layer_count-1] - 1;
        int example_counts = input_size / size_input_layer;
        int example_expected_result = output_size / size_output_layer;
        std::cout << "var helper" << std::endl;
        for(int i = 0 ; i < example_counts; i++){
            double* example_input = new double[size_input_layer];
            double* example_output = new double[size_output_layer];
            std::cout << "select range input/output for iteration" << std::endl;
            for(int k = 0 ; k < size_input_layer; k++)
                example_input[k] = input[i*size_input_layer+k];
            for(int k = 0 ; k < size_output_layer; k++)
                example_output[k] = output[i*size_output_layer+k];
            // predict_deep_model_xxx(model, example_input, size_input_layer)
//            calculus_regression_delta_last_layer(model, output);
//            calculus_delta_layers(model);
            delete example_input;
            delete example_output;
        }
        return true;
    }

    bool train_deep_classification_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size) {
        int size_input_layer = model->d[0] - 1;
        int size_output_layer = model->d[model->layer_count-1] - 1;
        int example_counts = input_size / size_input_layer;
        int example_expected_result = output_size / size_output_layer;
        std::cout << "var helper" << std::endl;
        for(int i = 0 ; i < example_counts; i++){
            double* example_input = new double[size_input_layer];
            double* example_output = new double[size_output_layer];
            std::cout << "select range input/output for iteration" << std::endl;
            for(int k = 0 ; k < size_input_layer; k++)
                example_input[k] = input[i*size_input_layer+k];
            for(int k = 0 ; k < size_output_layer; k++)
                example_output[k] = output[i*size_output_layer+k];
            // predict_deep_model_xxx(model, example_input, size_input_layer)
//            calculus_classification_delta_last_layer(model, output);
//            calculus_delta_layers(model);
            delete example_input;
            delete example_output;
        }
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
