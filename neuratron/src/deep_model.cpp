#include "neuratron.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdio>

// std::srand(std::time(nullptr));
using std::cout;
using std::endl;


void print_a(const double* array, size_t size){
    for(size_t i = 0 ; i < size; i++)
        printf("%4.6lf ", array[i]);
    cout << endl;
}

void calculus_regression_delta_last_layer(struct DeepModel* model, double* output){
    const int id_last_layer = model->layer_count - 1;
    int count_neuron_last_layer = model->d[id_last_layer];
    for(int i = 1; i < count_neuron_last_layer; i++){
        printf("neuron: %d || %d, %lf\n", i, model->x[id_last_layer][i], output[i-1]);
        model->deltas[id_last_layer][i] = model->x[id_last_layer][i] - output[i-1];
    }
//    std::cout << "end calculus last layer" << std::endl;
}

void calculus_classification_delta_last_layer(struct DeepModel* model, double* output){
    cout << "pp" << endl;
    const int id_last_layer = model->layer_count - 1;
    int count_neuron_last_layer = model->d[id_last_layer];
    printf("l: %d, count_neurons: %d\n", id_last_layer, count_neuron_last_layer);
    for(int i = 1; i < count_neuron_last_layer; i++){
        const double output_Lj = model->x[id_last_layer][i];
        printf("neuron: %d\n", i);
        model->deltas[id_last_layer][i] = (1 - output_Lj*output_Lj) * (output_Lj - output[i-1]);
    }
}

void calculus_delta_layers(struct DeepModel* model){
    for(int layer_l = model->layer_count-1; layer_l > 0; layer_l--){
        for(int i_neuron = 0 ; i_neuron < model->d[layer_l-1]; i_neuron++){
            double sigma_w_delta = 0.;
            for(int j_neuron = 0; j_neuron < model->d[layer_l]; j_neuron++){
//                std::cout << ", l: " << layer_l << "j: " << j_neuron << ", i: " << i_neuron << std::endl;
                const double delta_l_j = model->deltas[layer_l][j_neuron];
                const double weight_l_ij = model->w[layer_l-1][i_neuron][j_neuron];
                sigma_w_delta+= delta_l_j * weight_l_ij;
            }
//            std::cout << "sigma: "<< sigma_w_delta << std::endl;
            double output_layer_of_i_neuron = model->x[layer_l-1][i_neuron];
            double delta_layer_of_i_neuron = (1 - output_layer_of_i_neuron * output_layer_of_i_neuron) * sigma_w_delta;
            model->deltas[layer_l][i_neuron] = delta_layer_of_i_neuron;
        }
    }
}

void update_weights(DeepModel* model, double learning_rate){
    std::cout << "========= UPDATE WEIGHTS ========" << std::endl;
    for(int l = 1 ; l < model->layer_count; l++)
        for(int i = 0; i < model->d[l - 1]; i++){
            for(int j = 0; j < model->d[l]; j++){
                const double output_prev_layer_i = model->x[l-1][i];
                const double dot_output_delta = output_prev_layer_i * model->deltas[l][j];
                model->w[l-1][i][j] -= learning_rate * dot_output_delta;
                printf("l:%d, i:%d, j:%d || x[%d][%d]: %lf,", l, i, j, (l-1), i, output_prev_layer_i);
                printf(" x[%d][%d]*delta[%d][%d]=%lf\n", (l-1), i, l, j, model->w[l-1][i][j]);
            }
        }
}


void generate_xs_model(struct DeepModel* model, double* input, std::vector<Eigen::MatrixXd>& matrices) {
    Eigen::MatrixXd xi(1, model->d[0]);

    xi(0, 0) = 1;
    for(int i = 0; i < model->d[0] - 1; ++i) {
        xi(0, i + 1) = input[i];
    }

    matrices.push_back(xi);

//    std::cout << "x0" << std::endl << xi << std::endl;

    for(int i = 1; i < model->layer_count; ++i) {
//        std::cout << "Map from <=> to : " << model->d[i - 1] << " <=> " << model->d[i] << std::endl;

        Eigen::MatrixXd wi(model->d[i - 1], model->d[i]);
        for(int k = 0; k < model->d[i - 1]; ++k) {
            for(int j = 0; j < model->d[i]; ++j) {
                wi(k, j) = model->w[i - 1][k][j];
            }
        }

//        std::cout << "w" << i << std::endl << wi << std::endl;

        xi = (xi * wi);
        xi = xi.unaryExpr([](double x){ return std::tanh(x); });
        xi(0, 0) = 1;
        matrices.push_back(xi);

//        std::cout << "x" << i << std::endl << xi << std::endl;
    }
}

void deep_weights_to_matrice(double*** w, int* sizes, int size, std::vector<Eigen::MatrixXd>& matrices) {
    for (size_t i = 0; i < size - 1; i++) {
        Eigen::MatrixXd matrice (sizes[i], sizes[i + 1]);
        for (size_t a = 0; a < sizes[i]; a++) {
            for (size_t b = 0; b < sizes[i + 1]; b++) {
                matrice(a, b) = w[i][a][b];
            }
        }
        matrices.push_back(matrice);
    }
}

void update_neurons_outputs(DeepModel* model, const std::vector<Eigen::MatrixXd>& outputs){
    std::cout << "======= Before output update ========" << std::endl;
    for(int l = 0; l < outputs.size(); l++){
        const auto output_l = outputs[l];
        cout << "[" << l << "]: ";
        for(int i = 1 ; i < output_l.cols(); i++)
            printf("%lf, ", output_l(0,i));
        cout << endl;
    }
    cout << "====== Start output update =======" << endl;
    for(int layer_l = 0 ; layer_l < model->layer_count; layer_l++){
        const auto layer_output = outputs[layer_l];
//        std::cout << "rows: " << layer_output.rows() << ", cols: " << layer_output.cols() << std::endl;
        const int count_neuron = model->d[layer_l];
//        printf("l: %d, neurons_layer: %d\n", layer_l, count_neuron);
        for(int neuron_j = 1; neuron_j < count_neuron; neuron_j++){
//            printf("l: %d, neuron: %d\n", layer_l, neuron_j);
//            cout<< layer_output(0, neuron_j) << " ";
            model->x[layer_l][neuron_j] =  layer_output(0, neuron_j);
        }
        print_a(model->x[layer_l], count_neuron);
        std::cout << std::endl;
    }
    std::cout << "end update" << std::endl;
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
            model->x[i] = new double[size_layer];
            for (int j = 0; j < size_layer; j++) {
                model->w[i][j] = new double[neurons_per_layers[i + 1]];
                for (int k = 0; k < neurons_per_layers[i + 1]; k++)
                    model->w[i][j][k] = std::rand() % 2 < 1 ? -1 : 1;

//                for(int n = 0 ; n < size_layer; n++)
//                    model->x[i][n] = 0;
            }
        }
        const int size_last_layer = neurons_per_layers[size - 1] + 1;
        model->d[size - 1] = size_last_layer;
        model->deltas[size - 1] = new double[size_last_layer];
        model->x[size - 1] = new double[size_last_layer];
//            for(int k = 0 ; k < size_last_layer; k++)
//                model->x[j][k] = 0;

        return model;
    }

    bool train_deep_regression_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate) {
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
            auto example_neuron_outputs = std::vector<Eigen::MatrixXd> {};
            generate_xs_model(model, example_input, example_neuron_outputs);
            update_neurons_outputs(model, example_neuron_outputs);
            calculus_regression_delta_last_layer(model, example_output);
            calculus_delta_layers(model);
            update_weights(model, training_rate);
            delete example_input;
            delete example_output;
            std::cout << "example " << i << ", weights:" << std::endl;
//            for(int l = 0 ; l < model->layer_count-1; l++)
//                for(int k = 0; k < model->d[l]-1; k++){
//                    for(int j = 0; j < model->d[l+1]-1; j++)
//                        std::cout<<"weight: layer["<< l << "](" << k << " to " << j << ")" << model->w[l][k][j] << std::endl;

            std::cout << std::endl;
        }
        return true;
    }

    bool train_deep_classification_model(struct DeepModel* model, double* input, int input_size, double* output, int output_size, double training_rate) {
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
            calculus_classification_delta_last_layer(model, output);
            calculus_delta_layers(model);
            update_weights(model, training_rate);
            delete example_input;
            delete example_output;
        }
        return true;
    }


    // struct DeepModel{
    //     int layer_count;
    //     double*** w;
    //     double** x;
    //     double** deltas;
    //     int* d;
    // };

    double* predict_deep_model_regression(struct DeepModel* model, double* input) {
        int output_size = model->d[model->layer_count - 1] - 1;

        std::vector<Eigen::MatrixXd> matrices {};
        generate_xs_model(model, input, matrices);

        double* results = new double[output_size + 1];
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
