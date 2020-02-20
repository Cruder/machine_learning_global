#include "neuratron.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdio>

// std::srand(std::time(nullptr));
using std::cout;
using std::endl;

void deep_model_to_weights_matrice(DeepModel* model, std::vector<Eigen::MatrixXd>& matrices);

void print_a(const double* array, size_t size){
    for(size_t i = 0 ; i < size; i++)
        printf("%4.6lf ", array[i]);
    cout << endl;
}

void print_deltas(const DeepModel* model){
    cout << "===== PRINT DELTAS =====" << endl;
    for(int layer = 0; layer < model->layer_count; layer++) {
        cout << "[" << layer << "]: ";
        for (int neuron = 1; neuron < model->d[layer]; neuron++)
            cout << model->deltas[layer][neuron] << " ";
        cout << endl;
    }
    cout << "================" << endl << endl;
}

void print_weights(const DeepModel* model){
    cout << "===== PRINT WEIGHTS =====" << endl;
    for(int layer = 0; layer < model->layer_count-1; layer++) {
        cout << "w[" << layer << "]: ";
        for (int neuron_i = 0; neuron_i < model->d[layer-1]; neuron_i++){
            cout << "[" << neuron_i << "]";
            for(int neuron_j = 0; neuron_j < model->d[layer]; neuron_j++)
                cout << model->w[layer][neuron_i][neuron_j] << " ";
        }
        cout << endl;
    }
    cout << "================" << endl << endl;
}

void calculus_regression_delta_last_layer(struct DeepModel* model, double* output){
    const int id_last_layer = model->layer_count - 1;
    int count_neuron_last_layer = model->d[id_last_layer];
    for(int i = 1; i < count_neuron_last_layer; i++){
        model->deltas[id_last_layer][i] = model->x[id_last_layer][i] - output[i-1];
    }
}

void calculus_classification_delta_last_layer(struct DeepModel* model, double* output){

    const int id_last_layer = model->layer_count - 1;
    int count_neuron_last_layer = model->d[id_last_layer];
    for(int i = 1; i < count_neuron_last_layer; i++){
        const double output_Lj = model->x[id_last_layer][i];
        model->deltas[id_last_layer][i] = (1 - output_Lj*output_Lj) * (output_Lj - output[i-1]);
    }
}

void update_weights_matrix(DeepModel* model, const std::vector<Eigen::MatrixXd>& matXs, const std::vector<Eigen::MatrixXd>& matDeltas, double learning_rate){
    cout << "===== Start update weights matrix =====" << endl;
    const int last_layer = model->layer_count - 1;
    std::vector<Eigen::MatrixXd> new_weights(model->layer_count-1);
    auto current_weights = std::vector<Eigen::MatrixXd>{};
    deep_model_to_weights_matrice(model, current_weights);
    for(int layer = 1 ; layer < last_layer + 1; layer++){
        const auto old_layer_weight = current_weights[layer - 1];
        const auto layer_delta = matDeltas[layer];
        const auto layer_output = matXs[layer-1];
        Eigen::MatrixXd new_layer_weight(old_layer_weight.rows(), old_layer_weight.cols());
        cout << "old_layer_weight(" << old_layer_weight.rows() << ", " << old_layer_weight.cols() << ")" << endl;
        cout << "layer_output(" << layer_output.rows() << ", " << layer_output.cols() << ")" << endl;
        cout << "layer_delta(" << layer_delta.rows() << ", " << layer_delta.cols() << ")" << endl;
        new_layer_weight = old_layer_weight - learning_rate * layer_output.transpose() * layer_delta.transpose();
        new_weights[layer-1] = new_layer_weight;
    }
    cout << "start editing model weight" << endl;
    for(int layer = 1; layer < last_layer + 1; layer++){
        const auto layer_weight = new_weights[layer-1];
        for(int neuron_i = 0; neuron_i < model->d[layer-1]; neuron_i++)
            for(int neuron_j = 0; neuron_j < model->d[layer]; neuron_j++){
                model->w[layer-1][neuron_i][neuron_j] = layer_weight(neuron_i, neuron_j);
            }
    }
    cout << "===== end update weights matrix =====" << endl;
//    print_weights(model);
}

void calculus_regression_delta(DeepModel* model, const std::vector<Eigen::MatrixXd>& matXs, const Eigen::MatrixXd& matY, double learning_rate){
    const int last_layer = model->layer_count - 1;
    std::vector<Eigen::MatrixXd> deltas(model->layer_count);
    cout << deltas.size() << endl;
    std::cout << "matXs last layer = " << std::endl << matXs[last_layer] << std::endl;
    std::cout << "matY = " << std::endl << matY << std::endl;


    deltas[last_layer] = (matXs[last_layer] - matY).transpose();


    cout << "deltas[" << last_layer << "] = " << endl << deltas[last_layer] << endl;
    auto weights = std::vector<Eigen::MatrixXd>{};
    deep_model_to_weights_matrice(model, weights);
    for(int layer = last_layer; layer > 0 ; layer--){
        std::cout << "+=============================+\n\n\nStart with layer = " << layer << "\n\n\n+=============================+" << endl;
        Eigen::MatrixXd prev_layer_delta(1, model->d[layer - 1]);
        const Eigen::MatrixXd layer_output = matXs[layer-1];
        const Eigen::MatrixXd one = Eigen::MatrixXd::Constant(layer_output.rows(), layer_output.cols(), 1.);
        // const Eigen::MatrixXd layer_weight(1,1);
        std::cout << "Iteration layer " << layer << endl;
        std::cout << "one = " << std::endl << one << std::endl;
        std::cout << "layer_output = " << std::endl << layer_output << std::endl;
        std::cout << "weights[layer -1] = " << std::endl << weights[layer -1] << std::endl;
        std::cout << "deltas[layer (" << layer << ")] = " << std::endl << deltas[layer] << std::endl;

        Eigen::MatrixXd left = one - layer_output.unaryExpr([](double x){ return x * x; });
        std::cout << "(left) one - layer_output*layer_output = " << std::endl << left << std::endl;
        Eigen::MatrixXd right = weights[layer -1] * deltas[layer];
        std::cout << "(right) weights[layer -1] * deltas[layer (" << layer << ")] = " << std::endl << right << std::endl;

        prev_layer_delta = (left.transpose() * right.transpose()).diagonal();
        // prev_layer_delta = left * right;
        std::cout << "prev_layer_delta = " << std::endl << prev_layer_delta << std::endl;
        std::cout << "Add to " << layer - 1 << endl;
        // deltas.insert(std::begin(deltas) + layer, prev_layer_delta);
        deltas[layer-1] = prev_layer_delta;
    }
    update_weights_matrix(model, matXs, deltas, learning_rate);
 }

 void calculus_classification_delta(DeepModel* model, const std::vector<Eigen::MatrixXd>& matXs, const Eigen::MatrixXd& matY, double learning_rate){
    const int last_layer = model->layer_count - 1;
    std::vector<Eigen::MatrixXd> deltas(model->layer_count);
    cout << deltas.size() << endl;
    std::cout << "matXs last layer = " << std::endl << matXs[last_layer] << std::endl;
    std::cout << "matY = " << std::endl << matY << std::endl;

    Eigen::MatrixXd left1 = matXs[last_layer].unaryExpr([](double x){ return 1 - x * x; });
    Eigen::MatrixXd right1 = (matXs[last_layer] - matY);
    std::cout << "Left 1 " << std::endl << left1 << std::endl << std::endl << "Right 1 " << std::endl << right1 << std::endl;
    deltas[last_layer] = (left1.transpose() * right1).diagonal();

    cout << "deltas[" << last_layer << "] = " << endl << deltas[last_layer] << endl;
    auto weights = std::vector<Eigen::MatrixXd>{};
    deep_model_to_weights_matrice(model, weights);
    for(int layer = last_layer; layer > 0 ; layer--){
        std::cout << "+=============================+\n\n\nStart with layer = " << layer << "\n\n\n+=============================+" << endl;
        Eigen::MatrixXd prev_layer_delta(1, model->d[layer - 1]);
        const Eigen::MatrixXd layer_output = matXs[layer-1];
        const Eigen::MatrixXd one = Eigen::MatrixXd::Constant(layer_output.rows(), layer_output.cols(), 1.);
        // const Eigen::MatrixXd layer_weight(1,1);
        std::cout << "Iteration layer " << layer << endl;
        std::cout << "one = " << std::endl << one << std::endl;
        std::cout << "layer_output = " << std::endl << layer_output << std::endl;
        std::cout << "weights[layer -1] = " << std::endl << weights[layer -1] << std::endl;
        std::cout << "deltas[layer (" << layer << ")] = " << std::endl << deltas[layer] << std::endl;

        Eigen::MatrixXd left = one - layer_output.unaryExpr([](double x){ return x * x; });
        std::cout << "(left) one - layer_output*layer_output = " << std::endl << left << std::endl;
        Eigen::MatrixXd right = weights[layer -1] * deltas[layer];
        std::cout << "(right) weights[layer -1] * deltas[layer (" << layer << ")] = " << std::endl << right << std::endl;

        prev_layer_delta = (left.transpose() * right.transpose()).diagonal();
        // prev_layer_delta = left * right;
        std::cout << "prev_layer_delta = " << std::endl << prev_layer_delta << std::endl;
        std::cout << "Add to " << layer - 1 << endl;
        // deltas.insert(std::begin(deltas) + layer, prev_layer_delta);
        deltas[layer-1] = prev_layer_delta;
    }
    update_weights_matrix(model, matXs, deltas, learning_rate);
 }


void calculus_delta_layers(struct DeepModel* model){
    for(int layer_l = model->layer_count-1; layer_l > 0; layer_l--){
        for(int i_neuron = 0 ; i_neuron < model->d[layer_l-1]; i_neuron++){
            double sigma_w_delta = 0.;
            for(int j_neuron = 0; j_neuron < model->d[layer_l]; j_neuron++){
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

void update_weights(DeepModel* model, double learning_rate){
    for(int l = 1 ; l < model->layer_count; l++)
        for(int i = 0; i < model->d[l - 1]; i++){
            for(int j = 0; j < model->d[l]; j++){
                const double output_prev_layer_i = model->x[l-1][i];
                const double dot_output_delta = output_prev_layer_i * model->deltas[l][j];
                model->w[l-1][i][j] -= learning_rate * dot_output_delta;
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

void deep_model_to_weights_matrice(DeepModel* model, std::vector<Eigen::MatrixXd>& matrices) {
    for (size_t i = 0; i < model->layer_count - 1; i++) {
        Eigen::MatrixXd matrice (model->d[i], model->d[i + 1]);
        for (size_t a = 0; a < model->d[i]; a++) {
            for (size_t b = 0; b < model->d[i + 1]; b++) {
                matrice(a, b) = model->w[i][a][b];
            }
        }
        matrices.push_back(matrice);
    }
}

void update_neurons_outputs(DeepModel* model, const std::vector<Eigen::MatrixXd>& outputs){
    for(int l = 0; l < outputs.size(); l++){
        const auto output_l = outputs[l];
        cout << "[" << l << "]: ";
        for(int i = 1 ; i < output_l.cols(); i++)
            printf("%lf, ", output_l(0,i));
        cout << endl;
    }
    for(int layer_l = 0 ; layer_l < model->layer_count; layer_l++){
        const auto layer_output = outputs[layer_l];
        const int count_neuron = model->d[layer_l];
        for(int neuron_j = 1; neuron_j < count_neuron; neuron_j++){
            model->x[layer_l][neuron_j] =  layer_output(0, neuron_j);
        }
    }
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
            auto example_expected_output = Eigen::MatrixXd(1, size_output_layer + 1);
            std::cout << "select range input/output for iteration" << std::endl;
            for(int k = 0 ; k < size_input_layer; k++)
                example_input[k] = input[i*size_input_layer+k];
            example_expected_output(0, 0) = 1;
            for(int k = 0 ; k < size_output_layer; k++)
                example_expected_output(0, k + 1) = output[i * size_output_layer + k];
            cout << "=====| Example input/output |=====" << endl;
            cout << endl;
            auto example_neuron_outputs = std::vector<Eigen::MatrixXd> {};
            generate_xs_model(model, example_input, example_neuron_outputs);
            calculus_regression_delta(model, example_neuron_outputs, example_expected_output, training_rate);

            delete example_input;
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
            auto example_expected_output = Eigen::MatrixXd(1, size_output_layer + 1);
            std::cout << "select range input/output for iteration" << std::endl;
            for(int k = 0 ; k < size_input_layer; k++)
                example_input[k] = input[i*size_input_layer+k];
            example_expected_output(0, 0) = 1;
            for(int k = 0 ; k < size_output_layer; k++)
                example_expected_output(0, k + 1) = output[i * size_output_layer + k];
            cout << "=====| Example input/output |=====" << endl;
            cout << endl;
            auto example_neuron_outputs = std::vector<Eigen::MatrixXd> {};
            generate_xs_model(model, example_input, example_neuron_outputs);
            calculus_classification_delta(model, example_neuron_outputs, example_expected_output, training_rate);

            delete example_input;
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
