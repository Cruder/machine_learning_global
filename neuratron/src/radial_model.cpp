#include "neuratron.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdio>
#include <random>

using std::cout;
using std::endl;

double* predict_point(const RadialModel* model, const Eigen::MatrixXd& point);
bool train(RadialModel* model, double* expected_output, int iteration);

double norm_sq(Eigen::MatrixXd pointA, Eigen::MatrixXd pointB, int count_coordinates){
    double norm = 0.;
    for(int i = 0 ; i < count_coordinates; i++){
        const double diff = pointB(i,0) - pointA(i, 0);
        norm += diff*diff;
    }
    return norm;
}

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


extern "C" {
    RadialModel* create_radial_model(double* examples, int count_example, int size_input, int size_output, double gamma){
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);
        RadialModel* model = new RadialModel;
        model->examples_count = count_example;
        model->gamma = gamma;
        model->size_input = size_input;
        model->size_output = size_output;
        model->w = new double*[count_example];
        for(int k = 0 ; k < count_example;k++){
            model->w[k] = new double[size_output];
            for(int o = 0 ; o < size_output; o++){
                model->w[k][o] = distribution(generator);
            }
        }
        model->examples = examples;
        cout << "Init (" << model->examples_count << ", " << model->size_input << ", " << model->size_output << endl;
        return model;
    }

    bool train_radial_regression(RadialModel* model, double* expected_outputs){
        return train(model, expected_outputs, 1);
    }

    bool train_radial_classification(RadialModel* model, double* expected_outputs, int iteration){
        return train(model, expected_outputs, iteration);
    }
//     }
    double* predict_radial_regression(const RadialModel* model, double *batch_input, int size_batch){
        cout << "====== Start predictions =======" << endl;
        const int count_point_in_batch = size_batch / model->size_input;
        double* predictions = new double[count_point_in_batch*model->size_output];
        for(int i = 0 ; i < count_point_in_batch; i++){
            cout << "===== point " << i << " ======" << endl;
            Eigen::MatrixXd point(model->size_input, 1);
            for(int i  = 0 ; i < model->size_input; i++){
                point(i, 0) = batch_input[i*count_point_in_batch];
            }
            cout << "===== predict point " << i << " =======" << endl;
            double* prediction = predict_point(model, point);
            cout << "===== add prediction point " << i << " =======" << endl;
            for(int o = 0 ; o < model->size_output; o++){
                predictions[i*count_point_in_batch+o] = prediction[o];
            }
        }
        cout << "====== End predictions =======" << endl;
        return predictions;
    }

    double* predict_radial_classification(const RadialModel* model, double* batch_input, int size_batch){
        const int count_point_in_batch = size_batch / model->size_input;
        double* predictions = predict_radial_regression(model, batch_input, size_batch);
        for(int i  = 0 ; i < count_point_in_batch; i++ ){
            predictions[i] = predictions[i] > 0 ? 1.0 : -1.0;
        }
        return predictions;
    }

}

bool train(RadialModel* model, double* expected_output, int iteration){
    int* indexes = new int[model->examples_count];
    for(int i = 0 ; i < model->examples_count; i++){
        indexes[i]= i;
    }
    cout << "========== Start training ==========" << endl;
    Eigen::MatrixXd phi(model->examples_count, model->examples_count);
    Eigen::MatrixXd matY(model->examples_count, model->size_output);
    std::vector<Eigen::MatrixXd> points(model->examples_count);
    for(int r = 0 ; r < iteration; r++) {
//        cout << "====== Iteration " << r << " =====" << endl;
//        cout << "====== puts point to matrix " << r << " =====" << endl;
        for (int i = 0; i < model->examples_count; i++) {
            Eigen::MatrixXd example_point(model->size_input, 1);
//            cout << "===== example point " << i << " =====" << endl;
            for (int j = 0; j < model->size_input; j++) {
                example_point(j, 0) = model->examples[indexes[i] * model->size_input + j];
            }
//            cout << "===== example point expected output =====" << endl;
            for (int j = 0; j < model->size_output; j++) {
                matY(i, j) = expected_output[indexes[i] * model->size_output + j];
            }
            points[i] = example_point;
        }
//        cout << "===== fill phi =====" << endl;
        for (int i = 0; i < model->examples_count; i++) {
            const Eigen::MatrixXd input_point = points[i];
//            cout << "input point input(" << input_point.rows() << ", " << input_point.cols() << ")" << endl;
            for (int j = 0; j < model->examples_count; j++) {
                const Eigen::MatrixXd ref_example = points[j];
//                cout << "output point ref(" << ref_example.rows() << ", " << ref_example.cols() << ")" << endl;
                const double norm = norm_sq(input_point, ref_example, model->size_input);
                phi(i, j) = exp(-model->gamma * norm);
            }
        }
//        cout << "===== weight calculus =====" << endl;

//        cout << "phi(" << phi.rows() << ", " << phi.cols() << ")" << endl;
        const Eigen::MatrixXd inv_phi = phi.inverse();
//        cout << "phi inv(" << inv_phi.rows() << ", " << inv_phi.cols() << ")" << endl;
        const Eigen::MatrixXd weights = inv_phi * matY;
//        cout << "weights(" << weights.rows() << ", " << weights.cols() << ")" << endl;
//        cout << "====== To check/test, it might crash from here" << endl;

        for (int j = 0; j < model->examples_count; j++) {
            for(int k = 0 ; k < model->size_output; k++) {
                model->w[j][k] += weights(j, k);
            }
        }
//        cout << "====== Shuffle start =======" << endl;
        shuffle(indexes, model->examples_count);
//        cout << "====== Shuffle end =======" << endl;
    }
    cout << "============= end train ================" << endl;
    return true;
}

double* predict_point(const RadialModel* model, const Eigen::MatrixXd& point){
    double* sigma_outputs = new double[model->size_output];
    for(int o = 0 ; o < model->size_output; o++) {
        sigma_outputs[o] = 0.;
        for (int n = 0; n < model->examples_count; n++) {
            const double weight = model->w[n][o];
            Eigen::MatrixXd example_point(model->size_input, 1);
            for (int i = 0; i < model->size_input; i++) {
                example_point(i, 0) = model->examples[n * model->size_input + i];
            }
            const double norm = norm_sq(point, example_point, model->size_input);
            sigma_outputs[o] = exp(-model->gamma * norm);
        }
    }
    return sigma_outputs;
}
