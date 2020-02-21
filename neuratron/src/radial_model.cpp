#include "neuratron.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdio>

using std::cout;
using std::endl;

double norm_sq(Eigen::MatrixXd pointA, Eigen::MatrixXd pointB, int count_coordinates){
    double norm = 0.;
    for(int i = 0 ; i < count_coordinates; i++){
        const double diff = pointB(i,0) - pointA(i, 0);
        norm += diff*diff;
    }
    return norm;
}

extern "C" {
    RadialModel* create_radial_model(double* examples, int count_example, int size_input, int size_output, double gamma){
        RadialModel* model = new RadialModel;
        model->examples_count = count_example;
        model->gamma = gamma;
        model->size_input = size_input;
        model->size_output = size_output;
        model->w = new double[count_example];
        for(int k = 0 ; k < count_example;k++){
            model->w[k] = (((double) std::rand()) / RAND_MAX) * 2 - 1;
        }

        model->examples = examples;

    }

    double* train_regression(RadialModel* model, double* expected_output){
        cout << "========== Start regression training ==========" << endl;
        Eigen::MatrixXd phi(model->examples_count, model->examples_count);
        Eigen::MatrixXd matY(model->examples_count, 1);
        std::vector<Eigen::MatrixXd> points(model->examples_count);
        for( int i = 0 ; i < model->examples_count; i++){
            Eigen::MatrixXd example_point(model->size_input, 1);
            for(int j = 0 ; j < model->examples_count; j++){
                example_point(j, 0) = model->examples[i * model->size_input + j];
            }

            matY(i, 0) = expected_output[i * model->size_output];
            points[i] = example_point;
        }

        for(int i = 0 ; i < model->examples_count; i++){
            const auto input_point = points[i];
            for(int j = 0 ; j < model->examples_count; j++){
                const auto ref_example = points[j];
                const double norm = norm_sq(input_point, ref_example, model->size_input);
                phi(i, j) = exp(-model->gamma*norm);
            }
        }

        const Eigen::MatrixXd inv_phi = phi.inverse();
        const Eigen::MatrixXd weights = inv_phi * matY;
        cout << "====== To check/test, it might crash from here" << endl;

        for(int j = 0 ; j < model->examples_count; j++){
            model->w[j] = weights(j,0);
        }

    }
}