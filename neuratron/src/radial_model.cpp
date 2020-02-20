#include "neuratron.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdio>

extern "C" {
    RadialModel* create_radial_model(int count_example, int size_input, int size_output, double gamma){
        RadialModel* model = new RadialModel;
        model->examples_count = count_example;
        model->gamma = gamma;
        model->sizeInput = size_input;
        model->sizeOutput = size_output;
        model->w[0] = new double*[size_input];
        for(int i = 0 ; i < size_input; i++){
            model->w[0][i] = new double[count_example];
            for(int k = 0 ; k < count_example;k++){
                model->w[1][i][k] = 0.3;
            }
        }
        model->w[1] = new double*[count_example];
        for(int i = 0 ; i < count_example; i++){
            model->w[1][i] = new double[size_output];
            for(int k = 0 ; k < size_output ; k++){
                model->w[1][i][k] = 0.3;
            }
        }

    }
}