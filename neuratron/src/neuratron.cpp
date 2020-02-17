#include "neuratron.h"
#include <stdlib.h>
#include <Eigen/Dense>

#include <iostream>

extern "C" {
  struct LinearModel* create_linear_model(int input_size, int output_size) {
    struct LinearModel* model = new LinearModel;
    model->sizeInput = input_size;
    model->sizeOutput = output_size;

    model->inputs = new double[(input_size + 1) * output_size];

    return model;
  }

  bool train_linear_model(struct LinearModel* model, double* input, int input_size, double* output, int output_size) {
    std::cout << input[0] << " " << input[1] << std::endl;
    std::cout << output[0] << std::endl;

    int example_counts = input_size / model->sizeInput;

    Eigen::MatrixXd matX(example_counts, model->sizeInput + 1);
    Eigen::MatrixXd matY(example_counts, model->sizeOutput);

    for(int i = 0; i < example_counts; ++i) {
      matX(i, 0) = 1;
    }

    for(int i = 0; i < input_size; ++i) {
      int idX = (i % model->sizeInput) + 1;
      int idY = i / model->sizeInput;

      matX(idY, idX) = input[i];
    }

    for(int i = 0; i < output_size; ++i) {
      int idX = (i % model->sizeOutput);
      int idY = i / model->sizeOutput;

      matY(idY, idX) = output[i];
    }

    std::cout << matX << std::endl << std::endl;
    std::cout << matY << std::endl << std::endl;


    auto res1 = matX.transpose() * (matX);
    auto res2 = res1.inverse();
    auto res3 = res2 * (matX.transpose());
    auto result = res3 * (matY);

    std::cout << result << std::endl;

    // model->inputs = result.data();

    Eigen::Map<Eigen::MatrixXd>(model->inputs, result.rows(), result.cols()) = result;

    // Eigen::MatrixX3d in = input.inputs;
    // Eigen::MatrixX3d in = input.inputs;
    // Eigen::VectorXf w = model.inputs;
    return true;
  }
}
