#include "neuratron.h"
#include <stdlib.h>
#include <Eigen/Dense>

extern "C" {
  struct LinearModel* create_linear_model(int input_size, int output_size) {
    struct LinearModel* model = new LinearModel;
    model->sizeInput = input_size;
    model->sizeOutput = output_size;

    model->inputs = new double[(input_size + 1) * output_size];

    return model;
  }

  int train_linear_model(int i) {
    // Eigen::MatrixX3d in = input.inputs;
    // Eigen::MatrixX3d in = input.inputs;
    // Eigen::VectorXf w = model.inputs;
    return i + 3;
  }
}
