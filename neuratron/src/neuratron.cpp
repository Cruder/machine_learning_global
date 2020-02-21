#include "neuratron.h"
#include <stdlib.h>
#include <Eigen/Dense>

#include <iostream>
#include <random>

extern "C" {
  struct LinearModel* create_linear_model(int input_size, int output_size) {
    struct LinearModel* model = new LinearModel;
    model->sizeInput = input_size;
    model->sizeOutput = output_size;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);

    model->inputs = new double[(input_size + 1) * output_size];
    for(int i = 0; i < (input_size+1) * output_size; i++){
        model->inputs[i] = distribution(generator) / (input_size+1);
    }

    return model;
  }

  bool train_linear_model(struct LinearModel* model, double* input, int input_size, double* output, int output_size) {
    // std::cout << input[0] << " " << input[1] << std::endl;
    // std::cout << output[0] << std::endl;

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

    // std::cout << matX << std::endl << std::endl;
    // std::cout << matY << std::endl << std::endl;


    auto res1 = matX.transpose() * (matX);
    if (res1.determinant() == 0) {
      return false;
    }
    auto res2 = res1.inverse();
    auto res3 = res2 * (matX.transpose());
    auto result = res3 * (matY);

    // std::cout << result << std::endl;

    // model->inputs = result.data();

    Eigen::Map<Eigen::MatrixXd>(model->inputs, result.rows(), result.cols()) = result;

    return true;
  }


  // W = W hadamard alpha * (Y - X1) * X0
  bool train_linear_model_classification(struct LinearModel* model, double* input, double* output, double alpha) {
    // std::cout << "Start train_linear_model_classification" << std::endl;
    // std::cout << std::endl << input[0] << std::endl;

    // std::cout << model->sizeInput + 1 << std::endl;
    // std::cout << model->sizeOutput << std::endl;

    Eigen::MatrixXd matX0(1, model->sizeInput + 1);
    matX0(0, 0) = 1;
    for(int i = 0; i < model->sizeInput; ++i) {
      matX0(0, i + 1) = input[i];
    }

    // std::cout << "matX0" << std::endl;
    // std::cout << matX0 << std::endl;

    Eigen::MatrixXd matW(model->sizeInput + 1, model->sizeOutput);
    for(int i = 0; i < model->sizeInput + 1; ++i) {
      for(int j = 0; j < model->sizeOutput; ++j) {
        // std::cout << "Access " << i << ", " << j << " => " << j * (model->sizeInput + 1) + i << std::endl;
        matW(i, j) = model->inputs[j * (model->sizeInput + 1) + i];
      }
    }

    // std::cout << "matW" << std::endl;
    // std::cout << matW << std::endl;

    Eigen::MatrixXd matY(1, model->sizeOutput);
    for(int i = 0; i < model->sizeOutput; ++i) {
      matY(0, i) = output[i];
    }

    // std::cout << "matY" << std::endl;
    // std::cout << matY << std::endl;

    double* x1_prediction = predict_linear_model_classification(model, input);
    Eigen::MatrixXd matX1(1, model->sizeOutput);
    for(int i = 0; i < model->sizeOutput; ++i) {
      matX1(0, i) = output[i];
    }

    // std::cout << "matX1" << std::endl;
    // std::cout << matX1 << std::endl;

    Eigen::MatrixXd temp = (alpha * (matY - matX1));

    // std::cout << "temp" << std::endl;
    // std::cout << temp << std::endl;

    // std::cout << "matW(" << matW.cols() << ", " << matW.rows() << ")" << std::endl;
    // std::cout << "temp(" << temp.cols() << ", " << temp.rows() << ")" << std::endl;
    // std::cout << "matX0(" << matX0.cols() << ", " << matX0.rows() << ")" << std::endl;

    Eigen::MatrixXd temp2 = matX0.transpose() * temp;

    // std::cout << "temp2" << std::endl;
    // std::cout << temp2 << std::endl;

    Eigen::MatrixXd result = matrix_hadamard(matW, temp2);

    // std::cout << "result" << std::endl;
    // std::cout << result << std::endl;

    for(int i = 0; i < model->sizeInput + 1; ++i) {
      for(int j = 0; j < model->sizeOutput; ++j) {
        model->inputs[j * model->sizeInput + i] = result(i, j);
      }
    }

    return true;
  }

  double* predict_linear_model_regression(struct LinearModel* model, double* input) {
    double result = 0;

    double* results = new double[model->sizeOutput];

    for(int i = 0; i < model->sizeOutput; ++i) {
      results[i] += model->inputs[i * (model->sizeInput)];
      for(int j = 0; j < model->sizeInput; ++j) {
        results[i] += model->inputs[i * (model->sizeInput) + j + 1] * input[j];
      }
    }

    return results;
  }

  double* predict_linear_model_classification(struct LinearModel* model, double* input) {
    double* results = predict_linear_model_regression(model, input);

    for(int i = 0; i < model->sizeOutput; ++i) {
      results[i] = results[i] > 0 ? 1.0 : -1.0;
    }

    return results;
  }


  int free_linear_model(LinearModel* model) {
    delete[] model->inputs;
    delete model;
    return 0;
  }
}
