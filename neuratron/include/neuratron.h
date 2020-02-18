#ifndef __NEURATRON_H__
#define __NEURATRON_H__

extern "C" {

    struct DeepModel{
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

    struct LinearModel* create_linear_model(int input_size, int output_size);
    bool train_linear_model(struct LinearModel* model, double* input, int input_size, double* output, int output_size);
    int free_linear_model(LinearModel* model);

    struct DeepModel* create_deep_model(int* neurons_per_layers, int size);
}

#endif // __NEURATRON_H__
