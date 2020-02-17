#ifndef __NEURATRON_H__
#define __NEURATRON_H__

extern "C" {
  struct LinearModel{
      double* inputs;
      int sizeInput;
      int sizeOutput;
  };

  extern struct LinearModel* create_linear_model(int input_size, int output_size);
}

#endif // __NEURATRON_H__
