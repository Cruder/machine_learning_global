#ifndef __NEURATRON_H__
#define __NEURATRON_H__

extern "C" {
  struct LinearModel{
      double* inputs;
      int sizeInputs;
  };

  extern struct LinearModel* create_linear_model(const double *inputs, int size_inputs);
}

#endif // __NEURATRON_H__
