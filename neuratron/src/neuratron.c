#include "neuratron.h"
#include <stdlib.h>
#include <stdio.h>



extern struct LinearModel* create_linear_model(const double *inputs, int size_inputs){
    struct LinearModel* model = malloc(sizeof(struct LinearModel));
    int sizeWithInputBiased = sizeof(double) * size_inputs + sizeof(double) * (size_inputs/2);
    model->inputs = malloc(sizeWithInputBiased);
    for(int i = 0 ; i < sizeWithInputBiased / 3; i++){
        model->inputs[i] = inputs[i];
        model->inputs[i+1] = inputs[i+1];
        model->inputs[i+2] = 1;
    }
    return model;
}


extern inline int init_bar() {
  return 5;
}

extern inline struct Foo* init_foo() {
  struct Foo* foo = malloc(sizeof(struct Foo));
  foo->array = malloc(sizeof(int) * 5);
  foo->array[0] = 0;
  foo->array[1] = 5;
  foo->array[2] = 2;
  foo->array[3] = 1;
  foo->array[4] = 9;
  foo->size = 5;

  return foo;
}

extern inline int* give_42() {
  int* foo = malloc(sizeof(int));
  (*foo) = 40;
  return foo;
}
