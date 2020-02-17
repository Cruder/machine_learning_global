#include "neuratron.h"
#include <stdlib.h>


int init_bar() {
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
