#include "neuratron.hpp"

extern "C" {
  Foo* init_foo() {
    Foo* foo = new Foo;
    foo->array = new int[25];
    foo->size = 25;

    return foo;
  }

  extern int* give_42() {
    int* foo = new int;
    (*foo) = 42;
    return foo;
  }
}
