#ifndef __NEURATRON_H__
#define __NEURATRON_H__

extern "C" {
  struct Foo {
    int* array;
    int size;
  };

  struct Foo* init_foo();
  int* give_42();
}

#endif // __NEURATRON_H__
