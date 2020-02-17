#ifndef __NEURATRON_H__
#define __NEURATRON_H__

struct Foo {
  int* array;
  int size;
};

extern int init_bar();
extern struct Foo* init_foo();
extern int* give_42();


#endif // __NEURATRON_H__
