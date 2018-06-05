#include <cstdio>

class democlass {
public:
  // Two member variables, a and b
  int a, b;

  void print_values() const {
    printf("Values are: a=%i, b=%i\n", a, b);
  };
};

int main(void)
{
  democlass demo;
  demo.a = 10;
  demo.b = 20;
  
  demo.print_values();

  return 0;
}
