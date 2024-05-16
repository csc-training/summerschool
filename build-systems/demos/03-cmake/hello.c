#include<stdlib.h>
#include<stdio.h>
#include"config.h"
#include"module.h"

int main(int argc, char *argv[])
{
  printf("Hello version %i.%i\nBuilt by %s\n", 
         HELLO_VERSION_MAJOR, 
         HELLO_VERSION_MINOR,
         HELLO_BUILDER_NAME);
  hello();
  return EXIT_SUCCESS;
}
