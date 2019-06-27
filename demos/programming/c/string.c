#include <stdio.h>
#include <stdlib.h>


//atof string to double
//atoi string to int


int main(int argc, char *argv[]){


  char message[]="Hello!";
  char* p = message;
  
  while (*p != '\0') {
    printf("\"%c\" ",*p);
    p++;
  }
  
  printf(" + Final null character \n");
  


}

