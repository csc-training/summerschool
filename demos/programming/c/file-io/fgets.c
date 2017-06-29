#include <stdio.h>

int main(){
  FILE *fp;
  char str[5];

  /* opening file for reading */
  fp = fopen("file.txt" , "r");
  if(fp == NULL) {
    fprintf(stderr, "Error: file.txt did not exist\n");

    return(-1);
  }
  fgets (str, 5, fp);
  printf("%s", str);

  fclose(fp);
   
  return(0);
}
