#include <stdio.h>

int main(){
  FILE *fp;
  int a,b,c;
  int n;
  
  /* opening file for reading */
  fp = fopen("file.txt" , "r");
  if(fp == NULL) {
    fprintf(stderr, "Error: file.txt did not exist\n");
    return(-1);
  }
  n=fscanf(fp, "%d\n%d\n%d\n",&a, &b, &c);
  printf("Read %d numbers: %d %d %d\n", n, a, b, c);

  fclose(fp);
   
  return(0);
}
