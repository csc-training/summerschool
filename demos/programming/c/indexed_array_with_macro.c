#include <stdio.h>
#include <stdlib.h>
#define INDX(x,y,nx) ( (x) + (y)*(nx))

int main(int argc, char *argv[]){
  
  if(argc != 3) {
    printf("instructions how to use...\n");
    return 1;
  }
  int nx=atoi(argv[1]);
  int ny=atoi(argv[2]);
  
  float* a= malloc (nx*ny*sizeof(float));

  for(int j=0;j< ny;j++)
    for(int i=0;i< nx;i++)
      a[INDX(i,j+1,nx)] = 0.1 * (i + j*nx);

  for(int i=0;i< ny*nx;i++)
    printf("%.1f ", a[i]);
  printf("\n");

  free(a);
  return 0;
}
