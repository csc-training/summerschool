#include<stdio.h>
#include<stdlib.h>
//Need to know dimensions when passing arrays
void test1(int a[2][3]){
  for(int i=0;i < 2;i++){
    for(int j=0;j < 3;j++){
      printf("%d ", a[i][j]);
      a[i][j]=-a[i][j];
    }
    printf("\n");
  }
  printf("\n");
}


//Need to know dimensions when passing arrays
//or at least all expect first
void test2(int a[][3]){
  for(int i=0;i < 2;i++){
    for(int j=0;j < 3;j++){
      printf("%d ", a[i][j]);

    }
    printf("\n");
  }
  printf("\n");
}


//In c99 one can also use variable length arrays
void test3(int nx, int ny,int a[ny][nx]){
  for(int i=0;i < ny;i++){
    for(int j=0;j < nx;j++){
      printf("%d ", a[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}


int main(int argc, char *argv[]){
  //fixed sized array
  int a[2][3]={{1,2,3},{4,5,6}};

  //variable length array C99!
  int nx=atoi(argv[1]);
  int ny=atoi(argv[2]);
  int b[ny][nx];


  int *c;

  for(int i=0;i < ny;i++){
    for(int j=0;j < nx;j++){
      b[i][j]=i*j;
    }
  }

  test1(a);
  test2(a);
  test3(nx,ny,b);
  
  c=&(a[0][0]);
  for(int i=0; i<2*3;i++){
    printf("%d ",c[i]);
  }
  

}
