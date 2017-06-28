#include <stdio.h>
#include <stdlib.h>


int main() {
  int n=2;
  int m=3;

  printf("--- Array\n");
  printf("x y: address\n");
  int arr0[2][3];

  for(int y = 0; y < n; y++)
    for(int x = 0; x < m; x++)
      printf("%d %d: %ld\n", x, y, &(arr0[y][x]));
  
  printf("--- Dynamic array \n");
  printf("x y: address\n");
  int **arr2 = malloc(n * sizeof(float *));
  arr2[0] = malloc(m * n * sizeof(float));
  for (int i = 0; i < n; i++)
    arr2[i] = arr2[0] + i * m;

  for(int y = 0; y < n; y++)
    for(int x = 0; x < m; x++)
      printf("%d %d: %ld\n", x, y, &(arr2[y][x]));

  free(arr2[0]);
  free(arr2);

  printf("--- Variable length array (C99)\n");
  printf("x y: address\n");
  int arr1[n][m];

  for(int y = 0; y < n; y++)
    for(int x = 0; x < m; x++)
      printf("%d %d: %ld\n", x, y, &(arr1[y][x]));
  
  printf("--- Dynamic variable length array (C99)\n");
  printf("x y: address\n");
  int (*arr3)[m] = malloc(sizeof(int[n][m]));

  for(int y = 0; y < n; y++)
    for(int x = 0; x < m; x++)
      printf("%d %d: %ld\n", x, y, &(arr3[y][x]));

  free(*arr3);
}
