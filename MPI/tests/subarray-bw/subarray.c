#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
                                                                                                               
// Utility routine for allocating a two dimensional array                                                      
double **malloc_2d(int nx, int ny)                                                                             
{
   double **array;
   int i;
   array = (double **) malloc(nx * sizeof(double *));
   array[0] = (double *) malloc(nx * ny * sizeof(double));
   for (i = 1; i < nx; i++) {
      array[i] = array[0] + i * ny;
   }
   return array;
}                                                                                                               
                                                                                                               
// Utility routine for deallocating a two dimensional array                                                    
void free_2d(double **array)                                                                                   
{
   free(array[0]);
   free(array);
}
                                                                                                              

int main(int argc, char *argv[])
{
   
   int i,j;
   int size,subsize;
   int array_size[2];
   int subarray_size[2];
   int subarray_start[2];
   
   MPI_Datatype subtype;
   int iter, warmup;
   int rank;
   double **array;
   double *temp;
   
   

   double t1,t2;
   

   MPI_Init(&argc, &argv);

   
   if(argc != 4) {
      printf("%s size subsize iters\n", argv[0]);
      exit(1);
   }
   

   size = atoi(argv[1]);
   subsize = atoi(argv[2]);
   iter = atoi(argv[3]);
   warmup = 100;
   
   

   array_size[0] = size;
   array_size[1] = size;

   subarray_start[0] = 1;
   subarray_start[1] = 1;
   array  = malloc_2d(array_size[0], array_size[1]);
   temp = malloc(sizeof(double) * size * size);
   
   
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
   for (i=0; i<array_size[0]; i++)
      for (j=0; j<array_size[1]; j++)
         array[i][j] = rank;

   for(subsize = 1 ; subsize < 600; subsize *= 2) {
   subarray_size[0]=subsize;
   subarray_size[1]=subsize;


      printf("subsize %d\n", subsize);
      
      
      t1 = MPI_Wtime();
      for(int i =0 ; i< warmup; i++) {
      MPI_Type_create_subarray(2, array_size, subarray_size, subarray_start, MPI_ORDER_C, MPI_DOUBLE, &subtype);
      MPI_Type_commit(&subtype);      
      if (rank==0)  MPI_Recv(array[0], 1, subtype, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (rank==1)   MPI_Send(array[0], 1, subtype, 0, 123, MPI_COMM_WORLD);
      MPI_Type_free(&subtype);
   }
   t2 = MPI_Wtime();



   t1 = MPI_Wtime();
   for(int i =0 ; i< iter; i++) {
      MPI_Type_create_subarray(2, array_size, subarray_size, subarray_start, MPI_ORDER_C, MPI_DOUBLE, &subtype);
      MPI_Type_commit(&subtype);      
      if (rank==0)  MPI_Recv(array[0], 1, subtype, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (rank==1)   MPI_Send(array[0], 1, subtype, 0, 123, MPI_COMM_WORLD);
      MPI_Type_free(&subtype);
   }
   t2 = MPI_Wtime();
   printf("Rank %d msgsize %d Create+commit+trans+free %g GB/s\n", 
          rank, 
          subarray_size[0] * subarray_size[1] * sizeof(double), 
          (subarray_size[0] * subarray_size[1] * sizeof(double))/ (( t2 - t1)/iter) * 1e-9 );



   MPI_Type_create_subarray(2, array_size, subarray_size, subarray_start, MPI_ORDER_C, MPI_DOUBLE, &subtype);
   MPI_Type_commit(&subtype);         

   for(int i =0 ; i< warmup; i++) {
      if (rank==0)  MPI_Recv(array[0], 1, subtype, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (rank==1)   MPI_Send(array[0], 1, subtype, 0, 123, MPI_COMM_WORLD);
   } 

   t1=MPI_Wtime();
   for(int i =0 ; i< iter; i++) {
      if (rank==0)  MPI_Recv(array[0], 1, subtype, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (rank==1)   MPI_Send(array[0], 1, subtype, 0, 123, MPI_COMM_WORLD);
   } 
   t2=MPI_Wtime();
   printf("Rank %d msgsize %d trans %g GB/s\n", 
          rank, 
          subarray_size[0] * subarray_size[1] * sizeof(double), 
          (subarray_size[0] * subarray_size[1] * sizeof(double))/ (( t2 - t1)/iter) * 1e-9 );

   MPI_Type_free(&subtype);



   for(int i =0 ; i< warmup; i++) {
      if (rank==0){
      
         MPI_Recv(temp, subsize * subsize, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         for (int y = 0; y < subsize; y++)
            for (int x = 0; x < subsize; x++)
               array[y + subarray_start[0]][ x + subarray_start[1]] = temp[x + y * subsize];
      }
      
      if (rank==1){
         for (int y = 0; y < subsize; y++)
            for (int x = 0; x < subsize; x++)
               temp[x + y * subsize] = array[y + subarray_start[0]][ x + subarray_start[1]]; 
         MPI_Send(temp, subsize * subsize, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD);
      }
   } 

   t1=MPI_Wtime();
   for(int i =0 ; i< iter; i++) {
      if (rank==0){
      
         MPI_Recv(temp, subsize * subsize, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         for (int y = 0; y < subsize; y++)
            for (int x = 0; x < subsize; x++)
               array[y + subarray_start[0]][ x + subarray_start[1]] = temp[x + y * subsize];
      }
      
      if (rank==1){
         for (int y = 0; y < subsize; y++)
            for (int x = 0; x < subsize; x++)
               temp[x + y * subsize] = array[y + subarray_start[0]][ x + subarray_start[1]]; 
         MPI_Send(temp, subsize * subsize, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD);
      }
   } 
   t2=MPI_Wtime();
   printf("Rank %d msgsize %d manual %g GB/s\n", 
          rank, 
          subarray_size[0] * subarray_size[1] * sizeof(double), 
          (subarray_size[0] * subarray_size[1] * sizeof(double))/ (( t2 - t1)/iter) * 1e-9 );
   }
   
}

