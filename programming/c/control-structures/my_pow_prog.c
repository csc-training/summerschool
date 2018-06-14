/* my_pow.c -- Print out powers of k: 1, k, k^2, k^3, .. up to k^N */

#include <stdio.h>

int main(void)
{
    int N = 16;
    int k = 2;

    int n;                      /* The current exponent */
    int val = 1;                /* The current power of k  */

    printf("\t  n  \t    2^n\n");
    printf("\t================\n");

    /* Add here a for loop that at each iteration prints out
       the current value of the power and updates val */
#error Add the missing loop

    return 0;
}

/* It prints out :

      n         2^n
    ================
      0           1
      1           2
      2           4
      3           8
      4          16
      5          32
      6          64
      7         128
      8         256
      9         512
     10        1024
     11        2048
     12        4096
     13        8192
     14       16384
     15       32768
     16       65536

*/
