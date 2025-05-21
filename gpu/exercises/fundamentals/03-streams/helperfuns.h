__device__ int factorial(int m) {
  int retval(1);
  for (int k = 2; k<=m; ++k) {
    retval *= k;
  }
  return retval;
}

__device__ int ipow_mod(int m, int n, int mod) {
  int ret(1);
  while ( n!=0) {
    if (n%2) ret = (ret*m) % mod;
    m = (m*m) % mod;
    n >>= 1;
  }
  return ret;
}

__device__ float S(int j,int n) {
  float s = 0.0;
  for (int k = 0; k<n;++k) s += ipow_mod(16, n-k, 8*k+j) / (8.0*k+j);
  float t = 0.0;
  int k = n;
  for (k = n; k< n+10; ++k) t += pow(16,n-k) / (8*k+j);

  float r = s+t;
  return s+t-(int)(s+t);
}
