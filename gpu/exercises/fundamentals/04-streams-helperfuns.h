__device__ float recip_factpow(float x, int n) {
  float retval(1);
  for (int k = 1; k<= n; ++k) {
    retval = retval * x/k;
  }
  return retval;
}

__device__ int factorial(int m) {
  float retval(1);
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
  for (k = n; k< n+3; ++k) t += pow(16,n-k) / (8*k+j);

  float r = s+t;
  return s+t-(int)(s+t);
}

// GPU kernel definition
__global__ void kernel_a(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  constexpr int Q = 10000;
  a[tid] = float(0);

  if (tid < n) {
    for (int l = 0; l < Q; ++l) {
      float x = (float)tid;
      float s = sinf(x+l);
      float c = cosf(x+l);
      a[tid] = a[tid] + sqrtf(s*s+c*c);
    }
  }
}

__global__ void kernel_b(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  float S1, S2, S3, S4;
  if (tid < n) {
      S1 = S(1,tid);
      S2 = S(4,tid);
      S3 = S(5,tid);
      S4 = S(6,tid),
    a[tid] = 4*S1-2*S2-S3-S4;
    a[tid] = (a[tid] > 0) ? (a[tid] - (int)a[tid]) : (a[tid]-(int)a[tid] + 1);
    a[tid] = a[tid]*16;
  }
}

__global__ void kernel_c(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  constexpr int Q = 100;
  a[tid] = float(0);
  if (tid < n) {
    float x = float(100)*float(tid)/n;
    for (size_t m = 0; m < Q; ++m) {
      a[tid] += pow(-1,m)*recip_factpow(x/2, m)*recip_factpow(x/2,m+1);
    }
  }
}
