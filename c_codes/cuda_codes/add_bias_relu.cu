/***********************************************************
  > File Name: add_bias_relu.cu
  > Author: stx
  > Mail: 18817608993@163.com
  > Created Time: 2023年11月12日 星期日 23时53分04秒
  > Modified Time:2023年11月12日 星期日 23时53分04秒
 *******************************************************/

// grid b
// block M
template <typename Type>
__global__ void AddBiasReluKernel0(const Type *x, const Type *bias, Type *y, const int N)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  for (auto id = tid; id < N; id += blockDim.x) {
    float v = (float)x[bid * N + id] + (float)bias[id];
    v = fmax(v, 0);
    y[bid * N + id] = (Type)v;
  }
}

int AddBiasRelu0(const float *x, const float *bias, float *y, const int B, const int N, cudaStream_t stream)
{
  dim3 grid(B);
  dim3 block(std::min(N, 1024));
  AddBiasReluKernel0<float><<<gird, block, 0, stream>>>(x, bias, y, N);
  if (cudaGetLastError() != cudaSuccess) {
    LogErr("launch kernel failed");
    return -1;
  }
  return 0;
}

template <typename Type>
__global__ void AddBiasReluKernel1(const Type *x, const Type *bias, Type *y, const int Num, const int N)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < Num) {
    int bias_id = id % N;
    float v = (float)x[id] + (float)bias[bias_id];
    v = fmax(v, 0);
    y[id] = (Type)v;
  }
}

int AddBiasRelu1(const float *x, const float *bias, float *y, const int B, const int N, cudaStream_t stream) {
  int Num = B * N;
  dim3 block(std::min(Num, 1024));
  dim3 grid((num + block - 1) / block);
  AddBiasReluKernel1<float><<<grid, block, 0, stream>>>(x, bias, y, Num, N);
  if (cudaGetLastError() != cudaSuccess) {
    LogErr("launch kernel failed");
    return -1;
  }
  return 0;
}
