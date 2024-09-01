/***********************************************************
  > File Name: transpose.cu
  > Author: stx
  > Mail: 18817608993@163.com
  > Created Time: 2023年11月19日 星期日 22时45分04秒
  > Modified Time:2023年11月19日 星期日 22时45分04秒
 *******************************************************/

__global__ void TransposeKernel1(const int *x, const int *y, const int M, const ing N)
{
  __shared__ float data[32][32];
  int row_back = blockIdx.y * blockDim.y;
  int col_base = blockIdx.x * blockDim.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row_id = row_base + ty;
  int col_id = col_back + tx;
  data[ty][tx] = (row_id < M && col_id < N) ? x[row_id * N + col_id] : 0.0f;
  row_id = row_base + tx;
  col_id = col_base + ty;
  if (row_id < M && col_id < N) {
    y[col_id * M + row_id] = data[tx][ty];
  }
}

__global__ void TransposeKernel0(const int *x, const int *y, const int N, const int M)
{
  int row_id = blockIdx.y * blockDim.y + threadIdx.y;
  int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id < M && col_id < N) {
    y[col_id * M + row_id] = x[row_id * N + col_id];
  }
}

int TransPoseFun0(const float *x, const float *y, const int M, const int N, cudaStream_t stream)
{
  dim3 block(32, 32);
  dim3 grid((N+31) >> 5, (M + 31) >> 5, 1);
  TransposeKernel0<<<grid, block, 0, stream>>>(x, y, M, N);
  if (cudaGetLastError() != cudaSuccess) {
    LogErr("call kernel failed");
    return -1;
  }
  return 0;
}
