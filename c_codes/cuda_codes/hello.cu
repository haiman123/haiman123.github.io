/***********************************************************
  > File Name: hello.cu
  > Author: stx
  > Mail: 18817608993@163.com
  > Created Time: 2023年11月10日 星期五 01时16分38秒
  > Modified Time:2023年11月10日 星期五 01时16分38秒
 *******************************************************/
#include <stdio.h>
#include <iostream>
#include <vector>
__global__ void HelloWorldKernel() { printf("[BlockIdx[%d, %d, %d], threadIdx:[%d, %d, %d] hello world\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z); }

int Test1() {
   // cudaSetDevice(1);
   dim3 grid(2, 1, 1);
   dim3 block{8, 1, 1};
   HelloWorldKernel<<<grid, block>>>();
   cudaDeviceSynchronize();
   return 0;
}

__global__ void VecAddKernel(const float *A, const float *B, float *C, const int N) {
//__global__ void VecAddKernel(const float *A, const float *B, float *C) {
  auto bid = blockIdx.x;
  auto tid = threadIdx.x;
  auto id = bid * blockDim.x + tid;
  // printf("A[%d}:%f\n", tid, A[tid]);
  if (id < N) 
    C[id] = A[id] + B[id];
  
}

int Test2() {
  // C = A + B
  int N = 200000;
  float *A(nullptr);
  float *B(nullptr);
  float *C(nullptr);
  cudaMalloc(&A, sizeof(float) * N);
  cudaMalloc(&B, sizeof(float) * N);
  cudaMalloc(&C, sizeof(float) * N);
  std::vector<float> cpu_a(N, 0);
  std::vector<float> cpu_b(N, 0);
  std::vector<float> cpu_c(N, 0);
  for (std::size_t i = 0; i != cpu_a.size(); i++) {
    cpu_a[i] = 0.1f * i;
    cpu_b[i] = 0.2f * i;
    cpu_c[i] = cpu_a[i] + cpu_b[i];
  }
  cudaMemcpyAsync(A, cpu_a.data(), sizeof(float) * N, cudaMemcpyHostToDevice, nullptr);
  cudaMemcpyAsync(B, cpu_b.data(), sizeof(float) * N, cudaMemcpyHostToDevice, nullptr);
  
  // printf("A ptr:%lld", (intptr_t)A);
  dim3 grid((N + 1023) / 1024, 1, 1);
  dim3 block(1024, 1, 1); // max 1024 thread
  cudaStream_t stream;
  cudaStreamCreate(& stream);
  VecAddKernel<<<grid, block, 0, stream>>>(A, B, C, N);
  // printf("============cuda error is %d\n", cudaGetLastError());
  if (cudaGetLastError() != cudaSuccess) {
    printf("launch kernel failed! \n");
    return -1;
  }
  cudaDeviceSynchronize();
  std::vector<float> copy_cpu_c(N, 0);
  cudaMemcpy(copy_cpu_c.data(), C, sizeof(float) * N, cudaMemcpyDeviceToHost);
  for (std::size_t i = 0; i != copy_cpu_c.size(); i++) {
    if (copy_cpu_c[i] != cpu_c[i]) {
      printf("result %d error\n", i);
      return -1;
    }
    // printf("%f + %f = c[%d] =  %f\n", cpu_a[i], cpu_b[i], i, copy_cpu_c[i]);

  }
  printf("Test Passed\n");
  return 0;
}

int main() {
  Test1();
  Test2();
  return 0;
};
