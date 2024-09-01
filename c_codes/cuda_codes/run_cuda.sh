#/***********************************************************
#  > File Name: run_cuda.sh
#  > Author: stx
#  > Mail: 18817608993@163.com
#  > Created Time: 2023年11月10日 星期五 01时19分45秒
#  > Modified Time:2023年11月10日 星期五 01时19分45秒
# *******************************************************/
#nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -gencode arch=compute_75,code=sm_75 hello.cu -o hello
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart hello.cu -o hello
