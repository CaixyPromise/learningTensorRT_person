#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int N)
{
    // 获取本线程的索引，blockIdx 指的是线程块的索引，blockDim 指的是线程块的大小，threadIdx 指的是本线程块中的线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("tid: %d blockIdx.x: %d blockDim.x: %d threadIdx.x: %d \n", tid, blockIdx.x, blockDim.x, threadIdx.x);
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char **argv)
{
    // 检查命令行参数
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./simple <N>");
    }
    int N = std::atoi(argv[1]);
    int a[N], b[N], c[N], c_from_gpu[N];
    int *dev_a, *dev_b, *dev_c;

    // 在设备端分配内存
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    // 初始化数组
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // 统计CPU上运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time spent on CPU: %f ms\n", time);

    // 将数据从主机端复制到设备端
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 调用kernel函数，在GPU上运行并发计算
    // 一个线程块包含256个线程
    int n_threads = 256;
    // 计算线程块的数量
    int n_blocks = std::ceil(N * 1.0f / n_threads);

    // 统计时间
    cudaEventRecord(start, 0);
    // 调用kernel函数，传递线程块数量和大小
    add<<<n_blocks, n_threads>>>(dev_a, dev_b, dev_c, N); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time spent on GPU: %f ms\n", time);

    // 将数据从设备端复制到主机端
    cudaMemcpy(c_from_gpu, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 检查结果是否一致
    for (int i = 0; i < N; i++)
    {
        if (c[i] != c_from_gpu[i])
        {
            printf("Error: inconsistent results!\n");
        }
    }

    // 释放设备端内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
