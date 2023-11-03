#include <cuda_runtime.h>
#include <vector>
#include <iostream>


template<typename T>
__inline__ __device__ T warpReduceSum(T x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}

__global__ void reduce_col(
    const int* in, //(m, n) 
    const int m, 
    const int n, 
    int* out //(1, n)
){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y; 

    __shared__ int cache[32][32];//为了方便实现和理解用32*32线程数，也可以改成16*64或其他配置
    
    //每一列先累加到一个block中
    int sum = 0;
    if(tidx < n){
        for(int i = tidy; i < m; i += blockDim.y){
            sum += in[i * n + tidx];
        }
    }
    
    //将累加结果做转置，方便做warp reduce
    cache[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();

    //block内每一行做reduce，的到32个结果
    int x = cache[threadIdx.y][threadIdx.x];	
	x = warpReduceSum<int>(x);
    __syncthreads();

    if(threadIdx.x == 0){
        out[blockIdx.x * blockDim.x + threadIdx.y] = x;
    }
}

void print(std::vector<int>& data, const int m, const int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            std::cout << data[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(){
    const int m = 32;
    const int n = 32;
    std::vector<int> h_in(m * n), h_out(n, 0);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            h_in[i * n + j] = i * n + j;
            h_out[j] += h_in[i * n + j];
        }
    }
    print(h_in, m, n);

    int *d_in, *d_out;
    cudaMalloc(&d_in, m*n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));

    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((n + 31) / 32, 1, 1);
    reduce_col<<<gridDim, blockDim>>>(d_in, m, n, d_out);

    std::vector<int> check_out(n);
    cudaMemcpy(check_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    print(h_out, 1, n);
    print(check_out, 1, n);
    return 0;
}
