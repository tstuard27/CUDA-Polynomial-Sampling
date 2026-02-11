#include <stdio.h>
#include <iostream> 
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Indexer.h"
#include "CudaError.h"
#include <chrono>
#include <cmath>

int size = 10001;
double w0 = -1;
double w1 = 1;
double w2 = -1;
double b = 1;



namespace Kernels{
    __global__ void Sum_Kernel(double* data, int size, double w0, double w1, double w2, double b){
        
        unsigned long long grid_index = Indexer::FlatIndex2(threadIdx.x, blockIdx.x, blockDim.x);
        if (grid_index > size){
            return;
        }
        
        double x_i = data[grid_index];

        double y_i = (w0*pow(x_i, 3)) + (w1*pow(x_i, 2)) + (w2*x_i) + b;

        data[grid_index] = y_i;
        
    }

    __global__ void FirstDiff(double* y_arr, double* first_diff, int size){
        unsigned long long grid_index = Indexer::FlatIndex2(threadIdx.x, blockIdx.x, blockDim.x);

        if (grid_index < size - 1){
            first_diff[grid_index] = y_arr[grid_index + 1] - y_arr[grid_index];
        }
        
    }

    __global__ void SecondDiff(double* first_diff, double* second_diff, int size){
        unsigned long long grid_index = Indexer::FlatIndex2(threadIdx.x, blockIdx.x, blockDim.x);

        if (grid_index < size-2){
            second_diff[grid_index] = first_diff[grid_index + 1] - first_diff[grid_index];
        }
        
        __syncthreads();

    }

    __global__ void FindInflection(double* second_diff, int* inflection_dev, int size){
        for (int i=0; i < size-3; i++){
            if((second_diff[i+1] < 0 && second_diff[i] > 0) || (second_diff[i+1] > 0 && second_diff[i] < 0)){
                //printf("inflection point between %f and %f\n", x_array[i+1], x_array[i+2]);
                *inflection_dev = i+1;
                break;
                //printf("second_diff: %f %f\n", device_points[i], device_points[i+1]);
                
            }
                
        }
    } 

    __global__ void PositiveSample(double* y_arr, int* total_count, int size){
        unsigned long long grid_index = Indexer::FlatIndex2(threadIdx.x, blockIdx.x, blockDim.x);
        __shared__ int block_count;


        if (grid_index > size){
            return;
        }

        if (threadIdx.x == 0) block_count = 0;
        __syncthreads();

        if (y_arr[grid_index] > 0){
            atomicAdd(&block_count, 1);
        }
        __syncthreads();

        if (threadIdx.x == 0) atomicAdd(total_count, block_count);
    }
}

void print_graph(double* x_arr, double* y_arr, int size, int inflection_idx){

   
    int count = 0;
    int starting_index;
    int ending_index;

    for(int i = 0; i < size; i++){
        if(y_arr[i] >= -5 && y_arr[i] <= 5){
            count++;
        }
    }

    for(int i = 1; i < size; i++){
        if((y_arr[i-1] < -5 && y_arr[i] >= -5) || (y_arr[i-1] > -5 && y_arr[i] <= -5)){
            starting_index =  i;
            if(i <= inflection_idx){
                continue;
            }
            break;
        }
    }

    for(int i = 1; i < size; i++){
        if((y_arr[i-1] < 5 && y_arr[i] >= 5) || (y_arr[i-1] > 5 && y_arr[i] <= 5)){
            
            ending_index =  i-1;
            break;
        }
    }

    if(ending_index < starting_index){
        int temp = starting_index;
        starting_index = ending_index;
        ending_index = temp;
    }
    
    int step = count/21;

    for(int j = 5; j >= -5; j--){
        for(int i = starting_index; i<= ending_index; i+=step){
            if(y_arr[i] >= (double)(j) && y_arr[i] < (double)(j+1)){
                std::cout << "#";
            }
            else{
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
    
}

struct PolySum{
    void Sum(){
        

        
        double* x_array = new double[size];
        double* gpu_array;

        double* first_diff;
        double* second_diff;

        int* dev_count;
        
        

        cudaMalloc(&first_diff, (size-1)*sizeof(double));
        cudaMalloc(&second_diff, (size-2)*sizeof(double));

        double x_i = -10;
        double space = 20.0/((double)(size-1));

        for (int i = 0; i < size; i++){
            x_array[i] = x_i;
            x_i += space;
            
        }
        

        cudaMalloc(&gpu_array, (size_t)(size * sizeof(double)));
        cudaMemcpy(gpu_array, x_array, size * sizeof(double), cudaMemcpyHostToDevice);

        dim3 threads_per_block(32, 1, 1);

        int block_count = (size + threads_per_block.x - 1) / threads_per_block.x;

        dim3 blocks_per_grid(block_count, 1, 1);
        
        //compute sample 
        auto begin0 = std::chrono::steady_clock::now();

        Kernels::Sum_Kernel<<<blocks_per_grid, threads_per_block>>>(gpu_array, size, w0, w1, w2, b);
        cudaDeviceSynchronize(); 
        double* y_array = new double[size];
        cudaMemcpy(y_array, gpu_array, size * sizeof(double), cudaMemcpyDeviceToHost);
        auto end0 = std::chrono::steady_clock::now();

        CudaError::CheckError((cudaError_enum)cudaPeekAtLastError(), __FILE__, __LINE__);
		CudaError::CheckError((cudaError_enum)cudaDeviceSynchronize(), __FILE__, __LINE__);

        

        int* inflection_dev;
        int inflection_index;

        cudaMalloc(&inflection_dev, sizeof(int));
        cudaMemset(inflection_dev, -1, sizeof(int));

        //compute gradiests for inflection points
        auto begin1 = std::chrono::steady_clock::now();
        Kernels::FirstDiff<<<blocks_per_grid, threads_per_block>>>(gpu_array, first_diff, size);
        cudaDeviceSynchronize(); 
        Kernels::SecondDiff<<<blocks_per_grid, threads_per_block>>>(first_diff, second_diff, size);
        cudaDeviceSynchronize(); 
        Kernels::FindInflection<<<blocks_per_grid, threads_per_block>>>(second_diff, inflection_dev, size);
        cudaDeviceSynchronize();  
        
        auto end1 = std::chrono::steady_clock::now();

        cudaMemcpy(&inflection_index, inflection_dev, sizeof(int), cudaMemcpyDeviceToHost);

        CudaError::CheckError((cudaError_enum)cudaPeekAtLastError(), __FILE__, __LINE__);
		CudaError::CheckError((cudaError_enum)cudaDeviceSynchronize(), __FILE__, __LINE__);

        

        cudaMalloc(&dev_count, sizeof(int));
        cudaMemset(dev_count, 0, sizeof(int));


        auto begin2 = std::chrono::steady_clock::now();
        Kernels::PositiveSample<<<blocks_per_grid, threads_per_block>>>(gpu_array, dev_count, size);
        cudaDeviceSynchronize();
        int host_count;
        cudaMemcpy(&host_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
        auto end2 = std::chrono::steady_clock::now();

        CudaError::CheckError((cudaError_enum)cudaPeekAtLastError(), __FILE__, __LINE__);
		CudaError::CheckError((cudaError_enum)cudaDeviceSynchronize(), __FILE__, __LINE__);


        

        print_graph(x_array, y_array, size, inflection_index);
        if (inflection_index < 0){
            printf("No inflection points");
        } else{
            printf("Inflection points: (%.3f, %.3f)\n", x_array[inflection_index], y_array[inflection_index]);
        }
        
        printf("Positives: %d\n", host_count);
        float time_lapsed0 = std::chrono::duration<float>(end0 - begin0).count();
        float time_lapsed1 = std::chrono::duration<float>(end1 - begin1).count();
        float time_lapsed2 = std::chrono::duration<float>(end2 - begin2).count();
        printf("Sample Generation Runtime: %.5f\n", time_lapsed0);
        
        printf("Inflection Point Search Runtime: %.5f\n", time_lapsed1);
        printf("Positive Sample Count Runtime: %.5f\n", time_lapsed2);
    }

    
};