
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M_MAT_A_ROW_NUM         4023 //how many rows in A
#define K_MAT_A_COLUMN_NUM      1024 //how many column in A
#define K_MAT_B_ROW_NUM         K_MAT_A_COLUMN_NUM //how many rows in B
#define N_MAT_B_COLUMN_NUM      1025 //how many column in B

#define RTX3060
#ifdef RTX3060
#define SM_NUM                  30
#define CUDA_CORE_PER_SM        128
#define CUDA_CORE_PER_WARP      16
#else 
#ifdef V100
#define SM_NUM   80
#define CUDA_CORE_PER_SM        64
#define CUDA_CORE_PER_WARP      16
#else
#define SM_NUM   40
#define CUDA_CORE_PER_SM        128
#define CUDA_CORE_PER_WARP      16
#endif
#endif
#define M_MAT_C_ROW_NUM         M_MAT_A_ROW_NUM        //how many rows in C
#define N_MAT_C_COLUMN_NUM      N_MAT_B_COLUMN_NUM     //how many column in C
#define MATRIX_GLOBAL_SIZE      (M_MAT_C_ROW_NUM * N_MAT_C_COLUMN_NUM)

void mulMatrixWithCpu(float* c, float* a, float* b);
cudaError_t mulMatrixWithCuda(float *c, float *a, float *b);

//C = aAB+bC
__global__ void mulKernel(float* c, float* a, float* b)
{
    int i = 0;
    int j = 0;
    for(int index = blockIdx.x * blockDim.x + threadIdx.x;index < MATRIX_GLOBAL_SIZE;index+=gridDim.x*blockDim.x)
    {
        i = index / N_MAT_C_COLUMN_NUM;
        j = index % N_MAT_C_COLUMN_NUM;
        for (int k = 0; k < K_MAT_A_COLUMN_NUM; k++)
        {
            c[index] += a[i + k * M_MAT_A_ROW_NUM] * b[K_MAT_A_COLUMN_NUM * j + k];
        }
    }
}


int main()
{
    float* a = (float*)malloc(M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM*sizeof(float));
    float* b = (float*)malloc(K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM*sizeof(float));
    float* c_gpu_result = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* c_cpu_result = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));

    srand((unsigned)time(NULL));
    for(int i=0;i< MATRIX_GLOBAL_SIZE;i++)
    {
        c_gpu_result[i] = 0.0;
        c_cpu_result[i] = 0.0;
    }
    for (int i = 0; i < M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM; i++)
        a[i] = (rand() % 3)/3.0;// (111.1f / (float)i) % 2.4f;
    for (int i = 0; i < K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM; i++)
        b[i] = (rand() % 3)/3.0;// (i / 111.0) % 3.0;
    
    mulMatrixWithCpu(c_cpu_result, a, b);

    // Multiply matrix in parallel.
    cudaError_t cudaStatus = mulMatrixWithCuda(c_gpu_result, a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("Sample results:\n");
    for(int i=0;i<7;i++)
        printf("cpu[%d]: %5f,gpu[%d]:%5f\n",i, c_cpu_result[i],i, c_gpu_result[i]);

    printf("\n\nStarting validation the result...\n ");
    for (int i = 0; i < MATRIX_GLOBAL_SIZE; i++)
        if (c_cpu_result[i] - c_gpu_result[i] > 0.01 || c_gpu_result[i] - c_cpu_result[i] > 0.01)
            printf("c[%d] is not correct\n",i);
    printf("Validation is completed.\n ");
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
//A: column major; B: column major; C: row major;
void mulMatrixWithCpu(float* c, float* a, float* b)
{
    int i_x = 0, i_y = 0;

    for (int i = 0;i < M_MAT_A_ROW_NUM * N_MAT_B_COLUMN_NUM;i++)
    {
        i_x = i / N_MAT_C_COLUMN_NUM;  //i_x line of A,
        i_y = i % N_MAT_C_COLUMN_NUM;  //i_y column of B;
        for (int j = 0;j < K_MAT_A_COLUMN_NUM;j++)
        {
            c[i] += a[i_x + j * M_MAT_A_ROW_NUM] * b[j + i_y * K_MAT_B_ROW_NUM];
        }   
    }
}

__global__ void blank_warmingGPU() {}

cudaError_t mulMatrixWithCuda(float* c, float* a, float* b)
{
    float *dev_a = NULL;
    float *dev_b = NULL;
    float* dev_c = NULL;
    cudaError_t cudaStatus = cudaSuccess;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess){
        printf("cudaStatus_1=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((float**)&dev_c, MATRIX_GLOBAL_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess){
        printf("cudaStatus_2=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((float**)&dev_a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus_3=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus=cudaMalloc((float**)&dev_b,K_MAT_B_ROW_NUM*N_MAT_B_COLUMN_NUM*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus_4=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus_5=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus_6=%d \n", cudaStatus);
        goto Error;
    }

    blank_warmingGPU << <1, 1 >> > ();
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    mulKernel << <SM_NUM, 32*CUDA_CORE_PER_SM / CUDA_CORE_PER_WARP >> > (dev_c, dev_a, dev_b);//<<<30,256>>>LL::30个sm，30个block；32t/w  * (128/16)
   // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time_mulKernel is %f ms.\n\n", time);
    // clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){
        printf("cudaStatus_7=%d \n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess){
        printf("cudaStatus_8=%d \n", cudaStatus);
        goto Error;
    }
        
    cudaStatus = cudaMemcpy(c, dev_c, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){
        printf("cudaStatus_9=%d \n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return cudaStatus;
}