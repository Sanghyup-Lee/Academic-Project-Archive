#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 6
#define MASK_RADIUS (MASK_WIDTH/2)
#define TILE_SIZE (TILE_WIDTH + (MASK_RADIUS * 2))

//@@ Define constant memory for device kernel here
__constant__ float Mem[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N_ds[TILE_SIZE][TILE_SIZE][TILE_SIZE];
  int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
  int Dep = blockIdx.z*TILE_WIDTH + threadIdx.z;


  if((Row-MASK_RADIUS)>=0 && (Row-MASK_RADIUS)<y_size && (Col-MASK_RADIUS)>=0 && (Col-MASK_RADIUS)<x_size && (Dep-MASK_RADIUS)>=0 && (Dep-MASK_RADIUS)<z_size)
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = input[(Dep-MASK_RADIUS)*x_size*y_size + (Row-MASK_RADIUS)*x_size + (Col-MASK_RADIUS)];

  else
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
  __syncthreads();

  float Pvalue = 0.0;
  if(threadIdx.x<TILE_WIDTH && threadIdx.y<TILE_WIDTH && threadIdx.z<TILE_WIDTH)
  {
    for (int i=0; i<MASK_WIDTH; i++)
    {
      for (int j=0; j<MASK_WIDTH; j++)
      {
        for (int k=0; k<MASK_WIDTH; k++)
        {
          Pvalue += Mem[i][j][k] * N_ds[threadIdx.z+i][threadIdx.y+j][threadIdx.x+k];
        }
      }
    }
    if (Row<y_size && Col<x_size && Dep<z_size)
    {
      output[Dep*x_size*y_size + Row*x_size + Col] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, x_size*y_size*z_size*sizeof(float));
  cudaMalloc((void**) &deviceOutput, x_size*y_size*z_size*sizeof(float));



  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput,hostInput+3,x_size*y_size*z_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mem,hostKernel,kernelLength*sizeof(float),0,cudaMemcpyHostToDevice);


  //@@ Initialize grid and block dimensions here
  dim3 grid_size(ceil(x_size/float(TILE_WIDTH)), ceil(y_size/float(TILE_WIDTH)),ceil(z_size/float(TILE_WIDTH)));
  dim3 block_size(TILE_SIZE,TILE_SIZE,TILE_SIZE);

  //@@ Launch the GPU kernel here
  conv3d<<<grid_size,block_size>>>(deviceInput,deviceOutput,z_size,y_size,x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3,deviceOutput,x_size*y_size*z_size*sizeof(float),cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

