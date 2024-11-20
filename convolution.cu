%%writefile convolution.cu

#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>


//defining filter radius and size as constant
#define FILTER_RADIUS 2
#define FILTER_SIZE 2 * FILTER_RADIUS + 1


//constant for F_h
const float F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};


//constant for F_d
__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];




//CHECK call
#define CHECK(call) \
{ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}


//verification that cpu matches other results
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols)   {
    const float relativeTolerance = 1e0;
    for(int i=0; i<nRows; i++)  {
        for(int j=0; j<nCols; j++)  {
            float relativeError = ((float)answer1.at<unsigned char>(i,j) - (float)answer2.at<unsigned char>(i,j))/255;
            if (relativeError > relativeTolerance || relativeError < -relativeTolerance)    {
                printf("TEST FAILED at (%d, %d) with relativeError: %f\n", i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d): %u\n", i, j, answer1.at<unsigned char>(i,j));
                printf("    answer2.at<unsigned char>(%d, %d): %u\n\n", i, j, answer2.at<unsigned char>(i,j));
                return false;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return true;
}


//cpu timer
double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ( (double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}


//cpu blur image function
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            float sum = 0.0;
            for (int k = -FILTER_RADIUS; k <= FILTER_RADIUS; ++k) {
                for (int l = -FILTER_RADIUS; l <= FILTER_RADIUS; ++l) {
                    int row = i + k;
                    int col = j + l;
                    if (row >= 0 && row < nRows && col >= 0 && col < nCols) {
                        sum += static_cast<float>(Pin_Mat_h.at<unsigned char>(row, col)) * F_h[k + FILTER_RADIUS][l + FILTER_RADIUS];
                    }
                }
            }
            Pout_Mat_h.at<unsigned char>(i, j) = static_cast<unsigned char>(sum);
        }
    }
}


//gpu blur image kernel
__global__ void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;


    if (colIdx < width && rowIdx < height) {
        float sumPixVal = 0.0f;
        float normalizationFactor = 0.0f;  // Updated to handle variable filter sizes

        for (int blurRowOffset = -FILTER_RADIUS; blurRowOffset <= FILTER_RADIUS; blurRowOffset++) {
            for (int blurColOffset = -FILTER_RADIUS; blurColOffset <= FILTER_RADIUS; blurColOffset++) {
                int curRowIdx = rowIdx + blurRowOffset;
                int curColIdx = colIdx + blurColOffset;


                if (curRowIdx >= 0 && curRowIdx < height && curColIdx >= 0 && curColIdx < width) {
                    float pixelValue = static_cast<float>(Pin[curRowIdx * width + curColIdx]);
                    float filterValue = F_d[blurRowOffset + FILTER_RADIUS][blurColOffset + FILTER_RADIUS];
                    sumPixVal += pixelValue * filterValue;
                    normalizationFactor += filterValue;
                }
            }
        }


        Pout[rowIdx * width + colIdx] = static_cast<unsigned char>(sumPixVal / normalizationFactor);  // Updated to use normalization factor
    }
}


//driver function for gpu kernel
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {

    unsigned char *Pin_d, *Pout_d;


    //malloc
    CHECK(cudaMalloc((void**)&Pin_d, nRows * nCols *sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&Pout_d, nRows * nCols *sizeof(unsigned char)));


    //memcpy
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));


    //memcpy to symbol
    CHECK(cudaMemcpyToSymbol(F_d, F_h, (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1)*sizeof(float)));


    dim3 blockSize(32, 32);
    dim3 gridSize((nCols+blockSize.x-1)/ blockSize.x, (nRows+blockSize.y -1)/ blockSize.y);


    //launch
    blurImage_Kernel<<<gridSize, blockSize>>>(Pout_d, Pin_d, nCols, nRows);


    //memcpy back
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows* nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));


    //mem free
    cudaFree(Pin_d);
    cudaFree(Pout_d);

}


//define input and output tile dimensions
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


//tiled gpu blur kernel
__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;


    //using shared memory
    __shared__ unsigned char N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = Pin[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0;  // Assuming unsigned char uses 0 for default value
    }
    //synchronize
    __syncthreads();


    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;


   // loops for actual blurring
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            Pout[row * width + col] = (unsigned char)Pvalue;  // Assuming casting float to unsigned char is acceptable
        }
    }
}


//driver function for tiled gpu blurring
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {

    unsigned char *Pin_d, *Pout_d;

    //malloc
    CHECK(cudaMalloc((void**)&Pin_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&Pout_d, nRows * nCols * sizeof(unsigned char)));


    //memcpy
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows*nCols*sizeof(unsigned char), cudaMemcpyHostToDevice));


    //memcpy to symbol
    CHECK(cudaMemcpyToSymbol(F, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float)));


    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridSize((nCols + OUT_TILE_DIM -1)/ OUT_TILE_DIM, (nRows+OUT_TILE_DIM-1)/OUT_TILE_DIM);


    //kernel launch
    blurImage_tiled_Kernel<<<gridSize, blockSize>>>(Pout_d, Pin_d, nCols, nRows);


    //memcpy back
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows*nCols*sizeof(unsigned char), cudaMemcpyDeviceToHost));


    //mem free
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}


    //host function serves as entry into program
int main(int argc, char** argv) {
    cudaDeviceSynchronize();


    //timers
    double startTime, endTime;


    //usage
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return -1;
    }


    //error handling
    std::string inputFileName = argv[1];
    cv::Mat grayImg = cv::imread(inputFileName, cv::IMREAD_GRAYSCALE);
    if(grayImg.empty()) {
        std::cerr << "Error: Couldn't load the input image." << std::endl;
        return -1;
    }


    unsigned int nRows = grayImg.rows, nCols = grayImg.cols;


    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    cv::blur(grayImg, blurredImg_opencv, cv::Size(2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    endTime = myCPUTimer();
    printf("openCV's blur (CPU):                    %f s\n\n", endTime - startTime); fflush(stdout);


    cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on CPU:                       %f s\n\n", endTime - startTime); fflush(stdout);


    cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on GPU:                       %f s\n\n", endTime - startTime); fflush(stdout);


    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("(tiled) blurImage on GPU:               %f s\n\n", endTime - startTime); fflush(stdout);



    bool check = cv::imwrite("./blurredImg_opencv.jpg", blurredImg_opencv);
    if(!check)  {std::cerr << "Error writing blurredImg_opencv.jpg" << std::endl; return -1;}

    check = cv::imwrite("./blurredImg_cpu.jpg", blurredImg_cpu);
    if(!check)  {std::cerr << "Error writing blurredImg_cpu.jpg" << std::endl; return -1;}

    check = cv::imwrite("./blurredImg_gpu.jpg", blurredImg_gpu);
    if(!check)  {std::cerr << "Error writing blurredImg_gpu.jpg" << std::endl; return -1;}

    check = cv::imwrite("./blurredImg_tiled_gpu.jpg", blurredImg_tiled_gpu);
    if(!check)  {std::cerr << "Error writing blurredImg_tiled_gpu.jpg" << std::endl; return -1;}

    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

    return 0;
}
