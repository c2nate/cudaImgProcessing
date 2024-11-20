%%writefile blurring.cu

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iomanip> // For formatting the output
#include <cstdlib>  // For handling exit

#define CHECK_CUDA_CALL(call) \
    { cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }}

__global__ void gaussian_blur(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_kernel = kernel_size / 2;
    float kernel[15][15]; // Max kernel size of 15x15 for simplicity
    float kernel_sum = 0.0f;

    // Generate Gaussian kernel
    for (int i = -half_kernel; i <= half_kernel; i++) {
        for (int j = -half_kernel; j <= half_kernel; j++) {
            float dist_sq = i * i + j * j;
            kernel[i + half_kernel][j + half_kernel] = expf(-dist_sq / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);
            kernel_sum += kernel[i + half_kernel][j + half_kernel];
        }
    }

    // Normalize kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= kernel_sum;
        }
    }

    float pixel_value = 0.0f;

    // Apply Gaussian kernel
    for (int i = -half_kernel; i <= half_kernel; i++) {
        for (int j = -half_kernel; j <= half_kernel; j++) {
            int ni = y + i;
            int nj = x + j;
            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                pixel_value += input[ni * width + nj] * kernel[i + half_kernel][j + half_kernel];
            }
        }
    }

    output[y * width + x] = static_cast<unsigned char>(pixel_value);
}

void apply_gaussian_blur(const cv::Mat& input, cv::Mat& output, int kernel_size, float sigma, float& elapsed_time) {
    int width = input.cols;
    int height = input.rows;

    // Allocate memory on GPU
    unsigned char *d_input, *d_output;
    size_t size = width * height * sizeof(unsigned char);
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_output, size));

    // Copy input image to GPU
    CHECK_CUDA_CALL(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));

    // Configure CUDA kernel
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA_CALL(cudaEventRecord(start, 0));

    // Launch Gaussian blur kernel
    gaussian_blur<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size, sigma);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // Stop timing
    CHECK_CUDA_CALL(cudaEventRecord(stop, 0));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0.0f;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    elapsed_time = milliseconds / 1000.0f; // Convert to seconds

    // Copy output image back to CPU
    CHECK_CUDA_CALL(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));

    // Free GPU memory and destroy events
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    // Check if image path is provided via command line
    if (argc < 2) {
        std::cerr << "Error: Please provide the image path as a command line argument!" << std::endl;
        return -1;
    }

    // Load the input image as grayscale
    std::string image_path = argv[1];
    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Error: Could not load input image from path: " << image_path << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    // Define blur levels (9 levels of increasing blur)
    int blur_levels[9][2] = {
        {3, 1},   // Basic blur with kernel size 3, sigma 1
        {5, 1},   // Slightly stronger blur
        {7, 2},   // Medium blur
        {9, 4},   // Stronger blur
        {11, 4},  // Even stronger blur
        {13, 5},  // Strong blur
        {15, 6},  // Very strong blur
        {17, 7},   // Maximum blur
        {21, 8}   // Maximum MAXIMUM
    };

    // Apply Gaussian blur for each level and save the result
    for (int i = 0; i < 9; ++i) {
        int kernel_size = blur_levels[i][0];
        float sigma = static_cast<float>(blur_levels[i][1]);

        cv::Mat output(height, width, CV_8UC1);
        float elapsed_time = 0.0f;

        apply_gaussian_blur(input, output, kernel_size, sigma, elapsed_time);

        std::string output_filename = "blurred_image_level_" + std::to_string(i + 1) + ".jpg";
        cv::imwrite(output_filename, output);
        std::cout << "Saved: " << output_filename << std::endl;

        // Print timing information
        std::cout << "level " << (i + 1) << " blur:               ";
        std::cout << std::fixed << std::setprecision(6) << elapsed_time << " s" << std::endl;
    }

    return 0;
}
