# cudaImgProcessing
This is a foundational CUDA implementation of image blurring (done with google colab gpus).


convolution.cu is a series of blurring functions implemented via: cpu only, opencv, gpu, and tiled gpu to show the timing differences. 
blurring.cu is an extension using gpu to perform gaussian blurring at various levels to show the speed and strength of the gpu for this use case.

Each file is meant to show a variety of attempts at blurring. 
Where convolution.cu applies different methods of blurring.
1) Only the cpu's power to blur the image. 
2) opencv uses an open library that is quite effective for this task.
3) The gpu version uses only the gpu to apply the blur (a basic function to call the gpu only).
4) The tiled gpu version is a modified version of the basic gpu function to section out the image into tiles to further parallelize the problem, theoretically improving efficiency.


lib.jpg and sherk.jpg are provided sample images with which the program can be tested.
jpg images can be used to test this so long as they are stored within the same directory as the .cu files.

Each .cu file starts with "%%writefile (name.cu)" because it was done on google colab. This can be omitted or adjusted if testing on a different system.
