# cudaImgProcessing
basic CUDA implementation of image blurring (done with google colab gpus)


convolution.cu is a series of blurring functions implemented via: cpu only, opencv, gpu, and tiled gpu to show the timing differences. 
blurring.cu is an extension using gpu to perform gaussian blurring at various levels to show the speed and strength of the gpu for this use case.

lib.jpg and sherk.jpg are sample images to test this program with.


Each .cu file starts with "%%writefile (name.cu)" because it was done on google colab. This can be omitted or adjusted if testing on a different system.
