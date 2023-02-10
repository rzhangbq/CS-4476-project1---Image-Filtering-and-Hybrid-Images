from vision.part4 import my_conv2d_freq, my_deconv2d_freq
from vision.utils import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from vision.utils import load_image, save_image
from vision.part1 import create_Gaussian_kernel_2D

image = rgb2gray(load_image('data/1a_dog.bmp'))
kernel = create_Gaussian_kernel_2D(7)

fft_image, fft_kernel, fft_conv_result, conv_result = my_conv2d_freq(image,kernel)


plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/image.png',image, cmap='gray')

plt.subplot(1,2,2)
plt.imsave('temp/np.png',np.fft.ifftshift(np.log(np.abs(fft_image))))


plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/kernel.png',kernel)

plt.subplot(1,2,2)
plt.imsave('temp/kernel_freq.png',np.fft.ifftshift(np.log(np.abs(fft_kernel))))


plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/conv_result.png',conv_result, cmap='gray')

plt.subplot(1,2,2)
plt.imsave('temp/conv_result_freq.png',np.fft.ifftshift(np.log(np.abs(fft_conv_result))))

image_freq, filter_freq, deconv_freq, deconv = my_deconv2d_freq(conv_result, kernel)

plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/conv_result_2.png',conv_result, cmap='gray')

plt.subplot(1,2,2)
plt.imsave('temp/conv_result_freq_2.png',np.fft.ifftshift(np.log(np.abs(image_freq))))


plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/kernel_2.png',kernel)

plt.subplot(1,2,2)
plt.imsave('temp/kernel_freq_2.png',np.fft.ifftshift(np.log(np.abs(filter_freq))))


plt.figure(figsize=(11,6))

plt.subplot(1,2,1)
plt.imsave('temp/deconv_2.png',deconv, cmap='gray')

plt.subplot(1,2,2)
plt.imsave('temp/deconv_freq_2.png',np.fft.ifftshift(np.log(np.abs(deconv_freq))))
