
import numpy as np
import matplotlib.pyplot as plt
from vision.part1 import create_Gaussian_kernel_2D
kernel = create_Gaussian_kernel_2D(24)


fft_kernel = np.fft.fft2(kernel)

plt.imsave('./extra/kernel_without_mask.png', kernel)
plt.imsave('./extra/kernel_frequency_without_mask.png',
           np.fft.ifftshift(np.log(np.abs(fft_kernel))))

for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        if (i-kernel.shape[0]/2)**2 + (j-kernel.shape[1]/2)**2 > (kernel.shape[0]/2)**2:
            kernel[i][j] = kernel[kernel.shape[1]//2][kernel.shape[1]-1]

fft_kernel = np.fft.fft2(kernel)

plt.imsave('./extra/kernel_with_mask.png', kernel)
plt.imsave('./extra/kernel_frequency_with_mask.png',
           np.fft.ifftshift(np.log(np.abs(fft_kernel))))
