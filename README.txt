# Extra:
How to make the FFT of 2D Gaussian look the same as Gaussian?

## 1.Analyze the problem: There is a bright cross in the image:

![](./src/extra/kernel_without_mask.png)
![](./src/extra/kernel_frequency_without_mask.png)

## 2.Theory: This is probably because the image is square, not circular.

## 3.How to verify this theory? We can use a circular mask to cut the Gaussian into circle, and see whether the FFT still shows a bright cross.

## 4.Code:
```python
for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        if (i-kernel.shape[0]/2)**2 + (j-kernel.shape[1]/2)**2 > (kernel.shape[0]/2)**2:
            kernel[i][j] = kernel[kernel.shape[1]//2][kernel.shape[1]-1]
```
To exececute:

## 5.Result: The bright cross disappear after masking with a circular mask. The FFT generally follows Gaussian. The theory is correct.
![](./src/extra/kernel_with_mask.png)
![](./src/extra/kernel_frequency_with_mask.png)