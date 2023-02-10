import torch
import torchvision
import os

from vision.part2_datasets import HybridImageDataset
from vision.part2_models import HybridImageModel

if not os.path.exists('results/part2/'):
        os.makedirs('results/part2/')
        
data_root = 'data' # if you're using additional data, make sure to change this to '../additional_data'
cf_file = 'cutoff_frequencies.txt'
model = HybridImageModel()
dataset = HybridImageDataset(data_root, cf_file)
dataloader = torch.utils.data.DataLoader(dataset)

data_iter = iter(dataloader)
import matplotlib.pyplot as plt
for i in range(len(dataset)):
    image_a, image_b, cutoff_frequency = next(data_iter)
    low_frequencies, high_frequencies, hybrid_image = model(image_a, image_b, cutoff_frequency)
    
    # saves low frequencies, high frequencies, and hybrid image of each pair of images
    torchvision.utils.save_image(low_frequencies, 'results/part2/%d_low_frequencies.jpg' % i)
    torchvision.utils.save_image(high_frequencies+0.5, 'results/part2/%d_high_frequencies.jpg' % i)
    torchvision.utils.save_image(hybrid_image, 'results/part2/%d_hybrid_image.jpg' % i)