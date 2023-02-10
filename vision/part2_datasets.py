#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args:
        path: string specifying the directory containing images
    Returns:
        images_a: list of strings specifying the paths to the images in set A,
           in lexicographically-sorted order
        images_b: list of strings specifying the paths to the images in set B,
           in lexicographically-sorted order
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    images_a = [os.path.join(path,file_name) for file_name in os.listdir(path) if (file_name.__contains__('.') and file_name[1] == 'a')]
    images_b = [os.path.join(path,file_name) for file_name in os.listdir(path) if (file_name.__contains__('.') and file_name[1] == 'b')]
    images_a.sort()
    images_b.sort()
    assert len(images_a) == len(images_b)
    return images_a, images_b
    raise NotImplementedError(
        "`make_dataset` function in `part2_datasets.py` needs to be implemented"
    )
    ### END OF STUDENT CODE ####
    ############################



def get_cutoff_frequencies(path: str) -> List[int]:
    """
    Gets the cutoff frequencies corresponding to each pair of images.

    The cutoff frequencies are the values you discovered from experimenting in
    part 1.

    Args:
        path: string specifying the path to the .txt file with cutoff frequency
        values
    Returns:
        cutoff_frequencies: numpy array of ints. The array should have the same
            length as the number of image pairs in the dataset
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    f = open(path, "r")
    cutoff_frequencies = np.array(f.read().split(), int)
    return cutoff_frequencies
    raise NotImplementedError(
        "`get_cutoff_frequencies` function in "
        + "`part2_datasets.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################



class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You
        can specify additional transforms (e.g. image resizing) if you want to,
        but it's not necessary for the images we provide you since each pair has
        the same dimensions.

        Args:
            image_dir: string specifying the directory containing images
            cf_file: string specifying the path to the .txt file with cutoff
            frequency values
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)

        ############################
        ### TODO: YOUR CODE HERE ###

        # raise NotImplementedError(
        #     "`self.transform` function in `part2_datasets.py` needs to be implemented"
        # )
        self.transform = torchvision.transforms.ToTensor()

        ### END OF STUDENT CODE ####
        ############################

        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        ############################
        ### TODO: YOUR CODE HERE ###
        return len(self.images_a)
        raise NotImplementedError(
            "`__len__` function in `part2_datasets.py` needs to be implemented"
        )

        ### END OF STUDENT CODE ####
        ############################

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff frequency value at
        index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0
        and 1. Make sure you transpose the dimensions so that image_a and
        image_b are of shape (c, m, n) instead of the typical (m, n, c), and
        convert them to torch Tensors.

        Args:
            idx: int specifying the index at which data should be retrieved
        Returns:
            image_a: Tensor of shape (c, m, n)
            image_b: Tensor of shape (c, m, n)
            cutoff_frequency: int specifying the cutoff frequency corresponding
               to (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        ############################
        ### TODO: YOUR CODE HERE ###
        image_dir_a = self.images_a[idx]
        image_dir_b = self.images_b[idx]
        pil_image_a = PIL.Image.open(image_dir_a)
        pil_image_b = PIL.Image.open(image_dir_b)
        image_a = self.transform(pil_image_a)
        image_b = self.transform(pil_image_b)

        cutoff_frequency = self.cutoff_frequencies[idx]
        return image_a, image_b, cutoff_frequency
        raise NotImplementedError(
            "`__getitem__ function in `part2_datasets.py` needs to be implemented"
        )

        ### END OF STUDENT CODE ####
        ############################

