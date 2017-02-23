import os
import tensorflow as tf
from PIL import Image
import numpy as np


########################################################################
# Data processing
########################################################################


target_size = (1024, 512)

# Define function to read image and convert to one-hot vector


def process_image(path):
    img = Image.open(path)
    resized_image = img.resize(target_size)
    onehot_image = np.asarray(resized_image).flatten()
    return onehot_image

# Path to images containing hammerheads
hammerhead_filepath = 'hammerheads_classification/is_hammerhead'
hammerhead_images = [os.path.join(hammerhead_filepath, img_path)
                     for img_path
                     in os.listdir(hammerhead_filepath)]


# Path to images with other type of images
non_hammerhead_filepath = ['hammerheads_classification/manta_ray',
                           'hammerheads_classification/normal_shark',
                           'hammerheads_classification/sun_fish']
non_hammerhead_images = [os.path.join(folderpath, img_path)
                         for folderpath in non_hammerhead_filepath
                         for img_path in os.listdir(folderpath)]

# Read the data
hammerhead_data = np.stack([process_image(image)
                            for image in hammerhead_images])

non_hammerhead_data = np.stack([process_image(image)
                                for image in non_hammerhead_images])


# Create the data for training and labelling.
train_data = np.vstack([hammerhead_data, non_hammerhead_data])
train_label = np.repeat(np.array([[1, 0], [0, 1]]),
                        [len(hammerhead_data), len(non_hammerhead_data)],
                        axis=0)


########################################################################
# Tensorflow model specification
########################################################################
