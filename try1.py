import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# train_images = sorted(glob("../input/cityscapes-500x500/cityscapes500x500/cityscapes500x500/leftImg8bit/train/*/*Img8bit.png"))
# train_labels = sorted(glob("../input/cityscapes-500x500/cityscapes500x500/cityscapes500x500/gtFine/train/*/*labelIds.png"))
# val_images = sorted(glob("../input/cityscapes-500x500/cityscapes500x500/cityscapes500x500/leftImg8bit/val/*/*Img8bit.png"))
# val_labels = sorted(glob("../input/cityscapes-500x500/cityscapes500x500/cityscapes500x500/gtFine/val/*/*labelIds.png"))
# test_images = sorted(glob("../input/cityscapes-500x500/cityscapes500x500/cityscapes500x500/leftImg8bit/test/berlin/*"))
# print(len(train_images))
# print(len(val_images))
# print(len(test_images)) 

dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
dataset_train, dataset_val
