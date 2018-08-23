
from mnist_reader import load_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

DATASET_PATH = '/home/third-meow/datasets/fashion_mnist/'


#load training images and labels
xtrain, ytrain = load_mnist(DATASET_PATH, 'train')
#load testing images and labels (t10k because t=test 10k=10000 images/labels)
xtest, ytest = load_mnist(DATASET_PATH, 't10k')

import matplotlib.pyplot as plt

plt.imshow(xtrain[0])
plt.show()
