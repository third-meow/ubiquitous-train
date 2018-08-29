
from mnist_reader import load_mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

DATASET_PATH = '/home/third-meow/datasets/fashion_mnist/'

#load training images and labels
xtrain, ytrain = load_mnist(DATASET_PATH, 'train')
#load testing images and labels (t10k because t=test 10k=10000 images/labels)
xtest, ytest = load_mnist(DATASET_PATH, 't10k')

#normalize data
xtrain = keras.utils.normalize(xtrain, axis=1)
xtest = keras.utils.normalize(xtest, axis=1)

#reshape and display image with pyplot
def show_flat_img(img):
    plt.imshow(img.reshape((28,28)))
    plt.show()

#main script
def main():
    #create model
    mdl = keras.models.Sequential()

    #add hidden layers
    mdl.add(keras.layers.Dense(300, activation=tf.nn.relu))
    mdl.add(keras.layers.Dense(300, activation=tf.nn.relu))
    mdl.add(keras.layers.Dense(300, activation=tf.nn.relu))

    #add final layer
    mdl.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    mdl.fit(xtrain, ytrain, epochs=8, verbose=1)

    print(mdl.evaluate(xtest, ytest, verbose=1))

#if this is the main module, run main
if __name__ == '__main__':
    main()
            
