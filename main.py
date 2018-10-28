
import sys
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


#main script
def main():
    #create model
    mdl = keras.models.Sequential()

    #add hidden layers
    mdl.add(keras.layers.Dense(600, activation=tf.nn.tanh))
    mdl.add(keras.layers.Dense(350, activation=tf.nn.elu))
    mdl.add(keras.layers.Dense(350, activation=tf.nn.elu))

    #add final layer
    mdl.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))

    #complie model
    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    #train model
    mdl.fit(xtrain, ytrain, epochs=3, verbose=1)


    #evaluate 5 times, then average
    results = []
    for i in range(5):
        results.append(mdl.evaluate(xtest, ytest, verbose=1)[1])
    print(np.average(results))


#if this is the main module, run main
if __name__ == '__main__':
    main()
            
