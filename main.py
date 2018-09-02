
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

#reshape and display image with pyplot
def show_flat_img(img, title=None):
    if title is not None:
        plt.title(title)
    plt.imshow(img.reshape((28,28)))
    plt.show()

def display_mistakes(mdl):
    prediction = mdl.predict(xtest)
    for i in range(len(xtest)):
        if np.argmax(prediction[i]) != ytest[i]:
            try:
                show_flat_img(xtest[i], ytest[i])
            except KeyboardInterrupt:
                print('..Closing pyplot..')
                plt.close()
                break

#main script
def main():

    if sys.argv[1] == 'load':
        try:
            mdl = keras.models.load_model('latest_model.model')
        except Exception as e:
            print(e.message, e.args)
            print('Loading model failed, try running without "load" argument')
            quit()
    else:
        #create model
        mdl = keras.models.Sequential()

        #add hidden layers
        mdl.add(keras.layers.Dense(600, activation=tf.nn.tanh))
        mdl.add(keras.layers.Dense(350, activation=tf.nn.elu))
        mdl.add(keras.layers.Dense(350, activation=tf.nn.elu))

        #add final layer
        mdl.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))

        #TODO try other optimizer / loss
        #complie model
        mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        mdl.fit(xtrain, ytrain, epochs=8, verbose=1)

        mdl.save('latest_model.model')

    #evaluate 15 times, then average
    results = []
    for i in range(15):
        results.append(mdl.evaluate(xtest, ytest, verbose=1)[1])
    print(np.average(results))

    display_mistakes(mdl)


#if this is the main module, run main
if __name__ == '__main__':
    main()
            
