
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

#test a combination of layers, nodes per layer and num of epochs
def test_config(layer_n, node_n, epoch_n):
    #create model
    mdl = keras.models.Sequential()
    #add hidden layers
    for l in range(layer_n):
        mdl.add(keras.layers.Dense(node_n, activation=tf.nn.relu))
    #add final layers
    mdl.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    mdl.fit(xtrain, ytrain, epochs=epoch_n, verbose=0)

    return mdl.evaluate(xtest, ytest, verbose=0)[1]

#sort 2d list of values
def sort_by_acc(unsorted):
    return list(reversed(sorted(unsorted, key=lambda x: (x[3]))))

#main script
def main():
    #the outcomes of all the tests
    config_outcomes = []
    
    #test meny, meny configs
    for l in range(1, 8):
        for n in range(150, 351, 50):
            print('{}/8 - {}/351'.format(l, n))
            for e in range(3, 9):
                config_outcomes.append([])
                config_outcomes[-1].append(l)
                config_outcomes[-1].append(n)
                config_outcomes[-1].append(e)
                try:
                    config_outcomes[-1].append(test_config(l, n, e))
                except Exception as e:
                    print(e.message, e.args)
                    #config_outcomes[-1].append(-1)

    try:
        #sort the outcomes by accuracy for inspection by user
        sorted_config_outcomes = sort_by_acc(config_outcomes)
        
        #print best
        print()
        print(sorted_config_outcomes[0], end='\n\n')
        #print the next ten
        for i in sorted_config_outcomes[1:11]:
            print(i)

        #pickle dump sorted config outcomes
        pickle.dump(
                sorted_config_outcomes,
                open('sorted_config_outcomes.p','wb')
                )

    except Exception as e:
        #print exception
        print(e.message, e.agrs)
        #print custom err msg
        print('sorting failed, still outputing pickle file')
        #pickle dumb unsorted config outcomes
        pickle.dump(config_outcomes, open('config_outcomes.p', 'wb'))
    
        
#if this is the main module, run main
if __name__ == '__main__':
    main()
            
