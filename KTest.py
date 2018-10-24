# Code based on:
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_cnn.py, retrieved on 8/23/2018
# https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009, retrieved on 8/25/2018
# https://github.com/OmarAflak/Keras-Android-XOR/blob/master/keras/index.py, retrieved on 8/30/2018
import os
import math
import random
import numpy as np
import tensorflow as tf

import keras
import matplotlib.pylab as plt
from keras import initializers, regularizers, optimizers, losses
from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Activation, Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib


def get_data():
    # Loads dataset
    # RGB images rescaled to convert color scale to 0-1 for processing
    # Current directory is starting point of file search

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Default args from flow_from_directory:
    #   color_mode='rgb'
    #   shuffle=True
    #   class_mode='categorical'
    test_generator = test_datagen.flow_from_directory('./data/test', target_size=(256, 256), batch_size=1, class_mode=None, shuffle=False)

    return test_generator

def load_cnn():
	model_path = input('Model path: ')
	model = load_model(model_path)
	return model

def get_layer_output(model, layer_index, x):
    # Will return the output of a certain layer given a certain input
    # Learning phase for testing is 0

    get_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_index].output])
    layer_output = get_output([x, 0])[0]
    return layer_output

def get_activations(model, test_images):
    # Displays activations of convolutional layers

    if not os.path.isdir('./convolutional_activations'):
        os.mkdir('./convolutional_activations')

    for image_index in range(len(test_images)):
        conv1_output = get_layer_output(model, 0, test_images)
        conv2_output = get_layer_output(model, 4, test_images)
        conv3_output = get_layer_output(model, 8, test_images)
        conv4_output = get_layer_output(model, 11, test_images)
        conv5_output = get_layer_output(model, 14, test_images)

        filter1 = conv1_output[image_index]
        filter2 = conv2_output[image_index]
        filter3 = conv3_output[image_index]
        filter4 = conv4_output[image_index]
        filter5 = conv5_output[image_index]

        # filter_index = random.randint(0, len(conv1_output[0][0][0]))

        plt.imshow(filter1[:,:,image_index])
        plt.savefig('./convolutional_activations/conv1_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter2[:,:,image_index])
        plt.savefig('./convolutional_activations/conv2_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter3[:,:,image_index])
        plt.savefig('./convolutional_activations/conv3_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter4[:,:,image_index])
        plt.savefig('./convolutional_activations/conv4_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter5[:,:,image_index])
        plt.savefig('./convolutional_activations/conv5_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        if image_index == 30:
            break

if __name__ == '__main__':
    model = load_cnn()
    test_images, test_labels = next(test_generator)

    test_generator.reset()
    predictions = model.predict_generator(test_generator)
    predictions = np.argmax(predictions, axis=1)

    score = model.evaluate_generator(test_generator)
    get_activations(model, test_images)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.summary()
