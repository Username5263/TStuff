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
    # Loads dataset and augments them
    # RGB images rescaled to convert color scale to 0-1 for processing
    # Current directory is starting point of file search

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Default args from flow_from_directory:
    #   color_mode='rgb'
    #   shuffle=True
    #   class_mode='categorical'
    train_generator = train_datagen.flow_from_directory('./data/train', target_size=(256, 256), batch_size=32, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory('./data/validation', target_size=(256, 256), batch_size=32, class_mode='binary')
    test_generator = test_datagen.flow_from_directory('./data/test', target_size=(256, 256), batch_size=1, class_mode=None, shuffle=False)

    return train_generator, validation_generator, test_generator

def get_model(num_classes):
    # Constructs the CNN model for training
    # Model args with default values:
    #   padding='valid'
    #   use_bias=True
    #   bias_initializer='zeros'
    # Batch normalization applied before activation layers

    model = Sequential()

    model.add(Conv2D(96, kernel_size=(7, 7), strides=(2, 2), input_shape=(256, 256, 3),
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), bias_initializer='ones',
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='ones',
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='ones',
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(4096, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def train_cnn(num_classes, epochs):
    # Feeds the data to the model for training
    # Some model args:
    #   verbose: detailed information of the training is displayed in the console if 1
    #   callbacks: variable tracking during training; takes a list as an argument

    train_generator, validation_generator, test_generator = get_data()
    model = get_model(num_classes)

    # Model compilation
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    lr_metric = get_lr_metric(sgd)
    hist_metric = model.compile(loss=losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy', lr_metric])

    if not os.path.isdir('./training_log'):
        os.mkdir('./training_log')

    callbacks = [CSVLogger('./training_log/history.csv'), ModelCheckpoint('./training_log/best_epoch_{}.hdf5'.format(str(epochs).zfill(4)), 'val_acc', verbose=1, save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=0.00001)]

    print('Training initialized. Epoch: 0')
    model.fit_generator(train_generator, samples_per_epoch=int(len(train_generator)/batch_size), epochs=epochs, verbose=1,
                        validation_data=validation_generator, validation_steps=int(len(validation_generator)/batch_size), callbacks=callbacks)
    print('Training terminated. Epoch:', epochs)

    # Saving till last epoch performed
    model.save('./training_log/last_epoch_{}.hdf5'.format(str(epochs).zfill(4)))

    # Loop and conditional statement to give a choice whether to continue training or not
    while True:
        train_more = input('Continue training (Y/N)? ')
        if train_more.lower() == 'y':
            try:
                epoch_more = int(input('Specify additional epochs: '))
            except ValueError:
                print('Not a number!/n')

            print('Training initialized. Epoch:', epochs)
            model.fit_generator(train_generator, samples_per_epoch=int(len(train_generator)/batch_size), epochs=epochs+epoch_more,
                                verbose=1, validation_data=validation_generator, validation_steps=int(len(validation_generator)/batch_size),
                                callbacks=callbacks, initial_epoch=epochs)
            print('Training terminated. Epoch:', epochs+epoch_more)

            if input('Save model (Y/N)?').lower() == 'y':
                model.save('./training_log/last_epoch_{}.hdf5'.format(str(epochs).zfill(4)))
                print('Model saved./n')
            return model, test_generator, hist_metric, epochs
        elif train_more.lower() == 'n':
            return model, test_generator, hist_metric, epochs
        else:
            print('Invalid input./n')

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

        filter_index = random.randint(0, len(conv1_output[0][0][0]))

        plt.imshow(filter1[:,:,filter_index])
        plt.savefig('./convolutional_activations/conv1_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter2[:,:,filter_index])
        plt.savefig('./convolutional_activations/conv2_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter3[:,:,filter_index])
        plt.savefig('./convolutional_activations/conv3_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter4[:,:,filter_index])
        plt.savefig('./convolutional_activations/conv4_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        plt.imshow(filter5[:,:,filter_index])
        plt.savefig('./convolutional_activations/conv5_{}.jpg'.format(str(image_index).zfill(4)))
        plt.close

        if image_index == 30:
            break

def export_model():
    # Exporting Keras model to a Tensorflow model

    if not os.path.isdir('./model_out'):
        os.mkdir('./model_out')

    # Writes a graph proto
    tf.train.write_graph(K.get_session().graph_def, logdir='./model_out', name='graph.pbtxt')

    # Saves variables
    tf.train.Saver().save(K.get_session(), save_path='./model_out/checkpoint.chkp')

    # Converts all variables in a graph and checkpoint into constants
    # input_graph - GraphDef file to load
    # input_binary=False indicates .pbtxt
    # input_checkpoint - result of tf.train.Saver().save()
    # output_graph - where to write frozen GraphDef
    # clear_devices - a bool whether to remove device specifications
    freeze_graph.freeze_graph(input_graph='./model_out/graph.pbtxt', input_binary=False, input_checkpoint='./model_out/checkpoint.chkp',
                              output_node_names='dense_3/Softmax', output_graph='./model_out/frozen_graph.pb', clear_devices=True)

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('./model_out/frozen_graph.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    # Returns an optimized version of the input graph
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names=['conv2d_1_input'], output_node_names=['dense_3/Softmax'],
                                                                         placeholder_type_enum=tf.float32.as_datatype_enum)
    # SerializeToString() - serializes message and returns it as string
    with tf.gfile.FastGFile('./model_out/string_graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph saved!")

if __name__ == '__main__':
    num_classes = 2
    epochs = 100

    model, test_generator, hist_metric, new_epochs = train_cnn(num_classes, epochs)
    test_images, test_labels = next(test_generator)

    # Plotting training accuracy with validation
    plt.figure(figsize=(10, 6))
    plt.axis((-10,new_epochs,0.8,0.99))
    plt.plot(hist_metric.history['acc'])
    plt.plot(hist_metric.history['val_acc'])
    plt.title('Training Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='lower right')

    # Plotting training loss with validation
    plt.figure(figsize=(10, 6))
    plt.axis((-10,new_epochs,0,0.09))
    plt.plot(hist_metric.history['loss'])
    plt.plot(hist_metric.history['val_loss'])
    plt.title('Training Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()

    test_generator.reset()
    predictions = model.predict_generator(test_generator)
    predictions = np.argmax(predictions, axis=1)

    score = model.evaluate_generator(test_generator)
    export_model()
    get_activations(model, test_images)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.summary()
