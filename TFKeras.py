# Code based on:
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_cnn.py, retrieved on 8/23/2018
# https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009, retrieved on 8/25/2018
# https://github.com/OmarAflak/Keras-Android-XOR/blob/master/keras/index.py, retrieved on 8/30/2018

import math
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
    train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(256, 256), batch_size=32, class_mode='categorical') ######### CHANGE TO BINARY
    validation_generator = validation_datagen.flow_from_directory('dataset/validate', target_size=(256, 256), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('dataset/test', target_size=(256, 256), batch_size=1, class_mode=None, shuffle=False)

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
    hist_metric = model.compile(loss=losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy', lr_metric]) ######### CHANGE TO BINARY_CROSSENTROPY

    callbacks = [CSVLogger('training_log/history.csv'), ModelCheckpoint('training_log/best_epoch.hdf5', 'val_acc', verbose=1, save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=0.00001)]

    print('Training initialized. Epoch: 0')
    model.fit_generator(train_generator, samples_per_epoch=int(len(train_generator)/batch_size), epochs=epochs, verbose=1,
                        validation_data=validation_generator, validation_steps=int(len(validation_generator)/batch_size), callbacks=callbacks)
    print('Training terminated. Epoch:', epochs)

    # Saving till last epoch performed
    model.save('training_log/last_epoch.hdf5')

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

            if input('Save model (Y/N)?') == 'y':
                model.save('training_log/last_epoch.hdf5')
                print('Model saved./n')
            return model, test_generator, hist_metric
        elif train_more.lower() == 'n':
            return model, test_generator, hist_metric
        else:
            print('Invalid input./n')

def export_model():
    # Exporting Keras model to a Tensorflow model

    # Writes a graph proto
    tf.train.write_graph(K.get_session().graph_def, logdir='model_out', name='graph.pbtxt')

    # Saves variables
    tf.train.Saver().save(K.get_session(), save_path='model_out/checkpoint.chkp')

    # Converts all variables in a graph and checkpoint into constants
    # input_graph - GraphDef file to load
    # input_binary=False indicates .pbtxt
    # input_checkpoint - result of tf.train.Saver().save()
    # output_graph - where to write frozen GraphDef
    # clear_devices - a bool whether to remove device specifications
    freeze_graph.freeze_graph(input_graph='model_out/graph.pbtxt', input_binary=False, input_checkpoint='model_out/checkpoint.chkp',
                              output_node_names='dense_3/Softmax', output_graph='model_out/frozen_graph.pb', clear_devices=True)

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('model_out/frozen_graph.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    # Returns an optimized version of the input graph
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names=['conv2d_1_input'], output_node_names=['dense_3/Softmax'],
                                                                         placeholder_type_enum=tf.float32.as_datatype_enum)
    # SerializeToString() - serializes message and returns it as string
    with tf.gfile.FastGFile('model_out/string_graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph saved!")

if __name__ == '__main__':
    num_classes = 10
    epochs = 10

    model, test_generator, hist_metric = train_cnn(num_classes, epochs)
    model.summary()

    # Plotting training accuracy with validation
    plt.figure(figsize=(10, 6))
    plt.axis((-10,310,0.8,0.99))
    plt.plot(hist_metric.history['acc'])
    plt.plot(hist_metric.history['val_acc'])
    plt.title('Training Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='lower right')

    # Plotting training loss with validation
    plt.figure(figsize=(10, 6))
    plt.axis((-10,310,0,0.09))
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
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
