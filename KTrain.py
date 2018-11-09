# Code based on:
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_cnn.py, retrieved on 8/23/2018
# https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009, retrieved on 8/25/2018
# https://github.com/OmarAflak/Keras-Android-XOR/blob/master/keras/index.py, retrieved on 8/30/2018
# https://keras.io/

import os
import math
import random
import datetime
import numpy as np

import keras
import matplotlib.pyplot as plt
from keras import initializers, regularizers, optimizers, losses
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Activation, Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

def get_data():
    # Loads dataset and augments them
    # RGB images rescaled to convert color scale to 0-1 for processing
    # Current directory is starting point of file search

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=40, zoom_range=0.2)
    validation_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=40, zoom_range=0.2)

    # Default args from flow_from_directory:
    #   color_mode='rgb'
    #   shuffle=True
    #   class_mode='categorical'
    train_augmented = './results/'+date+'/augmented/train'
    validation_augmented = './results/'+date+'/augmented/validation'
    os.mkdir('./results/'+date+'/augmented')
    os.mkdir(train_augmented)
    os.mkdir(validation_augmented)
    train_generator = train_datagen.flow_from_directory('./data/train', target_size=(128, 128), batch_size=32, save_to_dir=train_augmented, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory('./data/validation', target_size=(128, 128), batch_size=32, save_to_dir=validation_augmented, class_mode='categorical')

    return train_generator, validation_generator

def get_model(num_classes):
    # Constructs the CNN model for training
    # Model args with default values:
    #   padding='valid'
    #   use_bias=True
    #   bias_initializer='zeros'
    # Batch normalization applied before activation layers

    model = Sequential()

    model.add(Conv2D(96, kernel_size=(7, 7), strides=(2, 2), input_shape=(128, 128, 3),
                     kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
                     kernel_regularizer=regularizers.l2(0.0005), name='input_node'))
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
    model.add(Dense(2048, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, bias_initializer='ones', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model

def train_cnn(num_classes, epochs, date):
    # Feeds the data to the model for training
    # Some model args:
    #   verbose: detailed information of the training is displayed in the console if 1
    #   callbacks: variable tracking during training; takes a list as an argument
    if not os.path.isdir('./results/'):
        os.mkdir('./results/')

    os.mkdir('./results/'+date+'/')
    os.mkdir('./results/'+date+'/training_log/')

    train_generator, validation_generator = get_data()
    model = get_model(num_classes)

    model.summary()
    # print(model.output.op.name)

    # Model compilation
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss=losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    callbacks = [CSVLogger('./results/'+date+'/training_log/history.csv', append=True),
                 ModelCheckpoint('./results/'+date+'/training_log/best_epoch.hdf5', 'val_acc', verbose=1, save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.00001)]

    print('Training initialized. Epoch: 0')
    # model.fit_generator(train_generator, samples_per_epoch=int(len(train_generator)/batch_size), epochs=epochs, verbose=1, validation_data=validation_generator, validation_steps=int(len(validation_generator)/batch_size), callbacks=callbacks)
    hist_metric = model.fit_generator(train_generator, epochs=epochs, verbose=1, validation_data=validation_generator, callbacks=callbacks)
    print('Training terminated. Epoch:', epochs)

    # Saving till last epoch performed
    model.save('./results/'+date+'/training_log/last_epoch_{}.hdf5'.format(str(epochs).zfill(4)))
    print('Model saved.')
    return model, hist_metric

def acc_loss_graph(hist_metric, epochs):
    # Plotting training accuracy with validation
    min_acc = min(min(hist_metric.history['acc']), min(hist_metric.history['val_acc']))
    plt.figure()
    plt.plot(hist_metric.history['acc'])
    plt.plot(hist_metric.history['val_acc'])
    plt.xticks(range(0,epochs,100))
    plt.yticks(np.arange(round(min_acc,1)-0.1,1,0.1))
    plt.title('Training and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('./results/'+date+'/model_out/accuracy.png')
    plt.close()
    print('Accuracy graph saved.')

    # Plotting training loss with validation
    min_loss = min(min(hist_metric.history['loss']), min(hist_metric.history['val_loss']))
    max_loss = max(max(hist_metric.history['loss']), max(hist_metric.history['val_loss']))
    plt.figure()
    plt.plot(hist_metric.history['loss'])
    plt.plot(hist_metric.history['val_loss'])
    plt.xticks(range(0,epochs,100))
    plt.yticks(np.arange(int(min_loss)-0.5, np.ceil(max_loss)+0.5,0.5))
    plt.title('Training and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('./results/'+date+'/model_out/loss.png')
    plt.close()
    print('Loss graph saved.')

def export_model():
    # Exporting Keras model to a Tensorflow model
    # Graph stores architecture information of the network
    # Checkpoint files contain weight values at different epochs

    # Freezing model: producing a singular file containing information
    # about the graph and checkpoint variables, but saving these
    # hyperparameters as constants within the graph structure.

    output_node_names = 'activation_8/Softmax'
    input_node_names = model.input.name

    # Writes a graph proto
    tf.train.write_graph(K.get_session().graph_def, logdir='./results/'+date+'/model_out/', name='graph.pbtxt')

    # Saves variables
    tf.train.Saver().save(K.get_session(), save_path='./results/'+date+'/model_out/checkpoint.chkp')

    # Converts all variables in a graph and checkpoint into constants
    # input_graph - GraphDef file to load
    # input_binary=False indicates .pbtxt
    # input_checkpoint - result of tf.train.Saver().save()
    # output_graph - where to write frozen GraphDef
    # clear_devices - a bool whether to remove device specifications
    freeze_graph.freeze_graph(input_graph='./results/'+date+'/model_out/graph.pbtxt', input_saver=None, input_binary=False,
                              input_checkpoint='./results/'+date+'/model_out/checkpoint.chkp', output_node_names=output_node_names,
                              restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph='./results/'+date+'/model_out/frozen_graph.pb', clear_devices=True, initializer_nodes='')

    # input_graph_def = tf.GraphDef()
    # with tf.gfile.Open('./results/'+date+'/model_out/frozen_graph.pb', "rb") as f:
    #     input_graph_def.ParseFromString(f.read())
    #
    # # Returns an optimized version of the input graph
    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names=[input_node_names], output_node_names=[output_node_names],
    #                                                                      placeholder_type_enum=tf.float32.as_datatype_enum)
    # # SerializeToString() - serializes message and returns it as string
    # with tf.gfile.FastGFile('./results/'+date+'/model_out/string_graph.pb', "wb") as f:
    #     f.write(output_graph_def.SerializeToString())

    print('Graph saved!')

if __name__ == '__main__':
    num_classes = 2
    epochs = 250

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    model, hist_metric = train_cnn(num_classes, epochs, date)

    os.mkdir('./results/'+date+'/model_out/')
    acc_loss_graph(hist_metric, epochs)

    K.clear_session()
    model = load_model('./results/'+date+'/training_log/best_epoch.hdf5')
    export_model()
