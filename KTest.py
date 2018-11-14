# Code based on:
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_cnn.py, retrieved on 8/23/2018
# https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009, retrieved on 8/25/2018
# https://github.com/OmarAflak/Keras-Android-XOR/blob/master/keras/index.py, retrieved on 8/30/2018

import os
import cv2
import numpy as np
import tensorflow as tf

import keras
import matplotlib.pylab as plt
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

def get_data():
    # Loads dataset
    # RGB images rescaled to convert color scale to 0-1 for processing
    # Current directory is starting point of file search

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Default args from flow_from_directory:
    #   color_mode='rgb'
    #   shuffle=True
    #   class_mode='categorical'
    test_generator = test_datagen.flow_from_directory('./test', target_size=(128, 128), batch_size=1, shuffle=False)

    return test_generator

def get_cnn():
	# model_path = input('Model path: ')
    model_path = './results/2018-11-12_17-08/training_log/best_epoch.hdf5'
    model = load_model(model_path)
    return model

def save_predicted_images(test_generator, predictions):
    # Saves the wrong predicted images in a folder

    folder_names = os.listdir('./test')
    # print(folder_names)
    if not os.path.isdir('./results/2018-11-12_17-08/wrong_predictions/'):
        os.mkdir('./results/2018-11-12_17-08/wrong_predictions/')

    image_number = 1
    for image_index in range(len(test_generator.classes)):
        if(test_generator.classes[image_index] != predictions[image_index]):
            file_name = './results/2018-11-12_17-08/wrong_predictions/'+str(image_number).zfill(4)+'_wrong.jpg'

            if(image_index < 153):
                source_file = './test/'+str(folder_names[1])+'/'+str(image_index+1).zfill(4)+'_out.jpg'
            else:
                source_file = './test/'+str(folder_names[0])+'/'+str(image_index+1-153).zfill(4)+'_out.jpg'

            file = cv2.imread(source_file)
            cv2.imwrite(file_name, file)
            image_number += 1

if __name__ == '__main__':
    test_generator = get_data()
    model = get_cnn()

    predictions = model.predict_generator(test_generator, verbose=1)
    predictions = np.argmax(predictions, axis=1)

    score = model.evaluate_generator(test_generator, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    save_predicted_images(test_generator, predictions)
    print(confusion_matrix(test_generator.classes, predictions))
    # model.summary()
