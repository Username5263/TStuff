import os
import cv2
import math
import random
import numpy as np

def file_names():
    # Returns image file names as list
    filenames_one = []
    filenames_two = []
    folder_names = os.listdir('./test')

    # For first recognized classification
    for file in os.listdir(os.path.join('./test', folder_names[0])):
        filenames_one.append(file)

    # For second recognized classification
    for file in os.listdir(os.path.join('./test', folder_names[1])):
        filenames_two.append(file)

    return filenames_one, filenames_two, folder_names

def cut_and_scale(output_size):
    # Cuts images to squares (center squares) and resizes them to (output_size, output_size) pixels
    filenames_one, filenames_two, folder_names = file_names()

    file_number = 1
    for file in filenames_one:
        image = cv2.imread('./test/' + str(folder_names[0]) + '/' + file)

        rows, columns, _ = image.shape
        min_size = min(rows, columns)
        max_size = max(rows, columns)
        new_image = np.zeros((min_size, min_size, 3), dtype='uint8')

        if min_size == rows:
            topmost = 0
            bottomost = min_size
            leftmost = int((max_size - min_size) / 2)
            rightmost = int(max_size - (max_size - min_size) / 2)
        else:
            topmost = int((max_size - min_size) / 2)
            bottomost = int(max_size - (max_size - min_size) / 2)
            leftmost = 0
            rightmost = min_size

        new_image[0:min_size+1, 0:min_size+1] = image[topmost:bottomost, leftmost:rightmost+1]

        resized_image = cv2.resize(new_image, (output_size,output_size))
        image_name = './test/' + str(folder_names[0]) + '/' + str(file_number).zfill(4) + '_out.jpg'
        file_number += 1
        cv2.imwrite(image_name, resized_image)
        os.remove('./test/' + str(folder_names[0]) + '/' + file)

    file_number = 1
    for file in filenames_two:
        image = cv2.imread('./test/' + str(folder_names[1]) + '/' + file)

        rows, columns, _ = image.shape
        min_size = min(rows, columns)
        max_size = max(rows, columns)
        new_image = np.zeros((min_size, min_size, 3), dtype='uint8')

        if min_size == rows:
            topmost = 0
            bottomost = min_size
            leftmost = int((max_size - min_size) / 2)
            rightmost = int(max_size - (max_size - min_size) / 2)
        else:
            topmost = int((max_size - min_size) / 2)
            bottomost = int(max_size - (max_size - min_size) / 2)
            leftmost = 0
            rightmost = min_size

        new_image[0:min_size+1, 0:min_size+1] = image[topmost:bottomost, leftmost:rightmost+1]

        resized_image = cv2.resize(new_image, (output_size,output_size))
        image_name = './test/' + str(folder_names[1]) + '/' + str(file_number).zfill(4) + '_out.jpg'
        file_number += 1
        cv2.imwrite(image_name, resized_image)
        os.remove('./test/' + str(folder_names[1]) + '/' + file)

if __name__ == '__main__':
    cut_and_scale(128)
    print('Image resized to (128,128).')
