import os
import random

def file_names():
    # Returns image file names for each class as lists
    filenames_one = []
    filenames_two = []
    folder_names = os.listdir('./data')

    # For first recognized classification
    for file in os.listdir(os.path.join('./data', folder_names[0])):
        filenames_one.append(file)

    # For second recognized classification
    for file in os.listdir(os.path.join('./data', folder_names[1])):
        filenames_two.append(file)

    return filenames_one, filenames_two, folder_names

def divide_data():
    # Divides the images to train, validation, and test folders
    filenames_one, filenames_two, folder_names = file_names()

    # train_one = int(len(filenames_one) * 0.8)
    # validation_one = int(len(filenames_one) * 0.1)
    # test_one = len(filenames_one) - train_one - validation_one

    # train_two = int(len(filenames_two) * 0.8)
    # validation_two = int(len(filenames_two) * 0.1)
    # test_two = len(filenames_two) - train_two - validation_two

    train_one = int(len(filenames_one) * 0.85)
    validation_one = len(filenames_one) - train_one

    train_two = int(len(filenames_two) * 0.85)
    validation_two = len(filenames_two) - train_two

    if not os.path.isdir('./data/train'):
        os.mkdir('./data/train')

    if not os.path.isdir('./data/validation'):
        os.mkdir('./data/validation')

    # if not os.path.isdir('./data/test'):
    #     os.mkdir('./data/test')

    for i in range(train_one):
        file_index = random.randint(0, len(filenames_one)-1)
        file_name = filenames_one[file_index]
        if not os.path.isdir('./data/train/'+folder_names[0]):
            os.mkdir('./data/train/'+folder_names[0])
        os.rename('./data/'+folder_names[0]+'/'+file_name, './data/train/'+folder_names[0]+'/'+file_name)
        del filenames_one[file_index]

    for i in range(validation_one):
        file_index = random.randint(0, len(filenames_one)-1)
        file_name = filenames_one[file_index]
        if not os.path.isdir('./data/validation/'+folder_names[0]):
            os.mkdir('./data/validation/'+folder_names[0])
        os.rename('./data/'+folder_names[0]+'/'+file_name, './data/validation/'+folder_names[0]+'/'+file_name)
        del filenames_one[file_index]

    # for i in range(test_one):
    #     file_index = random.randint(0, len(filenames_one)-1)
    #     file_name = filenames_one[file_index]
    #     if not os.path.isdir('./data/test/'+folder_names[0]):
    #         os.mkdir('./data/test/'+folder_names[0])
    #     os.rename('./data/'+folder_names[0]+'/'+file_name, './data/test/'+folder_names[0]+'/'+file_name)
    #     del filenames_one[file_index]

    for i in range(train_two):
        file_index = random.randint(0, len(filenames_two)-1)
        file_name = filenames_two[file_index]
        if not os.path.isdir('./data/train/'+folder_names[1]):
            os.mkdir('./data/train/'+folder_names[1])
        os.rename('./data/'+folder_names[1]+'/'+file_name, './data/train/'+folder_names[1]+'/'+file_name)
        del filenames_two[file_index]

    for i in range(validation_two):
        file_index = random.randint(0, len(filenames_two)-1)
        file_name = filenames_two[file_index]
        if not os.path.isdir('./data/validation/'+folder_names[1]):
            os.mkdir('./data/validation/'+folder_names[1])
        os.rename('./data/'+folder_names[1]+'/'+file_name, './data/validation/'+folder_names[1]+'/'+file_name)
        del filenames_two[file_index]

    # for i in range(test_two):
    #     file_index = random.randint(0, len(filenames_two)-1)
    #     file_name = filenames_two[file_index]
    #     if not os.path.isdir('./data/test/'+folder_names[1]):
    #         os.mkdir('./data/test/'+folder_names[1])
    #     os.rename('./data/'+folder_names[1]+'/'+file_name, './data/test/'+folder_names[1]+'/'+file_name)
    #     del filenames_two[file_index]

    os.rmdir('./data/'+folder_names[0])
    os.rmdir('./data/'+folder_names[1])

divide_data()
