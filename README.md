# TStuff
CNN Image Classification using Keras with TF backend (5 convolutional, 3 fully-connected layers)

IProcess.py: If acquired input image is rectangular, this python code gets the center square of that image and rewrites the file.

IDivide.py: Divides the data to train and validation sets.

KTrain.py: Given a 'data' folder in the same directory, having divided training ('train') and validation ('validation'), it trains the CNN model with 200 epochs then saves and exports model.

ITest.py: Does the same thing as IProcess.py, but for the test folder.

KTest.py: Tests the model trained by KTrain.py. Model path must point to saved hdf5 file.
