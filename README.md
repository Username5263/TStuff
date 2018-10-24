# TStuff
CNN Image Classification using Keras with TF backend (5 convolutional, 3 fully-connected layers)

IProcess.py: If acquired input image is rectangular, this python code gets the center square of that image and rewrites the file.

KTrain.py: Given a 'data' folder in the same directory, having divided training ('train'), validation ('validation'), and testing ('test') data, it trains the CNN model with an initial 100 epochs then saves and exports model if no further epochs are run.

KTest.py: Tests the model trained by KTrain.py. Model path must point to saved pb file.
