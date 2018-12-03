from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# Simple feed forward NN
def def_model_simple_ff():
    model = Sequential()
    model.add(Dense(784, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model


# Deep feed forward NN
def def_model_deep_ff():
    model = Sequential()
    model.add(Dense(784, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model

# Convolutional?
# def def_model_cnn():


