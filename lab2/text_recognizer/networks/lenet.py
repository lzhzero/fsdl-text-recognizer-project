from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    model = Sequential()
    model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
    input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape))
    model.add(MaxPooling2D(poolsize(2,2)))
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(poolsize(2,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes))
    ##### Your code above (Lab 2)

    return model

