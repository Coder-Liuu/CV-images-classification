from tensorflow.keras import layers,models
from tensorflow.keras.layers import *


def CNN(num_class=2):
    # 一个普通的CNN
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(32,kernel_size=(3,3))(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,kernel_size=(5,5))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32,activation='relu')(x)
    outputs = layers.Dense(num_class,activation = 'softmax')(x)
    model = models.Model(inputs = inputs,outputs = outputs)
    return model

if __name__ == "__main__":
    model = CNN()
    model.summary()
