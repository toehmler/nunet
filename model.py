from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import initializers
import numpy as np
import keras.backend as K
import tensorflow as tf
from metrics import dice_coef, dice_coef_loss

def unet(input_size = (240,240,4)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv5),conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv6),conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv7),conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv8),conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)

    conv10 = Conv2D(5, (1, 1), activation='relu',
                    kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)
    conv10 = Activation('softmax')(conv10)
    model = Model(inputs=[inputs], outputs=[conv10])

    lr = 1e-4
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == "__main__":
#    model_name = input('Model Name: ')
    unet_model = unet()
    '''
    model_json = unet_model.to_json()
    with open('models/architectures/test_model.json','w') as json_file:
        json_file.write(model_json)
    unet_model.save_weights('models/weights/test_model_weights')
    '''
    unet_model.save('models/test')
    unet_model.save_weights('models/test_w')
