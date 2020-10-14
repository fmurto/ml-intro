import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from models import WideResNet
from activations import Mish
from learning_rates import OneCycleLR
from preprocessing import normalize, get_random_eraser

# load cifar-10 data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# global contrast normalization
X_train, X_test = normalize(X_train, X_test)

# one hot encoding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# data augmentation by shifiting, hflipping, cutout & random erasing (pixel-level)
generator = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=True,
	preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))

# model parameters (wideresnet-10-28)
width = 10
depth = 4 # 6n + 4
classes = 10
filters = 16
input_shape = (32, 32, 3)
activation = 'Mish'
dropout_rate = 0.2
initializer = 'he_normal'
weight_decay = 5e-4
regularizer = l2(weight_decay)

# training parameters
epochs = 200
batch_size = 32
learning_rate = 0.01
max_learning_rate = 0.1
clr = OneCycleLR(num_samples=X_train.shape[0], batch_size=batch_size, max_lr=max_learning_rate)
chk = ModelCheckpoint(filepath='results/wrn1028', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

# fit the model
model = WideResNet(width, depth, classes, filters, input_shape, activation, dropout_rate, initializer, regularizer).get_model()
model.compile(optimizer=SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(generator.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, Y_test), callbacks=[clr, chk])