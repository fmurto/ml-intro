from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input)
from keras.regularizers import l2

class VGGNet:
	# shorthand for adding convolutional layers
	def _conv(self, filters):
		return Conv2D(filters, 3, padding='same', activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
		
	def __init__(self, classes, filters, input_shape, activation='elu', dropout_rate=0.2, initializer='he_normal', regularizer=None):
		self.classes = classes # number of data classes
		self.filters = filters # number of starting filters
		self.input_shape = input_shape # data shape
		self.activation = activation # activation function
		self.dropout_rate = dropout_rate # post-last BN dropout rate (values below 0.5 work best)
		self.initializer = initializer # initialization method
		self.regularizer = regularizer # regularization method
		
		# 8+1 layer mini vgg w/ sequential model building
		# convolutional layer filters increase by powers of two
		# output size 16x16
		self.model = Sequential()
		self.model.add(Input(input_shape))
		self.model.add(self._conv(self.filters))
		self.model.add(self._conv(self.filters))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(strides=2))

		# output size 8x8
		self.model.add(self._conv(self.filters*2))
		self.model.add(self._conv(self.filters*2))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(strides=2))

		# output size 4x4
		self.model.add(self._conv(self.filters*4))
		self.model.add(self._conv(self.filters*4))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(strides=2))

		# output size 2x2
		self.model.add(self._conv(self.filters*8))
		self.model.add(self._conv(self.filters*8))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(strides=2))

		# flatten & fully connected layer
		self.model.add(Flatten())
		self.model.add(Dense(self.filters*16, activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
		
		# post-last BN dropout improves performance
		self.model.add(BatchNormalization())
		self.model.add(Dropout(rate=self.dropout_rate))
		
		# output layer
		self.model.add(Dense(self.classes, activation='softmax', kernel_initializer=self.initializer))
	
	def get_model(self):
		return self.model

class WideResNet:
	# shorthand for adding convolutional layers
	def _conv(self, filters, size, strides):
		return Conv2D(filters, size, strides=strides, padding='same', use_bias=False, activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
	
	# basic & basic-wide block (widening factor k)
	def _basic(self, input, k):
		# two [3x3, filters*k] blocks
		x = BatchNormalization()(input)
		x = self._conv(self.filters * k * self.width, 3, 1)(x)
		x = BatchNormalization()(x)
		x = self._conv(self.filters * k * self.width, 3, 1)(x)
		x = Add()([input, x])
		
		return x

	# bottleneck block (widening factor k)
	def _bottleneck(self, input, k, strides):
		# two [3x3, 16*k] blocks & one [1x1, filter*k]
		x = self._conv(self.filters * k * self.width, 3, strides)(input)
		x = BatchNormalization()(x)
		x = self._conv(self.filters * k * self.width, 3, 1)(x)
		shortcut = self._conv(self.filters * k * self.width, 1, strides)(input)
		x = Add()([x, shortcut])
		
		return x
		
	def __init__(self, width, depth, classes, filters, input_shape, activation='elu', dropout_rate=0.2, initializer='he_normal', regularizer=None):
		self.width = width # network width
		self.depth = depth # network depth 6n + 4
		self.classes = classes # number of data classes
		self.filters = filters # number of starting filters
		self.input_shape = input_shape # data shape
		self.activation = activation # activation function
		self.dropout_rate = dropout_rate # last layer dropout rate (values below 0.5 work best)
		self.initializer = initializer # initialization method
		self.regularizer = regularizer # regularization method
		
		# width-depth wideresnet w/ functional model building
		# conv block 1, output size 32x32
		input = Input(shape=self.input_shape)
		x = self._conv(self.filters, 3, 1)(input)
		x = BatchNormalization()(x)
		x = self._bottleneck(x, 1, 1) # k=1
		
		# conv block 2, output size 32x32
		for i in range(self.depth - 1):
			x = self._basic(x, 1)
		x = BatchNormalization()(x)
		x = self._bottleneck(x, 2, 2)
		
		# conv block 3, output size 16x16
		for i in range(self.depth - 1):
			x = self._basic(x, 2)
		x = BatchNormalization()(x)
		x = self._bottleneck(x, 3, 2)
		
		# conv block 4, output size 8x8
		for i in range(self.depth - 1):
			x = self._basic(x, 3)
		
		# post-last BN dropout outperforms dropout blocks & no dropout
		x = BatchNormalization()(x)
		x = Dropout(self.dropout_rate)(x)
		
		# average pooling outperforms max pooling, output size 1x1
		x = AveragePooling2D(pool_size=8, strides=1, padding="same")(x)
		
		# flatten & output layer
		x = Flatten()(x)
		x = Dense(self.classes, activation='softmax', kernel_regularizer=self.regularizer)(x)
		
		self.model = Model(input, x)
		
	def get_model(self):
		return self.model
