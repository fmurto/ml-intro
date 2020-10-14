import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

# Mish activation function
# https://arxiv.org/abs/1908.08681
class Mish(Activation):
	def __init__(self, activation, **kwargs):
		super(Mish, self).__init__(activation, **kwargs)
		self.__name__ = 'Mish'


def mish(inputs):
	return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})