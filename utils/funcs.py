import io
import os
import numpy as np
import pandas as pd
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# GLOBAL SETTINGS
os.environ["KERAS_BACKEND"] = "tensorflow"
DATASET_URL = r"..\urlset.csv"
# REPRODUCIBILITY
np.random.seed(17)
# The dimension of our random noise vector.
random_dim = 100


def scale_data(df):
	# get data
	data = df.values[:,1:]
	x_train, x_test, y_train, y_test= train_test_split(data[:, :12], data[:, 12])

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_xtrain = scaler.fit_transform(x_train)
	return scaled_xtrain, scaler


def split_cyber_data(gan, disc):

	gan_xtrain, gan_xtest, gan_ytrain, gan_ytest = train_test_split(
		gan.values[:, 1:12], gan.values[:, 13], test_size=.33)

	disc_xtrain, disc_xtest, disc_ytrain, disc_ytest = train_test_split(
		disc.values[:, 1:12], disc.values[:, 13], test_size=.33)

	disc = [disc_xtrain, disc_xtest, disc_ytrain, disc_ytest]
	gan = [gan_xtrain, gan_xtest, gan_ytrain, gan_ytest]

	return gan, disc


def load_cyber_data(url=DATASET_URL):
	"""This function will take file url path
	and return split values for use in training and testing.
	Note:
		this will return the data for the gan and disc separately as
		[ (gan_x, gan_y, gan_xtest, gan_ytest), (disc_x, disc_y, disc_xtest, disc_ytest)]
	"""
	df = pd.read_csv(
		io.StringIO(
			open("..\\urlset.csv", errors="ignore").read()),
		error_bad_lines=False,
		escapechar='\\'
	)
	# split data 50-50 for gan and disc
	disc = df.sample(frac=0.5)
	gan = df[~df.index.isin(disc.index)]

	gan, disc = split_cyber_data(gan, disc)

	return gan, disc


# You will use the Adam optimizer
def get_optimizer():
	return Adam(lr=0.0002, beta_1=0.5)


def create_gen_model():
	"""This will create the generator model which will use tanh activation for output
	Notes:
		https://stackoverflow.com/questions/41489907/generative-adversarial-networks-tanh
	Args:
		input_dim (tuple): shape representing inputs to feed
		opt (keras.optimizers): optimizer to use (adam by default)
	Returns:
		generator (keras.SequentialModel): generator model
	"""
	optimizer = get_optimizer()
	generator = Sequential()
	generator.add(Dense(128, input_dim=100))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(256))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(512))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(12, activation='tanh'))

	generator.compile(loss='binary_crossentropy', optimizer=optimizer)
	return generator


def create_disc_model():
	"""This will create a discriminator model for our GAN network using Keras
	Args:
		Optimizer (keras.optimizers): Adam
	Returns:
		discriminator (keras.SequentialModel): discriminator model
	"""
	optimizer = get_optimizer()
	discriminator = Sequential()
	# first layer
	discriminator.add(Dense(64, input_dim=12))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))
	# second layer
	discriminator.add(Dense(128))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))
	# third layer
	discriminator.add(Dense(64))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))

	discriminator.add(Dense(1, activation='sigmoid'))
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
	return discriminator


def create_gan_network(discriminator, random_dim, generator, optimizer):
	# We initially set trainable to False since we only want to train either the
	# generator or discriminator at a time
	discriminator.trainable = False
	# gan input (noise) will be 100-dimensional vectors
	gan_input = Input(shape=(random_dim,))
	# the output of the generator (an image)
	x = generator(gan_input)

	# get the output of the discriminator (probability if the image is real or not)
	gan_output = discriminator(generator(gan_input))
	gan = Model(inputs=gan_input, outputs=gan_output)
	gan.compile(loss='binary_crossentropy', optimizer=optimizer)
	return gan
