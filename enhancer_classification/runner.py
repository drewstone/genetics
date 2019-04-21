import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import util
import nn

def run_kmer_for_dataset(dataset="dog", model_path="./kmer-models", ksize=21):
	positives, negatives = util.parse_data_for_animal(dataset)
	X, Y = util.kmer_encode_dataset(positives, negatives, ksize)
	print(X.shape, Y.shape)

	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state=42)

	train_X = train_X.reshape(train_X.shape[0], 500 - ksize + 1, ksize, 4, 1)
	test_X = test_X.reshape(test_X.shape[0], 500 - ksize + 1, ksize, 4, 1)

	model = nn.build_kmer_cnn_model(train_X)
	history = model.fit(train_X, train_Y, epochs=10, validation_data=(test_X, test_Y))

	save_model(model, dataset, model_path)
	return model, history

def run_all():
	for dirname, dirnames, filenames in os.walk('./Enhancers_vs_negative'):
		for subdirname in dirnames:
			if subdirname != "peaks_fasta_files":
				model, history = run_for_dataset(subdirname)


def run_for_dataset(dataset="dog", model_path="./models"):
	positives, negatives = util.parse_data_for_animal(dataset)
	X, Y = util.one_hot_encode_dataset(positives, negatives)
	print(X.shape, Y.shape)

	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state=42)

	train_X = train_X.reshape(train_X.shape[0], 500, 4 , 1)
	test_X = test_X.reshape(test_X.shape[0], 500, 4 , 1)

	model = nn.build_cnn_model(train_X)
	history = model.fit(train_X, train_Y, epochs=10, validation_data=(test_X, test_Y))
	save_model(model, dataset, model_path)
	return model, history

def save_model(model, dataset, model_path):
	# serialize model to JSON
	model_json = model.to_json()
	with open("{}/{}.json".format(model_path, dataset), "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("{}/{}.h5".format(model_path, dataset))
	print("Saved model to disk")

def load_all():
	for dirname, dirnames, filenames in os.walk('./Enhancers_vs_negative'):
		for subdirname in dirnames:
			if subdirname != "peaks_fasta_files":
				model = load_for_dataset(subdirname)


def load_for_dataset(dataset="dog", model_path="./models"):
	# load json and create model
	json_file = open('{}/{}.json'.format(model_path, dataset), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("{}/{}.h5".format(model_path, dataset))
	print("Loaded {} model from disk".format(dataset))