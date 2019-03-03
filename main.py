import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
from sklearn.model_selection import train_test_split
import util
import svm
import cnn

if __name__ == '__main__':
	np.random.seed(0)

	positives, negatives = util.parse_data_for_animal("dog");
	X, Y = util.one_hot_encode_dataset(positives, negatives);
	print(X.shape, Y.shape)

	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25, random_state=42)

	train_X = train_X.reshape(train_X.shape[0], 500, 4 , 1)
	test_X = test_X.reshape(test_X.shape[0], 500, 4 , 1)

	model = cnn.build_model(train_X)
	history = model.fit(train_X, train_Y, epochs=20, validation_data=(test_X, test_Y))