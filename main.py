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
	print(train_X.shape, train_Y.shape)

	model = cnn.build_model(train_X)
	history = model.fit(train_X, train_Y, epochs=5, validation_split=0.25)