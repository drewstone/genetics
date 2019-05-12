import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import util


if __name__ == '__main__':
	np.random.seed(101)
	data = util.read_data_into_mem()
	print(data)
