import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import runner


if __name__ == '__main__':
	np.random.seed(101)
	# runner.run_for_dataset()
	runner.read_data_into_mem()
	# load_all()