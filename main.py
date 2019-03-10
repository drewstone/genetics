import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import runner


if __name__ == '__main__':
	np.random.seed(101)
	# runner.run_all()
	runner.run_kmer()
	# load_all()