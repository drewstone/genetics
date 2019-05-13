import os
import numpy as np
from scipy.stats import moment

def read_data_into_mem(raw=True, result=True, meta=False, segments=False):
	data = {}

	if raw:
		for dirname, dirnames, filenames in os.walk('./chiron/raw'):
			for file in filenames:
				name = file.split('.')[0]
				
				if name not in data:
					data[name] = {}

				with open('./chiron/raw/{}'.format(file)) as f:
					data[name] = {
						"raw": np.array(list(map(
                            lambda v: int(v), f.readlines()[0].split(' ')
                        ))),
                        **data[name]
					}

	if result:
		for dirname, dirnames, filenames in os.walk('./chiron/result'):
			for file in filenames:
				name = file.split('.')[0]
				
				if name not in data:
					data[name] = {}

				with open('./chiron/result/{}'.format(file)) as f:
					lines = f.readlines()
					data[name] = {
						"name": lines[0],
						"sequence": one_hot_encode_sequence(lines[1]),
						# "quality": lines[3],
                        **data[name]
					}

	if meta:
		for dirname, dirnames, filenames in os.walk('./chiron/meta'):
			pass

	if segments:
		for dirname, dirnames, filenames in os.walk('./chiron/segments'):
			pass

	return data

def one_hot_encode_sequence(sequence):
	matrix = np.zeros((len(sequence), 4), dtype=int)
	for (inx, letter) in enumerate(sequence):
		if letter == "A":
			matrix[inx][0] = 1
		if letter == "G":
			matrix[inx][1] = 1
		if letter == "C":
			matrix[inx][2] = 1
		if letter == "T":
			matrix[inx][3] = 1
	return matrix

def kmer_expansion(sequence, k):
	table = []
	for i in range(len(sequence) - k + 1):
		table.append(sequence[i:k+i])
	return table

# Returns the moments of a normalized signal sequence
def get_moment_features(signal_seq, max_moment, norm_ord=1):
	# make a numpy array if not
	arr = np.array(signal_seq)
	# normalize the sequence
	normalized_arr = arr / np.linalg.norm(arr, ord=norm_ord)
	# calculate an array of moments up to "max_moment"
	return np.array(list(map(
		lambda k: moment(normalized_arr, moment=k), [ i for i in range(max_moment)]
	)))