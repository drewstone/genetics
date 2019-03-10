import numpy as np
import khmer

def parse_data_for_animal(animal):
	positives, negatives = [], []
	with open("./Enhancers_vs_negative/{}/positive_samples".format(animal.lower().capitalize())) as f:
		positives = list(map(lambda line: line.replace('\n', ''), f.readlines()))

	with open("./Enhancers_vs_negative/{}/negative_samples".format(animal.lower().capitalize())) as f:
		negatives = list(map(lambda line: line.replace('\n', ''), f.readlines()))

	return positives, negatives

def one_hot_encode_sequence(sequence):
	matrix = np.zeros((500, 4), dtype=int)
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

def one_hot_encode_dataset(positives, negatives):
	train_X, train_Y = [], []
	for sequence in positives:
		matrix = one_hot_encode_sequence(sequence)
		train_X.append(matrix)
		train_Y.append([1,0])

	for sequence in negatives:
		matrix = one_hot_encode_sequence(sequence)
		train_X.append(matrix)
		train_Y.append([0,1])

	return np.array(train_X), np.array(train_Y)


def kmer_expansion(sequence, k):
	ksize = 21
	target_table_size = 5e8
	num_tables = 4

	counts = khmer.Counttable(ksize, target_table_size, num_tables)
	return counts.get_kmers()

def kmer_encode_dataset(positives, negatives, k):
	train_X, train_Y = [], []
	for sequence in positives:
		kmer_list = []
		for kmer in kmer_expansion(sequence, k):
			kmer_list.append(one_hot_encode_sequence(kmer))

		train_X.append(np.array(kmer_list))
		train_Y.append([1,0])

	for sequence in negatives:
		kmer_list = []
		for kmer in kmer_expansion(sequence, k):
			kmer_list.append(one_hot_encode_sequence(kmer))

		train_X.append(np.array(kmer_list))
		train_Y.append([0,1])
