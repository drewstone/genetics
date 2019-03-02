import numpy as np

def parse_data_for_animal(animal):
	positives, negatives = [], []
	with open("./Enhancers_vs_negative/{}/positive_samples".format(animal.lower().capitalize())) as f:
		positives = list(map(lambda line: line.replace('\n', ''), f.readlines()))

	with open("./Enhancers_vs_negative/{}/negative_samples".format(animal.lower().capitalize())) as f:
		negatives = list(map(lambda line: line.replace('\n', ''), f.readlines()))

	return positives, negatives

def one_hot_encode_dataset(positives, negatives):
	train_X, train_Y = [], []
	for sequence in positives:
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

		train_X.append(matrix)
		train_Y.append(np.array([1,0]))

	for sequence in positives:
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

		train_X.append(matrix)
		train_Y.append(np.array([0,1]))

	# top = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))))
	# bottom = np.concatenate((np.zeros(len(positives)), np.ones(len(negatives))))
	# train_Y = np.array([top, bottom])

	return np.array(train_X), np.array(train_Y)

def split_data_for_experiment(positives, negatives, train, test, val):
	if train + test + val < 1: raise ValueError("Invalid dataset split distribution")
	test_breakpoint_pos = [int(len(positives)*train), int(len(positives)*train + len(positives)*test)]
	test_breakpoint_neg = [int(len(negatives)*train), int(len(negatives)*train + len(negatives)*test)]
	train_pos, train_neg = positives[:int(train*len(positives))], negatives[:int(train*len(negatives))]
	test_pos, test_neg = positives[test_breakpoint_pos[0]:test_breakpoint_pos[1]], negatives[test_breakpoint_neg[0]:test_breakpoint_neg[1]]
	val_pos, val_neg = positives[test_breakpoint_pos[1]:], negatives[test_breakpoint_neg[1]:]

	train_X = np.concatenate((train_pos, train_neg), axis=0)
	train_Y = np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg))), axis=0)

	test_X = np.concatenate((test_pos, test_neg), axis=0)
	test_Y = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))), axis=0)

	val_X = np.concatenate((val_pos, val_neg), axis=0)
	val_Y = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))), axis=0)

	return [(train_X, train_Y), (test_X, test_Y), (val_X, val_Y)]
