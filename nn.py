from keras.layers import Conv2D, Conv3D, Flatten, Dropout, Dense
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.models import Sequential

def build_cnn_model(train_features):
	model = Sequential()
	model.add(Conv2D(
		filters=20,
		kernel_size=(9,4),
		input_shape=(
			train_features.shape[1],
			train_features.shape[2],
			1),
		activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(3,1)))
	model.add(Conv2D(
		filters=30,
		kernel_size=(20,5),
		border_mode='same',
		activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(4,1)))
	model.add(Conv2D(
		filters=40,
		kernel_size=(30,3),
		border_mode='same',
		activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(4,1)))
	model.add(Flatten())
	model.add(Dense(90, activation='relu'))
	model.add(Dense(45, activation='relu'))
	model.add(Dropout(0.85))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', 
	              metrics=['binary_accuracy'])
	model.summary()
	return model

# sequences are 500 characters long
# There are 500 - k + 1 kmers of length k
# 	- This creates a matrix of 4 * k for each one hot encoded kmer
# 	- In aggregate we have a matrix of (500 - k + 1) * 4 * k
# 	- Sampels are 3D rather than 2D as above
# Use 3D convolution nn
def build_kmer_cnn_model(train_features):
	print(train_features.shape)
	model = Sequential()
	model.add(Conv3D(
		filters=20,
		kernel_size=(9, train_features.shape[2], train_features.shape[3]),
		input_shape=(
			train_features.shape[1],
			train_features.shape[2],
			train_features.shape[3],
			1),
		padding = 'same',
		activation = 'relu'
	))
	model.add(MaxPooling3D(pool_size=(3,2,2)))
	model.add(Conv3D(filters=30, kernel_size=(20, 5, 5), activation="relu", padding="same"))
	model.add(MaxPooling3D(pool_size=(4,3,2)))
	model.add(Conv3D(filters=40, kernel_size=(30, 4, 4), activation="relu", padding="same"))
	model.add(MaxPooling3D(pool_size=(4,1,1)))
	model.add(Flatten())
	model.add(Dense(90, activation='relu'))
	model.add(Dense(45, activation='relu'))
	model.add(Dropout(0.85))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', 
	              metrics=['binary_accuracy'])
	model.summary()
	return model

def build_attention_model(train_features):
	pass