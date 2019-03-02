from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

def build_model(train_features):
	model = Sequential()
	model.add(Conv1D(filters=20, kernel_size=32, 
	                 input_shape=(train_features.shape[1], 4)))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', 
	              metrics=['binary_accuracy'])
	model.summary()
	return model;