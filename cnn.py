from keras.layers import Conv1D, Conv2D, Dense, MaxPooling1D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential

def build_model(train_features):
	model = Sequential()
	model.add(Conv2D(filters=20, kernel_size=(9,4), input_shape=(500,4,1), activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(3,1)))
	model.add(Conv2D(filters=30, kernel_size=(20,5), border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(4,1)))
	model.add(Conv2D(filters=40, kernel_size=(30,3), border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(4,1)))
	model.add(Flatten())
	model.add(Dense(90, activation='relu'))
	model.add(Dense(45, activation='relu'))
	model.add(Dropout(0.85))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', 
	              metrics=['binary_accuracy'])
	model.summary()
	return model;