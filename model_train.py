from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.utils import class_weight
import numpy as np
import math
# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
def define_model(length, vocab_size, num_classes=None):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(num_classes, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

def run_training():
	# load training dataset
	trainLines, trainLabels = load_dataset('processed_data/train.pkl')

	# create tokenizer
	tokenizer = create_tokenizer(trainLines)

	# calculate max document length
	length = max_length(trainLines)

	# calculate vocabulary size
	vocab_size = len(tokenizer.word_index) + 1
	print('Max document length: %d' % length)
	print('Vocabulary size: %d' % vocab_size)

	# encode data
	trainX = encode_text(tokenizer, trainLines, length)
	print(trainX.shape)

	# define model
	print('Number of Y classes: %d' % trainLabels.shape[1])
	model = define_model(length, vocab_size, trainLabels.shape[1])

	# Set class weights
	labels_dict = {0: 131, 1: 3226, 2: 62, 3: 186}

	labels_dict = {0: 6.92548077 ,1: 0.27916667, 2: 14.69897959, 3: 4.86655405}
	#labels_dict = {0: 96.36615811, 1: 10.51317614, 2: 98.28016644, 3: 94.84049931}
	# Fit the model
	model.fit([trainX,trainX,trainX], trainLabels, epochs=7, batch_size=16, class_weight=labels_dict)
	# save the model
	model.save('models/multichannel_cnn.h5') 
	# RUN: python model_train.py

# REFS:
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras