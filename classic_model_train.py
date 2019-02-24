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
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression
from sklearn import svm


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

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

def load_data_classic_models():
	# load training dataset
	trainLines, trainLabelsEncoded = load_dataset('processed_data/train.pkl')
	testLines, testLabelsEncoded = load_dataset('processed_data/test.pkl')
	trainY, testY = load_dataset('processed_data/labels.pkl')
	# create tokenizer
	tokenizer = create_tokenizer(trainLines)
	# calculate max document length
	length = max_length(trainLines)
	# calculate vocabulary size
	vocab_size = len(tokenizer.word_index) + 1
	#print('Max document length: %d' % length)
	#print('Vocabulary size: %d' % vocab_size)
	# encode data
	trainX = encode_text(tokenizer, trainLines, length)
	testX = encode_text(tokenizer, testLines, length)
	#print('X shape: ', trainX.shape)
	#print('Y shape: ', trainY.shape)
	return  trainX, trainY, testY, testX

def print_metrics(y,yhat):
	score = accuracy_score(y,yhat)
	print('\n[info] Accuracy: ', score)
	print('\n[INFO] confusion matrix: ')
	print(confusion_matrix(y,yhat))  
	print('\n[INFO] classification report:')
	print(classification_report(y,yhat))  

def evalutate_MultinomialNB(trainX, trainY, testY, testX):
	print('[info] MultinomialNB model')
	model = MultinomialNB().fit(trainX, trainY)
	yhat = model.predict(testX)
	print_metrics(testY,yhat)

def evalutate_LogisticRegression(trainX, trainY, testY, testX):
	print('[info] LogisticRegression model')
	model = LogisticRegression(class_weight='balanced').fit(trainX, trainY)
	yhat = model.predict(testX)
	print_metrics(testY,yhat)

def evalutate_ComplementNB(trainX, trainY, testY, testX):
	print('[info] ComplementNB model')
	model = ComplementNB().fit(trainX, trainY)
	yhat = model.predict(testX)
	print_metrics(testY,yhat)

def evaluate_SVM(trainX, trainY, testY, testX):
	print('[info] SVM model')
	model = svm.LinearSVC().fit(trainX, trainY)
	yhat = model.predict(testX)
	print_metrics(testY,yhat)

def run_classic_models_evaluation():
	trainX, trainY, testY, testX = load_data_classic_models()
	evalutate_MultinomialNB(trainX, trainY, testY, testX)
	evalutate_LogisticRegression(trainX, trainY, testY, testX)
	evalutate_ComplementNB(trainX, trainY, testY, testX)
	evaluate_SVM(trainX, trainY, testY, testX)

# RUN: python classic_model_train.py
# https://www.kaggle.com/hungdo1291/keras-dnn-multi-class
# http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
# https://www.cs.waikato.ac.nz/~eibe/pubs/FrankAndBouckaertPKDD06new.pdf

'''
LogisticRegression
class_weight='balanced'
                  precision    recall  f1-score   support**
negativo_maioria       0.26      0.30      0.28        27
negativo_unânime       0.93      0.92      0.93       646
positivo_maioria       0.09      0.08      0.08        13
positivo_unânime       0.23      0.26      0.24        38

       micro avg       0.85      0.85      0.85       724
       macro avg       0.38      0.39      0.38       724
    weighted avg       0.86      0.85      0.85       724

MultinomialNB
                  precision    recall  f1-score   support
negativo_maioria       0.00      0.00      0.00        27
negativo_unânime       0.92      0.98      0.95       646
positivo_maioria       0.17      0.08      0.11        13
positivo_unânime       0.20      0.11      0.14        38

       micro avg       0.89      0.89      0.89       724
       macro avg       0.32      0.29      0.30       724
    weighted avg       0.83      0.89      0.86       724

ComplementNB
                  precision    recall  f1-score   support
negativo_maioria       0.00      0.00      0.00        27
negativo_unânime       0.91      0.99      0.95       646
positivo_maioria       0.20      0.15      0.17        13
positivo_unânime       0.27      0.08      0.12        38

       micro avg       0.89      0.89      0.89       724
       macro avg       0.35      0.31      0.31       724
    weighted avg       0.83      0.89      0.86       724
'''