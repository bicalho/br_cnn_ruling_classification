from pickle import load
from numpy import array
import numpy as np
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
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight




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

def evaluate_MultinomialNB(trainX, trainY, testY, testX):
	print('[info] MultinomialNB model')
	model = MultinomialNB()
	parameters = {  
                'alpha': [1,0.1,0.01,0.001],
                 'fit_prior': [True, False],
            	}
	clf = GridSearchCV(estimator=model,param_grid=parameters,scoring='f1_macro')
	clf.fit(trainX, trainY)
	yhat = clf.predict(testX)
	print(clf.best_params_)
	print_metrics(testY,yhat)
	precision, recall, _ =  precision_recall_curve(testY, yhat)
	plt.xlabel = ('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall Curve')
	plt.plot(recall, precision)

def evaluate_LogisticRegression(trainX, trainY, testY, testX):
	print('[info] LogisticRegression model')
	model = LogisticRegression(class_weight='balanced', multi_class='auto')
	parameters = {
				'C': np.logspace(-3,3,7), 
				'penalty': ['l1','l2']
				}
	clf = GridSearchCV(estimator=model,param_grid=parameters,scoring='f1_macro')
	clf.fit(trainX, trainY)
	yhat = clf.predict(testX)
	print(clf.best_params_)
	print_metrics(testY,yhat)
	

def evaluate_ComplementNB(trainX, trainY, testY, testX):
	print('[info] ComplementNB model')
	model = ComplementNB()
	parameters = {  
                'alpha': [1,0.1,0.01,0.001],
                 'fit_prior': [True, False],
            	}
	clf = GridSearchCV(estimator=model,param_grid=parameters,scoring='f1_macro')
	clf.fit(trainX, trainY)
	yhat = clf.predict(testX)
	print(clf.best_params_)
	print_metrics(testY,yhat)

def evaluate_SVM(trainX, trainY, testY, testX):
	print('[info] SVM model')
	
	cw = compute_class_weight("balanced", np.unique(trainY), trainY)
	print('SVM class weight: ', cw)
	
	model = svm.LinearSVC(class_weight='balanced')
	parameters = { 
				'C': [1, 10]
				}
	clf = GridSearchCV(estimator=model,param_grid=parameters,scoring='f1_macro')
	clf.fit(trainX, trainY)
	yhat = clf.predict(testX)
	print(clf.best_params_)
	print_metrics(testY,yhat)

def run_classic_models_evaluation():
	trainX, trainY, testY, testX = load_data_classic_models()
	evaluate_MultinomialNB(trainX, trainY, testY, testX)
	evaluate_LogisticRegression(trainX, trainY, testY, testX)
	evaluate_ComplementNB(trainX, trainY, testY, testX)
	evaluate_SVM(trainX, trainY, testY, testX)

#run_classic_models_evaluation()

# RUN: python classic_model_train.py
# https://www.kaggle.com/hungdo1291/keras-dnn-multi-class
# http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
# https://www.cs.waikato.ac.nz/~eibe/pubs/FrankAndBouckaertPKDD06new.pdf

'''
[info] LogisticRegression model
{'C': 100.0, 'penalty': 'l1'}

[info] Accuracy:  0.8715469613259669

[INFO] confusion matrix: 
[[  8  15   2   2]
 [ 12 614   6  14]
 [  3   4   2   4]
 [  7  21   3   7]]

[INFO] classification report:
                  precision    recall  f1-score   support

negativo_maioria       0.27      0.30      0.28        27
negativo_unânime       0.94      0.95      0.94       646
positivo_maioria       0.15      0.15      0.15        13
positivo_unânime       0.26      0.18      0.22        38

       micro avg       0.87      0.87      0.87       724
       macro avg       0.40      0.40      0.40       724
    weighted avg       0.86      0.87      0.87       724


[info] MultinomialNB model
{'alpha': 1, 'fit_prior': True}

[info] Accuracy:  0.8853591160220995

[INFO] confusion matrix: 
[[  0  19   3   5]
 [  1 636   1   8]
 [  1   8   1   3]
 [  1  32   1   4]]

[INFO] classification report:
                  precision    recall  f1-score   support

negativo_maioria       0.00      0.00      0.00        27
negativo_unânime       0.92      0.98      0.95       646
positivo_maioria       0.17      0.08      0.11        13
positivo_unânime       0.20      0.11      0.14        38

       micro avg       0.89      0.89      0.89       724
       macro avg       0.32      0.29      0.30       724
    weighted avg       0.83      0.89      0.86       724

[info] ComplementNB model
{'alpha': 1, 'fit_prior': True}

[info] Accuracy:  0.888121546961326

[INFO] confusion matrix: 
[[  0  21   3   3]
 [  1 638   3   4]
 [  1   9   2   1]
 [  1  32   2   3]]

[INFO] classification report:
                  precision    recall  f1-score   support

negativo_maioria       0.00      0.00      0.00        27
negativo_unânime       0.91      0.99      0.95       646
positivo_maioria       0.20      0.15      0.17        13
positivo_unânime       0.27      0.08      0.12        38

       micro avg       0.89      0.89      0.89       724
       macro avg       0.35      0.31      0.31       724
    weighted avg       0.83      0.89      0.86       724

[info] SVM model
SVM class weight:  [ 6.92548077  0.27916667 14.69897959  4.86655405]
{'C': 10}

[info] Accuracy:  0.8756906077348067

[INFO] confusion matrix: 
[[  7  14   3   3]
 [ 12 613   2  19]
 [  3   4   0   6]
 [  4  19   1  14]]

[INFO] classification report:
                  precision    recall  f1-score   support

negativo_maioria       0.27      0.26      0.26        27
negativo_unânime       0.94      0.95      0.95       646
positivo_maioria       0.00      0.00      0.00        13
positivo_unânime       0.33      0.37      0.35        38

       micro avg       0.88      0.88      0.88       724
       macro avg       0.39      0.39      0.39       724
    weighted avg       0.87      0.88      0.87       724
'''