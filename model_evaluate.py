from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.preprocessing import LabelBinarizer

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

# Inverse transform one hote enconded label to class vector
def to_categorical(y):
	nb_classes = ['negativo_maioria','negativo_unânime','positivo_maioria','positivo_unânime']
	binarize = LabelBinarizer()
	binarize.fit_transform(nb_classes)
	return binarize.inverse_transform(y)

def run_evaluation():
	# load datasets
	trainLines, trainLabels = load_dataset('processed_data/train.pkl')
	testLines, testLabels = load_dataset('processed_data/test.pkl')
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
	testX = encode_text(tokenizer, testLines, length)
	print(trainX.shape, testX.shape)

	# load the model
	model = load_model('models/multichannel_cnn.h5')

	# evaluate model on training dataset
	loss, acc = model.evaluate([trainX,trainX,trainX], trainLabels, verbose=0)
	print('Train categorical accuracy: %f' % (acc*100))

	# evaluate model on test dataset dataset
	loss, acc = model.evaluate([testX,testX,testX], testLabels, verbose=0)
	print('Test categorical accuracy: %f' % (acc*100))

	# print train metrics
	print('** TRAIN DATA')
	yhat = model.predict([trainX,trainX,trainX])
	score = accuracy_score(to_categorical(trainLabels),to_categorical(yhat))
	print('\n[INFO] Accuracy: ', score)
	print('\n[INFO] confusion matrix: ')
	print(confusion_matrix(to_categorical(trainLabels),to_categorical(yhat)))  
	print('\n[INFO] classification report:')
	print(classification_report(to_categorical(trainLabels),to_categorical(yhat)))  

	# Print test metrics
	print('** TEST DATA')
	yhat = model.predict([testX,testX,testX])
	score = accuracy_score(to_categorical(testLabels), to_categorical(yhat))
	print('\n[INFO] Accuracy: ', score)
	print('\n[INFO] confusion matrix: ')
	print(confusion_matrix(to_categorical(testLabels),to_categorical(yhat)))  
	print('\n[INFO] classification report:')
	print(classification_report(to_categorical(testLabels),to_categorical(yhat)))  

# REFERENCES:
# https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin2
# https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin2
# 

'''
Accuracy:  0.9571823204419889
[[ 23   1   3   0]
 [  6 631   1   8]
 [  3   0   7   3]
 [  1   4   1  32]]
                  precision    recall  f1-score   support

negativo_maioria       0.70      0.85      0.77        27
negativo_unânime       0.99      0.98      0.98       646
positivo_maioria       0.58      0.54      0.56        13
positivo_unânime       0.74      0.84      0.79        38

       micro avg       0.96      0.96      0.96       724
       macro avg       0.75      0.80      0.78       724
    weighted avg       0.96      0.96      0.96       724
'''