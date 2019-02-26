from pickle import load
from numpy import array
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from nltk.corpus import stopwords
import string

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	# saving tokenizer
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return tokenizer

def load_tokenizer():
	# loading
	tokenizer = None
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# turn a doc into clean tokens
def clean_doc(doc):
	doc = doc.replace(".", " ")
	doc = doc.replace(",", " ")
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('portuguese'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	# to lower case
	tokens = [w.lower() for w in tokens]
	tokens = ' '.join(tokens)
	return tokens

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

def map_class(idx):
	clss = ['negativo_maioria','negativo_unânime','positivo_maioria','positivo_unânime']
	return clss[idx]

def build_tokenizer():
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
	return trainX,testX

def predict(doc_array=[]):
	#df = pd.DataFrame({'doc' : doc_array})
	X = map(clean_doc, doc_array)
	#df['doc'] = df['doc'].map(clean_doc)
	#X = df['doc'].tolist()
	tokenizer = load_tokenizer()
	X = encode_text(tokenizer, X, 260)
	model = load_model('models/multichannel_cnn.h5')
	yhat = model.predict([X,X,X])
	clss = np.argmax(yhat[0])
	clss_str = map_class(clss)
	print('[INFO] prediction: ', clss_str)
	return clss_str

def test_predictions():
	doc1 = "Negado provimento ao agravo regimental, nos termos do voto do Relator. Decisão unânime. Ausentes, justificadamente, neste julgamento, a Senhora Ministra Ellen Gracie e o Senhor Ministro Joaquim Barbosa. 2ª Turma, 21.06.2011."
	doc2 = "O Tribunal, por unanimidade e nos termos do voto da Relatora, Ministra Cármen Lúcia (Presidente), acolheu os embargos de declaração para reconsiderar a decisão monocrática de fl. 452 e o acórdão embargado de fls. 463-466, tornando-os sem efeito, edeterminar o regular processamento do recurso. Plenário, sessão virtual de 18 a 24.11.2016."
	doc3 = "A Turma, por maioria, negou provimento ao agravo interno, nos termos do voto do Relator, vencido o Ministro Marco Aurélio. 1ª Turma, Sessão Virtual de 18 a 24.11.2016."
	predict([doc1])

test_predictions()






