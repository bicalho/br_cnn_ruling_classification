from pickle import load
from numpy import array
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_clean_util import clean_doc
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix  

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

def map_class(idx):
	clss = ['negativo_maioria','negativo_unânime','positivo_maioria','positivo_unânime']
	return clss[idx]

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


doc = "Negado provimento ao agravo regimental, nos termos do voto do Relator. Decisão unânime. Ausentes, justificadamente, neste julgamento, a Senhora Ministra Ellen Gracie e o Senhor Ministro Joaquim Barbosa. 2ª Turma, 21.06.2011."
#doc = "O Tribunal, por unanimidade e nos termos do voto da Relatora, Ministra Cármen Lúcia (Presidente), acolheu os embargos de declaração para reconsiderar a decisão monocrática de fl. 452 e o acórdão embargado de fls. 463-466, tornando-os sem efeito, edeterminar o regular processamento do recurso. Plenário, sessão virtual de 18 a 24.11.2016."
doc2 = "A Turma, por maioria, negou provimento ao agravo interno, nos termos do voto do Relator, vencido o Ministro Marco Aurélio. 1ª Turma, Sessão Virtual de 18 a 24.11.2016."
df = pd.DataFrame({'doc' : [doc, doc2]})
df['doc'] = df['doc'].map(clean_doc)
print(df['doc'].head())
X = df['doc'].tolist()
X = encode_text(tokenizer, X, length)
# load the model
model = load_model('model_multiclasses.h5')
yhat = model.predict([X,X,X])

clss = numpy.argmax(yhat[0])
print(map_class(clss))
clss = numpy.argmax(yhat[1])
print(map_class(clss))

