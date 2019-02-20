from nltk.corpus import stopwords
from pickle import dump
import string
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import numpy

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

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

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# Read documents as a dataframe
def get_dataframes():
	sqlite_con = sqlite3.connect('../../legaltech/data/stf_acordaos_db')

	positive_sql = '''
		select calsse as classe, decisao, data, codigo, turma, 1 as label from acordoes
		where decisao <> ''  and not
  		(lower(decisao) like '%negou%' or
		lower(decisao) like '%rejeitaram-se%' or
		lower(decisao) like '%negou-lhe %' or
		lower(decisao) like '%rejeitou%' or
		lower(decisao) like '%desproveu%' or
		lower(decisao) like '%rejeitados%' or
		lower(decisao) like '%rejeitado%' or
		lower(decisao) like '%negando%' or
		lower(decisao) like '%indeferiu%' or
		lower(decisao) like '%não conheceu%' or
		lower(decisao) like '%Não se conheceu%' or
		lower(decisao) like '%não tomaram%' or
		lower(decisao) like '%prejudicada%' or
		lower(decisao) like '%não conhecendo%' or
		lower(decisao) like '%improcedente%')
		and calsse = 'Mandado de Segurança';
	'''

	negative_sql = '''
		select calsse as classe, decisao, data, codigo, turma, 0 as label from acordoes
		where decisao <> ''  and
  		(lower(decisao) like '%negou%' or
		lower(decisao) like '%rejeitaram-se%' or
		lower(decisao) like '%negou-lhe %' or
		lower(decisao) like '%rejeitou%' or
		lower(decisao) like '%desproveu%' or
		lower(decisao) like '%rejeitados%' or
		lower(decisao) like '%rejeitado%' or
		lower(decisao) like '%negando%' or
		lower(decisao) like '%indeferiu%' or
		lower(decisao) like '%não conheceu%' or
		lower(decisao) like '%Não se conheceu%' or
		lower(decisao) like '%não tomaram%' or
		lower(decisao) like '%prejudicada%' or
		lower(decisao) like '%não conhecendo%' or
		lower(decisao) like '%improcedente%')
		and calsse = 'Mandado de Segurança';
	'''

	df_p = pd.read_sql_query(positive_sql, sqlite_con)
	df_n = pd.read_sql_query(negative_sql, sqlite_con)

	postive_len = len(df_n)
	negative_len = len(df_p)

	print('Class 0:', negative_len)
	print('Class 1:', postive_len)
	print('Proportion:', round(negative_len/ postive_len, 2), ': 1')

	# Undersappling the dataframe positive dataframe
	df_n =  df_n.sample(postive_len, random_state=5)

	return df_p, df_n

# python -c 'from data_clean_util import *; get_dataframes_re()'
# python -c 'import numpy as np; l = np.array([131, 3226, 62, 186]); print(l/l.sum()*100); print(l/l.sum()*100-100);'
# REFERENCES: https://github.com/keras-team/keras/issues/741
def get_dataframes_re():
	df = pd.read_csv('data/RE.csv')
	df["label_str"] = df["sentimento"] + '_' + df["mérito_modo"]
	print("All classes")
	print(df.groupby(['label_str']).size())
	print("-------------------")
	filter_idxs =  df['label_str'].isin(['negativo_maioria','negativo_unânime','positivo_maioria','positivo_unânime'])
	df = df[filter_idxs]
	df.label_str = pd.Categorical(df.label_str)
	df['label'] = df.label_str.cat.codes
	df['doc'] = df['10']
	df = df[['doc', 'label', 'label_str']]
	#print(df[['10','label_str']].head())
	print("Filtered")
	print(df.head)
	print(df.groupby(['label']).size())

	df['doc'] = df['doc'].map(clean_doc)
	labels = df.label.unique()
	trains = []
	tests = []
	for label in labels:
		train, test = train_test_split(df[df.label == label], test_size=0.2)
		tests.append(test)
		trains.append(train)

	train = pd.concat(trains, keys=labels)
	test= pd.concat(tests, keys=labels)
	print('Splitted: ')
	print(test.groupby(['label_str']).size())
	print(train.groupby(['label_str']).size()) 

	# Get Y
	binarizer = LabelBinarizer()
	trainY = binarizer.fit_transform(train.label_str)
	print(binarizer.classes_)
	#trainY = multilabel_binarizer.classes_
	binarizer = LabelBinarizer()
	testY = binarizer.fit_transform(test.label_str)
	#testY = multilabel_binarizer.classes_
	print('multilabel_binarizer example: ')
	#print(testY[0:5])
	# Get X 
	trainX = train['doc'].tolist()
	testX = test['doc'].tolist()
	# Save processed data
	save_dataset([train.label_str,test.label_str], 'processed_data/labels.pkl')
	save_dataset([trainX,trainY], 'processed_data/train.pkl')
	save_dataset([testX,testY], 'processed_data/test.pkl')

#MAIN
#df = pd.read_csv('mc_acordoes_data.csv')
# python -c "import data_clean_util;  clean_acordoes_data"
def clean_acordoes_data():
	sqlite_con = sqlite3.connect('../../legaltech/data/stf_acordaos_db')
	df_p, df_n = get_dataframes()
	df_p['doc']= df_p['decisao'].map(clean_doc)
	df_n['doc']= df_n['decisao'].map(clean_doc)

	train_p, test_p = train_test_split(df_p, test_size=0.1)
	train_n, test_n = train_test_split(df_n, test_size=0.1)

	# load all training reviews
	trainX = train_p['doc'].tolist() + train_n['doc'].tolist()
	trainY = train_p['label'].tolist() + train_n['label'].tolist()
	save_dataset([trainX,trainY], 'train.pkl')

	# load all test reviews
	testX = test_p['doc'].tolist() + test_n['doc'].tolist()
	testY = test_p['label'].tolist() + test_n['label'].tolist()
	save_dataset([testX,testY], 'test.pkl')