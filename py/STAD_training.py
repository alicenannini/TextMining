import nltk
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
import joblib
import stemming
import test_utils

dataset = pd.read_csv("training_set.csv",sep='\t',names=['tweets','target'])

clf = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', LinearSVC()),
])


clf.fit(dataset.tweets, dataset.target)

def testModel():
	data = pd.read_csv("fileFeatures1.csv",sep='\t',names=['tweets','datetime'])
	#data['datetime'] = pd.to_datetime(data['datetime'],format="%Y/%m/%d")
	
	cl0 = 0
	cl1 = 0
	cl2 = 0

	data['predicted'] = clf.predict(data.tweets)
	
	i = 0
	for w in data['predicted']:
		if w == 0:
		    cl0 = cl0 +1
		if w == 1:
		    cl1 = cl1 +1
		    print('1 : ' + str(data.iloc[i,0]))
		if w == 2:
		    cl2 = cl2 +1
		    print('2 : ' + str(data.iloc[i,0]))
		i = i+1
		    
	print("\ndataset len: " + str(len(data)))
	print("class 0 len: " + str(cl0))
	print("class 1 len: " + str(cl1))
	print("class 2 len: " + str(cl2))
	
	data = data.set_index('datetime')
	
	#test_utils.create_hist(data)
	test_utils.print_results(data)
	'''
	data = data.drop(['tweets'],axis = 1)
	data.plot()
	plt.show()
	
	with open("histoFile.csv","w") as histFile:
		histFile.write(test_utils.toString(data))
	'''


#testModel()

joblib.dump(clf, 'STAD_model.pkl', compress=9)

