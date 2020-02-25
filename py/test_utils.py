import nltk
import csv
import sys
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
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

def count_elements(seq) -> dict:
	hist = {}
	for i in seq:
		hist[i] = hist.get(i, 0) + 1
	return hist
	
def divide_classes(data) -> dict:
	class0 = data[data.predicted == 0]
	class1 = data[data.predicted == 1]
	class2 = data[data.predicted == 2]
	
	#return class0['datetime'], class1['datetime'], class2['datetime']
	return class0, class1, class2

def toString(data):
	s = ''
	dt = [d for d in data['datetime']]
	tw = [t for t in data['tweets']]
	pr = [p for p in data['predicted']]
	i = 0
	while i < len(dt):
		s = s + tw[i] + '\t'+ str(dt[i]) + '\t'+ str(pr[i])
		s = s + '\n'
		i = i+1
	return s
	
def print_results(data):
	class0, class1, class2 = divide_classes(data)
	
	df = class0.groupby("datetime")["predicted"].count()
	print('\nclass0:')
	print(df)
	df = class1.groupby("datetime")["predicted"].count()
	print('\nclass1:')
	print(df)
	df = class2.groupby("datetime")["predicted"].count()
	print('\nclass2:')
	print(df)
	
def create_hist(data):
	#data['datetime'] = mdates.epoch2num(data['datetime'])
	class0, class1, class2 = divide_classes(data)
	classes = [class0,class1,class2]
	
	plt.figure(figsize=(12, 6))
	colors = ['lightblue', 'darkblue', 'black']
	plt.hist(classes,bins=5, rwidth=0.8,histtype='bar', stacked=True,color=colors, label=['0','1','2'])
	plt.show()
	"""
	fig, ax = plt.subplots(1,1)
	ax.hist(class0['datetime'], bins=20, color='lightblue', )
	ax.hist(class1['datetime'], bins=20, color='blue', )
	ax.hist(class2['datetime'], bins=20, color='black')
	locator = mdates.AutoDateLocator()
	ax.xaxis.set_major_locator(locator)
	ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
	plt.show()
	
	
	data = data.drop(['tweets'],axis = 1)
	data.plot()
	plt.show()
	"""
