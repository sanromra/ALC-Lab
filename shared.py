import xml.etree.ElementTree as ET
import os
import csv
import sys
import argparse
from nltk.tokenize import TweetTokenizer

import re
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report


import scipy
import nltk
from nltk.util import ngrams
from sklearn.decomposition import PCA, TruncatedSVD
import spacy
from es_lemmatizer import lemmatize
from spacy_spanish_lemmatizer import SpacyCustomLemmatizer
import twokenize
import pickle
from emoticons import analyze_tweet
import pandas as pd 
import pickle
import joblib


def parse(filename, xml_flag, test=False):
	data_train = []
	label_train = []

	if xml_flag:
		tree = ET.parse(filename)
		root = tree.getroot()
		data_train = []
		label_train = []
		for tweet in root.getchildren():
			data_train.append(tweet.find("content").text.lower())
			if not test:
				label_train.append(tweet.find("sentiment").find("polarity").find("value").text)
			else:
				label_train.append(tweet.find("tweetid").text)
	else:
		df = pd.read_csv(filename, sep="\t")
		for x in df["Text"]:
			data_train.append(x)

		for x in df["Label"]:
			label_train.append(x)

		print("Data_train type: " + str(type(data_train)))

	return data_train, label_train


def vectorize(samples, fit, vectorizer):

	if fit:
		X = vectorizer.fit_transform(samples)
	else:
		X = vectorizer.transform(samples)
	return X


def mi_tokenizador_twokenizer(s): 
	xx = twokenize.tokenize(s)
	if ("happy" or "tongue" or "sad" or "wink" or "other") in xx:
		print("ORIGINAL: " + str(s))
		print("TOKENS: " + str(xx))
	xxx = []
	for t in xx:
		t = re.sub('@.*', "arroba", t)
		t = re.sub('#(.*)', "hashtag",t) 
		t = re.sub('((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}))', "web", t) 
		t = re.sub('[0-9].*', "num", t)
		xxx.append(t)
	return (xxx)


def cargar_lexico(filename):
	file = open(filename, "r")
	lexico = {}
	for line in file:
		if not line.startswith("#"):
			terminos = line.split()
			if len(terminos) != 0:
				lexico[terminos[0]] = terminos[1]

	file.close()
	return lexico

def counters_pos_neg(lexico, muestras):
	pos = []
	neg = []
	sentiment = []
	dic_sents = {"happy":0, "sad":1, "tongue":2, "wink":2, "other":2}
	for muestra in muestras:
		counter_pos = 0
		counter_neg = 0
		error_count = 0
		token = nltk.word_tokenize(muestra)
		for n in range(1,4):
			grams = list(ngrams(token, n))
			if n == 1:
				for word in grams:
					try:
						if lexico[word[0]] == "positive":
							counter_pos = counter_pos + 1
						elif lexico[word[0]] == "negative":
							counter_neg = counter_neg + 1
					except:
						error_count = error_count + 1
			else:
				for word in grams:
					clave = "_".join(word)
					try:
						if lexico[clave] == "positive":
							counter_pos = counter_pos + 1
						elif lexico[clave] == "negative":
							counter_neg = counter_neg + 1
					except:
						error_count = error_count + 1

		pos.append(counter_pos)
		neg.append(counter_neg)
		sentiment.append(dic_sents[analyze_tweet(muestra).lower()])

	return pos, neg, sentiment


def main():

	parser = argparse.ArgumentParser(description="Parsing command line argunments...")
	parser.add_argument("--train", default=None, type=str, required=False, help="Route to input file")
	parser.add_argument("--dev", default=None, type=str, required=False, help="Route to input file")
	parser.add_argument("--test", default=None, type=str, required=False, help="Route to input file")
	parser.add_argument("--lex", default=None, type=str, required=False, help="Route to input file")
	parser.add_argument("--saveLemmas", default=False, type=bool, required=False, help="Route to input file")


	args = parser.parse_args()

	flag_lex = False
	xml_flag = False

	print(xml_flag)

	if args.lex is not None:
		lexico = cargar_lexico(args.lex)
		flag_lex = True


	#vectorizer = TfidfVectorizer(tokenizer=mi_tokenizador_twokenizer, ngram_range=(1,2), min_df=0.0)
	#vectorizer2 = TfidfVectorizer(tokenizer=mi_tokenizador_twokenizer, ngram_range=(1,2), min_df=0.0)
	#vectorizer = HashingVectorizer(tokenizer=mi_tokenizador)
	#vectorizer = CountVectorizer(tokenizer=mi_tokenizador_twokenizer)
	#pca = TruncatedSVD(n_components=400)
	#lexico = cargar_lexico("lexico.lex")
	vocabulary = "tfidfEN.pickle"
	vectorizer = TfidfVectorizer(tokenizer=mi_tokenizador_twokenizer, ngram_range=(1,2), min_df=0.0, vocabulary=pickle.load(open(vocabulary, 'rb')))

	if args.train is not None:
		data_train, label_train = parse(args.train, xml_flag)
		print(type(data_train[0]))
		vector1 = vectorize(data_train, True, vectorizer)
		print("LENGTH VECTOR: " + str(vector1.shape))
		pickle.dump(vectorizer.vocabulary_, open("tfidfEN.pickle", "wb"))
		if args.saveLemmas:
			with open("lemma_train.txt", "rb") as fp:   #Pickling
				lemmatized = pickle.load(fp)
				lemmatized = [" ".join(x) for x in lemmatized]
				print(type(lemmatized[0]))

		if flag_lex:
			pos, neg, sent = counters_pos_neg(lexico, data_train)

		all_vector_train = scipy.sparse.hstack((vector1, np.asarray(pos)[:,None], np.asarray(neg)[:,None], np.asarray(sent)[:,None]))
		#pca.fit(all_vector_train)

	if args.dev is not None:
		
		data_dev, label_dev = parse(args.dev, xml_flag)
		vector1 = vectorize(data_dev, False, vectorizer)

		if args.saveLemmas:
			with open("lemma_dev.txt", "rb") as fp:   #Pickling
				lemmatized = pickle.load(fp)
				lemmatized = [" ".join(x) for x in lemmatized]

		if flag_lex:
			pos, neg, sent = counters_pos_neg(lexico, data_dev)

		all_vector_dev = scipy.sparse.hstack((vector1, np.asarray(pos)[:,None], np.asarray(neg)[:,None], np.asarray(sent)[:,None]))
		#all_vector_dev = pca.transform(all_vector_dev)

	if args.test is not None:
		data_test, label_test = parse(args.test, xml_flag, test=True)
		vector1 = vectorize(data_test, True, vectorizer)

		if args.saveLemmas:
			with open("lemma_test.txt", "rb") as fp:   #Pickling
				lemmatized = pickle.load(fp)
				lemmatized = [" ".join(x) for x in lemmatized]



		if flag_lex:
			pos, neg, sent = counters_pos_neg(lexico, data_test)
		all_vector_test = scipy.sparse.hstack((vector1, np.asarray(pos)[:,None], np.asarray(neg)[:,None], np.asarray(sent)[:,None]))
		#all_vector_test = pca.transform(all_vector_test)



	for c in [17]:

		#clf = svm.SVC(C=1000, kernel="sigmoid")
		#clf = MLPClassifier(max_iter=1000000, alpha=0.01, learning_rate_init=0.0001, random_state=c, solver="sgd")
		

		#clf.fit(all_vector_train, label_train)
		clf = joblib.load(open("svm_ALC_EN.sav", 'rb')) 
		print(clf.__getstate__()['_sklearn_version'])
		prediccion = clf.predict(all_vector_test)
		print(classification_report(label_test, prediccion))

		#prediccion = clf.predict(all_vector_test)
		"""
		file1 = open("resultadoTest.txt","w") 
		for i in range(len(id_tweets)):
			file1.write(str(id_tweets[i]) + "\t" + str(prediccion[i] + "\n"))

		file1.close()
		"""
	#filename = 'svm_ALC_EN.sav'
	#pickle.dump(clasificador, open(filename, 'wb'))
 


main()
