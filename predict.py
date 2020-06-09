import xml.etree.ElementTree as ET
import os
import csv
import argparse
import tensorflow
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pickle
import numpy as np 
import pandas as pd 

from emoticons import analyze_tweet
import spacy
from es_lemmatizer import lemmatize
from spacy_spanish_lemmatizer import SpacyCustomLemmatizer
import twokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
import re
from nltk.util import ngrams
import scipy

def vectorize(samples, fit, vectorizer):

	if fit:
		X = vectorizer.fit_transform(samples)
	else:
		X = vectorizer.transform(samples)

	#print(str(X.shape))

	return X

def cargar_lexico(filename):
	file = open(filename, "r")
	lexico = {}
	for line in file:
		#print(line)
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
						#print("KEY: " + word[0] + " FOUND")
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
						#print("KEY: " + clave + " FOUND")
					except:
						error_count = error_count + 1

		pos.append(counter_pos)
		neg.append(counter_neg)
		sentiment.append(dic_sents[analyze_tweet(muestra).lower()])
		#print("TWEET: " + muestra)
		#print("SENTIMENT: " + analyze_tweet(muestra).lower())

	return pos, neg, sentiment

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
	#print(str(xxx))
	return (xxx)

def limpiar_corpus(datos):
	muestras = []

	for x in datos:
		#Quitamos contenido HTML
		texto = BeautifulSoup(x).get_text()
		#Quitamos todo lo que no son letras
		texto = re.sub("RT", "", texto)
		texto = re.sub("#USER#", "", texto)
		texto = re.sub("#HASHTAG#", "", texto)
		texto = re.sub("#URL#", "", texto)
		texto = re.sub("[^a-zA-Z]", " ", texto)
		#Tokenizado y pasamos a minusculas
		palabras = word_tokenize(texto.lower())

		#Mapeamos cada palabra a su lema (a ver si así mejoramos)
		lemas = [lemmatizer.lemmatize(i) for i in palabras]

		muestras.append(lemas)
	return(muestras)

def main():
	parser = argparse.ArgumentParser(description="Parsing command line argunments...")
	parser.add_argument("-i", default=None, type=str, required=True, help="Route to input folder")
	parser.add_argument("-o", default=None, type=str, required=True, help="Route to output folder")

	args = parser.parse_args()
	print(args.i)
	directories = [d for d in os.listdir(args.i) if os.path.isdir(os.path.join(args.i, d))]
	print(directories)

	for d in directories:
		files = [f for f in os.listdir(os.path.join(args.i, d)) if os.path.isfile(os.path.join(os.path.join(args.i, d), f))]
		
		model = ""
		vocabulary = ""

		if d == "es":
			#model = "svm_ALC_ES.sav"
			model = "mlp_ALC_ES.sav"
			#vocabulary = "tfidfES.pickle"
			vocabulary = "countES.pickle"
		else:
			#model = "svm_ALC_EN.sav"
			model = "mlp_ALC_EN.sav"
			#vocabulary = "tfidfEN.pickle"
			vocabulary = "countEN.pickle"

				

		for file in files:
			if file.startswith("."):
				continue
			print(os.path.join(os.path.join(args.i, d), file))
			tree = ET.parse(os.path.join(os.path.join(args.i, d), file))
			root = tree.getroot()
			language = root.attrib["lang"]
			
			textos = []	
			for documents in root:
				for document in documents:
					textos.append(document.text)

			lexico = cargar_lexico("lexico.lex")
			#vectorizer = TfidfVectorizer(tokenizer=mi_tokenizador_twokenizer, ngram_range=(1,2), min_df=0.0, vocabulary=pickle.load(open(vocabulary, 'rb')))
			vectorizer = CountVectorizer(tokenizer=mi_tokenizador_twokenizer, vocabulary=pickle.load(open(vocabulary, 'rb')))
			vector1 = vectorize(textos, True, vectorizer)
			pos, neg, sent = counters_pos_neg(lexico, textos)
			all_vector_train = scipy.sparse.hstack((vector1, np.asarray(pos)[:,None], np.asarray(neg)[:,None], np.asarray(sent)[:,None]))
			loaded_model = joblib.load(open(model, 'rb'))
			print(str(type(loaded_model)))

			prediccion = loaded_model.predict(all_vector_train)

			print(prediccion)
			votaciones = {0:0, 1:0}

			for voto in prediccion:
				votaciones[voto] = votaciones[voto] + 1

			print(votaciones)
			ganador = 0

			if votaciones[1] > votaciones[0]:
				ganador = 1

			output = '<author id="' + file.split(".")[0] + '" lang="' + language + '" type="' + str(ganador) + '"/>'

			try:
				# Create target Directory
				os.mkdir(os.path.join(args.o, d))
				print("Directory " , os.path.join(args.o, d) ,  " Created ") 
			except FileExistsError:
				print("Directory " , os.path.join(args.o, d) ,  " already exists")

			with open(os.path.join(os.path.join(args.o, d), file), "w") as archivo_salida:
				archivo_salida.write(output)
				archivo_salida.close()

			
	print("FINISHED")


main()