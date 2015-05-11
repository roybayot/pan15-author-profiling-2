#!/usr/bin/python
# This script is not yet complete. 

import sys, getopt
import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

from scipy import sparse

from myTrainingSoftware2 import getCount, review_to_words
from myTrainingSoftware2 import function_words_dict, stylistic_features

from sklearn.externals import joblib

import bleach
import os
import re
import csv
import pickle



import pandas as pd
import numpy as np
import re
import timeit

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from bs4 import BeautifulSoup

from treetagger import TreeTagger

reload(sys)
sys.setdefaultencoding("ISO-8859-1")


def getRelevantDirectories(argv):
   inputDir = ''
   outputDir = ''
   modelDir = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","mfile="])
   except getopt.GetoptError:
      print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputDir = arg
      elif opt in ("-m", "--mfile"):
      	 modelDir = arg
      elif opt in ("-o", "--ofile"):
         outputDir = arg   
   return inputDir, outputDir, modelDir

def getAllModels(modelDir):
	filename = modelDir + "/models.pkl"
	f = open(filename,'rb')
	models = pickle.load(f)
	f.close()
	return models

def dirExists(inputDir):
	if os.path.exists(inputDir):
		return True
	elif os.access(os.path.dirname(inputDir), os.W_OK):
		print "Cannot access the directory. Check for privileges."
		return False
	else:
		print "Directory does not exist."
		return False

def isXML(f):
	a = f.strip().split('.')
	if a[1] == 'xml':
		return True
	else:
		return False

def absoluteFilePaths(directory):
	allPaths = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			onePath = os.path.abspath(os.path.join(dirpath, f))
			allPaths.append(onePath)
# 			yield os.path.abspath(os.path.join(dirpath, f))
	return allPaths

def getAllFilenamesWithAbsPath(inputDir):
	if dirExists(inputDir):
		allPaths = absoluteFilePaths(inputDir)
		return allPaths
	else:
		sys.exit()

def getAllTestFiles(inputDir):
	if dirExists(inputDir):
		allTestFiles = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]
		allTestFiles = [ f for f in allTestFiles if isXML(f) ]
		return allTestFiles
	else:
		sys.exit()

def getAllXmlFiles(allTestFiles):
	allTestFiles = [ f for f in allTestFiles if isfile(f) ]
	allTestFiles = [ f for f in allTestFiles if isXML(f) ]
	return allTestFiles	

def getLanguage(oneFile):
	tree = ET.parse(oneFile)
	root = tree.getroot()
	a = root.attrib
	return a['lang']

def getTweetsToLine(oneFile):
	allText = ""
	try:
		tree = ET.parse(oneFile)
#  		print "Filename: %s SUCCESS!" % oneFile
	except:
		e = sys.exc_info()[0]
		print "Filename: %s Error: %s" % (oneFile, e)
	else:
		allDocs = tree.getroot().findall("document")		
		for doc in allDocs:
			clean = bleach.clean(doc.text, tags=[], strip=True)
 			allText = allText + clean
	 	allText = allText.encode('utf-8')
	return allText

def getCount(onePattern, inputString):
	return inputString.count(onePattern)
	 	
def getFeatureVecFromFunctionWords(oneLine, test_patterns):
	vector_for_one_entry = []
	for pattern in test_patterns:
		count = getCount(pattern, oneLine)
		vector_for_one_entry.append(count)
	return np.array(vector_for_one_entry)


def getFeatureVecFromStylisticFeatures(oneLine, stylistic_features):
	return getFeatureVecFromFunctionWords(oneLine, stylistic_features)


def getFeatureVecFromPOS(oneLine, lang, n_gram_range, ngram_vec):
	clean_train_reviews = review_to_words( oneLine, lang )
	tt = TreeTagger(encoding='latin-1',language=lang)
	train_reviews_pos_tags = []
	
	train_reviews_pos_tags = tt.tag(clean_train_reviews)
	a = [col[1] for col in train_reviews_pos_tags]
	pos_line = " ".join(a)
	X = ngram_vec.transform(pos_line).toarray()
	return X

	
def classifyTestFiles(models, inputDir):
	results = {}
	 
	base_models = {'nl': { 'gender'		 : 'male', \
					  'age'			 : 'XX-XX', \
					  'extroverted'	 : '0.2', \
					  'stable'		 : '0.4', \
					  'agreeable'	 : '0.1', \
					  'open'		 : '0.1', \
					  'conscientious': '0.4'
					}, \
			  'en': { 'gender'		 : 'male', \
					  'age'			 : '25-34', \
					  'extroverted'	 : '0.1', \
					  'stable'		 : '0.2', \
					  'agreeable'	 : '0.2', \
					  'open'		 : '0.1', \
					  'conscientious': '0.1' \
			  		}, \
			  'it': { 'gender'		 : 'male', \
					  'age'			 : 'XX-XX', \
					  'extroverted'	 : '0.1', \
					  'stable'		 : '0.1', \
					  'agreeable'	 : '0.1', \
					  'open'		 : '0.1', \
					  'conscientious': '0.1'
					}, \
			  'es': { 'gender'		 : 'male', \
					  'age'			 : '25-34', \
					  'extroverted'	 : '0.2', \
					  'stable'		 : '-0.1', \
					  'agreeable'	 : '0.2', \
					  'open'		 : '0.4', \
					  'conscientious': '0.1' \
					}, \
			  }	
	allTestFiles = getAllFilenamesWithAbsPath(inputDir)
	allTestFiles = getAllXmlFiles(allTestFiles)
	
	tasks = ["gender", "age", "extroverted", "stable", "agreeable", "open", "conscientious"]
	
	for oneFile in allTestFiles:
		lang = getLanguage(oneFile)
		aa = oneFile.strip().split("/")
		aa = aa[-1].strip().split(".")
		thisId					= aa[0]
# 		print oneFile
		thisType				= 'twitter'
		thisLanguage			= lang
		
		
		if lang == 'en':
			tempLang = 'english'
		if lang == 'nl':
			tempLang = 'dutch'
		if lang == 'it':
			tempLang = 'italian'
		if lang == 'es':
			tempLang = 'spanish'
		
		cwd = os.getcwd()
		if tempLang == 'english':
			path = cwd + '/model_dir/tfidf-english-models.pkl'
			vec = joblib.load(path)
		if tempLang == 'dutch':
			path = cwd + '/model_dir/tfidf-dutch-models.pkl'
			vec = joblib.load(path)
		if tempLang == 'italian':
			path = cwd + '/model_dir/tfidf-italian-models.pkl'
			vec = joblib.load(path)
		if tempLang == 'spanish':
			path = cwd + '/model_dir/tfidf-spanish-models.pkl'
			vec = joblib.load(path)

			
									
# 		for x in models: 
# 			if (x.keys()[0] == 'tfidf_vectorizer') and (x.values()[0].keys()[0] == tempLang):
#  				vec = x.values()[0].values()[0]
# 
# 		for x in models: 
# 			if (x.keys()[0] == 'unigram_vectorizer') and (x.values()[0].keys()[0] == tempLang):
#  				uni_vec = x.values()[0].values()[0]
# 
# 		for x in models: 
# 			if (x.keys()[0] == 'bigram_vectorizer') and (x.values()[0].keys()[0] == tempLang):
#  				bi_vec = x.values()[0].values()[0]

# 		import pdb; pdb.set_trace()
		
		oneLine = getTweetsToLine(oneFile)
		oneLine = review_to_words(oneLine, tempLang)
		
# 		pdb.set_trace()
		X1 = vec.transform([oneLine]).toarray()
# 		import pdb; pdb.set_trace()
# 		why is X1 still at 1x10000
# 		X1 = np.array(X1)
		print X1.shape
		X2 = getFeatureVecFromStylisticFeatures(oneLine, stylistic_features)
# 		X2 = np.array(X2)
		X2 = np.matrix(X2)
# 		X3 = getFeatureVecFromPOS(oneLine, tempLang, (1,1), uni_vec)
# 		X4 = getFeatureVecFromPOS(oneLine, tempLang, (1,2), bi_vec)
		
		
		temp = {}
		for task in tasks:
			if (lang == 'nl') and (task == 'age'):
				predictedAge = base_models['nl']['age']
			elif (lang == 'it') and (task == 'age'):
				predictedAge = base_models['it']['age']
			else:
# 				import pdb; pdb.set_trace()						
				for model in models:
# 					import pdb; pdb.set_trace()
# 					print model.keys()[0], lang
# 					print model.values()[0].keys()[0], task
					if (model.keys()[0] == tempLang) and (model.values()[0].keys()[0] == task):
						X5 = getFeatureVecFromFunctionWords(oneLine, function_words_dict[tempLang][task])
# 						X5 = np.array(X5)
						X5 = np.matrix(X5)						
# 						print X1.shape, X2.shape, X3.shape, X4.shape, X5.shape
						print X1.shape, X2.shape, X5.shape						
# 						import pdb; pdb.set_trace()						
# 						descriptors = np.concatenate((X1,X2,X3,X4,X5), axis=1)
# 						descriptors = np.concatenate((X1,np.matrix(X2),np.matrix(X5)))						
# 						descriptors = np.concatenate((X1,X2,X5))
						descriptors = np.hstack((X1,X2,X5))
# 						descriptors = np.hstack((X2,X5))						
# 						descriptors = np.hstack((X1,descriptors))
						clf = model[tempLang][task]
						# import pdb; pdb.set_trace()						
						
						pred_value = clf.predict(descriptors)
						temp[task] = pred_value
		temp['thisId']       = thisId
		temp['thisType']     = thisType
		temp['thisLanguage'] = thisLanguage
		results[oneFile] =  temp
	return results

def writeOneResult(key, value, outputDir):
	key = key.strip().split("/")
	cwd = os.getcwd()
	path = cwd + "/" + outputDir + "/" + key[-1]
# 	import pdb; pdb.set_trace()
	thisId					= value['thisId']
	thisType				= value['thisType']
	thisLanguage			= value['thisLanguage']
	predictedGender 	 	= value['gender']
	predictedAge    	 	= value['age']
	predictedExtroverted 	= value['extroverted']
	predictedStable 		= value['stable']
	predictedAgreeable	  	= value['agreeable']
	predictedOpen		  	= value['open']
	predictedConscientious 	= value['conscientious']

	
	text_to_write = """<author id='%s'\n\ttype='%s'\n\tlang='%s'\n\tage_group='%s'\n\tgender='%s'\n\textroverted='%s'\n\tstable='%s'\n\tagreeable='%s'\n\tconscientious='%s'\n\topen='%s'\n/>"""% (thisId, thisType, thisLanguage, predictedAge, predictedGender, \
  		  predictedExtroverted, predictedStable, predictedAgreeable, \
  		  predictedConscientious, predictedOpen)
	# Open a file
	fo = open(path, "w")
	fo.write( text_to_write );
	fo.close()
	
def makeDirectory(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
		else:
			print "\nBE CAREFUL! Directory %s already exists." % path
		
def writeAllResults(results, outputDir):
	if (not dirExists(outputDir)):
		print "Creating new directory."
		makeDirectory(outputDir)
	for key, value in results.iteritems():
#  		print key, ":", value
		writeOneResult(key, value, outputDir)	
		
def main(argv):
	inputDir, outputDir, modelDir = getRelevantDirectories(argv)
# 	print 'Input directory is "',  inputDir
# 	print 'Model directory is "',  modelDir   
# 	print 'Output directory is "', outputDir
	
	models = getAllModels(modelDir)
	results = classifyTestFiles(models, inputDir)
# 	print results
	writeAllResults(results, outputDir)

   
if __name__ == "__main__":
	main(sys.argv[1:])