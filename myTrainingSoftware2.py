#!/usr/bin/python

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import csv
import pickle

from scipy import sparse

import pandas as pd
import numpy as np
import re
import timeit



from nltk.corpus import stopwords

from sklearn.externals import joblib
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


function_words_dict = {
'english':{
			'age':	[ 
						"co",
						"wanna",
						"us",
						"haha",
						"username",
						"fitbit",
						"et",
						"bowl",
						"academia",
						"bitch",
						"happened",
						"even",
						"year",
						"reach",
						"free",
						"times",
						"speech",
						"top",
						"add",
						"social",
						"think",
						"nothing",
						"financial",
						"pop",
						"inspiring",
						"lil",
						"complicated",
						"aa"
						],
			'gender': [ 
						"close",
						"love",
						"mention",
						"co",
						"wife",
						"lanka",	
						"believe",
						"video",
						"cute",
						"phone",
						"le",
						"day",
						"urban",
						"round",
						"thank",
						"bird",
						"wouldn",
						"aa"
						],
			'extroverted':[
							"co",
							"username",
							"million",
							"liked",
							"facebook",
							"last",
							"better",
							"de",
							"music",
							"around",
							"let",
							"book",
							"happy",
							"friends",
							"used",
							"inside",
							"really",
							"di",
							"work",
							"google",
							"opinion",
							"phd",
							"racist",
							"things",
							"forget",
							"via",
							"need",
							"nice",
							"http",
							"application",
							"slides",
							"sign",
							"sun",
							"sell",
							"years",
							"latest",
							"starbucks",
							"jullie",
							"interesante",
							"minute",
							"screen",
							"model",
							"shirt",
							"ziglar",
							], 
			'stable':[
							"like",
							"re",
							"god",
							"computer",
							"cause",
							"android",
							"follow",
							"waiting",
							"well",
							"school",
							"ever",
							"rock",
							"part",
							"photo",
							"want",
							"years",
							"mind",
							"need",
							"bring",
							"original",
							"says",
							"back",
							"colleagues",
							"last",
							"finally",
							"bu",
							"according",
							"experience",
							"work",
							"real",
							"sour",
							"sometimes",
							"many",
							"savigny",
							"play",
							"st",
							"silly",
							"similar",
							"birthday",
							"dz",
							"holds",
							"today",
							"gerrard",
							"middle",
							"song",
							"ve"			
			], 
			'agreeable':[
							"https",
							"birthday",
							"made",
							"google",
							"important",
							"need",
							"church",
							"oh",
							"haha",
							"early",
							"hearts",
							"personal",
							"one",
							"eat",
							"girl",
							"go",
							"mo",
							"ly",
							"facebook",
							"amazing",
							"keeping",
							"speak",
							"iv",
							"secret",
							"room",
							"fate",
							"sit",
							"married",
							"background",
							"sharedleadership",
							"ward",
							"anyone",
							"dream",
							"succes",
							"needs",
							"views",
							"annoyed",
							"habit",
							"walk"			
			], 
			'open':[
							"love",
							"time",
							"years",
							"http",
							"goes",
							"dreams",
							"birthday",
							"high",
							"win",
							"world",
							"wanna",
							"digital",
							"replies",
							"would",
							"women",
							"ready",
							"get",
							"wall",
							"point",
							"lot",
							"project",
							"mean",
							"meet",
							"right",
							"people",
							"page",
							"season",
							"bit",
							"fall",
							"qenbj",
							"er",
							"looks",
							"year",
							"go",
							"want",
							"midnight",
							"username",
							"attention",
							"cold",
							"like",
							"little",
							"psd",			
			], 
			'conscientious':[
							"awesome",
							"party",
							"maybe",
							"crazy",
							"ff",
							"using",
							"thanks",
							"little",
							"new",
							"could",
							"tears",
							"long",
							"thirty",
							"saying",
							"system",
							"find",
							"wtf",
							"one",
							"someone",
							"reason",
							"john",
							"lasting",
							"re",
							"five",
							"reat",
							"http",
							"via",
							"thrones",
							"words",
							"furious",
							"sjgy",
							"bout",
							"thank",
							"mini",
							"qw",
							"central",
							"looks",
							"playing"			
			]
			},
'dutch':{
			'age': [
						"zit",
						"heel",
						"best",
						"geeft",
						"idee",
						"nooit",
						"weer",
						"binnen",
						"goed",
						"avond",
						"bijwerken",
						"dag",
						"laatste",
						"man",
						"voelt",
						"hart",
						"toekomst",
						"boeit",
						"dh",
						"feestje",
						"ging",
						"meisje",
						"morgen",
						"muzikanten",
						"onderweg",
						"onderzoeksjournalistiek",
						"onzin",
						"proficiat",
						"ten",
						"verdient",
						"verzuurde",
						"werkt"			
			],
			'gender':[
						"username",
						"goed",
						"bent",
						"saai",			
					],
			'extroverted':[
							"dingen",
							"blijft",
							"bijna",
							"mr",
							"zeker",
							"vallen",
							"doet",
							"xkwktrd",
							"zoek"			
			], 
			'stable':[
							"username",
							"snel",
							"misschien",
							"ergens",
							"blijft",
							"namelijk",
							"jaar",
							"vrijdag",
							"terwijl",
							"hashtag",
							"interviewee",			
			], 
			'agreeable':[
							"rt",
							"terug",
							"snel",
							"bedankt",
							"smh",
							"terwijl",
							"the",
							"heerlijk",
							"hallo"			
			], 
			'open':[
							"hahaha",
							"week",
							"tijd",
							"username",
							"we",
							"kaviaarbehandeling",
							"jeeeej",
							"can"			
			], 
			'conscientious':[
							"mag",
							"fietsen",
							"mn",
							"dacht",
							"zet",
							"moddermanstraat"			
			]
			},
'italian':{
			'age':[
						"domani",
						"fa",
						"poi",
						"pezzo",
						"immagini",
						"quel",
						"ultimo",
						"binari",
						"bravo",
						"foto",
						"is",
						"sentito",
						"stato",
						"pi",
						"seguire",
						"borgo",
						"elected",
						"federico",
						"riusciamo",
						"super",
						"tassoni",
						"agendadigitale",
						"casalinga",
						"cc",
						"de",
						"dio",
						"eccomi",
						"esempio",
						"novit",
						"oscena",
						"pard",
						"piazza",
						"preso",
						"pu",
						"rispetto",
						"yg"		
				],
			'gender':[
						"co",
						"campagna",
						"ottimo",
						"conoscessi",
						"voci"			
					],
			'extroverted':[
							"design",
							"hotel",
							"ore",
							"dopo",
							"oppure",
							"ariosto",
							"scaccia",
							"son",
							"date"			
			], 
			'stable':[
							"co",
							"design",
							"sostenibile",
							"andare",
							"me",
							"esempio",
							"at",
							"buone",
							"semplicissima",
							"incapace",
							"tv"			
			], 
			'agreeable':[
							"bologna",
							"via",
							"twitter",
							"style",
							"co",
							"sento",
							"monti",
							"disegni"			
			], 
			'open':[
							"qualcosa",
							"anni",
							"bel",
							"ricerca",
							"sangue",
							"zagaria",
							"sento",
							"striati"			
			], 
			'conscientious':[
							"design",
							"ore",
							"username",
							"anni",
							"sembra",
							"oppure",
							"massimo",
							"purtroppo",
							"confermo"			
			]			
			},
'spanish':{
			'age':[
						"http",
						"ma",
						"dijo",
						"momento",
						"cil",
						"as",
						"buenos",
						"mala",
						"bieber",
						"falta",
						"buscan",
						"facebook",
						"info",
						"todas",
						"favor",
						"cula",
						"nom",
						"ofpbmahc"		
					],
			'gender':[
						"vida",
						"alguien",
						"corrupci",
						"ciudades",
						"si",
						"temprano",
						"puro",
						"meta",
						"foto",
						"dio"			
						],
			'extroverted':[
						"xico",
						"alguien",
						"escribir",
						"tambi",
						"nueva",
						"pe",
						"gusto",
						"http",
						"comen",
						"mujeres",
						"fico",
						"toda",
						"quiero",
						"sue",
						"aunque",
						"ahora",
						"chistes",
						"mano",
						"ser",
						"luz",
						"verdad",
						"dar",
						"hoy",
						"cticas",
						"che",
						"suicidio",
						"portugal",
						"recuerdo",
						"responsabilidad",			
			], 
			'stable':[
						"amigos",
						"is",
						"quiero",
						"ja",
						"despertar",
						"noches",
						"buenos",
						"ah",
						"mayor",
						"quieres",
						"bado",
						"iphone",
						"est",
						"culo",
						"sesi",
						"cient",
						"pel",
						"you",
						"sab",
						"internet",
						"torno",
						"tardando",
						"podemos",
						"tampoco",
						"nnjutigybf",
						"corriendo",
						"va",
						"acompa",
						"hacer",
						"papaya",
						"vas",
						"bonitas",			
			], 
			'agreeable':[
						"sabes",
						"cc",
						"dif",
						"quedan",
						"username",
						"despedida",
						"estudiar",
						"vez",
						"pesar",
						"vamos",
						"esperar",
						"tambi",
						"solo",
						"sociales",
						"hacen",
						"luego",
						"ngelamaria",
						"fin",
						"acordaba",
						"terror",
						"ja",
						"bellas",
						"firmad",
						"fr",			
			], 
			'open':[
						"puta",
						"jajaja",
						"interesante",
						"luego",
						"espa",
						"esperar",
						"dia",
						"acuerdo",
						"grande",
						"ma",
						"amigo",
						"siempre",
						"sonrisa",
						"haber",
						"pista",
						"buenos",
						"penlties",
						"aburrida",
						"burra",
						"venes",
						"pelotita",
						"crisis",
						"youtube",
						"social",
						"hombres",
						"plana",
						"serie",			
			], 
			'conscientious':[
						"siempre",
						"fer",
						"cc",
						"rtela",
						"tico",
						"corrupci",
						"solo",
						"momento",
						"mundo",
						"mal",
						"empleo",
						"do",
						"pone",
						"va",
						"transici",
						"veces",
						"pa",
						"escuchar",
						"mayor",
						"meses",
						"puede",
						"ciento",
						"andar",
						"article",
						"gt",
						"moralmente",
						"preguntar",
						"online"			
			]
			}
}


stylistic_features = [ 
				"#",
				"@username",
				"http://",
				":)",
				";)",
				"o_O",
				"!",
				"!!",
				"!!!",
				":("
				]
				
def trainOne(X,y,task):
	n_folds = 10
# 	import pdb; pdb.set_trace()
	if task in ['gender', 'age']:
		clf = svm.SVC(kernel='linear', C=1)
		clf.fit(X,y)		
# 		scores = cross_validation.cross_val_score( clf, X, y, cv=n_folds)
	else:
		clf = svm.SVR(kernel='linear', C=1)
		clf.fit(X,y)		
# 		scores = cross_validation.cross_val_score( clf, X, y, cv=n_folds)
	return clf
	
				
def getCount(onePattern, inputString):
	return inputString.count(onePattern)

def review_to_words(raw_review, language):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words(language))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


def writeModels(models, outputDir):
	fileName = outputDir + "/models.pkl"
	f = open(fileName, 'wb')
	pickle.dump(models, f)
	f.close()
# 	pickle.dump( favorite_color, open( fileName, "wb" ) )

def getFeatureVecFromTFIDF(fileName, lang):
	n_words = 10000
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	num_text = train["text"].size

	clean_train_reviews = []
	print "Looping through all text.\n" 

	for i in xrange( 0, num_text):
		clean_train_reviews.append( review_to_words( train["text"][i], lang ) )

	vectorizer = TfidfVectorizer(analyzer = "word",\
								 tokenizer = None,      \
								 preprocessor = None,   \
								 stop_words = None,     \
								 max_features = n_words) 

	X = vectorizer.fit_transform(clean_train_reviews)
	X = X.toarray()
	return X, vectorizer
	
def	getFeatureVecFromFunctionWords(fileName, test_patterns):
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	train_reviews = train["text"]
	
	X = []
	for line in train_reviews:
		vector_for_one_entry = []
		for pattern in test_patterns:
			count = getCount(pattern, line)
			vector_for_one_entry.append(count)
		X.append(vector_for_one_entry)
	X = np.array(X)
	return X

	
def	getFeatureVecFromStylisticFeatures(fileName, stylistic_features):
	return getFeatureVecFromFunctionWords(fileName, stylistic_features)

def	getFeatureVecFromPOS(fileName, lang, n_gram_range):
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	num_text = train["text"].size

	clean_train_reviews = []
# 	print "Looping through all text.\n" 

	for i in xrange( 0, num_text):
		clean_train_reviews.append( review_to_words( train["text"][i], lang ) )
	
	tt = TreeTagger(encoding='latin-1',language=lang)
	train_reviews_pos_tags = []
	
	for line in clean_train_reviews:
		a = tt.tag(line)
		a = [col[1] for col in a]
		pos_line = " ".join(a)
		train_reviews_pos_tags.append(pos_line)

	ngram_vectorizer = CountVectorizer(ngram_range=n_gram_range, min_df=1)
	X = ngram_vectorizer.fit_transform(train_reviews_pos_tags).toarray()
	return X, ngram_vectorizer


def getDescriptorsForOne(outputFilename, lang, task):
	X1, vectorizer = getFeatureVecFromTFIDF(outputFilename, lang)
	X2 = getFeatureVecFromFunctionWords(outputFilename, function_words_dict[lang][task])
	X3 = getFeatureVecFromStylisticFeatures(outputFilename, stylistic_features)
# 	X4 = getFeatureVecFromPOS(outputFilename, lang, (1,1))
# 	X5 = getFeatureVecFromPOS(outputFilename, lang, (1,2))
# 	X = np.concatenate((X1,X2,X3,X4,X5), axis=1)	
	X = np.concatenate((X1,X2,X3), axis=1)	

	return X, vectorizer

def saveDescriptors(fileName,descriptors):
	np.savetxt(fileName, descriptors, delimiter=",", fmt="%s")
	
def dirExists(inputDir):
	if os.path.exists(inputDir):
		return True
	elif os.access(os.path.dirname(inputDir), os.W_OK):
		print "Cannot access the directory. Check for privileges."
		return False
	else:
		print "Directory does not exist."
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

def isTruthTextFile(f):
	return 'truth.txt' in f
	
def getTruthTextFiles(allPaths):
	return [f for f in allPaths if isTruthTextFile(f)]

def getRelevantDirectories(argv):
   inputDir = ''
   outputDir = ''
   modelDir = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
      print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
         print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputDir = arg
      elif opt in ("-o", "--ofile"):
         outputDir = arg   
   return inputDir, outputDir

def tsv_writer(data, path):
    """
    Write data to a TSV file path
    """
    with open(path, "a") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(data)


def writeOneSummary(outputFilename, f, allPaths):
	data = ["filename", "gender", \
			"age", "extroverted", \
			"stable", "agreeable", \
			"open", "conscientious", \
			"text"]
	path = outputFilename.strip().split("/")
	outputFilename = path[-1]
	print "Output filename: ",outputFilename

	path = '/'.join(path[0:-1])

	tsv_writer(data, outputFilename)
	gender = {'M': 0, 'F':1}
	ageGroup = {'18-24': 0, \
				'25-34': 1, \
				'35-49': 2, \
				'50-XX': 3, \
				'50-64': 3, \
				'XX-XX': None}
	file = open(f, 'r')
	
	for line in file:
		a = line.strip().split(":::")
		fileName 		  = path+ "/" + a[0] + ".xml"
# 		print fileName
		thisGender 	 	  = gender[a[1]]
		thisAgeGroup 	  = ageGroup[a[2]]
		thisExtroverted   = float(a[3])
		thisStable 		  = float(a[4])
		thisAgreeable	  = float(a[5])
		thisOpen		  = float(a[6])
		thisConscientious = float(a[7])
		
# 		print "%s %d %d %f %f %f %f %f" % (fileName, thisGender, thisAgeGroup, thisExtroverted, thisStable, thisAgreeable, thisOpen, thisConscientious)
		
		try:
			tree = ET.parse(fileName)
# 			print "Filename: %s SUCCESS!" % fileName
		
		except:
			e = sys.exc_info()[0]
			print "Filename: %s Error: %s" % (fileName, e)
		else:
# 			print "In Else"
			allDocs = tree.getroot().findall("document")
 			allText = ""
			
# 			print "Going in for loop"
			for doc in allDocs:
				clean = bleach.clean(doc.text, tags=[], strip=True)
 				allText = allText + clean
	 		
			# 	clean = clean.encode('utf-8')
			allText = allText.encode('utf-8')
# 			print "Out of loop, writing"								
			data = [fileName, thisGender, thisAgeGroup, thisExtroverted, \
					thisStable, thisAgreeable, thisOpen, thisConscientious, \
					allText]
			tsv_writer(data, outputFilename)
# 			print "Finish writing one line"

def getTarget(filename, task):
	train = pd.read_csv(filename, header=0, delimiter="\t", quoting=1)
	y = train[task]
	
	if task in ['age', 'gender']:
		le = preprocessing.LabelEncoder()
		le.fit(list(set(y)))
		y = le.transform(y)
	y = np.array(y)
	return y
	   
def main(argv):
# 	print "Getting relevant directories"
	inputDir, outputDir = getRelevantDirectories(argv)


# 	print "Getting all filenames with absolute paths"
	allPaths = getAllFilenamesWithAbsPath(inputDir)


# 	print "Getting all truth files"
	allTruthText = getTruthTextFiles(allPaths)



	tasks = ["gender", "age", "extroverted", "stable", "agreeable", "open", "conscientious"]
	langs = ["english", "dutch", "spanish", "italian"]

# 	tasks = ["gender"]
# 	langs = ["english", "dutch"]
	
	tempLangs = []
	summaryFiles = []
	
	for f in allTruthText:
		a = f.strip().split("/")
		lang = [ lang for lang in langs if lang in f]
		print "Processing: ", lang[0]
# 		import pdb; pdb.set_trace()
		outputFilename = '/'.join(a[0:-1]) + '/summary-' + lang[0] + '-' + a[-1]
# 		print "Output filename: ", outputFilename
		print "Writing one summary"
		writeOneSummary(outputFilename, f, allPaths)
		
		oneSummaryFile = 'summary-' + lang[0] + '-' + a[-1]
		
		tempLangs.append(lang[0])
		summaryFiles.append(oneSummaryFile)
	
	descriptorFilenames = []
	models = []

	for lang, outputFilename in zip(tempLangs, summaryFiles):
		X1, vec			= getFeatureVecFromTFIDF(outputFilename, lang)
		X2      	  	= getFeatureVecFromStylisticFeatures(outputFilename, stylistic_features)
# 		X3, uni_vec   	= getFeatureVecFromPOS(outputFilename, lang, (1,1))
# 		X4, bi_vec    	= getFeatureVecFromPOS(outputFilename, lang, (1,2))
		
		fileName = outputDir + "/tfidf-"+lang+"-models.pkl"
		f = open(fileName, 'wb')
		joblib.dump(vec, fileName)
		f.close()
		
		
		models.append({'tfidf_vectorizer': {lang : vec}})
# 		models.append({'unigram_vectorizer':{lang: uni_vec}})
# 		models.append({'bigram_vectorizer':{lang: bi_vec}})
		
		for task in tasks:
			X5 = getFeatureVecFromFunctionWords(outputFilename, function_words_dict[lang][task])
# 			descriptors = np.concatenate((X1,X2,X3,X4,X5), axis=1)	
			descriptors = np.concatenate((X1,X2,X5), axis=1)	
			print lang, task, descriptors.shape
# 			test_x1_descriptor = vec.transform(["Hello world."])
# 			import pdb; pdb.set_trace()
# 			X1 size n x 8509			
# 			descriptor size = 69 x 8523
			filename = lang + "_" + task + "_descriptors.csv"
			print filename
			descriptorFilenames.append({lang:{task: filename}})
			saveDescriptors(filename, descriptors)

	
	lang_to_file_correspondence = dict(zip(tempLangs, summaryFiles))
	for file in descriptorFilenames:
		model_for_one = {}
		language = file.keys()[0]
		task = file.values()[0].keys()[0]
		filename = file.values()[0].values()[0]
		
		print filename
		X = pd.read_csv(filename, header=None, delimiter=",", quoting=1)
# 		import pdb; pdb.set_trace()
		
		outputFilename = lang_to_file_correspondence[language]
		
		
# 		import pdb; pdb.set_trace()
		if (language in ['dutch', 'italian']) and (task in ['age']):
			models.append({language: {task: []}})
		else:
			y = getTarget(outputFilename, task)
			one_clf = trainOne(X,y,task)
			models.append({language: {task: one_clf}})
	
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)			
	writeModels(models, outputDir)

if __name__ == "__main__":
   main(sys.argv[1:])