from numpy import *


def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', \
					'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', \
					'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', \
					'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how',\
					'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec


def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:print("the word:  %s is not in my Vocabulary! " % word)
	return returnVec
	
	
	
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = zeros(numWords); p1Num = zeros(numWords)
	p0Denom = 0.0; plDenom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			plNum += trainMatrix[i]
			plDenom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = plNum/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive




#计算出现频率
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),\
				reverse=True)
	return sortedFreq[:30]
#计算出现频率



#
def localWords(feed1,feed0):
	import feedparser
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList, fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList|remove(pairW[0])
	trainingSet = range(2*minLen); testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != \
			classList[docIndex]:
			errorCount += 1
	print('the error rate is: ',float(errorCount)/len(testSet)
	return vocabList,p0V,p1V
#