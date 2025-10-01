import os
import operator
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from math import log10, sqrt

words = {}
dfs = {}
idf = {}
stemmer = PorterStemmer()
corpusroot = './US_Inaugural_Addresses'

def getidf(token):
    token = stemmer.stem(token)
    if token not in idf:
        return -1
    else:
        return idf[token]
    
def getweight(filename, token):
    token = stemmer.stem(token)
    if filename not in words or token not in words[filename]:
        return 0
    else:
        return words[filename][token]

# Function to calculate weight following ltc (logarithmic tf, logarithmic idf, cosine normalization) weighting scheme
def calculateWeight():
    for filename, tfVector in words.items():
        veclen = 0.0
        for token, tfValue in tfVector.items():
            tfValue = (1 + log10(tfValue)) * idf[token]
            tfVector[token] = tfValue
            veclen += pow(tfValue, 2)
        
        veclen = sqrt(veclen)
        if veclen > 0:
            for token in tfVector:
                tfVector[token] = tfVector[token] / veclen
    return words

# Function to calculate dfs and idf
def calculateIdfs():
    for filename, tfVector in words.items():
        for token in tfVector:
            if token not in dfs:
                dfs[token] = 1
            else:
                dfs[token] += 1

    for token, df in dfs.items():
        idf[token] = log10(len(words) / df)
    return idf

# Function to calculate query weight following lnc (logarithmic tf, no idf, cosine normalization) weighting scheme
def calculateQueryWeight(queryVector):
    vectorLength = 0.0
    for token, tfValue in queryVector.items():
        tfValue = 1 + log10(tfValue)
        queryVector[token] = tfValue
        vectorLength += pow(tfValue, 2)
    
    vectorLength = sqrt(vectorLength)
    if vectorLength > 0:
        for token in queryVector:
            queryVector[token] = queryVector[token] / vectorLength
    return queryVector

def readfiles(corpusroot):
    for filename in os.listdir(corpusroot):
        try:
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                doc = file.read().lower()
                tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
                tokens = tokenizer.tokenize(doc)
                stopWords = set(stopwords.words('english'))
                stemmedTokens = [stemmer.stem(token) for token in tokens if token not in stopWords]
                tfVector = Counter(stemmedTokens)
                words[filename] = tfVector
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return words

def query(qstring):
    queryDoc = qstring.lower()
    tokenizer = RegexpTokenizer(r'[a-z]+')
    tokens = tokenizer.tokenize(queryDoc)
    stopWords = set(stopwords.words('english'))
    queryTokens = [stemmer.stem(token) for token in tokens if token not in stopWords]
    queryVector = Counter(queryTokens)
    queryTfidf = calculateQueryWeight(queryVector)

    scores = {}
    for token in queryTfidf:
        for filename, tfidfVector in words.items():
            if token in tfidfVector:
                if filename not in scores:
                    scores[filename] = 0
                scores[filename] += tfidfVector[token] * queryTfidf[token]

    if not scores:
        return (None, 0)

    filename, score = max(scores.items(), key=operator.itemgetter(1))
    return (filename, score)

# Read files and process data
words = readfiles(corpusroot)
idf = calculateIdfs()
tfidfScores = calculateWeight()

# Test cases
print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))