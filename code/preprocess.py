import pandas as pd
import gensim
import numpy as np
import nltk
import re
from nltk.corpus import stopwords as stp
from textblob import TextBlob



stop_words = set(stp.words('english'))
punctuations= ["\"","(",")","*",",","-","_",".","~","%","^","&","!","#",'@'
               "=","\'","\\","+","/",":","[","]","«","»","،","؛","?",".","…","$",
               "|","{","}","٫",";",">","<","1","2","3","4","5","6","7","8","9","0"]

def load_data(address):

    train = pd.read_csv('address',encoding='ISO-8859-1')

    ## Split into val and train set
    X_valid, Y_valid = X_train[-2000:], Y_train[-2000:]
    X_train, Y_train = X_train[:-2000], Y_train[:-2000]
    n=len(Y_train)

    return X_train,Y_train,X_valid,Y_valid

def load_google_vector():
    model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)
    return model

def tweet2v(list_words, model):
    num_features = 300
    #sentence_vec = np.zeros(num_features)
    sentence_vec = []
    if len(list_words)!=0:
        for word in list_words:
            if word in model:
                sentence_vec.append(model[word])
    return sentence_vec

def tweets2tokens(tweet_text,model):
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tweet_text.lower()))
    words=[]
    for token in tokens:
        if token.startswith( 'http' ):
            url=1
        else:
            url=0
            if  '@' not in token and token in model and token not in stop_words and token != "" and token not in punctuations:
            # if  '@' not in token and token not in stop_words and token != "" and token not in punctuations:
                words.append(token)
    return tokens,url

def tweet2features(tweet_text,model):

    tokens,url=tweets2tokens(tweet_text,model)

    features=(tweet2v(tokens,model)).tolist()
    list=punctuationanalysis(tweet_text)
    for item in list:
        features.append(item)
    features.append(negativewordcount(tokens))
    features.append(positivewordcount(tokens))
    features.append(capitalratio(tweet_text))
    features.append(contentlength(tokens))
    features.append(sentimentscore(tweet_text))
    list=poscount(tweet_text)
    for item in list:
        features.append(item)
    features.append(url)
    # print("features",features)
    return features

#punctuations
def punctuationanalysis(tweet_text):
    hasqmark =sum(c =='?' for c in tweet_text)
    hasemark =sum(c =='!' for c in tweet_text)
    hasperiod=sum(c =='.' for c in tweet_text)
    number_punct=sum(c in punctuations for c in tweet_text)
    return hasqmark,hasemark,hasperiod,number_punct

def negativewordcount(tokens):
    count = 0
    negativeFeel = ['tired', 'sick', 'bord', 'uninterested', 'nervous', 'stressed',
                     'afraid', 'scared', 'frightened', 'boring','bad',
                     'distress', 'uneasy', 'angry', 'annoyed', 'pissed',"hate",
                     'sad', 'bitter', 'down', 'depressed', 'unhappy','heartbroken','jealous']
    for negative in negativeFeel:
        if negative in tokens:
            count += 1
    return count

def positivewordcount(tokens):
    count = 0
    positivewords = ['joy', ' happy', 'hope', 'kind', 'surprise'
                     , 'excite', ' interest', 'admire',"delight","yummy",
                     'confidenc', 'good', 'satisf', 'pleasant',
                     'proud', 'amus', 'amazing', 'awesome',"love","passion","great","like","wow","delicious"]
    for pos in positivewords:
        if pos in tokens:
            count += 1
    return count

def capitalratio(tweet_text):
    uppers = [l for l in tweet_text if l.isupper()]
    capitalratio = len(uppers) / len(tweet_text)
    return capitalratio

def contentlength(words):
    wordcount = len(words)
    return wordcount

def sentimentscore(tweet_text):
    analysis = TextBlob(tweet_text)
    return analysis.sentiment.polarity

def poscount(tweet_text):
    postag = []
    poscount = {}
    poscount['Noun']=0
    poscount['Verb']=0
    poscount['Adjective'] = 0
    poscount['Pronoun']=0
    poscount['Adverb']=0
    Nouns = {'NN','NNS','NNP','NNPS'}
    Verbs={'VB','VBP','VBZ','VBN','VBG','VBD','To'}
    word_tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tweet_text))
    postag = nltk.pos_tag(word_tokens)
    for g1 in postag:
     if g1[1] in Nouns:
        poscount['Noun'] += 1
     elif g1[1] in Verbs:
         poscount['Verb']+= 1
     elif g1[1]=='ADJ'or g1[1]=='JJ':
         poscount['Adjective']+=1
     elif g1[1]=='PRP' or g1[1]=='PRON':
         poscount['Pronoun']+=1
     elif g1[1]=='ADV':
         poscount['Adverb']+=1
    return poscount.values()

def main():
    x_train , y_train,x_test, y_test =load_data()


    model = load_google_vector()
    print("Vectors are loaded!")

    tweet_vectors_train= np.zeros((len(y_train),315))
    i=0
    for tweet_text in x_train:
        tweet_vectors_train[i,:]=tweet2features(tweet_text,model)
        i+=1
    #
    # print("tweet vectors",tweet_vectors)

    tweet_vectors_test= np.zeros((len(y_test),315))
    i=0
    for tweet_text in x_test:
        tweet_vectors_test[i,:]=tweet2features(tweet_text,model)
        i+=1

    return tweet_vectors_train,y_train ,tweet_vectors_test,y_test

def load_train_data():
    

def load_test_data():
    

X_train, Y_train, X_test,Y_test= main()


np.save("f_trainx",X_train)
np.save("f_trainy",Y_train)
np.save("f_testx",X_test)
np.save("f_testy",Y_test)