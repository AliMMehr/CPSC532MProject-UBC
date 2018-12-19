# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
import nltk
import re
from nltk.corpus import stopwords as stp
from textblob import TextBlob
import multiprocessing
from multiprocessing import Process
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../clean_data/"))
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(os.listdir("../"))

stop_words = set(stp.words('english'))
punctuations= ["\"","(",")","*",",","-","_",".","~","%","^","&","!","#",'@'
               "=","\'","\\","+","/",":","[","]","«","»","،","؛","?",".","…","$",
               "|","{","}","٫",";",">","<","1","2","3","4","5","6","7","8","9","0"]

def load_data(filename):

    data = pd.read_csv('../input/%s' % filename  #, encoding='ISO-8859-1'
                        , engine="python")

    return data

def load_google_vector():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin.gz',binary=True)
    return model

def tweet2v(list_words, model):
    sentence_vec = []
    if len(list_words)!=0:
        for word in list_words:
            if word in model:
                sentence_vec.append(model[word].tolist())
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

def tweet_text2features(tweet_text,model):
    tokens,url=tweets2tokens(tweet_text,model)
    
    features=[]
    
    sentence_vec=tweet2v(tokens,model)
    list1=punctuationanalysis(tweet_text)
    for item in list1:
        features.append(item)
    features.append(negativewordcount(tokens))
    features.append(positivewordcount(tokens))
    features.append(capitalratio(tweet_text))
    features.append(contentlength(tokens))
    features.append(sentimentscore(tweet_text))
    list1=poscount(tweet_text)
    for item in list1:
        features.append(item)
    features.append(url)
    qfeatures={'word_vectors':sentence_vec,'additional_features':features}
    return qfeatures

def batch_of_items2json_files(q_batch,model,batch_number,run_id):
    print('starting run:%s batch:%s' % (run_id,batch_number))
    batch_clean_data={}
    for index, sample in q_batch.iterrows():
        tweet_text=sample['question_text']
        qid=sample['qid']
        target=sample['target']
        qfeatures=tweet_text2features(tweet_text,model)
        #print(qfeatures)
        batch_clean_data[qid]={'qfeatures':qfeatures,'target':target}
    
    with open('../clean_data/%s-%s.json' % (run_id,batch_number), 'w') as fp:
        json.dump(batch_clean_data, fp)
        print('Done batch %s'% batch_number)

#punctuations
def punctuationanalysis(tweet_text):
    hasqmark =sum(c =='?' for c in tweet_text)
    hasemark =sum(c =='!' for c in tweet_text)
    hasperiod=sum(c =='.' for c in tweet_text)
    hasstar=sum(c =='*' for c in tweet_text)
    number_punct=sum(c in punctuations for c in tweet_text)
    return hasqmark,hasemark,hasperiod,hasstar,number_punct

def negativewordcount(tokens):
    count = 0
    negativeFeel = ['dick','penis','god']
    for negative in negativeFeel:
        if negative in tokens:
            count += 1
    return count

def positivewordcount(tokens):
    count = 0
    positivewords = []
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

def store_features_for_data(model,data,run_id):
    batch_size=3
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    
    i=0
    processes=[]
    for batch in chunker(data,batch_size):
        batch_of_items2json_files(batch,model,i,run_id)
        #p=Process(target=batch_of_items2json_iles,args=(batch,model,i,run_id))
        #p.start()
        #processes.append(p)
        if i>2:
            break
        i+=1
    for p in processes:
        p.join()

def load_test_data():
    pass


data =load_data('train.csv')
print(data.columns)
print(data.head())
print(data.describe())


model = load_google_vector()
print("load_google_vector loaded!")

store_features_for_data(model,data,'ali')