import pandas as pd
import numpy as np
#import progressbar
#bar = progressbar.ProgressBar()
#import spacy
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.feature_extraction.text import TfidfTransformer,HashingVectorizer
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
#from tqdm import tqdm
from os import path
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_selection import mutual_info_classif
import copy
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("/home/zhenzuo2/project-stat/text/clean_data.csv")
print("Reading data Done")

def feature_selection(df, length, minimum, maximum, ngram, n):
    df = df.loc[df.num_words > length]
    vectorizer = CountVectorizer(ngram_range=ngram, max_df = maximum, min_df = minimum)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(df.cleaned.astype('U'))
    Y = df.voted_up
    tfidf = transformer.fit_transform(X)
    total_tf_idf = tfidf.sum(axis = 0)
    selected_index = sorted(range(total_tf_idf.shape[1]), key=lambda k: total_tf_idf[0,k])[-n:]
    words = vectorizer.get_feature_names()
    selected_words = [words[i] for i in selected_index]
    tfidf = tfidf[:,selected_index]
    simple = copy.deepcopy(tfidf.toarray())
    extra = df[["playtime_forever", "playtime_last_two_weeks", 'num_games_owned','num_reviews','votes_up','votes_funny']]
    tfidf = np.append(tfidf.toarray(),extra.values.reshape(tfidf.shape[0],6), 1)
    word = [words[i] for i in selected_index]
    return Y,simple,tfidf,word

def train_model(Y, tfidf, word, w, model, length, minimum, maximum, ngram, n):
    X_train, X_test, y_train, y_test = train_test_split(tfidf, Y, test_size=0.25)
    if model == "tree":
        t = tree.DecisionTreeClassifier()
        t.fit(X_train, y_train)
        y_pred = t.predict(X_test)
        c =  confusion_matrix(y_test, y_pred)
    if model == "gnb":
        gnb = MultinomialNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        c = confusion_matrix(y_test,y_pred)
    if model == "svm":
        s = LinearSVC()
        s.fit(X_train, y_train)
        y_pred = s.predict(X_test)
        c =  confusion_matrix(y_test, y_pred)
    TN = c[0,0]
    FN = c[1,0]
    TP = c[1,1]
    FP = c[0,1]
    result = pd.DataFrame([[length, minimum, maximum, ngram, n, "Tfidf", model, w, w, w, w, w, w, TN, FN, TP, FP, word]])
    columns = ['Minimum_Length','Minimum','Maximum','ngram','N', 'Method', "Model", "playtime_forever", "playtime_last_two_weeks", 'num_games_owned','num_reviews','voted_up','votes_funny','TN', 'FN', 'TP', 'FP', "Words"]
    result.columns = columns
    return result

columns = ['Minimum_Length','Minimum','Maximum','ngram','N', 'Method', "Model", "playtime_forever", "playtime_last_two_weeks", 'num_games_owned','num_reviews','voted_up','votes_funny','TN', 'FN', 'TP', 'FP', "Words"]
output = pd.DataFrame(columns=columns)
i = 0
for l in [5,10]:
    for mi in [10,50]:
        for ma in [20000,100000]:
            for gram in [(1,1),(1,2),(1,3)]:
                for n in [100,500]:
                    Y,simple,tfidf,word = feature_selection(df, l, mi, ma, gram, n)
                    for model in ['tree','gnb']:
                        output = output.append(train_model(Y,simple,word, 0, model, l, mi, ma, gram, n))
                        print("tree done")
                        output = output.append(train_model(Y,tfidf,word, 1, model, l, mi, ma, gram, n))
                        print("gnb done")
                        output.to_csv("/home/zhenzuo2/project-stat/text/result2.csv", index=False)
                        print(i)
                        print("finished")
                        i = i + 1

def feature_selection2(df, length, minimum, maximum, ngram, n):
    df = df.loc[df.num_words > length]
    vectorizer = CountVectorizer(ngram_range=ngram, max_df = maximum, min_df = minimum)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(df.cleaned.astype('U'))
    Y = df.voted_up
    IG = []
    r_c = X.shape
    for i in range(r_c[1]):
        a = np.sum(X[:,i] != 0)
        b = sum(Y[(X[:,i] != 0).toarray().reshape(r_c[0])] == True)
        c = r_c[0] - a
        d = sum(Y[(X[:,i] == 0).toarray().reshape(r_c[0])] == True)
        if b == 0 or a == b:
            first = 0
        else:
            first = -b/a * np.log(b/a) - (a-b)/a * (np.log((a-b)/a))
        if d == 0 or c == d :
            second = 0
        else:
            second = -d/c * np.log(d/c) - (a-b)/a * (np.log((c-d)/c))
        IG = IG + [-(first*a/r_c[0] + second*c/r_c[0])]
    selected_index = IG.argsort()[-n:][::-1]
    words = vectorizer.get_feature_names()
    selected_words = [words[i] for i in selected_index]
    tfidf = tfidf[:,selected_index]
    simple = copy.deepcopy(tfidf.toarray())
    extra = df[["playtime_forever", "playtime_last_two_weeks", 'num_games_owned','num_reviews','votes_up','votes_funny']]
    tfidf = np.append(tfidf.toarray(),extra.values.reshape(tfidf.shape[0],6), 1)
    word = [words[i] for i in selected_index]
    return Y,simple,tfidf,word
