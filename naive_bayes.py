import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def run_models(data):
    stopwords = [True, False]
    gram = [1, 2]
    
    data = pd.read_csv("./data/cleaned_headlines.csv")
    
    # Split out the y column
    y = data.is_sarcastic
    
    print("****** Running Models *******")
    for s in stopwords:
        
        for g in gram:
            
            vectorizer = CountVectorizer(ngram_range = (1, g))
            
            # Vectorize the input column
            if s:
                X = vectorizer.fit_transform(data.tokenized)
            else:
                X = vectorizer.fit_transform(data.tokenized_no_stopwords)

                
                
            # Split the data into train, validation, and testing
            trainX, valX, trainy, valy = train_test_split(X, y, train_size=0.8, random_state=487)
            valX, testX, valy, testy = train_test_split(valX, valy, train_size=0.5, random_state=487)
            
            
            # Train the Naive Bayes classifier
            clf = MultinomialNB()
            clf.fit(trainX, trainy)
            
            if s:
                print(f"Using gram = {g} and stopwords:")
            else:
                print(f"Using gram = {g} and no stopwords:")
            
            print(f"f1_score: {f1_score(valy, clf.predict(valX))}")
            
            print(f"accuracy_score: {accuracy_score(valy, clf.predict(valX))}")
            
            print(f"roc_auc_score: {roc_auc_score(valy, clf.predict_proba(valX)[:, 1])}")
            
            print("________________________")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            