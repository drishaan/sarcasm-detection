import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# on should be one of {"f1", "acc", "roc"}
def run_models(data, gram=2, on="f1"):
    # Parameters to iterate over
    stopwords = [True, False]
    gram = [g+1 for g in range(gram)]
    
    # Split out the y column
    y = data.is_sarcastic
    
    # Create variables to track best params
    top_score = 0
    top_params = None
    
    print("****** Running Models *******")
    for s in stopwords:
        
        for g in gram:
            # Get ngrams
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
            clf = LogisticRegression(random_state=487, max_iter=1000)
            clf.fit(trainX, trainy)
            
            if s:
                print(f"Using gram = {g} and stopwords:")
            else:
                print(f"Using gram = {g} and no stopwords:")
            
            f1 = f1_score(valy, clf.predict(valX))
            print(f"f1_score: {f1}")
            
            acc = accuracy_score(valy, clf.predict(valX))
            print(f"accuracy_score: {acc}")
            
            roc = roc_auc_score(valy, clf.predict_proba(valX)[:, 1])
            print(f"roc_auc_score: {roc}")
            
            print("________________________")
            
            # Vary output depending on which metric is being tested for
            if on == "f1":
                if top_score < f1:
                    top_score = f1
                    top_params = {"stopwords": s, "gram_size": g}
            elif on == "acc":
                if top_score < acc:
                    top_score = acc
                    top_params = {"stopwords": s, "gram_size": g}
            else:
                if top_score < roc:
                    top_score = roc
                    top_params = {"stopwords": s, "gram_size": g}            
    
    # Print best parameters based on metric choosen                
    print(f"Best params based on {on} was {top_params} with a score of {top_score}")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            