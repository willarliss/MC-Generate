import os
import sys
import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from mc_upsample import MCUpsampling



def load_data(f='corpus.json'):
    
    with open(f, 'r') as file:
        corpus = json.load(file)
        
    pos = np.array([i[0] for i in corpus if i[1] == '1'])
    neg = np.array([i[0] for i in corpus if i[1] == '0'])
    
    ## POS IS THE MINORITY CLASS
    np.random.seed(42)
    # np.random.seed(None)
    pos = np.random.choice(pos, len(neg)//3)
        
    X, y = [], []
    for p in pos:
        X.append(p)
        y.append(1)
    for n in neg:
        X.append(n)
        y.append(0)
    
    return np.array(X), np.array(y)



def imbalanced(X_train, X_test, y_train, y_test):
        
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
        ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Imbalanced', accuracy_score(y_test, preds), recall_score(y_test, preds))



def markov_upsampling(X_train, X_test, y_train, y_test):
        
    full = np.c_[X_train, y_train]
    minority = np.array([i[0] for i in full if i[1] == '1'])
    
    mcu = MCUpsampling()
    mcu.build(minority)
    synthetic = mcu.generate(len(X_train)-(2*len(minority)))
    
    for s in synthetic:
        X_train = np.append(X_train, [s], axis=0)
        y_train = np.append(y_train, [1], axis=0)
    # print(np.unique(y_train, return_counts=True)[1])
    # print(len(y_train), len(X_train))
    
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
        ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('MCUpsampling', accuracy_score(y_test, preds), recall_score(y_test, preds))



def bootstrap_upsampling(X_train, X_test, y_train, y_test):
    
    full = np.c_[X_train, y_train]
    minority = np.array([i[0] for i in full if i[1] == '1'])
    
    bootstrapped = np.random.choice(minority, len(X_train)-(2*len(minority)), replace=True)
    
    for b in bootstrapped:
        X_train = np.append(X_train, [b], axis=0)
        y_train = np.append(y_train, [1], axis=0)
    # print(np.unique(y_train, return_counts=True)[1])
    # print(len(y_train), len(X_train))
    
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
        ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Bootstrap up', accuracy_score(y_test, preds), recall_score(y_test, preds))



def bootstrap_downsampling(X_train, X_test, y_train, y_test):
    
    full = np.c_[X_train, y_train]
    majority = np.array([i[0] for i in full if i[1] == '0'])
    
    bootstrapped = np.random.choice(majority, len(X_train)-len(majority), replace=True)
    
    for b in bootstrapped:
        X_train = np.append(X_train, [b], axis=0)
        y_train = np.append(y_train, [1], axis=0)
    # print(np.unique(y_train, return_counts=True)[1])
    # print(len(y_train), len(X_train))
    
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
        ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Bootstrap down', accuracy_score(y_test, preds), recall_score(y_test, preds))


def random_bag(X_train, X_test, y_train, y_test):
    
    full = np.c_[X_train, y_train]
    minority = np.array([i[0] for i in full if i[1] == '1'])
    
    bag = np.concatenate([i.split() for i in minority])
    min = np.min([len(i.split()) for i in minority]) + 1
    max = np.max([len(i.split()) for i in minority])
    
    for r in np.arange(len(X_train)-(2*(len(minority)))):
        length = np.random.randint(min, max)
        new = ' '.join(np.random.choice(bag, length))
        X_train = np.append(X_train, [new], axis=0)
        y_train = np.append(y_train, [1], axis=0)
        
    # print(np.unique(y_train, return_counts=True)[1])
    # print(len(y_train), len(X_train))
    
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
        ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Random bag', accuracy_score(y_test, preds), recall_score(y_test, preds))



def main():
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    imbalanced(X_train, X_test, y_train, y_test)
    markov_upsampling(X_train, X_test, y_train, y_test)
    bootstrap_upsampling(X_train, X_test, y_train, y_test)
    bootstrap_downsampling(X_train, X_test, y_train, y_test)
    random_bag(X_train, X_test, y_train, y_test)
    


if __name__ == '__main__':
    main()
    
    
    
