#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:36:20 2019

@author: salihemredevrim
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score,  precision_score, recall_score, f1_score
import nltk
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

#%%
#take dataset
data_all = pd.read_excel('MM_last.xlsx')
data_all['text'] = data_all['text'].astype(str)
#data_all['processed'] = data_all['processed'].astype(str)
data_all['Corpus'] = data_all['Corpus'].astype(str)

va = pd.crosstab(index = data_all['VERBAL ABUSE'], columns="Total count")
nva = pd.crosstab(index = data_all['NON-VERBAL ABUSE'], columns="Total count")
pa = pd.crosstab(index = data_all['PHYSICAL ABUSE'], columns="Total count")
spa = pd.crosstab(index = data_all['SERIOUS PHYSICAL ABUSE'], columns="Total count")
oa = pd.crosstab(index = data_all['OTHER ABUSE'], columns="Total count")

#all are unbalanced, we can make it balanced for each target below function
#spa is really low that's why serious physical abuse and physical abuse are merged

data_all['PHYSICAL ABUSE'] = data_all.apply(lambda x: 1 if (x['PHYSICAL ABUSE'] + x['SERIOUS PHYSICAL ABUSE']) > 0 else 0, axis=1) 
pa = pd.crosstab(index = data_all['PHYSICAL ABUSE'], columns="Total count")


#%%
del va, nva, pa, spa, oa

#%%

def data_prep(data1, target, text, test_percent):

    #Data preparation     
    data11 = data1[data1[target] == 1]
    data22 = data1[data1[target] == 0]
    
    #Balance 
    min1 = min(len(data11), len(data22));    
    data11 = data11.sample(n=min1, random_state=1905)
    data22 = data22.sample(n=min1, random_state=1905)
    
    data_all = data11.append(data22, ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(data_all[text], data_all[target], stratify=data_all[target], test_size=test_percent, random_state=1905)
   
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_test = pd.DataFrame(y_test).reset_index(drop=True)
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

#%%
#Functions for Doc2Vec 
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens  

#%%
def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

#%%
def log_reg_and_svm(XX_train, YY_train, XX_test, YY_test):     
    
    #Logistic Regression
    model1 = LogisticRegression()
    
    model1.fit(XX_train, YY_train)

    #Predict on test set
    predictions = model1.predict(XX_test)

    #score
    roc1 = roc_auc_score(YY_test, predictions)
    accuracy1 = accuracy_score(YY_test, predictions)
    precision1 = precision_score(YY_test, predictions)
    recall1 = recall_score(YY_test, predictions)
    f1_score1 = f1_score(YY_test, predictions)
    
    #F1 = 2 * (precision * recall) / (precision + recall)

    #SVM
    model2 = SVC()
    
    model2.fit(XX_train, YY_train)

    #Predict on test set
    predictions2 = model2.predict(XX_test)

    #score
    roc2 = roc_auc_score(YY_test, predictions2)
    accuracy2 = accuracy_score(YY_test, predictions2)
    precision2 = precision_score(YY_test, predictions2)
    recall2 = recall_score(YY_test, predictions2)
    f1_score2 = f1_score(YY_test, predictions2)
       
    output = {
         'Accuracy LR:': accuracy1,  
         'ROC LR:': roc1,    
         'Precision LR:': precision1,    
         'Recall LR:': recall1,    
         'F1-score LR:': f1_score1,
         
        'Accuracy SVM:': accuracy2,  
         'ROC SVM:': roc2,    
         'Precision SVM:': precision2,    
         'Recall SVM:': recall2,    
         'F1-score SVM:': f1_score2}
    
    return output

#%%
    
def all_techniques(data1, target, text, test_percent, min_df, max_df, ngram_range1, ngram_range2, vector_size1, window1, negative1, min_count1): 

    #Doc2vec parameters: vector_size, window, negative, min_count
    
    #train - test split 
    X_train, X_test, y_train, y_test = data_prep(data1, target, text, test_percent); 
    
    #Simple BOW ********************************************************************************************************
    
    #min_df is used for removing terms that appear too infrequently. For example:
    #min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
    #min_df = 5 means "ignore terms that appear in less than 5 documents".
    #max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
    #max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
    #max_df = 25 means "ignore terms that appear in more than 25 documents".

    #WITH PLAIN LYRICS
    TEXT = text;
    
    #tokenization
    vect = CountVectorizer(max_df=max_df, min_df=min_df).fit(X_train[TEXT])

    #transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train[TEXT])
    X_test_vectorized = vect.transform(X_test[TEXT])

    #models with only lyrics
    output1_bow_plain_lyrics = log_reg_and_svm(X_train_vectorized, y_train, X_test_vectorized, y_test);

    #Tf-Idf ******************************************************************************************************************

    vect_tf = TfidfVectorizer(min_df= min_df, max_df= max_df).fit(X_train[TEXT])

    #transform
    X_train_vectorized_tf = vect_tf.transform(X_train[TEXT])
    X_test_vectorized_tf = vect_tf.transform(X_test[TEXT])
    
    #models with only lyrics
    output3_tfidf_plain_lyrics = log_reg_and_svm(X_train_vectorized_tf, y_train, X_test_vectorized_tf, y_test);
        
    #N-grams ******************************************************************************************************************
    
    #document frequency of 5 and extracting 1-grams and 2-grams...
    vect3 = CountVectorizer(min_df=min_df, ngram_range=(ngram_range1, ngram_range2)).fit(X_train[TEXT])
    
    X_train_vectorized_ng = vect3.transform(X_train[TEXT])
    X_test_vectorized_ng = vect3.transform(X_test[TEXT])
    
    #models with only lyrics
    output5_ngram_plain_lyrics = log_reg_and_svm(X_train_vectorized_ng, y_train, X_test_vectorized_ng, y_test);
        
    #Doc2Vec ********************************************************************************************************************************************************
    
    train_doc = pd.DataFrame(pd.concat([X_train[TEXT], y_train], axis=1))
    test_doc = pd.DataFrame(pd.concat([X_test[TEXT], y_test], axis=1))
    
    train_doc2 = train_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x[target]]), axis=1)
    test_doc2 = test_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x[target]]), axis=1)

    #DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec. 
    #The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
    #We set the minimum word count to 2 in order to discard words with very few occurrences.

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0,  vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, hs=0, workers=cores, epochs=200)
    train_corpus = [x for x in train_doc2.values]
    model_dbow.build_vocab([x for x in train_doc2.values])
    
    model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
    
    y_train_doc, X_train_doc = get_vectors(model_dbow, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dbow, test_doc2)
    
    #models 
    output7_doc2vec_dbow_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);
 
    #dm = 1 model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, workers=cores, epochs=200)
    model_dmm.build_vocab([x for x in train_doc2.values])
    
    model_dmm.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    y_train_doc, X_train_doc = get_vectors(model_dmm, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dmm, test_doc2)
    
    #models 
    output8_doc2vec_dm_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);

    #Mix of dbow and dmm
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    
    y_train_doc, X_train_doc = get_vectors(new_model, train_doc2)
    y_test_doc, X_test_doc = get_vectors(new_model, test_doc2)
    
    #models 
    output9_doc2vec_dbowdb_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);


    xxd11 = pd.DataFrame(output1_bow_plain_lyrics, index=['bow_plain_lyrics'])
    xxd11 = xxd11.append(pd.DataFrame(output3_tfidf_plain_lyrics, index=['tfidf_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output5_ngram_plain_lyrics, index=['ngram_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output7_doc2vec_dbow_plain_lyrics, index=['doc2vec_dbow_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output8_doc2vec_dm_plain_lyrics, index=['doc2vec_dm_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output9_doc2vec_dbowdb_plain_lyrics, index=['doc2vec_dbowdb_plain_lyrics']))

    return xxd11

#%%

#corpus text
output_h_vs_not = all_techniques(data_all, 'Target', 'Corpus', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
output_verbal_vs_not = all_techniques(data_all, 'VERBAL ABUSE', 'Corpus', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
output_nverbal_vs_not = all_techniques(data_all, 'NON-VERBAL ABUSE', 'Corpus', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
output_pys_vs_not = all_techniques(data_all, 'PHYSICAL ABUSE', 'Corpus', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 

##plain text
#output_h_vs_not = all_techniques(data_all, 'Target', 'text', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#output_verbal_vs_not = all_techniques(data_all, 'VERBAL ABUSE', 'text', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#output_nverbal_vs_not = all_techniques(data_all, 'NON-VERBAL ABUSE', 'text', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#output_pys_vs_not = all_techniques(data_all, 'PHYSICAL ABUSE', 'text', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#
##processed
#p_output_h_vs_not = all_techniques(data_all, 'Target', 'processed', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#p_output_verbal_vs_not = all_techniques(data_all, 'VERBAL ABUSE', 'processed', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#p_output_nverbal_vs_not = all_techniques(data_all, 'NON-VERBAL ABUSE', 'processed', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 
#p_output_pys_vs_not = all_techniques(data_all, 'PHYSICAL ABUSE', 'processed', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 5) 

