#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:03:40 2019

@author: salihemredevrim
"""

#data sources:  
#safecity 
#safecity previous study: 
#https://github.com/swkarlekar/safecity
#https://www.kaggle.com/utathya/imdb-review-dataset#imdb_master.csv
#http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences#
#https://github.com/rmaestre/Sentiwordnet-BC/blob/master/test/testdata.manual.2009.06.14.csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix
#%%
#take datasets
negatives = pd.read_excel("negatives.xlsx") 
count1 = negatives['source'].value_counts()

#to upload github, first x rows have been taken
#writer = pd.ExcelWriter('negatives_sample.xlsx', engine='xlsxwriter');
#negatives_sample.to_excel(writer, sheet_name= 'sample');
#writer.save();

safecity_new = pd.read_excel("safecity.xlsx", sheetname='Full_Data', password='Safecitymap@123')
#safecity_old1 = pd.read_csv("safecity_previous1.csv")
#safecity_old2 = pd.read_csv("safecity_previous2.csv")

#%%
#new safecity data
#target 1's
keep_columns = ('INCIDENT TITLE', 'DESCRIPTION', 'VERBAL ABUSE', 'NON-VERBAL ABUSE',
       'PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE') 

safecity1 = safecity_new.filter(items=keep_columns)
safecity1['DESCRIPTION'] = safecity1['DESCRIPTION'].astype(str)
safecity1['INCIDENT TITLE'] = safecity1['INCIDENT TITLE'].astype(str)


count2 = safecity1['INCIDENT TITLE'].value_counts()

#length comparison of description and incident title
safecity1['len_1'] = safecity1.apply(lambda x: len(x['INCIDENT TITLE'].strip()) if len(x['INCIDENT TITLE'].strip()) > 0 else 0, axis=1) 
safecity1['len_2'] = safecity1.apply(lambda x: len(x['DESCRIPTION'].strip()) if len(x['DESCRIPTION'].strip()) > 0 else 0, axis=1) 

#keep longer one if description is shorter than 30 chars
safecity1['text'] = safecity1.apply(lambda x: x['DESCRIPTION'].strip() if (x.len_2 >= x.len_1 or x.len_2 >= 30)  else x['INCIDENT TITLE'].strip(), axis=1) 

#remove weird duplicates
safecity2 = safecity1.drop_duplicates()

safecity2 = pd.DataFrame(safecity2)
safecity2 = safecity2.rename(columns={0: "text"})

#check weird characters
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
 
        return True
    
#safecity2['check'] = 1
#for k in range(len(safecity2)):
#    s = safecity2['text'].iloc[k]
#    safecity2['check'].iloc[k] = isEnglish(s)

#remove weird chars
safecity2['text2'] = safecity2['text'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

#remove less than 30 chars
safecity2['len_3'] = safecity2.apply(lambda x: len(x['text2'].strip()) if len(x['text2'].strip()) > 0 else 0, axis=1) 
safecity3 = safecity2[safecity2.len_3 >= 30]

safecity4 = safecity3[['text2', 'VERBAL ABUSE', 'NON-VERBAL ABUSE',
       'PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE']]

safecity4 = safecity4.rename(columns={'text2': 'text'}).reset_index(drop=True)

#%%
del safecity1, safecity2, safecity3
del keep_columns, count1, count2
del safecity_new

#%%
#target 0's
count2 = negatives['source'].value_counts()

negatives['text'] = negatives.astype(str)
negatives['len_1'] = negatives.apply(lambda x: len(x['text'].strip()) if len(x['text'].strip()) > 0 else 0, axis=1) 

#remove less than 30 chars
negatives2 = negatives[negatives.len_1 >= 30]

#sampled for now (to make it ~ 1/2)
negatives3 = negatives2[negatives2['source'] == 'imdb_2'].sample(n=7000, random_state=1905)
negatives4 = negatives2[negatives2['source'] != 'imdb_2']

negatives5 = negatives3.append(negatives4, ignore_index=True)
negatives5 = negatives5[['text', 'target_harassment']].drop_duplicates()

#remove weird chars
negatives5['text2'] = negatives5['text'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
negatives5['len_2'] = negatives5.apply(lambda x: len(x['text2'].strip()) if len(x['text2'].strip()) > 0 else 0, axis=1) 

negatives6 = negatives5.copy()
negatives6['VERBAL ABUSE'] = 0
negatives6['NON-VERBAL ABUSE'] = 0
negatives6['PHYSICAL ABUSE'] = 0
negatives6['SERIOUS PHYSICAL ABUSE'] = 0
negatives6['OTHER ABUSE'] = 0

negatives6 = negatives6[['text2', 'VERBAL ABUSE', 'NON-VERBAL ABUSE',
       'PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE']]

negatives6 = negatives6.rename(columns={'text2': 'text'}).reset_index(drop=True)

#%%

data_all = safecity4.append(negatives6, ignore_index=True)
data_all = data_all.reset_index(drop=True)

va = pd.crosstab(index = data_all['VERBAL ABUSE'], columns="Total count")
nva = pd.crosstab(index = data_all['NON-VERBAL ABUSE'], columns="Total count")
pa = pd.crosstab(index = data_all['PHYSICAL ABUSE'], columns="Total count")
spa = pd.crosstab(index = data_all['SERIOUS PHYSICAL ABUSE'], columns="Total count")
oa = pd.crosstab(index = data_all['OTHER ABUSE'], columns="Total count")

#all are unbalanced, we can make it balanced for each target below function
#spa is really low that's why serious physical abuse and physical abuse are merged

data_all['PHYSICAL ABUSE'] = data_all.apply(lambda x: 1 if (x['PHYSICAL ABUSE'] + x['SERIOUS PHYSICAL ABUSE']) > 0 else 0, axis=1) 

pa = pd.crosstab(index = data_all['PHYSICAL ABUSE'], columns="Total count")

#to_excel
#writer = pd.ExcelWriter('model_data.xlsx', engine='xlsxwriter');
#data_all.to_excel(writer, sheet_name= 'model_data');
#writer.save();

#%%
del negatives, negatives2, negatives3, negatives4, negatives5, negatives6
del count2, safecity4
del va, nva, pa, spa, oa

#%%
#train - test split 

def per_abuse(data1, target):
    
    #make it balanced 
    data2 = data1[['text', target]]
    
    positives = data2[data2[target] == 1]
    len1 = len(positives)
    
    negatives = data2[data2[target] == 0].sample(n=len1, random_state=1905)
    
    data3 = positives.append(negatives, ignore_index=True)
    data3 = data3.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(data3['text'], data3[target], stratify=data3[target], test_size=0.3, random_state=1905)

    #tokenization
    vect = CountVectorizer().fit(X_train)
    #check1 = vect.get_feature_names()

    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)


    #initial model (Logistic regression)

    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Predict test documents
    predictions = model.predict(X_test_vectorized)

    #score
    print('AUC: ', roc_auc_score(y_test, predictions))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print('accuracy: ', accuracy_score(y_test, predictions))

    confusion = confusion_matrix(y_test, predictions)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


#Tf-Idf

#min_df is used for removing terms that appear too infrequently. For example:
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".

#max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".

    vect_tf = TfidfVectorizer(min_df=5, max_df=0.8).fit(X_train)
    #check2 = vect_tf.get_feature_names()
    
    X_train_vectorized_tf = vect_tf.transform(X_train)
    model2 = LogisticRegression()
    model2.fit(X_train_vectorized_tf, y_train)

    predictions_tf = model2.predict(vect_tf.transform(X_test))

    #score
    print('AUC: ', roc_auc_score(y_test, predictions_tf))
    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_tf)
    #roc_auc2 = auc(false_positive_rate, true_positive_rate)

    print('accuracy: ', accuracy_score(y_test, predictions_tf))
    confusion2 = confusion_matrix(y_test, predictions_tf)

    return confusion, confusion2

#%%
confusion1_va, confusion2_va = per_abuse(data_all, 'VERBAL ABUSE')
confusion1_nva, confusion2_nva = per_abuse(data_all, 'NON-VERBAL ABUSE')
confusion1_pa, confusion2_pa = per_abuse(data_all, 'PHYSICAL ABUSE')
confusion1_oa, confusion2_oa = per_abuse(data_all, 'OTHER ABUSE')

#%%
##ngrams
#
## document frequency of 5 and extracting 1-grams and 2-grams
#vect3 = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
#X_train_vectorized_ng = vect3.transform(X_train)
#
#model3 = LogisticRegression()
#model3.fit(X_train_vectorized_ng, y_train)
#
#predictions_ng = model3.predict(vect3.transform(X_test))
#
##score
#print('AUC: ', roc_auc_score(y_test, predictions_ng))
##false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_ng)
##roc_auc3 = auc(false_positive_rate, true_positive_rate)
#
#print('accuracy: ', accuracy_score(y_test, predictions_ng))
#confusion3 = confusion_matrix(y_test, predictions_ng)