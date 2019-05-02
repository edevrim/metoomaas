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
safecity_old1 = pd.read_csv("safecity_previous1.csv")
safecity_old2 = pd.read_csv("safecity_previous2.csv")

#%%
#new safecity data
#target 1's
keep_columns = ('INCIDENT TITLE', 'DESCRIPTION', 'CATEGORY',
       'Catcalls/Whistles', 'Commenting', 'Sexual Invites',
       'Ogling/Facial Expressions/Staring', 'Taking Pictures',
       'Indecent Exposure', 'Touching /Groping', 'Stalking',
       'Rape / Sexual Assault', 'Poor / No Street Lighting', 'Chain Snatching',
       'North East India Report', 'Others', 'VERBAL ABUSE', 'NON-VERBAL ABUSE',
       'PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE') 

safecity1 = safecity_new.filter(items=keep_columns)

safecity2 = safecity1[['INCIDENT TITLE', 'DESCRIPTION']]
safecity2['DESCRIPTION'] = safecity2['DESCRIPTION'].astype(str)
safecity2['INCIDENT TITLE'] = safecity2['INCIDENT TITLE'].astype(str)


count2 = safecity2['INCIDENT TITLE'].value_counts()

#length comparison of description and incident title
safecity2['len_1'] = safecity2.apply(lambda x: len(x['INCIDENT TITLE'].strip()) if len(x['INCIDENT TITLE'].strip()) > 0 else 0, axis=1) 
safecity2['len_2'] = safecity2.apply(lambda x: len(x['DESCRIPTION'].strip()) if len(x['DESCRIPTION'].strip()) > 0 else 0, axis=1) 

#keep longer one if description is shorter than 30 chars
safecity2['text'] = safecity2.apply(lambda x: x['DESCRIPTION'].strip() if (x.len_2 >= x.len_1 or x.len_2 >= 30)  else x['INCIDENT TITLE'].strip(), axis=1) 

#append previous safecity datasets 
safecity3 = safecity2['text'].append(safecity_old1['Description'], ignore_index=True)
safecity4 = safecity3.append(safecity_old2['Description'], ignore_index=True)

#remove weird duplicates
safecity5 = safecity4.drop_duplicates()

safecity5 = pd.DataFrame(safecity5)
safecity5 = safecity5.rename(columns={0: "text"})

#check weird characters
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
 
        return True
    
safecity5['check'] = 1
for k in range(len(safecity5)):
    s = safecity5['text'].iloc[k]
    safecity5['check'].iloc[k] = isEnglish(s)

#remove weird chars
safecity5['text2'] = safecity5['text'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

#remove less than 30 chars
safecity5['len_3'] = safecity5.apply(lambda x: len(x['text2'].strip()) if len(x['text2'].strip()) > 0 else 0, axis=1) 
safecity6 = safecity5[safecity5.len_3 >= 30]

#remove duplicates again
safecity7 = safecity6['text2'].drop_duplicates()

safecity8 = pd.DataFrame(safecity7).rename(columns={'text2': 'text'}).reset_index(drop=True)
safecity8['target_harassment'] = 1; 

#%%
del safecity1, safecity2, safecity3, safecity4, safecity5, safecity6, safecity7
del keep_columns, k, s, count1, count2
del safecity_new, safecity_old1, safecity_old2

#%%
#target 0's
count2 = negatives['source'].value_counts()

negatives['text'] = negatives.astype(str)
negatives['len_1'] = negatives.apply(lambda x: len(x['text'].strip()) if len(x['text'].strip()) > 0 else 0, axis=1) 

#remove less than 30 chars
negatives2 = negatives[negatives.len_1 >= 30]

#sampled for now (to make it ~ 1/2)
negatives3 = negatives2[negatives2['source'] == 'imdb_2'].sample(n=8000, random_state=1905)
negatives4 = negatives2[negatives2['source'] != 'imdb_2']

negatives5 = negatives3.append(negatives4, ignore_index=True)
negatives5 = negatives5[['text', 'target_harassment']].drop_duplicates()

#remove weird chars
negatives5['text2'] = negatives5['text'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
negatives5['len_2'] = negatives5.apply(lambda x: len(x['text2'].strip()) if len(x['text2'].strip()) > 0 else 0, axis=1) 
#%%

data_all = safecity8.append(negatives5[['text', 'target_harassment']], ignore_index=True)
data_all = data_all.reset_index(drop=True)

pd.crosstab(index = data_all['target_harassment'], columns="Total count")

#to_excel
#writer = pd.ExcelWriter('model_data.xlsx', engine='xlsxwriter');
#data_all.to_excel(writer, sheet_name= 'model_data');
#writer.save();

#%%
del negatives, negatives2, negatives3, negatives4, negatives5
del safecity8, count2

#%%
#train - test split 
X_train, X_test, y_train, y_test = train_test_split(data_all['text'], data_all['target_harassment'], stratify=data_all['target_harassment'], test_size=0.3, random_state=1905)

#tokenization
vect = CountVectorizer().fit(X_train)
check1 = vect.get_feature_names()

# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

#%%
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

#%%
#Tf-Idf

#min_df is used for removing terms that appear too infrequently. For example:
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".

#max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".

vect_tf = TfidfVectorizer(min_df=5, max_df=0.8).fit(X_train)
check2 = vect_tf.get_feature_names()

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

#%%
#ngrams

# document frequency of 5 and extracting 1-grams and 2-grams
vect3 = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized_ng = vect3.transform(X_train)

model3 = LogisticRegression()
model3.fit(X_train_vectorized_ng, y_train)

predictions_ng = model3.predict(vect3.transform(X_test))

#score
print('AUC: ', roc_auc_score(y_test, predictions_ng))
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_ng)
#roc_auc3 = auc(false_positive_rate, true_positive_rate)

print('accuracy: ', accuracy_score(y_test, predictions_ng))
confusion3 = confusion_matrix(y_test, predictions_ng)