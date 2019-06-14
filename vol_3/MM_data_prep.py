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
#Preprocessing 

def clean_text(texto, min_char, text):
    
    texto[text] = texto.text.dropna()
    texto[text] = texto[text].astype(str)
    texto[text] = texto[text].replace({'\n': ' '}, regex=True)
    texto[text] = texto[text].replace(r'[\W_]+', ' ', regex=True) 
    
    #Let's make lowercase in the end
    #texto = texto.text.str.lower() # lowercase text
    
    #remove weird chars
    texto[text] = texto.text.apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    #texto = pd.DataFrame(texto)
    #keep longer one if description is shorter than 30 chars
    texto['len1'] = texto.apply(lambda x: len(x.text.strip()) if len(x.text.strip()) > 0 else 0, axis=1) 

    #drop if shorter than min_char
    texto = texto[texto['len1'] >= min_char]
    
    texto[text] = texto.text.str.strip()
    #texto[text] = texto.dropna()
    
    #remove weird duplicates
    texto = texto.drop_duplicates()
    texto = texto.reset_index(drop=True)

    return texto

#%%
#new safecity data
#target 1's
keep_columns = ('INCIDENT TITLE', 'DESCRIPTION', 'VERBAL ABUSE', 'NON-VERBAL ABUSE','PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE') 

safecity1 = safecity_new.filter(items=keep_columns)

#safecity2 = safecity1[['INCIDENT TITLE', 'DESCRIPTION']]
safecity2 = safecity1.copy()
safecity2['DESCRIPTION'] = safecity2['DESCRIPTION'].astype(str)
safecity2['INCIDENT TITLE'] = safecity2['INCIDENT TITLE'].astype(str)

count2 = safecity2['INCIDENT TITLE'].value_counts()

#length comparison of description and incident title
safecity2['len_1'] = safecity2.apply(lambda x: len(x['INCIDENT TITLE'].strip()) if len(x['INCIDENT TITLE'].strip()) > 0 else 0, axis=1) 
safecity2['len_2'] = safecity2.apply(lambda x: len(x['DESCRIPTION'].strip()) if len(x['DESCRIPTION'].strip()) > 0 else 0, axis=1) 

#keep longer one if description is shorter than 30 chars
safecity2['text'] = safecity2.apply(lambda x: x['DESCRIPTION'].strip() if (x.len_2 >= x.len_1 or x.len_2 >= 30)  else x['INCIDENT TITLE'].strip(), axis=1) 

#append previous safecity datasets 
#safecity3 = safecity2['text'].append(safecity_old1['Description'], ignore_index=True)
#safecity4 = safecity3.append(safecity_old2['Description'], ignore_index=True)
#safecity4 = pd.DataFrame(safecity4).rename(index=str, columns={0: "text"})

#Preprocessing 
safecity5 = clean_text(safecity2, 30, 'text')

#safecity5 = pd.DataFrame(safecity5)
safecity5 = safecity5.rename(columns={0: "text"})

safecity5 = safecity5[['text', 'VERBAL ABUSE', 'NON-VERBAL ABUSE','PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE']]

#%%
#del safecity1, safecity2, safecity3, safecity4
del safecity1, safecity2
del keep_columns, count1, count2
#del safecity_new, safecity_old1, safecity_old2
del safecity_new

#%%
#target 0's
count2 = negatives['source'].value_counts()

negatives['text'] = negatives.astype(str)
negatives['len_1'] = negatives.apply(lambda x: len(x['text'].strip()) if len(x['text'].strip()) > 0 else 0, axis=1) 

#sampled for now (to make it ~ 1/2)
negatives3 = negatives[negatives['source'] == 'imdb_2'].sample(n=10000, random_state=1905)
negatives4 = negatives[negatives['source'] != 'imdb_2']
negatives5 = negatives3.append(negatives4, ignore_index=True)
negatives5 = negatives5['text'].drop_duplicates()
negatives5 = pd.DataFrame(negatives5)

#Preprocessing 
negatives6 = clean_text(negatives5, 30, 'text')
negatives6 = pd.DataFrame(negatives6)

negatives6['VERBAL ABUSE'] = 0
negatives6['NON-VERBAL ABUSE'] = 0
negatives6['PHYSICAL ABUSE'] = 0
negatives6['SERIOUS PHYSICAL ABUSE'] = 0
negatives6['OTHER ABUSE'] = 0

negatives6 = negatives6[['text', 'VERBAL ABUSE', 'NON-VERBAL ABUSE','PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE']]
           

#%%
del negatives, negatives3, negatives4, negatives5
del count2

#%%
#All models 

def data_prep(data1, data2, text):
    #Data1: target 1 data 
    #Data2: target 0 data 
    #text: column for text    
    
    #Data preparation     
    #data1 = pd.DataFrame(data1[text])
    data1['Target'] = 1

    #data2 = pd.DataFrame(data2[text])
    data2['Target'] = 0
    
    #Balance 
    min1 = min(len(data1), len(data2));    
    data1 = data1.sample(n=min1, random_state=1905)
    data2 = data2.sample(n=min1, random_state=1905)
    
    data_all = data1.append(data2, ignore_index=True)
    
    data_ner = data_all.copy();
    
    data_class = data_all.copy();
    
    data_class[text] = data_class[text].str.lower();
    
    #to excel
    writer = pd.ExcelWriter('data_ner.xlsx', engine='xlsxwriter');
    data_ner.to_excel(writer, sheet_name= 'data_ner');
    writer.save();
    
    writer = pd.ExcelWriter('data_class.xlsx', engine='xlsxwriter');
    data_class.to_excel(writer, sheet_name= 'data_class');
    writer.save();
    
    #X_train, X_test, y_train, y_test = train_test_split(data_all[text], data_all['Target'], stratify=data_all['Target'], test_size=test_percent, random_state=1905)
    
    return data_ner, data_class 

#%%

data_ner, data_class = data_prep(safecity5, negatives6, 'text')  
