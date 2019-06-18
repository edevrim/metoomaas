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
import re
from replacers import AntonymReplacer
from spellchecker import SpellChecker
import string

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
def decontracted(phrase):
    # specific
    #phrase = phrase.lower() # lowercase text
    phrase = re.sub(r",", "", phrase)
    phrase = re.sub(r'i\'mma', 'i am going to', phrase)
    phrase = re.sub(r'i\'ma', 'i am going to', phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ain\'t", "are not", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"wanna", "want to", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r'dammit', 'damn it', phrase)
    phrase = re.sub(r'imma', 'i am going to', phrase)
    phrase = re.sub(r'gimme', 'give me', phrase)
    phrase = re.sub(r'luv', 'love', phrase)
    phrase = re.sub(r' dem ', 'them', phrase)
    phrase = re.sub(r' asap ', 'as soon as possible', phrase)
    phrase = re.sub(r' gyal ', 'girl', phrase)
    phrase = re.sub(r' dat ', ' that ', phrase)
    phrase = re.sub(r' skrrt ', ' ', phrase)
    phrase = re.sub(r' yea ', ' yeah ', phrase)
    phrase = re.sub(r' ayy ', '', phrase)
    phrase = re.sub(r' aye ', '', phrase)
    phrase = re.sub(r' ohoh ', '', phrase)
    phrase = re.sub(r' hol ', 'hold', phrase)
    phrase = re.sub(r' lil ', ' little ', phrase)
    phrase = re.sub(r' g ', ' gangster ', phrase)
    phrase = re.sub(r' gangsta ', ' gangster ', phrase)
    phrase = re.sub(r'thang', 'thing', phrase)
    phrase = re.sub(r'gotta', 'going to', phrase)
    phrase = re.sub(r' hook ', ' ', phrase)
    phrase = re.sub(r' intro ', ' ', phrase)
    phrase = re.sub(r' gon ', ' going to ', phrase)
    phrase = re.sub(r' shoulda ', ' should have ', phrase)
    phrase = re.sub(r' em ', ' them ', phrase)
    phrase = re.sub(r' ya ', ' you ', phrase)
    phrase = re.sub(r' da ', ' the ', phrase)
    phrase = re.sub(r' na na ', ' ', phrase)
    phrase = re.sub(r' hoe', ' whore', phrase)
    phrase = re.sub(r' oh ', ' ', phrase)
    phrase = re.sub(r'\b(\w+)( \1\b)+', r'\1', phrase)
    phrase = re.sub(r'\'til', 'till', phrase)
    phrase = re.sub(r'ooh', '', phrase)
    phrase = re.sub(r'lala', '', phrase)
    phrase = re.sub(r' ho ', ' whore ', phrase)
    phrase = re.sub(r' mm ', '  ', phrase)
    phrase = re.sub(r' yah ', '  ', phrase)
    phrase = re.sub(r' yeah ', '  ', phrase)
    phrase = re.sub(r'hitta', 'nigga', phrase)   
    #phrase = re.sub(r'u', 'you', phrase)   
    phrase = re.sub(r'\&', 'and', phrase)
    phrase = re.sub(r'nothin', 'nothing', phrase)
    phrase = re.sub(r'\$', 's', phrase)
    phrase = re.sub(r" c\'mon", "come on", phrase)
    phrase = re.sub(r" \'cause", " because", phrase)
    phrase = re.sub(r" cuz ", " because ", phrase)
    phrase = re.sub(r" \'cuz ", " because ", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'yo", "your", phrase)

    return phrase

#%%    
def spelling11(data1, text1): 
    
    spell = SpellChecker()
    spell.word_frequency.load_text_file('corporaForSpellCorrection.txt')
    sent = data1[text1].str.split()

    for k in range(len(sent)):     
        misspelled = spell.unknown(sent.iloc[k])
        xd1 = ''
        for word in sent.iloc[k]:
            if word in misspelled:
                # Get the one `most likely` answer
                word = spell.correction(word)
                xd1 = xd1+' '+word 
            else:
                xd1 = xd1+' '+word 
                
        data1[text1].iloc[k] = xd1 
        
    return data1

#%%
def clean_text(texto, min_char, text):
    
    texto[text] = texto[text].dropna()
    texto[text] = texto[text].astype(str)
    texto[text] = texto[text].replace({'\n': ' '}, regex=True)
    #texto[text] = texto[text].replace(r'[\W_]+', ' ', regex=True) 
    
    remove = string.punctuation
    remove = remove.replace(":", "")
    remove = remove.replace("'", "")
    
    texto[text] = texto[text].str.translate({ord(char): None for char in remove})
    
    #Let's make lowercase in the end
    #texto = texto.text.str.lower() # lowercase text
    
    #remove weird chars
    texto[text] = texto[text].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    #texto = pd.DataFrame(texto)
    #keep longer one if description is shorter than 30 chars
    texto['len1'] = texto.apply(lambda x: len(x.text.strip()) if len(x.text.strip()) > 0 else 0, axis=1) 

    #drop if shorter than min_char
    texto = texto[texto['len1'] >= min_char]
    
    texto[text] = texto[text].str.strip()
    #texto[text] = texto.dropna()
    
    #remove weird duplicates
    texto = texto.drop_duplicates()
    texto = texto.reset_index(drop=True)

    return texto

#%%
def negations(text):
    replacer = AntonymReplacer()
    
    sent = text.split()
    noneg = replacer.replace_negations(sent)
    separator = ' '
    out = separator.join(noneg)
    
    return out

#%%
#Test 
#text11 = pd.DataFrame(index=range(6),columns=range(1))
#text11['text'] = ''
#text11['text'].iloc[0] = "Im gonna go to cuz hell fuckin yeah!"  
#text11['text'].iloc[1] = "I am not unhappi today?" 
#text11['text'].iloc[2] = "You weere not respectful today" 
#text11['text'].iloc[5] = "I was not satisfied today" 
#text11['text'].iloc[3] = "I wasn't  dissatisfied today" 
#text11['text'].iloc[4] = "Hello          world, I am fuckin harrassed" 
#
#text11['text2'] = text11['text'].apply(decontracted) 
#text11 = clean_text(text11, 10, 'text2')
#text11 = spelling11(text11, 'text2') 
#text11['text2'] = text11['text2'].apply(negations) 

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
safecity2['text'] = safecity2['text'].apply(decontracted) 
safecity5 = clean_text(safecity2, 20, 'text')
#safecity5 = spelling11(safecity5, 'text') 
safecity5['text'] = safecity5['text'].apply(negations) 

#safecity5 = pd.DataFrame(safecity5)
#safecity5 = safecity5.rename(columns={0: "text"})
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
negatives5['text'] = negatives5['text'].apply(decontracted) 
negatives6 = clean_text(negatives5, 20, 'text')
#negatives6 = spelling11(negatives6, 'text') 
negatives6['text'] = negatives6['text'].apply(negations) 


#negatives6 = pd.DataFrame(negatives6)

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
