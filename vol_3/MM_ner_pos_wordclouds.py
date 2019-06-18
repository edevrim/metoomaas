#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:45:19 2019

@author: salihemredevrim
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from wordcloud import WordCloud

#%%

data_class = pd.read_excel('data_class.xlsx')
#data_ner = pd.read_excel('data_ner.xlsx')

#data_c1111 = data_class.head(10)
#data_ner2 = data_ner[data_ner['Target'] == 1]

#%%
#find verbs, adverbs, nouns...

def spacy_data(data1, lyrics):

    #init 
    #verbs = []
    #nouns = []
    #adverbs = []
    #adj = []
    corpus = []
    processed = []
    
    nlp = spacy.load('en_core_web_md')
    
    for i in range (0, len(data1)):
        #print('song', i)
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        spacy_data = pd.DataFrame()
        
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "word": token.text,
                "lemma": lemma,
                "pos": token.pos_,
                "stop_word": token.is_stop
            }  
            spacy_data = spacy_data.append(row, ignore_index = True)
            
            
            #check this code 
        
        list1313 = ('VERB', 'ADJ', 'NOUN', 'ADV')
        #adj.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "ADJ"].values))
        #verbs.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "VERB"].values))
        #nouns.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "NOUN"].values))
        #adverbs.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "ADV"].values))
        processed.append(" ".join(spacy_data["lemma"][spacy_data["pos"].isin(list1313)].values))
        
        #corpus_clean = " ".join(spacy_data["lemma"][spacy_data["stop_word"] == False].values)
        corpus_clean = " ".join(spacy_data["lemma"].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
        
    #data1['Verbs'] = verbs
    #data1['Nouns'] = nouns
    #data1['Adverbs'] = adverbs
    #data1['Adjectives'] = adj
    data1['Corpus'] = corpus
    data1['processed'] = processed
    
    return data1

#%%
#NER by Stanford   

##most frequent bigrams
#c_vec = CountVectorizer(ngram_range=(2,2), min_df=3)
#
## input to fit_transform() should be an iterable with strings
#lyrix = data_ner['text'].astype(str)
#ngrams = c_vec.fit_transform(lyrix)
#
## needs to happen after fit_transform()
#vocab = c_vec.vocabulary_
#
#count_values = ngrams.toarray().sum(axis=0)
#
#xd1 = pd.DataFrame(vocab, index=[0])
#xd1 = xd1.T.reset_index(drop=False) 
#
## new data frame with split value columns 
#new = xd1['index'].str.split(" ", n = 1, expand = True) 
#  
## making separate first name column from new data frame 
#list_consecutive = pd.DataFrame(xd1['index'])
#list_consecutive['First_word']= new[0] 
#  
## making separate last name column from new data frame 
#list_consecutive['Last_word']= new[1] 
#
#list_consecutive2 = list_consecutive['index'].tolist()
#%%    

#del count_values, lyrix, new, vocab, xd1, list_consecutive

#%%
#
#def ner_stanford(data1, lyrics, list_consecutive2): 
#    #print('NTLK Version: %s' % nltk.__version__)
#    stanford_ner_tagger = StanfordNERTagger(
#    'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
#    'stanford_ner/' + 'stanford-ner-3.9.2.jar')
#      
#    data1['LOC'] = ''; 
#    data1['DATE'] = '';
#    data1['PERSON'] = '';
#    data1['ORGANIZATION'] = '';
#     
#    for i in range(0, len(data1)):
#        
#        loc = '';
#        date = '';
#        org = '';
#        person = '';
#        prev_tag_type = 'HELLO'
#        prev_tag_value = 'HELLO'
#        
#        song = data1.iloc[i][lyrics]
#        results = stanford_ner_tagger.tag(song.split())
#        
#        for result in results:
#           
#            tag_value = result[0]
#            tag_type = result[1]
#            
#            compare1 = prev_tag_value + ' ' + tag_value
#            compare1 = compare1.lower()
#            
#            if tag_type == 'PERSON': 
#                if prev_tag_type == 'PERSON':
#                    if compare1 in list_consecutive2:
#                        person = person + '-' + tag_value
#                else:
#                    person = person + ' ' + tag_value
#                    
#            elif tag_type == 'LOCATION':
#                if prev_tag_type == 'LOCATION':
#                    if compare1 in list_consecutive2:
#                        loc =  loc+ '-' +tag_value
#                else:
#                    loc =  loc+' '+tag_value
#                    
#            elif tag_type == 'ORGANIZATION':
#                if prev_tag_type == 'ORGANIZATION':
#                    if compare1 in list_consecutive2:
#                        org = org+ '-' +tag_value
#                else:
#                    org = org+ ' ' +tag_value
#                    
#            #No need for date        
#            elif tag_type == 'DATE': 
#              date = date+ ' '+tag_value
#              
#            prev_tag_type = tag_type  
#            prev_tag_value = tag_value 
#              
#              
#        data1['LOC'].iloc[i] = loc;
#        data1['DATE'].iloc[i] = date;
#        data1['PERSON'].iloc[i] = person;
#        data1['ORGANIZATION'].iloc[i] = org;
#        
#        return data1
   
    
#%%
#NER for harassments
#data_ner3 = ner_stanford(data_ner2, 'text', list_consecutive2)        
#        
#%%
#All together 
        
def create_my_data(data_lower, lyrics):
    
    
    data1 = spacy_data(data_lower, lyrics); 

    #keep1 = [lyrics, 'Target','VERBAL ABUSE', 'NON-VERBAL ABUSE','PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE', 'Verbs','Nouns','Adverbs','Adjectives','processed']
    keep1 = [lyrics, 'Target','VERBAL ABUSE', 'NON-VERBAL ABUSE','PHYSICAL ABUSE', 'SERIOUS PHYSICAL ABUSE', 'OTHER ABUSE','processed', 'Corpus']
    data3 = data1[keep1]
    
    #to excel
    writer = pd.ExcelWriter('MM_last.xlsx', engine='xlsxwriter');
    data3.to_excel(writer, sheet_name= 'MM_last');
    writer.save();
    
    return data3

#%%%
mm_last = create_my_data(data_class, 'text')

#%%

#PLOTTING *****************************************************************************************************************************
#For yearly word counts per genre and pos

def word_counts(data1, pos, year, most_num):
    
    data11 = data1.copy();
    data11[year] = 2018;
    data11 = data11[data11['Target'] == 1]
    
    #most_num: most common x words per year
    #Year is for year column name
    #init
    freq = pd.DataFrame()
    common_words = []
    years = data11[year].unique().tolist()
    
    #frequencies per each year
    for i in range (0, len(years)):
        year_corpus = str(data11[pos][data11[year] == years[i]].tolist())
        tokens = year_corpus.split(" ")
        tokens = map(lambda foo: foo.replace('[', ''), tokens)
        tokens = map(lambda foo: foo.replace(',', ''), tokens)
        tokens = map(lambda foo: foo.replace(']', ''), tokens)
        tokens = map(lambda foo: foo.replace("'", ''), tokens)
        
        counts = Counter(tokens)
        freq = freq.append({
            year: years[i],
            "words": counts.most_common(n=most_num)
        }, ignore_index=True)
    freq[year] = freq[year].astype(int)
    
    #distinct words through years 
    for i in range (0, len(freq)): 
        for words in freq['words'][i]:
            common_words.append(words[0])
            
    common_words = list(set(common_words))
    
    #tabularize
    data2 = pd.DataFrame(dict.fromkeys(common_words, [0]))
    data2[year] = 0
    data3 = data2.copy()
    
    for j in freq[year]:
        row1 = data2.copy()
        row1[year] = j 
        data3 = data3.append(row1)
    
    data3 = data3[1:]
    data3 = data3.reset_index(drop=True)    
    
    
    for j in range(0, len(data3)):
            current_year = freq[year][j]
            current_terms = freq['words'][j]
            
            for words in current_terms:
                data3[words[0]] = data3.apply(lambda x: words[1] if x[year] == current_year else x[words[0]], axis=1)
  
    return freq

#%%
    
freq1 = word_counts(mm_last, 'Verbs', 'Year', 300)
freq2  = word_counts(mm_last, 'Nouns', 'Year', 300)
freq3  = word_counts(mm_last, 'Adjectives', 'Year', 300)
freq4  = word_counts(mm_last, 'Adverbs', 'Year', 300)

#%%   
#Word clouds
def wordcloud(freq, year_name, year, pos, max_words, genre):
    
    freq1 = freq[freq[year_name] == year].reset_index(drop=True)
    freq2 = pd.DataFrame(freq1['words'][0]).astype(str)
    freq2 = freq2.rename(index=str, columns={0: 'word', 1: 'count'})
    
    freq2['len1'] = freq2.apply(lambda x: len(x['word']), axis=1)
    
    freq2 = freq2[freq2['len1'] > 0]
    freq2 = freq2.drop('len1', axis=1).reset_index(drop=True)
    
    d = {}
    for a, x in freq2.values:
        d[a] = float(x)

    wordcloud = WordCloud( width = 4000,
                          height = 3000,
                          background_color="white",
                          max_words = max_words )
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    
    plt.savefig('WC_'+genre+'_'+pos+'_'+str(year), bbox_inches='tight')
    
    return;

#%%

wordcloud(freq1, 'Year', 2018, 'Verbs', 50, 'harras')  
wordcloud(freq2, 'Year', 2018, 'Nouns', 50, 'harras')  
wordcloud(freq3, 'Year', 2018, 'Adjectives', 50, 'harras')    
wordcloud(freq4, 'Year', 2018, 'Adverbs', 50, 'harras')  

#%%

del freq1, freq2, freq3, freq4




    