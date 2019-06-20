#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:58:25 2019

@author: salihemredevrim
"""

import MM_for_chatbot as mm
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

#%%
#Some test data 
text_0 = 'I was walking in Boullion street.'
text_01 = 'I was walking in Boullion street, a man called me bitch'
text_02 = 'When I was walking in the street I saw bitches'
text_1 = 'I was walking in Boullion street, an old guy groped me and masturbated!'
text_2 = 'I ate a pasta at this Chinese restaurant in Ocean Drive and I liked it so much it was fucking awesome mate'
text_3 = 'Yesterday, some people stalked a young girl in New York street.'
text_4 = "I kissed a girl and I liked it on Monday. The taste of her cherry chap stick. I hope my boyfriend don't mind it"
text_5 = "Hello"
text_6 = "I have a terrible headache please help me it fucks my brain at 22:45!"
text_7 = "Dun aksam, eve giderken orospu cocugunun teki gotumu elledi"
text_8 = "A son of a bitch called me bitch at 1 am near the Vrijtof"
text_9 = "Last night, when I was eating pasta in the restaurant, a guy called me bitch"
text_10 = "asddad sadads sadaadsa sdssxxx ddwqeq xxdas"
text_11 = 'I was not harassed'
text_12 = "I'm gonna take the bus at 12:13. I saw a guy yesterday and he looked like very disssatisfied and he weere groping me"
text_13 = "We weere watching a movie where old people masturbated"
text_14 = "Last night, I was not harassed"
text_15 = "Last night, we discussed about harassment issues in Maastricht"

#%%
out_0 = mm.finale(text_0, 10, 0.6) #ok
out_01 = mm.finale(text_01, 10, 0.6) #ok
out_02 = mm.finale(text_02, 10, 0.6) #ok
out_1 = mm.finale(text_1, 10, 0.6) #ok
out_2 = mm.finale(text_2, 10, 0.6) #ok
out_3 = mm.finale(text_3, 10, 0.6) #stalking is physical?
out_4 = mm.finale(text_4, 10, 0.6) #ok
out_5 = mm.finale(text_5, 10, 0.6) #ok 
out_6 = mm.finale(text_6, 10, 0.6) #ok 
out_7 = mm.finale(text_7, 10, 0.6) #lol it can understand turkish 
out_8 = mm.finale(text_8, 10, 0.6) #ok
out_9 = mm.finale(text_9, 10, 0.6) #NOK
out_10 = mm.finale(text_10, 10, 0.6) #ok
out_11 = mm.finale(text_11, 10, 0.6) #ok
out_12 = mm.finale(text_12, 10, 0.6) #ok
out_13 = mm.finale(text_13, 10, 0.6) #ok
out_14 = mm.finale(text_14, 10, 0.6) #OK
out_15 = mm.finale(text_15, 10, 0.6) #ok



