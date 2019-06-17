#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:58:25 2019

@author: salihemredevrim
"""

import MM_for_chatbot as mm
import pandas as pd 

#%%
#Some test data 

text_1 = 'I was walking in Boullion street, an old guy groped me and masturbated! :('
text_2 = 'I ate a pasta at this Chinese restaurant in Ocean Drive and I liked it so much it was fucking awesome mate'
text_3 = 'Yesterday, some people stalked a young girl in New York street.'
text_4 = "I kissed a girl and I liked it on Monday. The taste of her cherry chap stick. I hope my boyfriend don't mind it"
text_5 = "Hello"
text_6 = "I have a terrible headache please help me it fucks my brain at 22:45!"
text_7 = "Dun aksam, eve giderken orospu cocugunun teki gotumu elledi"
text_8 = "A son of a bitch called me bitch at 1 am near the Vrijtof"
text_9 = "Last night, when I was eating pasta in the restaurant, a guy called me bitch"
text_10 = "asddad sadads sadaadsa sdssxxx ddwqeq xxdas"
text_11 = 'I was not harassed you stupid bot'

#%%

out_1 = mm.finale(text_1, 10, 0.7)
out_2 = mm.finale(text_2, 10, 0.7)
out_3 = mm.finale(text_3, 10, 0.7)
out_4 = mm.finale(text_4, 10, 0.7)
out_5 = mm.finale(text_5, 10, 0.7)
out_6 = mm.finale(text_6, 10, 0.7)
out_7 = mm.finale(text_7, 10, 0.7)
out_8 = mm.finale(text_8, 10, 0.7)
out_9 = mm.finale(text_9, 10, 0.7)
out_10 = mm.finale(text_10, 10, 0.7)
out_11 = mm.finale(text_11, 10, 0.7)



