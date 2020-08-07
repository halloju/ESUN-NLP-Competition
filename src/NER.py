import pandas as pd
import re
import numpy as np
from snownlp import SnowNLP
#import jieba

def text2array(text, sequence_length):
    textArr = []#re.findall('.{' + str(sequence_length) + '}', text)
    textArr.append(text[(len(textArr) * sequence_length):])
    return [[c for c in text] for text in textArr]

## shorten sents
def sent_tokenize(x):
    sents_temp = re.split('(：|:|,|，|。|！|\!|？|\?|、| |)', x)
    sents = []
    for i in range(len(sents_temp)//2):
        sent = sents_temp[2*i] + sents_temp[2*i+1]
        sents.append(sent)
    sents.append(sents_temp[-1])
    return pd.Series(sents)

## get summarize text
def get_summarize(texts):
  if len(texts) > 850:
    key_sen = 50
  else:
    key_sen = np.round(len(texts)/15).astype('int')
  #key_sen = 50
  summ = SnowNLP(texts)
  t_keysen = summ.summary(key_sen)

  # sents = []
  # for t in t_keysen:
  #     sents.extend(sent_tokenize(t))

  return t_keysen

def get_names(ners):
  names=[]
  for la in ners:
    if len(la['labels']) > 0:
      for entity in la['labels']:
        if entity['entity'] == 'PER':
          if len(entity['value'])>3:
            if "男" in list(entity['value']) or "女" in list(entity['value']) or "嫌" in list(entity['value']): # 
              pass
            else: 
              names.append(entity['value'].replace(" ", ""))
  names = np.unique(np.array(names)).tolist()

  return names     


# %%
