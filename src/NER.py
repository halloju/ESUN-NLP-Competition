import re
from snownlp import SnowNLP

## format 
def text2array(text, sequence_length):
    textArr = re.findall('.{' + str(sequence_length) + '}', text)
    textArr.append(text[(len(textArr) * sequence_length):])
    return [[c for c in text] for text in textArr]

## shorten sents
def sent_tokenize(x):
    sents_temp = re.split('(：|:|,|，|。|！|\!|\.|？|\?|、)', x)
    sents = []
    for i in range(len(sents_temp)//2):
        sent = sents_temp[2*i] + sents_temp[2*i+1]
        sents.append(sent)
    sents.append(sents_temp[-1])
    return sents

## get summarize text
def get_summarize(texts):
  key_sen = np.round(len(texts)/100).astype('int')
  s = SnowNLP(texts)
  t_keysen = s.summary(key_sen)

  news = []
  for t in t_keysen:
      sents = sent_tokenize(t)
      for s in sents:
          out = text2array(s, sequence_length=len(t))
          news.extend(out)

  return news

def get_names(ners):
  names=[]
  for la in ners:
    if len(la['labels']) > 0:
      for entity in la['labels']:
        if entity['entity'] == 'PER':
          if len(entity['value'])>2:
            names.append(entity['value'])
  names = np.unique(np.array(names)).tolist()

  return names     
