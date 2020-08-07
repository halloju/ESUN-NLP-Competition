#%%
import requests as rq
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import os
import json
import io
import time


def crawler(url):
    nl_response = rq.get(url) 
    soup = BeautifulSoup(nl_response.text,'lxml')
    title = soup.find('title')
    content = soup.find_all('p')
    if title is not None:
        print('Found')
        title = title.text
        text = [k.text for k in content]
    else:
        print('Not found') 
        text = None
        title = None
    return(title, text)


tStart = time.time()
LinkList = pd.read_csv(os.path.join('download','tbrain_train_final_0610.csv'))

for i, link in enumerate(LinkList['hyperlink']):
    ID  = LinkList['news_ID'][i]
    names =  LinkList['name'][i]
    domain = urlparse(link).netloc
    print(ID)
    title, news = crawler(link)
    if news is not None:
        data = {
            'ID':int(ID),
            'Domain':domain,
            'Title':title,
            'news':news,
            'names':names
        }
        with open(os.path.join('data',str(ID)+'.json'), 'w+') as fp:
            json.dump(data, fp, ensure_ascii=False)
    else :
        pass
    
tEnd = time.time()

print ("It cost %f sec" % (tEnd - tStart))




# %%
