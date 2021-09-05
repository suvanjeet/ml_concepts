
import urllib.request
from bs4 import BeautifulSoup

#%%
arturl= 'https://www.nytimes.com/topic/subject/newspapers'
#%%
from  urllib.request import urlopen
page= urlopen(arturl).read().decode('utf8','ignore')
soup= BeautifulSoup(page, 'html.parser')
soup
#%%
soup.head
# <head><title>The Dormouse's story</title></head>
#%%
soup.title
# <title>The Dormouse's story</title>
#%%
for my_tag in soup.find_all(class_="byline"):
#    print(my_tag.text)
    #decompose means that class containts will not be imported
    my_tag.decompose() 

#%%
test_list = soup.find(class_='story-meta')
list_items = test_list.text

list_items
#%%
mylist=[]
  
for my_tag in soup.find_all(class_="story-meta"):
#    print(my_tag.text)
    mylist.append(my_tag.text)

#name_box = soup.find(‘p’, attrs={‘class’: ‘story-meta’})
#name = name_box.text.strip()
#name
mylist
#%%
'''import re
mylist = re.sub(r'\[[0-9]*\]', ' ',mylist)  
mylist = re.sub(r'\s+', ' ',mylist)   #removing extra space
mylist'''

'''for i in mylist:
    str(i).replace(' ','')
    str(i).replace('\n','')
mylist'''
k = [x.replace('  ', '') for x in mylist]
#k = [x.replace('\n', '') for x in mylist]
k
#%%
k = [x.replace('\n', ' ') for x in k]
k
#%%
from sklearn.feature_extraction.text  import TfidfVectorizer

#%%
vector1= TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
#%%
x=vector1.fit_transform(k)
x
#%%
from sklearn.cluster import KMeans
km= KMeans(n_clusters=3,init='k-means++',max_iter=100,n_init=1,verbose =True)
#%%
 km.fit(x) 
#%%
print(km.labels_)  
 #%%
 import numpy as np
 np.unique(km.labels_,return_counts=True)
#%%
text= {}
for i,cluster in enumerate(km.labels_):

    onedocument=k[i]
    if cluster not in text.keys():
        text[cluster]=onedocument
    else:
        text[cluster]+=onedocument

#%%
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest
import nltk
#%%

stopwds= set(stopwords.words('english')+list(punctuation))
#%%
keywords ={}
counts ={}
for cluster in range(3):
    word_sent=word_tokenize(text[cluster].lower())
    word_sent= [word for word in word_sent if word not in stopwds]
    freq =  FreqDist(word_sent)
    keywords[cluster]=nlargest(100,freq, key=freq.get) #return largest n elements
    counts[cluster]=freq
    

#%%
keywords
#%%
counts
#%%
'''for cluster in range(1):
    #print(set(range(3)))
    #print(set([cluster]))
    other_cluster= list(set(range(3))-set([cluster]))
    print(other_cluster)
    key_other_cluster=set(keywords[other_cluster[0]]).union(set(keywords[other_cluster[1]]))
    print(keywords[other_cluster[0]])
    print(keywords[other_cluster[1]])
    #print(key_other_cluster)'''
    
#%%
unique_keys ={}
for cluster in range(3):
    other_cluster=list(set(range(3))-set([cluster]))
    #print(other_cluster)
    key_other_cluster=set(keywords[other_cluster[0]]).union(set(keywords[other_cluster[1]]))
    unique=set(keywords[cluster])-key_other_cluster
    unique_keys[cluster]=nlargest(10,unique, key=counts[cluster].get)

#%%
unique_keys

#%%

article1='Computer malware attacks on infrastructure, while relatively rare, are hardly new: Russia has been credibly accused of shutting down power grids in Ukraine and a petrochemical plant in Saudi Arabia, Iran crippled a casino in Las Vegas, and the United States and Israel attacked a nuclear enrichment plant in Iran. But this would be the first known attack on major newspaper printing operations, and if politically motivated, it would define new territory in recent attacks on the media.'
#%%
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier()
classifier.fit(x,km.labels_)

#%%

vector2= TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
#%%
test1=vector2.fit_transform(article1)
test1


    