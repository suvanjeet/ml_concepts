
import nltk
nltk.download('punkt')
#%%
#Tokenization
text= 'Mary had a little Lamb. Her fleece was white as snow'
from nltk.tokenize import word_tokenize, sent_tokenize
#%%
# sentence Tokenization
sents= sent_tokenize (text)
print(sents)
#%%
#word tokenization
words= word_tokenize (text)
print(words)

#%%
#stemming
text2= 'Mary closed on closing night when she was in the mood to close'
from nltk.stem.lancaster import LancasterStemmer
st= LancasterStemmer()
#print (st.stem(word))
stemwords=[st.stem(word) for word in word_tokenize(text2)]
print(stemwords)

#%%
#POS tagging
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(word_tokenize(text2))

#%%
#list stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
stopwds= set(stopwords.words('english')+list(punctuation))
stopwds

#%%
#remove stopwords
wordsnostopwds=[word for word in word_tokenize (text) if word not in stopwds]
wordsnostopwds
#%%
#ngrams  bigrams
from nltk.collocations import *
bigram_measures=nltk.collocations.BigramAssocMeasures()
finder= BigramCollocationFinder.from_words(wordsnostopwds)
sorted(finder.ngram_fd.items())
#%%
# ngram  trigrams
from nltk.collocations import *
trigram_measures=nltk.collocations.TrigramAssocMeasures
finder= TrigramCollocationFinder.from_words(wordsnostopwds)
sorted(finder.ngram_fd.items())

#%%
nltk.download('wordnet')
from nltk.wsd import lesk
sense1= lesk(word_tokenize('sing in a lower tone, along with the bass'),'bass')
print(sense1,sense1.definition())
sense2= lesk(word_tokenize('This sea bass was really hard to catch'),'bass')
print(sense2,sense2.definition())

#%%

import urllib.request
from bs4 import BeautifulSoup

#%%
arturl= 'https://timesofindia.indiatimes.com/india/70-jobs-for-locals-in-mp-rahul-said-will-discuss-scheme-with-kamal-nath/articleshow/67141547.cms'
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
soup.find('arttextxml').text
#%%
text=' '.join(map(lambda p: p.text,soup.find_all('arttextxml')))
text

#%%
text.encode('ascii',errors='replace').replace("?"," ")

#%%
import re
# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', text)  
article_text = re.sub(r'\s+', ' ', article_text)   #removing extra space
article_text

#%%
article_text = re.sub(r'\'', '', article_text) 
article_text
#%%
article_text = re.sub(r'"', '', article_text) 
article_text
#%%
article_text = re.sub(r':', '', article_text) 
article_text = re.sub(r',', '', article_text) 
article_text = re.sub(r'\s+', ' ', article_text)   #removing extra space
article_text
#%%
# alternate for the below code
'''clean_text = article_text.lower()
clean_text = re.sub(r'\W',' ',clean_text)    #removing extra punctuation
clean_text = re.sub(r'\d',' ',clean_text)    #removing digits
clean_text = re.sub(r'\s+',' ',clean_text)   #removing space
clean_text'''

#%%
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

sents= sent_tokenize (article_text)
print(sents)

#%%

# Removing special characters and digits
'''formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)  
formatted_article_text'''

#%%
words= word_tokenize (article_text.lower())
print(words)
#%%
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
stopwds= set(stopwords.words('english')+list(punctuation))
stopwds
#%%
word_sent= [word for word in words if word not in stopwds]
word_sent
#%%
from nltk.probability import FreqDist # to construct freq dist of word
freq =  FreqDist(word_sent)
freq

#%%
from heapq import nlargest
nlargest(10,freq,key=freq.get)
#%%
from collections import defaultdict
ranking= defaultdict(int)

for i,sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
print(ranking)
#%%
sents_idx = nlargest(4,ranking, key=ranking.get)
sents_idx
#%%
sents_idx1=sorted(sents_idx)
sents_idx1
#%%
summary=[]
for j in sents_idx1:
    summary.append(sents[j])
summary

