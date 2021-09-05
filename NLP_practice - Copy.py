
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
''' rule based NLP''' 
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
sents_idx=nlargest(4,ranking, key=ranking.get)
sents_idx
#%%
[sents[j] for j in sorted(sents_idx)]

#%%
'''NLP using machine learning'''
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

def getAllDoxyDonkeyPosts(url,links):
    
    response= urlopen(url).read().decode('utf8','ignore')
    soup= BeautifulSoup(response,'html.parser')
    for a in soup.findAll('a'):
        try:
            url=a['href']
            title= a['title']
            if title == "older Posts":
                print (title) 
                print (url)
                links.append(url)
                getAllDoxyDonkeyPosts(url,links)
        except:
            title=""
    return

blogurl="http://doxydonkey.blogspot.com/"
links= []
getAllDoxyDonkeyPosts(blogurl,links)

#%%
'''from  urllib.request import urlopen
page= urlopen(arturl).read().decode('utf8','ignore')
soup= BeautifulSoup(page, 'html.parser')
soup'''
#%%
def getDoxyDonkeyTest(testurl):
    
    response= urlopen(testurl)
    soup= BeautifulSoup(response,'html.parser')
    mydivs=soup.findAll("div", {"class": 'post-body'})
    posts= []
    for div in mydivs:
        posts+=map(lambda p:p.text.encode('ascii' , error='replace').replace("?", " "), div.findAll("li"))
    return posts
#%%
doxyDonkeyPosts = []
for link in links:
    doxyDonkeyPosts += getDoxyDonkeyTest(link)

doxyDonkeyPosts


#------------------------------------------------------------------------------------
#%%
import requests
from bs4 import BeautifulSoup


page = requests.get('https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ1.htm')

soup = BeautifulSoup(page.text, 'html.parser')

# Remove bottom links
last_links = soup.find(class_='AlphaNav')
last_links.decompose()
#%%
artist_name_list = soup.find(class_='BodyText')
artist_name_list_items = artist_name_list.find_all('a')
#%%
for artist_name in artist_name_list_items:
    print(artist_name.prettify())

#%%
    
# Use .contents to pull out the <a> tag’s children
for artist_name in artist_name_list_items:
    names = artist_name.contents[0]
    print(names)
    
#%%
 for artist_name in artist_name_list_items:
     names = artist_name.contents[0]
     links = 'https://web.archive.org' + artist_name.get('href')
     print(names)
     print(links)
     
#%%
#save details in a csv
import requests
from bs4 import BeautifulSoup


page = requests.get('https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ1.htm')

soup = BeautifulSoup(page.text, 'html.parser')

# Remove bottom links
last_links = soup.find(class_='AlphaNav')
last_links.decompose()    
#%%
# Create a file to write to, add headers row
import csv
f = csv.writer(open('z-artist-names.csv', 'w')) 
f.writerow(['Name', 'Link'])

#%%
artist_name_list = soup.find(class_='BodyText')
artist_name_list_items = artist_name_list.find_all('a')

for artist_name in artist_name_list_items:
    names = artist_name.contents[0]
    links = 'https://web.archive.org' + artist_name.get('href')
    # Add each artist’s name and associated link to a row
    f.writerow([names, links])
    

#%%
#fetching data from multiple page   
import requests
import csv
from bs4 import BeautifulSoup


f = csv.writer(open('z-artist-names1.csv', 'w'))
f.writerow(['Name', 'Link'])

pages = []

for i in range(1,5):
    url = 'https://web.archive.org/web/20121007172955/https://www.nga.gov/collection/anZ'+str(i)+'.htm'
    pages.append(url)


for item in pages:
    page = requests.get(item)
    soup = BeautifulSoup(page.text, 'html.parser')

    last_links = soup.find(class_='AlphaNav')
    last_links.decompose()

    artist_name_list = soup.find(class_='BodyText')
    artist_name_list_items = artist_name_list.find_all('a')

    for artist_name in artist_name_list_items:
        names = artist_name.contents[0]
        links = 'https://web.archive.org' + artist_name.get('href')

        f.writerow([names, links])


#---------------------------------------------------------------------------------------------------------
#%%



















