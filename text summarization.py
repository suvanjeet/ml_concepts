# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:34:50 2018

@author: suvanjeet
"""

#%%

# Store url
url = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'

#%%

# Import `requests`
import requests

# Make the request and check object type
r = requests.get(url)
type(r)

#%%
# Extract HTML from Response object and print
html = r.text
print(html)

#%%

# Import BeautifulSoup from bs4# Impor 
from bs4 import BeautifulSoup


# Create a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, "html5lib")
type(soup)
print(soup)

#%%
# Get soup title
soup.title

#%%
# Get soup title as string
soup.title.string

#%%
# Get hyperlinks from soup and check out first 10
#soup.findAll('a')[:8]
soup.findAll('a')

#%%

# Get the text out of the soup and print it
text = soup.get_text()
print(text)
#%%


'''While not mandatory to do at this stage prior to tokenization (you'll find that this statement 
is the norm for the relatively flexible ordering of text data preprocessing tasks), replacing 
contractions with their expansions can be beneficial at this point, since our word tokenizer will
 split words like "didn't" into "did" and "n't." It's not impossible to remedy this tokenization at 
 a later stage, but doing so prior makes it easier and more straightforward.'''
 
#Contractions.download()
from pycontractions import Contractions

#%%
from pycontractions import Contractions
#import contractions 

 
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

text = replace_contractions(text)
print(text)

#%%
# Import regex package
import re

# Find all words in Moby Dick and print several
tokens = re.findall('\w+', text)
#tokens[:8]
print(tokens)

#%%
# Import RegexpTokenizer from nltk.tokenize
from nltk.tokenize import RegexpTokenizer

# Create tokenizer
tokenizer = RegexpTokenizer('\w+')
print(tokenizer)

#%%

# Create tokens
tokens = tokenizer.tokenize(text)
#tokens[:8]
print(tokens)

#%%

# Initialize new list
words = []


# Loop through list tokens and make lower case
for word in tokens:
    words.append(word.lower())


# Print several items from list as sanity check
words[:8]

#%%
import nltk

#%%
nltk.download()

#%%
# Import nltk
import nltk
#%%
# Get English stopwords and print some of them
sw = nltk.corpus.stopwords.words('english')
sw[:5]

#You want the list of all words in words that are not in sw. One way to get this 
#list is to loop over all elements of words and add the to a new list if they are not in sw:

#%%
# Initialize new list
words_ns = []

# Add to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)

# Print several list items as sanity check
words_ns[:5]

#%%
#Import datavis libraries
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Figures inline and set visualization style
%matplotlib inline
#%%
sns.set()
#%%
# Create freq dist and plot
freqdist1 = nltk.FreqDist(words_ns)
freqdist1.plot(25)


#%%

# Import stopwords from sklearn
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# Add sklearn stopwords to words_sw
sw = set(sw + list(ENGLISH_STOP_WORDS))

# Initialize new list
words_ns = []


# Add to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)


# Create freq dist and plot
freqdist2 = nltk.FreqDist(words_ns)

freqdist2.plot(25)

#%%
print(words_ns)

#%%
def plot_word_freq(url):
    """Takes a url (from Project Gutenberg) and plots a word frequency
    distribution"""
    # Make the request and check object type
    r = requests.get(url)
    # Extract HTML from Response object and print
    html = r.text
    # Create a BeautifulSoup object from the HTML
    soup = BeautifulSoup(html, "html5lib")
    # Get the text out of the soup and print it
    text = soup.get_text()
    # Create tokenizer
    tokenizer = RegexpTokenizer('\w+')
    # Create tokens
    tokens = tokenizer.tokenize(text)
    # Initialize new list
    words = []
    # Loop through list tokens and make lower case
    for word in tokens:
        words.append(word.lower())
    # Get English stopwords and print some of them
    sw = nltk.corpus.stopwords.words('english')
    # Initialize new list
    words_ns = []
    # Add to words_ns all words that are in words but not in sw
    for word in words:
        if word not in sw:
            words_ns.append(word)
    # Create freq dist and plot
    freqdist1 = nltk.FreqDist(words_ns)
    freqdist1.plot(25)
    
plot_word_freq('https://www.gutenberg.org/files/42671/42671-h/42671-h.htm')
    
#%%
import gensim
install --upgrade gensim

#%%   
#word to vector
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
#%%
# define training data
sentences = [words_ns]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()



















