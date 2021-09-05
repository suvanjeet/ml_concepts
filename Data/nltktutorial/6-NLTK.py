
# coding: utf-8

# In[2]:


import nltk
from bs4 import BeautifulSoup
 
import urllib.request
 
response = urllib.request.urlopen('https://wiki.python.org/moin/BeginnersGuide')
 
html = response.read()

#print(html)

#extract data in a clearer way
soup = BeautifulSoup(html,features="xml")
 
text = soup.get_text(strip=True)
 
print (text)


# In[8]:


# tokenize text using split()

tokens = [t for t in text.split()]
 
print (tokens)


# In[9]:


# count word frequency

freq = nltk.FreqDist(tokens)
print(freq)
#type(freq)
for key,val in freq.items():
 
    print (str(key) + ':' + str(val))


# In[10]:


freq.plot(20, cumulative=False)


# In[11]:


# removing stop words

from nltk.corpus import stopwords
 
clean_tokens = tokens[:]  # copy of the list
 
#sr = stopwords.words('english')
#print(sr) 
for token in tokens:
 
    if token in stopwords.words('english'):
 
        clean_tokens.remove(token)

newfreq = nltk.FreqDist(clean_tokens)
 
for key,val in newfreq.items():
 
    print (str(key) + ':' + str(val))
    


# In[12]:


#nltk.download()


# In[13]:


newfreq.plot(30, cumulative=False)


# In[14]:


# tokenize sentence

from nltk.tokenize import sent_tokenize
 
mytext = "Hello Nikita, how are you? I hope everything is going well. Have a good day, see you soon."
 
print(sent_tokenize(mytext))


# In[15]:


# tokenize words

from nltk.tokenize import word_tokenize
 
mytext = "Hello Ms. Nikita, how are you? I hope everything is going well. Have a good day, see you soon."
 
print(word_tokenize(mytext))


# In[16]:


# get synonyms

from nltk.corpus import wordnet
 
syn = wordnet.synsets("science")
 
print(syn[1].definition())
 
print(syn[1].examples())


# In[2]:


# stemming
# Word stemming means removing affixes from words and return the root word

from nltk.stem import PorterStemmer
 
stemmer = PorterStemmer()
 
print(stemmer.stem('beautiful'))


# In[18]:


from nltk.stem import PorterStemmer
 
stemmer = PorterStemmer()
 
print(stemmer.stem('able'))  #does not return the correct word


# In[19]:


#lemmatizing

from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
 
print(lemmatizer.lemmatize('increases'))


# In[20]:


from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
 
print(lemmatizer.lemmatize('brought', pos="v"))  #verb
 
print(lemmatizer.lemmatize('match', pos="n"))  #noun
 
print(lemmatizer.lemmatize('challenging', pos="a"))  #adjective
 
print(lemmatizer.lemmatize('wonderful', pos="r"))  #adverb


# In[21]:


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

sent = "We won the match because we played well."
text= pos_tag(word_tokenize(sent), tagset='universal')
print(text)


# In[26]:


#named entity recognizer

#my_sent = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
#my_sent= "In 1999, Vajpayee laid the foundation for the GoldenQuadrilateralHighway project, which would link four major cities: delhi, Mumbai, chennai and kolkata."
my_sent="“Indians is a football country now,” FIFA president Giani Infantino declared after arriving here to chair the FIFA Council meeting on Friday and attend the U-17 World Cup final."

#entity_list=nltk.ne_chunk(my_sent, binary=True)
#print(entity_list)

for sent in nltk.sent_tokenize(my_sent):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        if hasattr(chunk, 'label'):
            print(chunk.label(), ' '.join(c[0] for c in chunk))


# # counting no. of POS

# In[ ]:


# Import data and tagger
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents

# Load tokenized tweets
tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
#print(tweets_tokens)

# Tag tagged tweets
tweets_tagged = pos_tag_sents(tweets_tokens)
#print(tweets_tagged)

# Set accumulators
JJ_count = 0
NN_count = 0

# Loop through list of tweets
for tweet in tweets_tagged:
    for pair in tweet:
        tag = pair[1]
        if tag == 'JJ':
            JJ_count += 1
        elif tag == 'NN':
            NN_count += 1

# Print total numbers for each adjectives and nouns
print('Total number of adjectives = ', JJ_count)
print('Total number of nouns = ', NN_count)


# # Gender Prediction Program

# In[29]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def gender_features(word): 
    return {'last_letter': word[-1]} 
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
    [(name, 'female') for name in names.words('female.txt')])
 
#print(names)
featuresets = [(gender_features(n), g) for (n,g) in names] 
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
 
# Predict
name = input("Name: ")
print(classifier.classify(gender_features(name)))

