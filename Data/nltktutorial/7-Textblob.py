
# coding: utf-8

# In[ ]:


#pip install -U textblob  #at the prompt


# In[2]:


from textblob import TextBlob

#creating a textblob
wiki = TextBlob("Python is a high-level, general-purpose programming language.")


# In[3]:


#pos tagging

wiki.tags


# In[4]:


# word tokenize

mytext = TextBlob("""Life is beautiful. 
                  Enjoy the life you have before its too late. 
                  Travel a lot and make unforgettable memories. """)
mytext.words


# In[5]:


# sentence tokenize
mytext.sentences


# In[6]:


#word inflection

sentence = TextBlob('Use 4 spaces per indentation level.')
print(sentence.words)

print(sentence.words[2].singularize())
print(sentence.words[-1].pluralize())
print(sentence.words.pluralize())


# In[46]:


#lemmatization

from textblob import Word

w = Word("octopi")
print(w.lemmatize())

w = Word("went")
print(w.lemmatize("v"))  #pass POS


# In[47]:


# spelling correction

b = TextBlob("I havv goood speling!")
print(b.correct())


# In[49]:


# textblobs are like strings
print(wiki)
print(wiki[0:30])

print(wiki.upper())

print(wiki.find("programming"))


# In[7]:


# n grams

wiki.ngrams(n=4)


# In[9]:


#sentiment analysis

text = TextBlob("Life is beautiful")
text.sentiment


# In[53]:


for sentence in mytext.sentences:
    print(sentence.sentiment)


# # Building a text classification system

# In[16]:


#Loading data

train = [
        ('I love this sandwich.', 'pos'),
        ('this is an amazing place!', 'pos'),
        ('I feel very good about these beers.', 'pos'),
        ('this is my best work.', 'pos'),
        ("what an awesome view", 'pos'),
        ('I do not like this restaurant', 'neg'),
        ('I am tired of this stuff.', 'neg'),
        ("I can't deal with this", 'neg'),
        ('he is my sworn enemy!', 'neg'),
        ('my boss is horrible.', 'neg'),
        ('the beer was good.', 'pos'),
        ("I feel amazing!", 'pos'),
        ]
test = [
        ('the beer was good.', 'pos'),
        ('I do not enjoy my job', 'neg'),
        ("I ain't feeling dandy today.", 'neg'),
        ("I feel amazing!", 'pos'),
        ('Gary is a friend of mine.', 'pos'),
        ("I can't believe I'm doing this.", 'neg')
    ]


# In[17]:


#training the Naive Bayes Classifier
from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(train)


# In[29]:


#classify based on the training data
cl.classify("my boss is horrible")


# In[31]:


#find the probability distribution
prob_dist = cl.prob_classify("I am very happy with their customer support")
print(prob_dist.max())

print(round(prob_dist.prob("pos"), 2))
print(round(prob_dist.prob("neg"), 2))


# In[30]:


cl.accuracy(test)

