#%%
# lexicons are special dictionaries or vocabularies that have been created for analyzing sentiments.
#Most of these lexicons have a list of positive and negative polar words with some score associated with them,
# and using various techniques like the position of words, surrounding words, context, parts of speech, 
#phrases, and so on, scores are assigned to the text documents for which we want to compute the sentiment. 
#After aggregating these scores, we get the final sentiment.
'''Various popular lexicons are used for sentiment analysis, including the following.

AFINN lexicon
Bing Liu’s lexicon
MPQA subjectivity lexicon
SentiWordNet
VADER lexicon
TextBlob lexicon'''
#%%
#Simplifying Sentiment Analysis using VADER in Python (on Social Media Text)

import pandas as pd
import numpy as np
#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#%%
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
#%%
sentiment_analyzer_scores("The phone is super cool.")
#The Compound score is a metric that calculates the sum of all the lexicon ratings which have been 
#normalized between -1(most extreme negative) and +1 (most extreme positive)
#%%
#an increase in the number of (!), increases the magnitude accordingly.
print(sentiment_analyzer_scores("The food here is good"))
print(sentiment_analyzer_scores("The food here is good!"))
print(sentiment_analyzer_scores("The food here is good!!"))
sentiment_analyzer_scores("The food here is good!!!")

#%%
#Using upper case letters to emphasize a sentiment-relevant word in the presence of other 
#non-capitalized words, increases the magnitude of the sentiment intensity
print(sentiment_analyzer_scores("The food here is great!!!"))
sentiment_analyzer_scores("The food here is GREAT!!!")
#%%
#Degree modifiers: Also called intensifiers, they impact the sentiment intensity by either increasing 
#or decreasing the intensity. For example, “The service here is extremely good” is more intense than “The 
#service here is good”, whereas “The service here is marginally good” reduces the intensity.
print(sentiment_analyzer_scores("The service here is good"))
print(sentiment_analyzer_scores("The service here is extremely good"))
sentiment_analyzer_scores("The service here is marginally good")

#%%
'''Conjunctions: Use of conjunctions like “but” signals a shift in sentiment polarity,
 with the sentiment of the text following the conjunction being dominant. 
 “The food here is great, but the service is horrible” has mixed sentiment, with the 
 latter half dictating the overall rating.'''
sentiment_analyzer_scores("The food here is great but the service is horrible")

#%%
print(sentiment_analyzer_scores('I am 😄 today'))
print(sentiment_analyzer_scores('😊'))
print(sentiment_analyzer_scores('😥'))
print(sentiment_analyzer_scores('☹️'))
#%%
print(sentiment_analyzer_scores("Today SUX!"))
print(sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol"))
#%%
print(sentiment_analyzer_scores("Make sure you :) or :D today!"))





















