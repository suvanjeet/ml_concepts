#------------------------------------------------------------
# read in the libraries we're going to use
library(tidyverse) # general utility & workflow functions
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(NLP)
library(tm) # general text mining functions, making document term matrixes
library(SnowballC) # for stemming

install.packages("topicmodels")
install.packages("tm")
install.packages("NLP")

# read in our data
reviews <- read_csv("C:\Users\suvanjeet\Desktop/deceptive-opinion.csv")
MyData <- read_csv(file="c:/Users/suvanjeet/Desktop/deceptive-opinion.csv", header=TRUE, sep=",")
MyData<-read_csv(choose.files())
head(MyData)
MyData$text



# function to get & plot the most informative terms by a specificed number
# of topics, using LDA
top_terms_by_topic_LDA <- function(text, # should be a columm from a dataframe
                                   plot = T, # return a plot? TRUE by defult
                                   number_of_topics = 4) # number of topics (4 by default)
{    
  # create a corpus (type of object expected by tm) and document term matrix
  #A vector source interprets each element of the vector x as a document.
  #corpus are collections of documents containing (natural language) text
  Corpus <- Corpus(VectorSource(text)) # make a corpus object
  DTM <- DocumentTermMatrix(Corpus) # get the count of words/document
  
  # remove any empty rows in our document term matrix (if there are any 
  # we'll get an error when we try to run our LDA)
  unique_indexes <- unique(DTM$i) # get the index of each unique value
  DTM <- DTM[unique_indexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  # get the top ten terms for each topic
  top_terms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness
  
  # if the user asks for a plot (TRUE by default)
  if(plot == T){
    # plot the top ten terms for each topic in order
    top_terms %>% # take the top terms
      mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # as a bar plot
      facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
      labs(x = NULL, y = "Beta") + # no x label, change y label 
      coord_flip() # turn bars sideways
  }else{ 
    # if the user does not request a plot
    # return a list of sorted terms instead
    return(top_terms)
  }
}



# plot top ten terms in the hotel reviews by topic
top_terms_by_topic_LDA(MyData$text,number_of_topics = 6)

#words are not informative due to stop words
#try reprocessing our text data removing stop words



# create a document term matrix to clean
reviewsCorpus <- Corpus(VectorSource(MyData$text)) 
inspect(reviewsCorpus)
reviewsDTM <- DocumentTermMatrix(reviewsCorpus)
inspect(reviewsDTM)

# convert the document term matrix to a tidytext corpus
reviewsDTM_tidy <- tidy(reviewsDTM)
reviewsDTM_tidy
# I'm going to add my own custom stop words that I don't think will be
# very informative in hotel reviews
custom_stop_words <- tibble(word = c("hotel", "room"))
custom_stop_words
# remove stopwords
reviewsDTM_tidy_cleaned <- reviewsDTM_tidy %>% # take our tidy dtm and...
  anti_join(stop_words, by = c("term" = "word")) %>% # remove English stopwords and...
  anti_join(custom_stop_words, by = c("term" = "word")) # remove my custom stopwords
reviewsDTM_tidy_cleaned
# reconstruct cleaned documents (so that each word shows up the correct number of times)
cleaned_documents <- reviewsDTM_tidy_cleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(term, count))) %>%
  select(document, terms) %>%
  unique()

# check out what the cleaned documents look like (should just be a bunch of content words)
# in alphabetic order
head(cleaned_documents)

# now let's look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 6)





#Stemming: Removing all the inflection from words. For instance, the root form of 
#"horses", "horse", and "horsing [around]" is the same word: "horse". For some NLP tasks, 
#we want to count all of these words as the same word.

#Let's try our LDA topic modeling again, this time after stemming our data.

# stem the words (e.g. convert each word to its stem, where applicable)
reviewsDTM_tidy_cleaned <- reviewsDTM_tidy_cleaned %>% 
  mutate(stem = wordStem(term))

# reconstruct our documents
cleaned_documents <- reviewsDTM_tidy_cleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(stem, count))) %>%
  select(document, terms) %>%
  unique()

head(cleaned_documents)
# now let's look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 6)

#In this instance, it doesn't actually look like stemming was actually that helpful in terms 
#of generating informative topics. We also don't know which (if either) of these topics are 
#associated with deceptive reviews and which are associated with truthful ones.


#In general, unsupervised methods (like LDA) are helpful for data exploration but supervised 
#methods (like TF-IDF, which we're going to learn next) are usually a better choice if you have 
#access to labeled data. We've started with ab unsupervised method here in order to give you the 
#chance to directly compare LDA and TF-IDF on the same dataset.

#The general idea behind how TF-IDF works is this:
  
#Words that are very common in a specific document are probably important to the topic of that document
#Words that are very common in all documents probably aren't important to the topics of any of them
#So a term will recieve a high weight if it's common in a specific document and also uncommon across all documents.



# function that takes in a dataframe and the name of the columns
# with the document texts and the topic labels. If plot is set to
# false it will return the tf-idf output rather than a plot.
top_terms_by_topic_tfidf <- function(text_df, text_column, group_column, plot = T){
  # name for the column we're going to unnest_tokens_ to
  # (you only need to worry about enquo stuff if you're
  # writing a function using using tidyverse packages)
  group_column <- enquo(group_column)
  text_column <- enquo(text_column)
  
  # get the count of each word in each review
  words <- text_df %>%
    unnest_tokens(word, !!text_column) %>%
    count(!!group_column, word) %>% 
    ungroup()
  
  # get the number of words per text
  total_words <- words %>% 
    group_by(!!group_column) %>% 
    summarize(total = sum(n))
  
  # combine the two dataframes we just made
  words <- left_join(words, total_words)
  
  # get the tf_idf & order the words by degree of relevence
  tf_idf <- words %>%
    bind_tf_idf(word, !!group_column, n) %>%
    select(-total) %>%
    arrange(desc(tf_idf)) %>%
    mutate(word = factor(word, levels = rev(unique(word))))
  
  if(plot == T){
    # convert "group" into a quote of a name
    # (this is due to funkiness with calling ggplot2
    # in functions)
    group_name <- quo_name(group_column)
    
    # plot the 10 most informative terms per topic
    tf_idf %>% 
      group_by(!!group_column) %>% 
      top_n(10) %>% 
      ungroup %>%
      ggplot(aes(word, tf_idf, fill = as.factor(group_name))) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "tf-idf") +
      facet_wrap(reformulate(group_name), scales = "free") +
      coord_flip()
  }else{
    # return the entire tf_idf dataframe
    return(tf_idf)
  }
}




# let's see what our most informative deceptive words are
top_terms_by_topic_tfidf(text_df = MyData, # dataframe
                         text_column = text, # column with text
                         group_column = deceptive, # column with topic label
                         plot = T) # return a plot


#Interesting! So it looks like false reviews tend to use a lot of glowing praise in their reviews 
#("pampered", "exquisite"), while truthful reviews tend to talk about how they booked thier room 
#("priceline", "hotwire").

#Since we have a function written, we can easily check out the most informative words for other 
#topics. This dataset is also annotated for whether the reveiw was positive or negative, 
#so let's see which words are associated with which polarity.

# look for the most informative words for postive and negative reveiws
top_terms_by_topic_tfidf(text_df = MyData, 
                         text_column = text, 
                         group = polarity, 
                         plot = T)


#From this, we can see that negative reviews include words like "worst", "broken", "odor" and "stains"
#, while positive reviews really harp on the bathrobes (both "robes" and "bathrobes" show up in the 
#top ten words).

# get just the tf-idf output for the hotel topics
reviews_tfidf_byHotel <- top_terms_by_topic_tfidf(text_df = reviews, 
                                                  text_column = text, 
                                                  group = hotel, 
                                                  plot = F)

# do our own plotting
reviews_tfidf_byHotel  %>% 
  group_by(hotel) %>% 
  top_n(5) %>% 
  ungroup %>%
  ggplot(aes(word, tf_idf, fill = hotel)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~hotel, ncol = 4, scales = "free", ) +
  coord_flip()
















txt1 <- c("Hello to you.", "Blah me, too.")
txt1
library(tm)
VectorSource(txt1)
corp1 <- Corpus(VectorSource(txt1))
dtm1 <- DocumentTermMatrix(corp1)#, control = list(removePunctuation = TRUE, stopwords=TRUE))
inspect(corp1)
inspect(dtm1)
unique_indexes1 <- unique(dtm1$i) # get the index of each unique value
unique_indexes1
dtm1 <- dtm1[unique_indexes1,]
dtm1


docs <- c("This is a text.", "This another one.")
(vs <- VectorSource(docs))
vs
inspect(VCorpus(vs))
