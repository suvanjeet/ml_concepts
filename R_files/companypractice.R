library(tidyverse)      # data manipulation & plotting
library(stringr)        # text cleaning and regular expressions
library(tidytext)       # provides additional text mining functions
library(harrypotter)    # provides the first seven novels of the Harry Potter series
install.packages("tidyverse")
install.packages("tidytext")
install.packages("harrypotter")
install.packages("devtools")

if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}

devtools::install_github("bradleyboehmke/harrypotter")


library(dplyr)
sentiments
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

titles <- c("Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
            "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince",
            "Deathly Hallows")
titles

books <- list(philosophers_stone, chamber_of_secrets, prisoner_of_azkaban,
              goblet_of_fire, order_of_the_phoenix, half_blood_prince,
              deathly_hallows)
books[1]

series <- tibble()
series

for(i in seq_along(titles)) {
  
  clean <- tibble(chapter = seq_along(books[[i]]),
                  text = books[[i]]) %>%
    unnest_tokens(word, text) %>%       #split the column per word
    mutate(book = titles[i]) %>%        #to add new variable
    select(book,everything())
  
  series <- rbind(series, clean)
}
str(series)
series

# set factor to keep books in order of publication
series$book <- factor(series$book, levels = rev(titles))
#alternative
series$book <- as.factor(series$book)

#Now lets use the nrc sentiment data set to assess the different sentiments 
#that are represented across the Harry Potter series.
#We can see that there is a stronger negative presence than positive.
series %>%
  right_join(get_sentiments("nrc")) %>%
  filter(!is.na(sentiment)) %>%
  count(sentiment, sort = TRUE)

#create an index that breaks up each book by 500 words; this is the approximate number 
#of words on every two pages so this will allow us to assess changes in sentiment even 
#within chapters
#join the bing lexicon with inner_join to assess the positive vs. negative sentiment of each 
#word count up how many positive and negative words there are for every two pages"
#spread our data and.
#calculate a net sentiment (positive - negative)
#plot our data

series %>%
  group_by(book) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 500 + 1) %>% 
  inner_join(get_sentiments("bing")) %>%
  count(book, index = index , sentiment) %>%
  ungroup() %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative,
         book = factor(book, levels = titles)) %>%
  ggplot(aes(index, sentiment, fill = book)) +
  geom_bar(alpha = 0.5, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ book, ncol = 2, scales = "free_x")

#Lets use all three sentiment lexicons and examine how they differ for each novel.
afinn <- series %>%
  group_by(book) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 500 + 1) %>% 
  inner_join(get_sentiments("afinn")) %>%
  group_by(book, index) %>%
  summarise(sentiment = sum(score)) %>%
  mutate(method = "AFINN")

bing_and_nrc <- bind_rows(series %>%
                            group_by(book) %>% 
                            mutate(word_count = 1:n(),
                                   index = word_count %/% 500 + 1) %>% 
                            inner_join(get_sentiments("bing")) %>%
                            mutate(method = "Bing"),
                          series %>%
                            group_by(book) %>% 
                            mutate(word_count = 1:n(),
                                   index = word_count %/% 500 + 1) %>%
                            inner_join(get_sentiments("nrc") %>%
                                         filter(sentiment %in% c("positive", "negative"))) %>%
                            mutate(method = "NRC")) %>%
  count(book, method, index = index , sentiment) %>%
  ungroup() %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  select(book, index, method, sentiment)

bind_rows(afinn, 
          bing_and_nrc) %>%
  ungroup() %>%
  mutate(book = factor(book, levels = titles)) %>%
  ggplot(aes(index, sentiment, fill = method)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_grid(book ~ method)

#One advantage of having the data frame with both sentiment and word 
#is that we can analyze word counts that contribute to each sentiment.

bing_word_counts <- series %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts


#We can view this visually to assess the top n words for each sentiment:
  
  bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ggplot(aes(reorder(word, n), n, fill = sentiment)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip()
  
  tibble(text = philosophers_stone) %>% 
    unnest_tokens(sentence, text, token = "sentences")

  str(philosophers_stone)
  length(philosophers_stone)
  
  
  ps_sentences <- tibble(chapter = 1:length(philosophers_stone),
                         text = philosophers_stone) %>% 
    unnest_tokens(sentence, text, token = "sentences")
  
  
  

  
  
  
  
  
  
