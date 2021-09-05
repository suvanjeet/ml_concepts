data<-readLines(file.choose())
head(data)

#convert into corpus
library("tm")
corp<-Corpus(VectorSource(data))
corp<-tm_map(corp,tolower)
corp<-tm_map(corp,removePunctuation)
corp<-tm_map(corp,removeNumbers)
corp<-tm_map(corp,removeWords,stopwords("english"))
writeLines(as.character(corp[]))
corp
inspect(corp[1:3])
dtm<-TermDocumentMatrix(corp)
m<-as.matrix(dtm)
m
findFreqTerms(dtm)
findFreqTerms(dtm,10)
findAssocs(dtm,'fixing',0.30)
library(wordcloud)
install.packages("wordcloud")
v<-sort(rowSums(m),decreasing=TRUE)
mynames<- names(v)
mynames
d<-data.frame(word=mynames,freq=v)
d
pal2<- brewer.pal(8,"Dark2")
pal2
wordcloud(d$word,d$freq,random.order = FALSE, min.freq = 1,colors=pal2)
term.freq<-rowSums(m)
term.freq<-subset(term.freq,term.freq>=10)
term.freq
df<-data.frame(term=names(term.freq),freq=term.freq)
df
library(ggplot2)
ggplot(df,aes(x=term,y=freq))+geom_bar(stat="identity")+xlab("Terms")+ylab("count")+coord_flip()











