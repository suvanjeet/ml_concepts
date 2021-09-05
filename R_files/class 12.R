#market basket analysis
#sort by confidence
rules<-sort(rules,by="confidence",decreasing = TRUE)

#top 5 output will be sorted by confidence and thererfore the most relevant rules will be appear
options(digits = 2)
inspect(rules[1:5])

#to know what cutomers likely to buy before wholemilk
rules<-apriori(data=Groceries,parameter=list(supp=0.001,conf=0.8,minlen=2),
               appearance = list(default="lhs",rhs="whole milk"),control = list(verbose=F) )

rules<-sort(rules,decreasing=TRUE,by="confidence")
inspect(rules[1:5])

#to know what cutomers likely to buy after wholemilk
rules<-apriori(data=Groceries,parameter=list(supp=0.001,conf=0.15,minlen=2),
               appearance = list(default="rhs",lhs="whole milk"),control = list(verbose=F) )

rules<-sort(rules,decreasing=TRUE,by="confidence")
inspect(rules[1:5])

library(arulesViz)
#size of the bubble represents the strength of the relation
#interactive represents the naming of the units in graph
plot(rules,method="graph",interactive=TRUE,shading=NA)
plot(rules,method="graph",interactive=TRUE)  #shading relates with the lift value
plot(rules,method="graph")


#PCA/Factor analysis

NAR10<-read.csv(choose.files(), header = T, sep = "," )
NAR<-NAR10
head(NAR)
NAR<- subset(NAR,select=c(-Country))

# as some variables are mins and seconds so better to scale the wholw data set
#output is the ranking for al country and no result variable is not available use PCA
NAR1<- data.frame(scale(NAR))
NAR1
pc<-princomp(formula=~.,data=NAR1,cor=T)
summary(pc)
pc$loadings

#store scores of first pc in a variable called "performance"
pc$scores
NAR$performance<-pc$scores[,1]
NAR
rank<-rank(NAR$performance,na.last = TRUE)
NAR10$rank<-rank
NAR10$performance<-pc$scores[,1]
NAR10
NAR10[order(NAR10$performance),]
# to verify principle components are uncorrelated
round(cor(pc$scores))

salesp<-read.csv(choose.files(), header = T, sep = "," )
head(salesp)
model<-lm(SALES~AD+PRO+SALEXP+ADPRE+PROPRE,data = salesp)

summary(model)
vif(model)
model<-lm(SALES~SALEXP+PRO,data = salesp)

#now shift to PCA
salesp1<-salesp
salesp1$SALES<-NULL
salesp1$SRNO<-NULL

head(salesp1)
pc<-princomp(formula=~.,data=salesp1,cor=T)
summary(pc)

#summary decides the ncomp value for pcr regression
library(pls)
pcmodel<-pcr(SALES~AD+PRO+SALEXP+ADPRE+PROPRE,ncomp=3,data = salesp,scale=TRUE)
salesp$spread1<-predict(model,salesp)
salesp$spread2<-predict(pcmodel,salesp,ncomp=3)
head(salesp)
salesp$res_lm<-residuals(model)
salesp$res_pc<-(salesp$SALES-salesp$spread2)
head(salesp)
sqrt(mean(salesp$res_lm**2))
sqrt(mean(salesp$res_pc**2))
#as rmse of lm model is low so the 1st model(MLR)is better
#here we will consider the 2nd model as higher vif present in 1st model(and rmse value diff in both models are low)

