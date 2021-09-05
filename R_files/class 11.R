
#randomForest

empdata1<-read.csv(choose.files(), header = T, sep = "," )
head(empdata1)
install.packages("randomForest")
library(randomForest)

empdata1$status<-as.factor(empdata1$status)
str(empdata1)
empdata2<-subset(empdata1,select = c(-sn))
head(empdata2)
churn_rf<-randomForest(formula=status~function.+exp+gender+source,
    data=empdata2,mtry=2,ntree=100,importance=TRUE,cutoff=c(0.7,0.3))
churn_rf
churn_rf$predicted  #randomforest with predicted class
empdata2$predict<-churn_rf$predicted

plot(churn_rf)  # error for 0,1, and median

#increase no of trees
churn_rf<-randomForest(formula=status~function.+exp+gender+source,
                       data=empdata2,mtry=2,ntree=500,importance=TRUE,cutoff=c(0.7,0.3))
churn_rf

churn_rf$importance

##train data and test data  homework

index<-createDataPartition(empdata2$SN,p=0.7,list= FALSE)
head(index)
#Remove age,sn and defaulter from data as they are in factor format

traindata<-empdata2[index,]
traindata
testdata<-empdata2[-index,]
testdata

churn_rf<-randomForest(formula=status~function.+exp+gender+source,
                       data=empdata2,mtry=2,ntree=100,importance=TRUE,cutoff=c(0.7,0.3))
churn_rf



head(iris)
install.packages("rpart")
library(rpart)

fit<-rpart(Sepal.Length~Sepal.Width+Petal.Length+
             Petal.Width+Species,method="anova",data=iris)
fit

plot(fit,uniform=TRUE,main="Regression Tree for sepal length")
text(fit,use.n = TRUE)

printcp(fit)
par(mfrow=c(1,2))   # 1row and 2 columns for plot, two graphs can view in 1 row
rsq.rpart(fit)
testdata <-data.frame(Species='setosa',Sepal.Width=4,Petal.Length=1.2,Petal.Width=0.3)
predict(fit,testdata,method="anova")

#association rules
install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)

data("Groceries")
head(Groceries)
help(Groceries)
itemFrequencyPlot(Groceries,topN=20,type="absolute")

rules<-apriori(Groceries,parameter=list(supp=0.001,conf=0.8))
options(digits=2)
rules
inspect(rules[1:5])

#rest in class 12