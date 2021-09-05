#KNN Classification

loan<-read.csv(choose.files(), header = T, sep = "," )
head(loan)
str(loan)
library(caret)
index<-createDataPartition(loan$SN,p=0.7,list= FALSE)
head(index)
#Remove age,sn and defaulter from data as they are in factor format
bankloan2<-subset(loan,select = c(-AGE,-SN,-DEFAULTER))
bankloan2
bankloan3<-scale(bankloan2)
head(bankloan3)
traindata<-bankloan3[index,]
traindata
testdata<-bankloan3[-index,]
testdata

#create class vectors
Ytrain<-loan$DEFAULTER[index]
Ytrain
Ytest<-loan$DEFAULTER[-index]
Ytest

library(class)
model<-knn(traindata,testdata,k=21,cl=Ytrain)
model
table(Ytest,model)
confusionMatrix(as.factor(Ytest),as.factor(model),positive ="Yes")

#K means clustering method

bankloanca<-subset(loan,select = c(-AGE,-SN,-DEFAULTER))
cl<-kmeans(bankloanca,3)
cl
cl$cluster

bankloanca$clustor<-cl$cluster
bankloanca

bankloanca$clustor<-NULL

#decission tree

empdata<-read.csv(choose.files(), header = T, sep = "," )
head(empdata)
str(empdata)
churn<-glm(formula = status~function.+exp+gender+source,family = binomial,data=empdata)
summary(churn)

install.packages("partykit")
#or
install.packages("CHAID",repos="htttp://R-Forge.R-project.org",type="source")
library(CHAID)

empdata$status<-as.factor(empdata$status)
ctree<-partykit::ctree(formula = status~function.+exp+gender+source,data=empdata)
plot(ctree,type="simple")

#alternate
ctree<-CHAID(formula = status~function.+exp+gender+source,data=empdata)
plot(ctree,type="simple")

