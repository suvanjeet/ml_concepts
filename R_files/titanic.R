titanic <- read.csv(choose.files(), header = T, sep = ",")
head(titanic)
str(titanic)
colSums(is.na(titanic))
names(titanic)
titanic1<-titanic[,-c(3,8,10,12,13,14)]
#alternet
#
names(titanic1)
str(titanic1)
titanic1$pclass <- as.factor(titanic1$pclass)
titanic1$survived <- as.factor(titanic1$survived)


titanic1$pclass <- as.factor(titanic1$pclass)
titanic1$survived <- as.factor(titanic1$survived)
titanic1$sex <- as.factor(titanic1$sex)
titanic1$embarked <- as.factor(titanic1$embarked)

Agefit <- rpart(age ~ pclass + sex + sibsp + parch + fare + embarked ,
                data=titanic1[!is.na(titanic1$age),], 
                method="anova")
install.packages('rpart')
library(rpart)
titanic1$age[is.na(titanic1$age)] <- predict(Agefit, titanic1[is.na(titanic1$age),])

colSums(is.na(titanic1))
summary(titanic1)

titanic2<- titanic1[-which(is.na(titanic1$survived)), ]
summary(titanic2)

titanic2<- titanic2[-which(is.na(titanic1$fare)), ]
summary(titanic2)

titanic2<- titanic2[-which(is.na(titanic1$pclass)), ]
summary(titanic2)

titanic2<- titanic2[-which(is.na(titanic1$parch)), ]
summary(titanic2)

titanic2<- titanic2[-which(is.na(titanic1$sibsp)), ]
summary(titanic2)

colSums(is.na(titanic2))

library(randomForest)
random_model<-randomForest(formula=survived~pclass+sex+age+sibsp+parch+fare+embarked,
                           data=titanic2)
summary(random_model)
random_model

titanic2$cabin<-NULL
titanic2$boat<-NULL
titanic2$home.dest<-NULL
set.seed(123)
index<-createDataPartition(titanic2$survived,p=0.7,list = FALSE)
train<-titanic2[index,]
test<-titanic2[-index,]

colSums(is.na(test))
names(test)
library(data.table)
setnames(test, old=c("pclass","survived","sex","age","sibsp","parch","fare","embarked"), new=c("Pclass","Survived","Sex","Age","Sibsp","Parch","Fare","Embarked"))
setnames(test, old=c("Sibsp"), new=c("SibSp"))

head(test)
random_model<-randomForest(formula=survived~pclass+sex+age+sibsp+parch+fare+embarked,
                           data=train)
summary(random_model)
plot(random_model)

train$predict123<-predict(random_model,train,type='response')
head(train)

tuneRF(train,train$predict123,ntreeTry = 350,stepFactor=1,improve = 0.05,trace=TRUE,plot=TRUE)
#chargemean<-mean(insurancedata$charges, na.rm=TRUE)
#insurancedata$charges[is.na(insurancedata$charges)] <- chargemean

random_model1<-randomForest(formula=survived~pclass+sex+age+sibsp+parch+fare+embarked,
                           data=train, ntree=350,mtry=3,importance=TRUE,cutoff=c(0.4,0.6))
random_model1

varImpPlot(random_model1)
varUsed(random_model1)
confusionMatrix(as.factor(train$predict123), train$survived)


head(train)

