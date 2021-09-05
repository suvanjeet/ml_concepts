
install.packages('titanic')
library(titanic)

head(titanic_train)
summary(titanic_train)

colSums(is.na(titanic_train))
names(titanic_train)
titanic_train <- subset(titanic_train, select = -c(1,4,9,11))

str(titanic_train)

titanic_train$Pclass <- as.factor(titanic_train$Pclass)
titanic_train$Survived <- as.factor(titanic_train$Survived)
titanic_train$Sex <- as.factor(titanic_train$Sex)
titanic_train$Embarked <- as.factor(titanic_train$Embarked)

library(car)
library(caret)
library(rpart)
colSums(is.na(titanic_train))
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                data=titanic_train[!is.na(titanic_train$Age),], 
                method="anova")
titanic_train$Age[is.na(titanic_train$Age)] <- predict(Agefit, titanic_train[is.na(titanic_train$Age),])
summary(titanic_train)
set.seed(123)
library(randomForest)

RF <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=titanic_train)
RF
plot(RF)

## ntree = 300
## mtry = ?

pred <- predict(RF, titanic_train, type = "response")
confusionMatrix(pred, titanic_train$Survived)
set.seed(123)
tune_rf <- tuneRF(titanic_train, pred, ntreeTry = 350, stepFactor = 0.5, improve = 0.05, ttrace = T, plot = T)

RF1 <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, 
                    data=titanic_train, ntree = 350, 
                    mtry = 4, importance = T)
RF1
plot(RF1)

pred1 <- predict(RF1, titanic_train, type = "response")
confusionMatrix(pred1, titanic_train$Survived)

#########################################################


head(titanic_test)
titanic_test <- subset(titanic_test, select = -c(1,3,8,10))
colSums(is.na(titanic_test))
summary(titanic_test)
names(titanic_test)
#titanic_test <- subset(titanic_test, select = -c(1,3,8,10))

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                data=titanic_test[!is.na(titanic_test$Age),], 
                method="anova")
titanic_test$Age[is.na(titanic_test$Age)] <- predict(Agefit, titanic_test[is.na(titanic_test$Age),])


titanic_test<- titanic_test[-which(is.na(titanic_test$Fare)), ]
summary(titanic_test)
str(titanic_test)
str(titanic_train)

str(titanic_test)

titanic_test$Pclass <- as.factor(titanic_test$Pclass)
titanic_test$Sex <- as.factor(titanic_test$Sex)
titanic_test$Embarked <- as.factor(titanic_test$Embarked)
titanic_test$Survived <- as.factor(titanic_test$Survived)

names(titanic_test)
names(titanic_train)

titanic_test$Survived <- 0
head(titanic_test)

predTest<- predict(RF1, data = test,type="response")
predTest
titanic_test$Survived<-predTest

confusionMatrix(predTest, titanic_train$Survived)

hist(treesize(RF1))
varImpPlot(RF1)
importance(RF1)
varUsed(RF1)
attributes(RF1)


