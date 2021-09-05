iris

iris1<- iris

iris1[iris1]

library(plyr)
revalue(x, c("beta"="two", "gamma"="three"))
#> [1] alpha two   three alpha two  
#> Levels: alpha two three
library(plyr)
iris1$Species<-mapvalues(iris1$Species, from = c("setosa", "versicolor","virginica"), to = c(1,2,3))
str(iris1)
head(iris1)
iris1
set.seed(123)
library(caret)

index<-createDataPartition(iris1$Species,p=0.7,list=FALSE)

traindata<-iris1[index,]
traindata
testdata<-iris1[-index,]
testdata
train1<-traindata[-5]
test1<-testdata[-5]

train1
test1

ytrain<- traindata$Species
head(ytrain)
ytrain
ytest<- testdata$Species
ytest

library(class)
model<-knn(train1,test1,cl=ytrain,k=11)
model1<-knn(train1,test1,cl=ytrain,k=11,l = 6)
model2<-knn(train1,test1,cl=ytrain,k=11,prob = TRUE)
model3<-knn(train1,test1,cl=ytrain,k=11,prob = TRUE,use.all = TRUE)

model
model1
model2
summary(model3)
summary(model2)
summary(model)
table(model,ytest)
confusionMatrix(model,ytest)

train2<-scale(train1)
train2
test2<-scale(test1)
model5<-knn(train2,test2,cl=ytrain,k=11)
summary(model5)
confusionMatrix(model5,ytest)

model6<-knn(train2,test2,cl=ytrain,k=11,prob = TRUE,use.all = TRUE)
summary(model6)
confusionMatrix(model5,ytest)

#########################################
candata <- read.csv(choose.files(), header = T, sep = ",")
head(candata)
candata1<-candata[-1]
head(candata)

distace<-dist(candata1,method = "euclidean")
head(distace)

library(cluster)
fit<-hclust(distace,method = "complete")
plot(fit,hang=-1,labels = candata$PAM50.mRNA,col="blue")

candata$clust<-cutree(fit,k=4)
candata$clust
candata$clust<-cutree(fit,k=8)
names(candata)
candata[,c(1,58)]