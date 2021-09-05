names(getModelInfo())

#svm   support vector machines
bankloan10<-read.csv(choose.files(), header = T, sep = "," )
head(bankloan10)
str(bankloan10)

bankloan10$AGE<-as.factor((bankloan10$AGE))
library(e1071)
model<-svm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT+OTHDEBT,data=bankloan10,
           type="C",probability=TRUE,kernel="linear")

model
pred1<- predict(model,bankloan10,probability=TRUE)
pred1

pred2<-attr(pred1,"probabilities")[,1]
pred2
library(ROCR)
pred<-prediction(pred2,bankloan10$DEFAULTER)
pref<-performance(pred,"tpr","fpr")
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1)

par(mfrow=c(1,1))  

auc<-performance(pred,"auc")
auc@y.values

model2<-svm(formula=DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT+OTHDEBT,data=bankloan10,
            type="C",probability=TRUE,kernel="polynomial")
model2

library(e1071)
plot(iris,col=iris$Species)
plot(iris$Petal.Length,iris$Petal.Width,col=iris$Species) #better distungish result so we will consider the petal one
plot(iris$Sepal.Length,iris$Sepal.Width,col=iris$Species)

s<-sample(150,100)
col<-c("Petal.Length","Petal.Width","Species")
s

iris_train<-iris[s,col] # 3columns of 100 data set to train
iris_test<-iris[-s,col] # 3columns of 50 data set to test

#as species can be easily separable so taken linearly line 36
svmfit<-svm(Species~.,data=iris_train,kernel="linear",cost=.1,scale=F) 
print(svmfit)
plot(svmfit,iris_train[,col])
#table(,iris_train[,3])

tuned<-tune(svm,Species~.,data=iris_train,kernel="linear",ranges=list
            (cost=c(0.001,0.01,.1,10,100)))
summary(tuned)
# rerun the model in case best cost value changes and plot to check if misclassification value reduced

p<-predict(svmfit,iris_test[,col],type="class")
plot(svmfit,iris_test[,col])
plot(p)
table(p,iris_test[,3])
