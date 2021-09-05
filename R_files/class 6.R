#install package "corrplot" to find co ralation factor
library(corrplot)

test10<-read.csv(file.choose(), header = TRUE,sep = ",")
head(test10)
cr<-cor(test10)
cr
corrplot(cr,type = "full")
corrplot(cr,method  = "number")

library(car)
vif(jpimodel)
plot(test5$jpi_pred,test5$jpi_resi,col="blue")

qqnorm(test5$jpi_res, col="blue")
qqline(test5$jpi_res, col="blue")

shapiro.test(test5$jpi_res)

library(caret)


test11<-read.csv(file.choose(), header = TRUE,sep = ",")
test11
index<-createDataPartition(test11$claimamt,p=0.8,list = FALSE)
index
dim(index)
traindata<-test11[index,]
traindata
testdata<-test11[-index,]
testdata
dim(traindata)
dim(testdata)

model<-lm(claimamt~vehage+CC+Length+Weight,data = test11)
model
test11$claimamt_pred<-fitted(model)
test11$claimamt_res<-residuals(model)
test11
rmsetest11<-sqrt(mean(test11$claimamt_res**2))
rmsetest11


model1<-lm(claimamt~vehage+CC+Length+Weight,data = traindata)
model1
traindata$claimamt_pred<-fitted(model1)
traindata$claimamt_res<-residuals(model1)
traindata
rmsetraindata<-sqrt(mean(traindata$claimamt_res**2))
rmsetraindata

testdata$claimamt_pred<-predict(model1,testdata)
testdata$claimamt_res<- (testdata$claimamt-testdata$claimamt_pred)
testdata
rmsetestdata<-sqrt(mean(testdata$claimamt_res**2))
rmsetestdata

kfolds<-trainControl(method = "cv",number = 4)
kfolds
model3<-train(claimamt~vehage+CC+Length+Weight,data = test11,method="lm",trControl=kfolds)
model3



