sales<- read.csv(choose.files(), header = T, sep = "," )
sales
salesmodel<-lm(formula = sales~print+online,data = sales)
summary(salesmodel)

library(car)
durbinWatsonTest(salesmodel)
durbinWatsonTest(jpimodel)

banks<- read.csv(choose.files(), header = T, sep = "," )
influencePlot(jpimodel,id.method="identify",main="Influence Plot", sub="Circle size propertial to Cook's Distance")
peridex_inf<-test5[-(33),]

jpimodel20<-lm(jpi~aptitude+tol+technical+general,data=peridex_inf)
jpimodel20
summary(jpimodel20)
influencePlot(jpimodel20,id.method="identify",main="Influence Plot", sub="Circle size propertial to Cook's Distance")

#binary logistic regression
bankloan<-read.csv(choose.files(), header = T, sep = "," )
head(bankloan)
bankloan$AGE<- factor(bankloan$AGE)
str(bankloan)

riskmodel<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT+OTHDEBT,family = binomial,data=bankloan)
summary(riskmodel)

#removing OTHDEBT
riskmodel1<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT,family = binomial,data=bankloan)
summary(riskmodel1)

null<-glm(DEFAULTER~1,family = binomial,data=bankloan)
anova(null,riskmodel1,test="Chisq")

bankloan$predprob<-round(fitted(riskmodel1),2)
head(bankloan)

library(gmodels)
table(bankloan$DEFAULTER,fitted(riskmodel1)>0.5)
table(bankloan$DEFAULTER,fitted(riskmodel1)>0.4)
table(bankloan$DEFAULTER,fitted(riskmodel1)>0.3)
table(bankloan$DEFAULTER,fitted(riskmodel1)>0.2)
table(bankloan$DEFAULTER,fitted(riskmodel1)>0.1)

library(ROCR)

bankloan$predprob<-fitted(riskmodel1)
pred<-prediction(bankloan$predprob,bankloan$DEFAULTER)
pred
pref<-performance(pred,"tpr","fpr")
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1)
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.05))
abline(0,1)

auc<-performance(pred,"auc")
auc@y.values

riskmodel1
coef(riskmodel1)
exp(coef(riskmodel1))
bankloan1<-bankloan[-c(152,187,281),]

riskmodel2<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT,family = binomial,data=bankloan1)
summary(riskmodel1)
influencePlot(riskmodel2)
vif(riskmodel2)
dim(bankloan1)

bankloan1
library(caret)
index<-createDataPartition(bankloan1$DEFAULTER,p=0.7,list = FALSE)
index
dim(index)
traindata1<-bankloan1[index,]
traindata1
testdata1<-bankloan1[-index,]
testdata1
dim(traindata1)
dim(testdata1)

model<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT,family = binomial,data=bankloan1)
model
traindata1$predprob<-predict(model,traindata1,type='response')
traindata1$predY<-ifelse(traindata1$predprob>0.30,1,0)
confusionMatrix(as.factor(traindata1$predY),as.factor(traindata1$DEFAULTER),positive ="1")
pred<-prediction(traindata1$predprob,traindata1$DEFAULTER)
pref<-performance(pred,"tpr","fpr")
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1)

auc<-performance(pred,"auc")
auc@y.values


testdata1$predprob<-predict(model,testdata1,type='response')
testdata1$predY<-ifelse(testdata1$predprob>0.30,1,0)
confusionMatrix(as.factor(testdata1$predY),as.factor(testdata1$DEFAULTER),positive ="1")
pred1<-prediction(testdata1$predprob,testdata1$DEFAULTER)
pref1<-performance(pred1,"tpr","fpr")
plot(pref1,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1)

auc<-performance(pred,"auc")
auc@y.values





