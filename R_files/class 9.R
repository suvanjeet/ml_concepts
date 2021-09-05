#stepwise logistic regression
bankloan<-read.csv(choose.files(), header = T, sep = "," )
head(bankloan)
bankloan$AGE<- factor(bankloan$AGE)
str(bankloan)
null<-glm(DEFAULTER~1,family = binomial,data=bankloan)
full<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT+OTHDEBT,family = binomial,data=bankloan)
step(full,scpe=list(lower=null,upper=full),direction = "backward")


#case study on file Poject2_dataset
# change the loan status Y=1 and N=0 in excel or r
bankloan<-read.csv(choose.files(), header = T, sep = "," )

bankloan$Loan_Status<- as.factor(bankloan$Loan_Status) #as the dependant varible is in int so convert to factor
str(bankloan)
#in r to replace y=1 and n=0  bankloan$Loan_Status<-ifelse(bankloan$Loan_Status=="Y",1,0)

#create model summary to remove variable or process step method to remove(below)
modelloan<-glm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area,family = binomial,data=bankloan)
summary(modelloan)

#or step method to remove column
null<-glm(Loan_Status~1,family = binomial,data=bankloan)
anova(null,modelloan,test="Chisq")
full<-glm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area,family = binomial,data=bankloan)
step(full,scope=list(lower=null,upper=full),direction = "backward")

# after summary/step method the variable needs to keep are: Education   LoanAmount Property_Area  Married  Credit_History
modelloan1<-glm(Loan_Status~Education+LoanAmount+Property_Area+Married+Credit_History,family = binomial,data=bankloan)
summary(modelloan1)

influencePlot(modelloan1)
bankloan$predprob<-fitted(modelloan1)
head(bankloan)
table(bankloan$Loan_Status,fitted(modelloan1)>0.74)
table(bankloan$Loan_Status,fitted(modelloan1)>0.8)

library(ROCR)

bankloan$predprob<-fitted(modelloan1)
pred<-prediction(bankloan$predprob,bankloan$Loan_Status)
pref<-performance(pred,"tpr","fpr")
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1)
plot(pref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.05))
abline(0,1)


bankloan$predY<-ifelse(bankloan$predprob>0.75,1,0)
confusionMatrix(as.factor(bankloan$predY),as.factor(bankloan$Loan_Status),positive ="1")
coef(modelloan1)
exp(coef(modelloan1))

auc<-performance(pred,"auc")
auc@y.values

## conducting cross validation by finding the RMSE. method = repeated kfold ##
kfold<- trainControl(method = "repeatedcv", number = 4, repeats = 5)
modelloan2 <- train(Loan_Status~Education+LoanAmount+Property_Area+Married+Credit_History,data = bankloan, method = "glm", trControl = kfold )
modelloan2
vif(modelloan1)

head(bankloan)
p1 <- predict(modelloan1, data.frame(Education = 'Not Graduate', LoanAmount = 250, Property_Area = 'Rural', Married = 'Yes', Credit_History = 1))
p1



# need to complete

#Clusterization
townins<-read.csv(choose.files(), header = T, sep = "," )
townins
#as town is character so to remove town variable
townins2<-subset(townins,select = c(-Town))
d<- dist(townins2,method = "euclidean")
d
install.packages("cluster")
fit<-hclust(d,method = "single")
fit
plot(fit)
plot(fit,label=townins$Town,col="blue")

townins$segment<-cutree(fit,k=3)
townins
townins$segment<-cutree(fit,k=4)
townins


