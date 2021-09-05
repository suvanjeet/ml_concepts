insurancedata <- read.csv(choose.files(), header = T, sep = "," )
head(insurancedata)
#data("insurancedata")
#data()
#install.packages('arules')
#library(arules)
#data(Adult)
#edit(Adult)


## check if there is any outlier in the dependent variable ##
boxplot(insurancedata$charges)

summary(insurancedata$charges)
library(Hmisc)
describe(insurancedata$charges)
describe(insurancedata)

class(insurancedata)
# if want to delete all rows having null values column wise
insurancedata<- insurancedata[-which(is.na(insurancedata$charges)), ]
insurancedata<- insurancedata[-which(is.na(insurancedata$bmi)), ]

#alternate : if want to assign mean values to null for columns... do for all columns having null
chargemean<-mean(insurancedata$charges, na.rm=TRUE)
insurancedata$charges[is.na(insurancedata$charges)] <- chargemean

hist(insurancedata$charges, breaks = 10)
hist(insurancedata$charges)

quant<-fivenum(insurancedata$charges)
quant[2]
plot(insurancedata$charges,insurancedata$bmi)
pairs(~age+sex+bmi+children+smoker+region+charges, data = insurancedata)
names(insurancedata)

## removing the outliers from dependent variable ##
Q1 <- quantile(insurancedata$charges,na.rm = T)
Q1

#delta <- Q1[4]- Q1[2]
#delta

Rdelta <- 1.5*IQR(insurancedata$charges)
Rdelta

Llimit <- Q1[2]-Rdelta
Llimit

Ulimit <- Q1[4]+Rdelta
Ulimit

WOOL <- which(insurancedata$charges<=Ulimit & insurancedata$charges>=Llimit)
WOOL
#insurancedata[15,]
#WOOL1 <- which(insurancedata$charges>Ulimit | insurancedata$charges<Llimit)
#WOOL1
head(WOOL)
cleaninsurancedata <- insurancedata[WOOL,]
head(cleaninsurancedata)
tail(cleaninsurancedata)
str(cleaninsurancedata)
boxplot(cleaninsurancedata$charges)

## checking if the dependent variable is normally distributed or not ##
qqtestc <- qqnorm(cleaninsurancedata$charges,col = 3)
qqtestcl <- qqline(cleaninsurancedata$charges, col = 3)
m
## dependent variable is not normally distributed ##

## lets check AIC to select the variables 
nullinsurance <- lm(charges~1, data = cleaninsurancedata)
fullinsurance <- lm(charges~age+sex+bmi+children+smoker+region, data = cleaninsurancedata)
stepwiseinsurance <- step(nullinsurance, scope= list(lower = nullinsurance, upper = fullinsurance),  direction = "both")
## after AIC test selected variables are (smoker, age, bmi, children, region) ##

## so lets develop a linear model on 5 selected variable ##
insurancemodel <- lm(charges~age+bmi+children+smoker+region, data = cleaninsurancedata)
insurancemodel2 <- lm(charges~age+bmi+children+smoker+region, data = insurancedata)

## lets check VIF for each variable to find if there is any multicollinearity exists ##
library(car)
vif(insurancemodel)
## there is no multicollinearity between the variable ##

## lets check for Rsq of the model to know whether we can accept the model or not ##
summary(insurancemodel)
insurancemodel
summary(insurancemodel2)
coefficients(insurancemodel1)
## Rsq value is more than 0.6 for both the models, hence we can accept both the models ##

## lets conduct the RMSE test. Need to calculate the RMSE of total data and we will apply the repeated kfold test##
## calculating RMSE of total data ##
cleaninsurancedata$fit <- fitted(insurancemodel, data = cleaninsurancedata)
cleaninsurancedata$res <- residuals(insurancemodel, data = cleaninsurancedata)
head(cleaninsurancedata)
RMSEinsurancedata <- sqrt(mean(cleaninsurancedata$res**2))
RMSEinsurancedata

## conducting cross validation by finding the RMSE. method = repeated kfold ##
library(caret)
kfoldinsurance <- trainControl(method = "repeatedcv", number = 4, repeats = 5)
insurancemodelCV <- train(charges~age+bmi+children+smoker+region,data = cleaninsurancedata, method = "lm", trControl = kfoldinsurance )
insurancemodelCV

## comparison of two RMSE ##
RMSEinsurancedata
insurancemodelCV$results [1,2]

## checking if residuals are normally distributed ##
qqtest <- qqnorm(cleaninsurancedata$res,col = 3)
qqtestl <- qqline(cleaninsurancedata$res,  col = 3)
stest <- shapiro.test(cleaninsurancedata$res)
stest
## p value of shapiro test is very low, hence we can REJECT the NULL HYPOTHEIS. 
## conclusion - residuals are not normally distributed ##

## checking the relation between fitted and residuals ##
plot(cleaninsurancedata$res, cleaninsurancedata$fit, col = 3)
plot(cleaninsurancedata$fit, cleaninsurancedata$res, col = 3)
head(cleaninsurancedata)
p1 <- predict(insurancemodel, data.frame(age = 30, bmi = 24, children = 1, smoker = 1, region = 2))
p2 <- predict(insurancemodel, data.frame(age = 30, bmi = 24, children = 1, smoker = 0, region = 2))
p3 <- predict(insurancemodel, data.frame(age = 30, bmi = 24, children = 1, smoker = 0, region = 2))
p4 <- predict(insurancemodel, data.frame(age = 35, bmi = 24, children = 1, smoker = 0, region = 2))
p5 <- predict(insurancemodel, data.frame(age = 32, bmi = 28, children = 0, smoker = 0, region = 3))


p21 <- predict(insurancemodel2, data.frame(age = 30, bmi = 24, children = 1, smoker = 1, region = 2))
p22 <- predict(insurancemodel2, data.frame(age = 30, bmi = 24, children = 1, smoker = 0, region = 2))
p23 <- predict(insurancemodel2, data.frame(age = 30, bmi = 24, children = 1, smoker = 0, region = 2))
p24 <- predict(insurancemodel2, data.frame(age = 35, bmi = 24, children = 1, smoker = 0, region = 2))
p25 <- predict(insurancemodel2, data.frame(age = 32, bmi = 28, children = 0, smoker = 0, region = 3))

resultmodel1 <- c(p1,p2,p3,p4,p5)
resultmodel1

resultmodel2 <- c(p21,p22,p23,p24,p25)
resultmodel2
