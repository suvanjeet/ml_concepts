############################################################

frauddata <- read.csv(choose.files(), header = T, sep = ",")
head(frauddata)
str(frauddata)
frauddata$Class <- as.factor(frauddata$Class)
frauddata$Time <- NULL
colSums(is.na(frauddata))
summary(frauddata)

model <- glm(Class~., frauddata, family = "binomial")
model
summary(model)
names(frauddata)
#null<-glm(Class~1,family = binomial,data=frauddata)
#full<-glm(Class~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+Amount,family = binomial,data=frauddata)
#step(full,scpe=list(lower=null,upper=full),direction = "both")

##important parameters are V1, V4, V5,V8, V9, V10, V13, V14, V20,
## V21,V22, V27, V28, Amount.

model1 <- glm(Class~V1+V4+V8+V9+V13+V14+V20+V21+V22+Amount, data = frauddata, family = "binomial")
summary(model1)
library(car)
vif(model1)
plot(model1)

pre <- fitted(model1, frauddata)
pre1 <- ifelse(pre>0.5, 1,0)
library(caret)
confusionMatrix(as.factor(pre1), frauddata$Class)

res <- residuals(model1)
predictionvalues<- prediction(pre1, frauddata$Class)
perf <- performance(predictionvalues, "tpr", "fpr")

plot(perf, colorize =T, print.cutoffs.at = seq(0.1,by = 0.1))

pre_r <- fitted(model1, frauddata)
pre1_r <- ifelse(pre>0.002, 1,0)
confusionMatrix(as.factor(pre1_r), frauddata$Class)

head(frauddata)
head(pre1_r)

pr123 <- predict(model1, data.frame(Amount = 149.62
                                    , V1 = -1.359807134
                                    , V4 = 1.378155224
                                    , V5 = -0.33832077
                                    , V8 = 0.098697901
                                    , V9 = 0.36378697
                                    , V13 = -0.991389847
                                    , V14= -0.311169354
                                    , V20= 0.251412098
                                    , V21 = -0.018306778
                                    , V22 = 0.277837576
                                    , V27 = 0.133558377
                                    , V28 = -0.021053053), type = "response")

pr123

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
modelCV <- train(Class~V1+V4+V5+V8+V9+V13+V14+V20+V21+V22+V27+V28+Amount, data = frauddata, method = "glm", family = "binomial", trControl = ctrl)
modelCV
summary(modelCV)

exp(coef(model1))

###################################################################

