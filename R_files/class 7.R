kfolds1<-trainControl(method = "repeatedcv",number=4, repeats=5)
kfolds1
model4<-train(claimamt~vehage+CC+Length+Weight,data = test11,method="lm",trControl=kfolds1)
model4

kfolds2<-trainControl(method = "LOOCV")
kfolds2
model4<-train(claimamt~vehage+CC+Length+Weight,data = test11,method="lm",trControl=kfolds2)
model4

test15<-read.csv(file.choose(), header = TRUE,sep = ",")
head(test15)
full<-lm(jpi~aptitude+tol+technical+general,data=test15)
null<-lm(jpi~1,data=test15)
step(null,scope=list(lower=null,upper=full),direction = "forward")

step(full,scope=list(lower=null,upper=full),direction = "backward")

step(full,scope=list(lower=null,upper=full),direction = "both")

casest<-read.csv(file.choose(), header = TRUE,sep = ",")
head(casest)
pairs(~Losses.in.Thousands+Age+Years.of.Experience+Number.of.Vehicles+Gender+Married, data=casest,col="blue")

model10<-lm(Losses.in.Thousands~Age+Years.of.Experience+Number.of.Vehicles+Gender+Married,data=casest)
summary(model10)

vif(model10)
model11<-lm(Losses.in.Thousands~Years.of.Experience+Gender+Married,data=casest)
summary(model11)

casest$loss_pred<-fitted(model10)
casest$loss_res<-residuals(model10)
casest
rmsecasest<-sqrt(mean(casest$loss_res**2))
rmsecasest

plot(casest$loss_pred,casest$loss_res,col="blue")

qqnorm(casest$loss_res, col="blue")
qqline(casest$loss_res, col="blue")

kfolds<-trainControl(method = "cv",number = 3)
kfolds
model18<-train(Losses.in.Thousands~Years.of.Experience+Gender+Married,data = casest,method="lm",trControl=kfolds)
model18


ncvTest(jpimodel,~aptitude+tol+technical+general)
ncvTest(model11,~Years.of.Experience+Gender+Married)

