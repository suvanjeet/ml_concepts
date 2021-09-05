salary<-read.csv(file.choose(), header = TRUE,sep = ",")
glucose<- salary
glucose

t.test(glucose$C1,alternative = "greater", mu=90)

test1<-read.csv(file.choose(), header = TRUE,sep = ",")
test1
t.test(test1$X1,test1$X2,alternative = "two.sided", var.equal = TRUE)

test2<-read.csv(file.choose(), header = TRUE,sep = ",")
test2
t.test(test1$X1,test1$X2,alternative = "greater", paired  = TRUE)

test3<-read.csv(file.choose(), header = TRUE,sep = ",")
test3

test6<-read.csv(file.choose(), header = TRUE,sep = ",")
test6
test7<- melt (test6)
test7
test6<- dcast(test7)
test6
library(reshape2)

anov1<-aov(formula = Weight~comm,data = test3 )
summary(anov1)

test4<-read.csv(file.choose(), header = TRUE,sep = ",")
test4
annov2<-aov(formula = satindex~dept+exp+dept*exp,data=test4)
summary(annov2)

m<-mean(c(5,17,11,8,14,5))
m

test5<-read.csv(file.choose(), header = TRUE,sep = ",")
test5
pairs(~jpi+aptitude+tol+technical+general,data=test5,col="blue")
pairs(~tol+technical+general,data=test5,col="blue")
cor.test(test5$jpi,test5$aptitude,method = "pearson")
cor.test(test5$jpi,test5$tol,method = "pearson")
cor.test(test5$jpi,test5$technical,method = "pearson")
cor.test(test5$jpi,test5$general,method = "pearson")


jpimodel<-lm(jpi~aptitude+tol+technical+general,data=test5)
jpimodel
summary(jpimodel)

test5$jpi_pred<-fitted(jpimodel)
test5$jpi_res<-residuals(jpimodel)
head(test5)




