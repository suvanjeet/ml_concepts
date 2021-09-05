list1<-list(a=c(1,2,3,4,5), b=c(3,2,4,5,6,7), c=c(9,8,10,191,98))
sapply(list1,mean)
lapply(list1,mean)

d<-c(a,b)
d
random<-c("this","is","data","science","this","is","a","c")
lapply(random,nchar)
nchar(random)
sapply(random,nchar)

a<-c(1,2,3,4,1,2,3,4)
length(random)
length(a)

tapply(a,random,mean)


v1<-c(1,2,3)
v2<-c(10,11,12,13,14,15,16)
result<-array(c(v1,v2),dim<-c(3,3,2))
result
result[1,3,1]
result[1,3,2]
result[3,,2]
result[,,2]

n<-matrix(c(3:14),nrow=4,byrow=t)
n
n[1,3]<-15
n[1,3]<-NA

n<-matrix(c(3:15),nrow=4,byrow=t)
n

data<-c("ab","bc","ca","ab","bc")
class(data)
a<-factor(data)
a
class(a)



basicsal<-read.csv(file.choose(), header = TRUE,sep = ",")
basicsal

f<-function(x) c(mean=mean(x,na.rm = TRUE),median= median(x,na.rm = TRUE),sd=sd(x,na.rm = TRUE))
ba<-f(basicsal$ba)
ms<-f(basicsal$ms)
ab<-data.frame(ba,ms)
ab


aggregate(cbind(ba,ms)~Location, data = basicsal, FUN = f )
aggregate(cbind(ba,ms)~Location+Grade, data = basicsal, FUN = f )

abc<-table(basicsal$Location,basicsal$Grade)
abc
prop.table(abc)


prop.table(abc,1)
prop.table(abc,2)

basicsal

table1<- table(basicsal$Location,basicsal$Grade,basicsal$Function)
table1
ftable(table1)

quantile(basicsal$ba,na.rm = TRUE)
quantile(basicsal$ba,prob=c(0.1,0.5,0.8),na.rm = TRUE)

boxplot(ba~Grade,data=basicsal)
boxplot(ba~Grade+Location,data=basicsal,col=(c("blue","red","green","yellow")))
boxplot(ms~Location,data=basicsal,col=(c("blue","red")))
+

hist(basicsal$ba,col="blue")

install.packages("e1071")
library(e1071)

skewness(basicsal$ba,na.rm = T,type = 2)
kurtosis(basicsal$ba,na.rm = T,type = 2)