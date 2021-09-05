a<-10
a
e<-8
e
class(a)
b<-"test"
b
class(b)

c<-FALSE
d<-FALSE
c
d
c&d
c|d
c%d

salary<-read.csv(file.choose(), header = TRUE,sep = ",")
newemp<-read.csv(file.choose(), header = TRUE,sep = ",")
bonusdata<-read.csv(file.choose(), header = TRUE,sep = ",")
basicsalary<-read.csv(file.choose(), header = TRUE,sep = ",")

head(salary)
head(newemp)
head(bonusdata)
head(basicsalary)

nmiss<-sum(is.na(basicsalary$ba))
nmiss
mean(basicsalary$ba)
mean(basicsalary$ba,na.rm = TRUE)

basicsalary[c(5:10),]
basicsalary[c(1,3,5,9),]
basicsalary[,c(1,2)]

test<-basicsalary[c(1:15),c(1,6)]
test
basicsalary[,6]
basicsalary[,sum(6,7)]

basicsalary
subset(basicsalary,Location=="DELHI" | Location== "MUMBAI" & ba>10000)
subset(basicsalary,Location== "MUMBAI" & ms<10000)
subset(basicsalary,select = c(First_Name,Last_Name))
subset(basicsalary,select = c(1,2))
subset(basicsalary,Grade=="GR1" & ba>15000,select = c(1,2))
subset(basicsalary,select = c(1,2),Grade=="GR1" & ba>15000)
subset(basicsalary,!(Grade=="GR1") & !(Location=="MUMBAI"))
basicsalary[order(basicsalary$ba),]
attach(basicsalary)
basicsalary[order(ba),]
detach(basicsalary)
basicsalary[order(-basicsalary$ba),]
basicsalary[order(-ba),]
basicsalary[order(basicsalary$Grade),]
basicsalary[order(Grade),]
basicsalary[order(basicsalary$Grade,decreasing = TRUE),]
basicsalary[order(basicsalary$Grade,decreasing = !TRUE),]
basicsalary[order(basicsalary$Grade,basicsalary$ba),]
basicsalary[order(basicsalary$Location,basicsalary$ms),]

bonusdata
salary
merge(salary,bonusdata,by=c("Employee_ID"),all=TRUE)
merge(salary,bonusdata,by=c("Employee_ID"),all.x=TRUE)
merge(salary,bonusdata,by=c("Employee_ID"),all.y=TRUE)
merge(salary,bonusdata,by=c("Employee_ID"))

rbind(salary,newemp)
salary
newemp

salaryclmname<-read.csv(file.choose(), header = TRUE,sep = ",")
salaryclmnrmv<-read.csv(file.choose(), header = TRUE,sep = ",")

rbind(salaryclmname,newemp)
rbind(salaryclmnrmv,newemp)

aggregate(basicsalary$ba~basicsalary$Location, data = basicsalary, FUN = mean )
aggregate(basicsalary$ms~basicsalary$Grade, data = basicsalary, FUN = mean )
aggregate(basicsalary$ms~basicsalary$Location+Grade, data = basicsalary, FUN = mean )
