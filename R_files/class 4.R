
ONE.SAMPLE.t.TEST
t.test(ONE.SAMPLE.t.TEST$Time, alternative="greater",mu=90)

abc<-read.csv(file.choose(), header = TRUE,sep = ",")
abc
t.test(abc$time_g1,abc$time_g2,alternative = "two.sided",var.equal =TRUE)
var.test(abc$time_g1,abc$time_g2,alternative = "two.sided")


abc1<-read.csv(file.choose(), header = TRUE,sep = ",")
abc1
t.test(time~group, alternative="two.sided",,data=abc1,var.equal =TRUE)
abc2<-read.csv(file.choose(), header = TRUE,sep = ",")
abc2

cor.test(abc2$aptitude,abc2$job_prof, alternative="two.sided",method = "pearson")
plot(abc2$aptitude,abc2$job_prof)

abc3<-read.csv(file.choose(), header = TRUE,sep = ",")
abc3
t.test(abc3$time_before,abc3$time_after,alternative = "greater",paired  =TRUE)


abc4<-read.csv(file.choose(), header = TRUE,sep = ",")
abc4
test<-aov(formula = satindex~dept,data = abc4)
test
summary(test)

abc5<-read.csv(file.choose(), header = TRUE,sep = ",")
abc5
abc6<-read.csv(file.choose(), header = TRUE,sep = ",")
abc6
