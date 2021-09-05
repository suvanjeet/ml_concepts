data("AirPassengers")
AirPassengers
class(AirPassengers) #datatype
start(AirPassengers)
end(AirPassengers)

frequency(AirPassengers)
plot(AirPassengers)
boxplot(AirPassengers)
summary(AirPassengers)

cycle(AirPassengers)     # cycle is required to check seasonal effect

plot(AirPassengers)
model<-lm(AirPassengers~time(AirPassengers))  # time is rwquires to check trend effect
abline(model)

boxplot(AirPassengers~cycle(AirPassengers),col='red')

#---------------------------------------------------------------

gdp <- read.csv(choose.files(), header = T, sep = "," )
head(gdp)
gdpseries<-ts(gdp$GDP,start = 1950,end = 2006)
plot(gdpseries,col='red')     # non-stationary from the graph 
acf(gdpseries)       # auto corelation factor  1st line is 0th lag and so on
                      # autocorelation value is low at 12th point , slow decay represents non stationary
pacf(gdpseries,col='blue') #pacf is checking corelation after removing dependant component.
                            # pacf value is 2
gdpdiff<-diff(gdpseries,differences = 1)
plot(gdpdiff)

gdpdiff<-diff(gdpseries,differences = 2)  # 2 will be acceptable as on value 3 this is fully stationary
plot(gdpdiff)

gdpdiff<-diff(gdpseries,differences = 3)
plot(gdpdiff)

install.packages("forecast")
library(forecast)

ndiffs(gdpseries)   # gives value as 2

gdpdiff2<-diff(gdpseries, differences = 2)
plot(gdpdiff2,col='red')

acf(gdpdiff2)


#----------------------------------
sericegdp <- read.csv(choose.files(), header = T, sep = "," )
head(sericegdp)

servicets<-ts(sericegdp$Sales,start = c(2014,1),end = c(2015,12),frequency=12)
servicets
plot(servicets,col='red')

acf(servicets)              #q= 5 before diff
pacf(servicets,col='blue')  #d=differentiating factor

ndiffs(servicets)           #d=1 also pacf value

servicetsdiff<-diff(servicets, differences = 1)
plot(servicetsdiff,col='red')

acf(servicetsdiff)    #p= acf of diff series =1 

fit<-arima(log(servicets), c(1,1,5))   #in order as p,d,q
fit

pre
pred<- predict(fit,n.ahead= 10*12)
pred1<-2.718*pred$pred
pred1
