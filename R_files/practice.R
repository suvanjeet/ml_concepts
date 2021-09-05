# for text file read
read.table("test.txt",header=TRUE, sep="\t", stringsAsFactors= FALSE)



data()  #lists all predefined datas in r

dim(dataset)  # gives no of rows and columns
names(dataset)  # gives column names
str(dataset)  # gives column wise info

library(dplyr)
glimpse(dataset)  # better than str

summary(dataset)   #mean median quantile for each column
head(dataset,n=15)  # will display first 15 rows
tail(dataset,n=15)  # will display first 15 rows

# view histogram
hist(dataset$column)

#scatter plot to see relation between two variables
plot(dataset$col1, dataset$ col2)

#will provide  min value, 25%, median, 75%,max value
fivenum(dataset$col)
##will provide  min value, 25%, median, 75%,max value
hist(dataset$col, breaks = 10)
#inter quadrant range
IQR(dataset$col)

boxplot(dataset$col)
boxplot(dataset$col1 ~ dataset$col1, main="head text", xlab="xaxisname", ylab="yaxisname", col= "red")
colors()   # gives names of colors available

library(lattice)
#alternate for boxplot
bwplot(dataset$col1 ~ dataset$col1, main="head text", xlab="xaxisname", ylab="yaxisname", col= "red")

#to edit the uploaded data
dataset<-edit(dataset)
#cleaninsurancedata<- edit(cleaninsurancedata)

# describe function will give more info than summary( column wise missing value and % quadrile)
install.packages("Hmisc")
library(Hmisc)
describe(dataset)
describe(dataset$col)


# table format output
xtab(~col1 + col2 , data=dataset)


