a<-1
b<-2
c<-6
d<-11
a
b
c
d

# Arithmatic Operators
a+b
d-c
c*d
d/c

# Assignment operator
e<-15
e
f<<-2
f

25->g
g

# Relational operator
h<-10
h!=9


i<-FALSE
j<-TRUE
i
j
i&j
i|j
k<- 2:8
k
test<- "hello world"
test

#  vectors
  #classes of vectors
    #logical
a<-TRUE
class(a)
    #integer
a<-2L
class(a)
    #numeric
a<-2
class(a)
    #complex
a<-2+2i
class(a)
    #character
a<-"hello"
class(a)
    #raw
a<-charToRaw("hello")
class(a)



i<-c(1,2,3,4,5)
i
i<-c("test","test1","test2")
i

# Accessing vector elements using position.
t <- c("Sun","Mon","Tue","Wed","Thurs","Fri","Sat")
u <- t[c(2,3,6)]
print(u)

# Accessing vector elements using logical indexing.
v <- t[c(TRUE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE)]
print(v)

# Accessing vector elements using negative indexing.
x <- t[c(-2,-5)]
print(x)

v1 <- c(3,8,4,5,0,11)
v2 <- c(4,11)
# V2 becomes c(4,11,4,11,4,11)

add.result <- v1+v2
print(add.result)

sub.result <- v1-v2
print(sub.result)

# Create a list.
list1 <- list(c(2,5,3),21.3,sin)

# Print the list.
list1

c<- c(21,"test",23,"lol",11)
c
class(c)

c<- list(21,"test",23,"lol",11)
c
class(c)

# Create a list containing a vector, a matrix and a list.
list_data <- list(c("Jan","Feb","Mar"), matrix(c(3,9,5,1,-2,8), nrow = 2),
                  list("green",12.3))

# Give names to the elements in the list.
names(list_data) <- c("1st Quarter", "A_Matrix", "A Inner list")

# Show the list.
print(list_data)

list_data[3] <- "updated element"
print(list_data[3])

# Create two lists.
list1 <- list(1,2,3)
list2 <- list("Sun","Mon","Tue")

# Merge the two lists.
merged.list <- c(list1,list2)

# Print the merged list.
print(merged.list)

# Create lists.
list1 <- list(1:5)
print(list1)

list2 <-list(10:14)
print(list2)

# Convert the lists to vectors.
v1 <- unlist(list1)
v2 <- unlist(list2)



# Create a matrix.
M = matrix( c('a','a','b','c','b','a'), nrow = 2, ncol = 3, byrow = TRUE)
print(M)

M = matrix( c('a','a','b',2,3,5), nrow = 2, ncol = 3, byrow = TRUE)
print(M)

# Define the column and row names.
rownames = c("row1", "row2", "row3", "row4")
colnames = c("col1", "col2", "col3")

# Create the matrix.
P <- matrix(c(3:14), nrow = 4, byrow = TRUE, dimnames = list(rownames, colnames))

# Access the element at 3rd column and 1st row.
print(P[1,3])

# Access the element at 2nd column and 4th row.
print(P[4,2])

# Access only the  2nd row.
print(P[2,])

# Access only the 3rd column.
print(P[,3])




# Create an array.
a <- array(c('green','yellow'),dim = c(3,2,3))
print(a)

# Create two vectors of different lengths.
vector1 <- c(5,9,3)
vector2 <- c(10,11,12,13,14,15)
column.names <- c("COL1","COL2","COL3")
row.names <- c("ROW1","ROW2","ROW3")
matrix.names <- c("Matrix1","Matrix2")

# Take these vectors as input to the array.
result <- array(c(vector1,vector2),dim = c(3,3,2),dimnames = list(row.names,column.names,
                                                                  matrix.names))
print(result)


# Create a vector.
apple_colors <- c('green','green','yellow','red','red','red','green')

# Create a factor object.
factor_apple <- factor(apple_colors)

# Print the factor.
print(factor_apple)
print(nlevels(factor_apple))


# Create the data frame.
BMI <- 	data.frame(
  gender = c("Male", "Male","Female"), 
  height = c(152, 171.5, 165), 
  weight = c(81,93, 78),
  Age = c(42,38,26)
)
print(BMI)

# Create the data frame.
emp.data <- data.frame(
  emp_id = c (1:5), 
  emp_name = c("Rick","Dan","Michelle","Ryan","Gary"),
  salary = c(623.3,515.2,611.0,729.0,843.25), 
  
  start_date = as.Date(c("2012-01-01", "2013-09-23", "2014-11-15", "2014-05-11",
                         "2015-03-27")),
  stringsAsFactors = FALSE
)
# Print the data frame.			
print(emp.data)
emp.data[1:2,]
emp.data[,1:2]
emp.data[1,2]

var.1 = c(0,1,2,3)           

# Assignment using leftward operator.
var.2 <- c("learn","R")   

# Assignment using rightward operator.   
c(TRUE,1) -> var.3           

(var.1)
cat ("var.1 is ", var.1 ,"\n")
cat ("var.2 is ", var.2 ,"\n")
cat ("var.3 is ", var.3 ,"\n")

# to know all variables currently available
ls()

# List the variables starting with the pattern "text".
ls(pattern = "emp")

#to delete one variable

rm(a)
a

# All the variables can be deleted by using the rm() and ls() function together.
rm(list = ls())

#arithmatic operators for vectors
v <- c( 2,5.5,6)
t <- c(8, 3, 4)
v+t
v-t
v*t
v/t
v%%t
v%/%t

#relational operators for vectors

v <- c(2,5.5,6,9)
t <- c(8,2.5,14,9)
v<t
v>t
v<=t
v>=t
v==t
v!=t
v^t

#logical operators for vectors
v <- c(3,1,TRUE,2+3i)
t <- c(4,1,FALSE,2+3i)
v&t
v|t
!v

#assignment operators for vectors
v1 <- c(3,1,TRUE,2+3i)
v2 <<- c(3,1,TRUE,2+3i)
v3 = c(3,1,TRUE,2+3i)
v1
v2
v3

c(3,1,TRUE,2+3i) -> v1
c(3,1,TRUE,2+3i) ->> v2 
v1
v2

#miscellaneous operators for vectors
v <- 2:8
v
v1 <- 8
v2 <- 12
t <- 1:10
v1 %in% t 
v2 %in% t

# Create a sequence of numbers from 32 to 44.
print(seq(32,44))
print(seq(5, 9, by = 0.4))

# Find mean of numbers from 25 to 82.
print(mean(25:82))

# Find sum of numbers frm 41 to 68.
print(sum(41:68))

# Create a function to print squares of numbers in sequence.
ts <- function(a) {
  for(i in 1:a) {
    b <- i^2
    print(b)
  }
}

# Call the function new.function supplying 6 as an argument.
ts(6)

# Create a function with arguments.
new.function <- function(a,b,c) {
  result <- a * b + c
  print(result)
}

# Call the function by position of arguments.
new.function(5,3,11)

# Call the function by names of the arguments.
new.function(a = 11, b = 5, c = 3)

# Create a function with arguments.
new.function <- function(a, b) {
  print(a^2)
  print(a)
  print(b)
}

# Evaluate the function without supplying one of the arguments.
new.function(6)

# example of valid string

a <- 'Start and end with single quote'
print(a)

b <- "Start and end with double quotes"
print(b)

c <- "single quote ' in between double quotes"
print(c)

d <- 'Double quotes " in between single quote'
print(d)

# invalid string
f <- 'Single quote ' inside single quote'' 
print(f)

g <- "Double quotes " inside double quotes"    "
print(g)


# paste function
a <- "Hello"
b <- 'How'
c <- "are you? "

print(paste(a,b,c))

print(paste(a,b,c, sep = "-"))

print(paste(a,b,c, sep = "", collapse = ""))

# format functionality

# Total number of digits displayed. Last digit rounded off.
result <- format(23.123456789, digits = 9)
print(result)

# The minimum number of digits to the right of the decimal point.
result <- format(23.47, nsmall = 5)
print(result)

# Numbers are padded with blank in the beginning for width.
result <- format(13.7, width = 6)
print(result)

# Left justify strings.
result <- format("Hello", width = 8, justify = "l")
print(result)

# Justfy string with center.
result <- format("Hello", width = 8, justify = "c")
print(result)

# uppercase lowercase
# Changing to Upper case.
result <- toupper("Changing To Upper")
print(result)

# Changing to lower case.
result <- tolower("Changing To Lower")
print(result)

# no of characters
result <- nchar("Count the number of characters")
print(result)

# Extract characters from 5th to 7th position.
result <- substring("Extract", 5, 7)
print(result)

