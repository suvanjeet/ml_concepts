# -*- coding: utf-8 -*-

"""multiline
comments"""

'''this is amultiple comment
where code '''

#%%
a=5;d=10
b,c=3.2,"hello"
x=y=z="world"
#%%
print("value of a",a,"value of d",d)
print(b)
print(c)
print(x)
print(y,z)
#%%
print(1/3)   #factor output
print(1//3)  #integer output
print(7%3)

#%%
a=5
b=3.5
print(type(b))  #float
print(a+b)
#%%
a="5"
b="3.2"
print(a+b)      #53.2
print(type(b))  #str
#%%
#user input
name=input("enter your name")   # will take value in console
print("hello",name)
#%%
x=3;y=5
if x<y:
    print('x is less than y')
elif x>y:
    print(' x is greater than y')
else:
    print(' x and y are equal')
print('done')

#%%
x=3.5; y=3.4
if x==y:
    print('x and y are equal')
else:
    if x<y:
        print('x is less than y')
    else:
        print('x is greater than y')
print("done")
#%%
choice=int(float(input("enter 1/2/3: ")))     
if choice == 1:
    print('bad guess') 
elif choice == 2:
    print('good guess')    
elif choice == 3:
    print('close guess')
else:
    input1=input('enter a correct no')
    print('the correct no is ',input1)
print("done")

#%%
x=5
y=8

if x<7 and y<7:  # for and &  or |
#if x<7 or y<7:
    print(' x is a positive single digit no')
else:
    print('different')
  
    
#%%
#function
def square(a):       # def for defining function
    print("value is ",a)
    return a*a

value=input("enter a value")
square(int(value)) 
    
#%%
count=0
while(count<5):
    print(count)
    count+=1
else:
    print("count value reached",count)
    
#%%
friends=['a','b', 'c',3.6]    
for f in friends:
    print("happy diwali",f)
type(friends)
    
#%%
a=5
print(a, " is of type ",type(a))

a=30.6
print(a, " is of type ",type(a))

a=10+2j
print(a, " is of type ",type(a))

#%%
print(int(-2.8))  

print(float(5))  

 #%%
import math
print(math.sqrt(4))
print(math.floor(-10.7))
print(math.pow(2,3))
print(math.pi)
print(math.factorial(6))
type(math.pi)
  
 #%%
import random

print(random.random())  #default value for random function is 0 to 1
print(random.randrange(50,100))  #randrange function is used if donot want of range b/w 0 to 1

x=['a', 'b', 'c', 'd', 'e']
print(random.choice(x))
  
 #%%
my_string="hello world"
print(my_string)  
my_string= """python is a programmng language.
python is interesting.
this is it."""
print(my_string)
my_string="python is programming\n it is good"
print(my_string)


#%%

#slicing

my_string= "python is a programmng language.python is interesting."
print(my_string[0])
print(my_string[-3])    # from the end of the string

#[inclusive:exclusive]
print(my_string[3:10])  #9th is inclusive and 10th onwards exclusive
print(my_string[:25])
print(my_string[25:])
print(my_string[:])
#print(my_string[100])   # throw error as 100th character is not available in string (index out of range)
#%%
my_string[3] ='b' #will throw error,'str' object does not support item assignment

#%%
del my_string   # for deleting
print(my_string)    # name 'my_string' is not defined as deleted
    
 #%%
str1='hello'
str2='world'
str1=str1+" "+str2+" "+"done"# for concadination
print(str1)

 #%%
 print("PyTHon".lower())        # to lower case

string="python,is,s,programming language."
print(string.split(sep=","))            # separting by comma in list format

print('happy diwali'.replace('p','d'))    # replace p with d
str.count

 #%%
a= (5,'python',10+2j)
print(type(a))
print(a[0])

#%%
c=()
print(type(c))

c=tuple("python")   #tuple function will hild each character in text provided as an element
print(c)

#%%
n_tuple=("mouse",[8,4,6],[1,2,3])
print(n_tuple[1])

print(n_tuple[1][2])

n_tuple[1][2]=9  # points to 6 and can be replaced as it is in form of list
print(n_tuple)

# n_tuple[2][2]=0       # points to 2 and can not be replaced as it is in form of tuple

del n_tuple[1][2]       # can be deleted as in form of list
print(n_tuple)

#%%
mylist1=[1,3,5,4,1]
mylist2=[1,4,2,'python']
print(mylist1)
print(mylist2)

#%%
a=[10,20,30,40,50,60,70,80]
#accessing indivisual element
print("a[2]= ", a[2])
#including 0 but not 3
print("a[0:3]= ", a[0:3])
#include from 5 till end
print("a[5:]= ", a[5:])

#%%
a=[10,20,30,40,50,60,70,80]
#accessing indivisual element
print(a)
a[3]=55
print(a)

#%%

b= ['spam',2.0,5,[10,'raj']]
print(b[3][1])

#%%
a=[10,20,30,40,50,60,70,80]
b= ['spam',2.0,5,[10,'raj']]
c=a+b+[5,6]
c

#%%

#appending single element at once append function, append associated with only list
my_list=[1,2]
i=1

while(i<=5):
    value=(input("enter any element")) # input function always take input as string
    my_list.append(value)
    i+=1
print (my_list)

#%%

# appending multiple elements extend

mylist=[1,3,6,9]
my_newlist=[4,5,6,7,8]
my_newlist.extend(my_newlist)

print(my_newlist)

#%%
# if insert value in between

list1=['a','b','d','f','g','d','f']
list1.insert(1,'e')
print(list1)
print(list1.index('d')) # will return the 1st index no of the element

#%%
# reverse the list
list1=['a','b','d','f','g','d','f']
list1.reverse()
print (list1)

list1.sort(reverse=True)
print(list1)
list1.sort()
print(list1)

#%%
b=['spam',3,4.0]
b.sort  # for sorting should have same data type

#%%
list1=['a','b','d','f','g','d','f']
print(list1)
del[list1[4]]  # will delete the index
print(list1)

#%%
list1=['a','b','d','f','g','d','f']
print(list1)
list1.remove('d')  # will remove the 1st instance of d
print(list1)

#%%
list1=['a','b','d','f','g','d','f']
print(list1)
del list1[1:3] # will remove 12 index
print(list1)

list2=list1.copy()
print(list2)

list1.clear()
list1

#%%

#aggregate functions
num=[5,50,15,20,25,30]  # should be always same data type for aggregation
print(len(num))
print(max(num))
print(min(num))
print(sum(num))
print(sum(num)/len(num))

#%%
#lists and strings
x="python python"
y=list(x)
print(y)    # will print each char as list indexes
x=(1,2,3,4,5)
y=list(x)
print(y)

#%%

a=[10,20,30,40,50]
b=[5,6,7,8,9]
print(list(zip(a,b)))  # zip will ignore if extra value present in any variable, only print matching ones

#%%

# sets ane unique values, unordered and mutable in case of whole, not mutable in case of indivisual
#
my_sets={1,2,3,5,7,56,4,3,2,10,38,72,16}
print(my_sets)

my_set={1.0,"hello","manish",(1,2,3)}  # do not take repeatated valus and in order
print(my_set)


#%%

#my_set={1,2,[3,4]}
#print(my_set)    # list is not allowed as indivisual mode sets are not mutable where as list is mutable

my_set={1,2,(2,4)}
print(my_set)       # tupples are allowed
#my_set[1]       # no index concept in sets so error


#%%
a={}
print(type(a))    # will throw class as dictonary

a= set()
print(type(a))    # will represent empty set

#%%

my_set={1,3}
print(my_set)

my_set.add(2)   # always take single value
print(my_set)

my_set.update({6,8},[1,4,8],((2,9),(10,11)))  # as list within the tupples are not allowed in sets
print(my_set)   # tale all indivisdual items and place in sets

#%%

my_set={1,2,3,4,5,6}
print(my_set)
my_set.remove(6)
print(my_set)  # will remove the element 6 from sets

del my_set[2]
print(my_set)   # will throw error as index concept is not there

#%%

a={1,2,3,4,5}
b={4,5,6,7,8}

print(a|b)
print(a.union(b))    # will throw the same output as a|b (unquie values)

print(a&b)
print(a.intersection(b))   # both will throw common values

print(a-b)              #only a values that not present in b
print(b.difference(a))  #only b values that not present in a


#%%

'''Dictionaries, mutable data structures, unique keys present with values, data can be
 fetched by passing keys'''

my_d ={}
print(type(my_d))

my_d ={1:'a',2:'b'}
print(my_d)

my_d={'name': 'john','id':[2,3,4]}
print(my_d)

my_d= dict((('fruit','apple'),(2,'ball'), ('fruit','mango')))  '''in case of tupples, for 
each tupple two values should present (for key and another for value)    as key value should be unquie 
so mango will be replaced with apple for fruit key'''
print(my_d)

#%%
my_d ={'name':'suvanjeet','age':'25'}
print(my_d['name'])

#print(my_d['add'])   # error as no key add present in dict


#%%

#updation
my_d ={'name':'suvanjeet','age':'25'}
my_d['age']=27
print(my_d)    # in case key present then it will update the value

my_d['address']='mumbai'
print(my_d)     # in case key not  present then it will generate new entryin dict

#%%

squares={1:1,2:4, 3:6,4:16,5:25}
print(squares)
del squares[5]      # will delete the key and value from dict
print(squares)

squares.clear()
print(squares)          # will clear the dict



squares={1:1,2:4, 3:6,4:16,5:25}
print(squares.keys())
print(squares.values())


#%%

import numpy as np
np.__version__  # for version

#%%

a=np.array([1,2,3])  # single dimentional array
print(a)
print(type(a))

#%%

a=np.array([(1,2,3), (4,5,6)])  # multi dimentional array always represent in tupples inside list
print(a)

#%%

a=np.array([(1,2,3,4), (4,5,6)])  # as no of elements are not matching so create single dimensional
print(a)

print(a.ndim)

#%%

np.ones((2,3,4))  # 2 no of matrix, 3 no of rows and 4 no of columns   ones means all values as 1
a,b =np.ones((2,3,4))
print(a,"\n\n",b)

#%%

a,b=np.zeros((2,3,4),dtype=np.int)  # inputs as integers as by default its float   , all zero value
print(a)
print()
print(b)

#%%

np.full((2,2),7.5)  # 2*2 matrix with all values as 7.5


#%%
a=np.identity(3,dtype=np.int)   # 3*3 matrix with int value diagonal with 1 and rest as 0
print(a)

#%%

a=np.random.random((5,2))   # 5 rows and 2 cols with random values
print(a)

#%%

a=np.array([(1,2,3,4),(4,5,6,7)])
print(a)
print(a.size)       # total no of elements
print(a.shape)      # no of rows and cols

#%%
a=np.array([(1,2,3,4),(4,5,6,7)])
a=np.resize(a,(6,3))
print(a)            # keep repeating the values until it reached the speified dimensions


#a=np.reshape(a,(2,10))  # should match the elements count to convert
#print(a)                   will throw error as element counts donot match


a=np.reshape(a,(2,9))  # should match the elements count to convert
print(a)  


#%%

# index and slicing
a=np.array([(1,2,3,4),(4,5,6,7)])
print(a[0,2])
print(a[:,2])
print(a[0:2,:])

#%%

a=np.linspace(1,10,5)
print(a)        #  1 lower limit 10 upper limit and 5= equally devide

#%%

a=np.arange(0,25,5)  #0 is lower value 25 is upper limit but not included and 5 is the step increment
print(a)

a=np.arange(25,0,5)  #0 is lower value 25 is upper limit but not included and 5 is the step decrement
print(a)

#%%
a=np.array([(1,0,3),(3,4,5)])
print(np.sqrt(a))       # in array format
print(np.std(a))    # single value
print(np.log(a))       # also in array format

#%%

x=np.array([(1,2),(3,4)])
y=np.array([(5,6),(3,4)])

print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x%y)
print(np.dot(x,y))       # matrix multiplication
print(x@y)              # matrix multiplication
print(np.add(x,y))
print(np.subtract(x,y))
print(np.multiply(x,y))
print(np.divide(x,y))
print(np.remainder(x,y))

#%%

?np.dot()
#%%
# merging and joining

x=np.array([(1,2),(3,4)])
y=np.array([(5,6),(3,4)])

d=np.concatenate((x,y))
print(d)

print(np.vstack((x,y)))
print(np.hstack((x,y)))

#%%

#converting arrary into single array  ravel function
x=np.array([(1,2),(3,4)])
print(x)

print(x.ravel())       # to single array
print(x.reshape(2,2))   # to reshape the array

#%%
x=np.array([(1,2),(3,4)])
y=np.array([(5,6),(3,4)])
new_array= np.append(x,y)
print(new_array)            # it will convert the array to single dimension

new_array=np.reshape(new_array,(2,4)) # now reshping acc to requirment
# alternate
#new_array=new_array.reshape(2,4)
print(new_array)

#%%

x=np.array([(1,2),(3,4)])

new_array=np.insert(x,[1,4],5)  # 5 will be insert in 1 and 4th index
print(new_array)     # will convert to single dimension and change 


new_array=np.delete(new_array,[1,2])
print(new_array)

#%%

my_array2 = np.genfromtxt('array_data1.txt',
skip_header=1,delimiter=",",
filling_values=0)
print(my_array2)

#%%

#class 3






   
    
    


