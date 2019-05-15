#ex2 week2 from the Andrew Ng course
#Linear regression with more than 1 variable using normal functions
import numpy as np
from math import *
def normalize(x):
	ma=max(x)
	mi=min(x)
	for i in x:
		i=(i-mi)/(ma-mi)
	return x
size=[]
rooms=[]
price=[]
with open("/home/jenit1/Desktop/Manas/AndrewNg/Assignments/Week2/ex1/ex1/ex1data2.txt") as fr:
	for x in fr.readlines():
		x=x.strip().split(',')
		size.append(int(x[0]))
		rooms.append(int(x[1]))
		price.append(int(x[2]))
size=normalize(size)
rooms=normalize(rooms)
price=normalize(price)
m=len(price)
size=np.mat(size).transpose()   #141*1
rooms=np.mat(rooms).transpose()	#141*1
price=np.mat(price).transpose()	#141*1
x=np.ones([m,1])		#141*3
x=np.hstack((x,size))
x=np.hstack((x,rooms))
temp1=x.transpose()*x		#3*3
temp1=np.linalg.pinv(temp1)
temp2=x.transpose()*price		#3*1
theta=temp1*temp2
temp1=x*theta-price
cost=(temp1.transpose()*temp1)/(2*m)
print("cost"+str(cost))
print("theta: "+str(theta))
