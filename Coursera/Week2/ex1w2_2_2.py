#ex2 week2 from the Andrew Ng course
#Linear regression with more than 1 variable using gradient descent
import numpy as np
from math import *
def normalize(x):
	ma=max(x)
	mi=min(x)
	for i in range(len(x)):
		x[i]=(x[i]-mi)/float(ma-mi)
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
x=np.ones([m,1])	
x=np.hstack((x,size))
x=np.hstack((x,rooms))		#141*3	
theta=np.zeros([3,1])		#3*1	
alpha=0.01
temp=[0,0,0]
for j in range(1000):
	for i in range(m):
		temp[0]+=(theta.transpose()*x[i].transpose())-price[i]
		temp[1]+=((theta.transpose()*x[i].transpose())-price[i])*float(size[i])
		temp[2]=((theta.transpose()*x[i].transpose())-price[i])*float(rooms[i])
	for y in temp:
		y*=float(alpha/m)
	for y in range(3):
		theta[y][0]-=temp[y]
cost=0
for i in range(m):
	cost+=pow(((np.dot(x[i],theta))-price[i]),2)/(2*m)
print("Cost: "+str(cost))
print("theta: "+str(theta))

