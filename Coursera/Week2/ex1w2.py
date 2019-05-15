#ex1 of week2 Andrew Ng course
#Liner regression with 1 variable
import numpy as np
from math import *
def normalize(x):
	ma=max(x)
	mi=min(x)
	for i in range(len(x)):
		x[i]=(x[i]-mi)/(ma-mi)
	return x
profit=[]
population=[]
with open("/home/jenit1/Desktop/Manas/AndrewNg/Assignments/Week2/ex1/ex1/ex1data1.txt") as fr:
	for x in fr.readlines():
		x=x.strip().split(',')
		population.append(float(x[0]))	
		profit.append(float(x[1]))
m=len(profit)
population=normalize(population)
profit=normalize(profit)	
population=np.mat(population).transpose()	#194*1
profit=np.mat(profit).transpose()		#194*1
theta=np.zeros([2,1],dtype='f')			#2*1
x=np.ones([m,1])
x=np.hstack((x,population)) #194*2
alpha=0.01
for j in range(400):
	temp1=0
	temp0=0
	for i in range(m):
		temp1+= ((theta.transpose()*(x[i].transpose()))-profit[i])*float(population[i])
		temp0+=(theta.transpose()*(x[i].transpose()))-profit[i]
	temp1*=(alpha/m)
	temp0*=(alpha/m)
	theta[0]=theta[0]-list(temp0)
	theta[1]=theta[1]-list(temp1)
'''cost=0
for i in range(m):
	cost+=pow(((theta.transpose()*x[i].transpose())-profit[i]),2)/(2*m)
print("Cost: "+str(cost))'''
print(theta)

