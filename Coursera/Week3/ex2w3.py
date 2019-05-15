#if y=1 then the student gets admission in the university
#if y=0 then the student doesnt get admission in the uinversity
import numpy as np

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

labels=[]
exam1=[]
exam2=[]
with open("/home/jenit1/Desktop/Manas/AndrewNg/Assignments/Week3/ex2/ex2data1.txt") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		exam1.append(float(x[0]))
		exam2.append(float(x[1]))
		labels.append(int(x[2]))
theta=np.ones((3,1))			#3*1
m=len(labels)
labels=np.mat(labels).transpose()
for y in range(m):
	exam1[y]/=100
for y in range(m):
	exam2[y]/=100
exam1=np.mat(exam1).transpose()
exam2=np.mat(exam2).transpose()
x=np.hstack((np.ones((m,1)),exam1))
x=np.hstack((x,exam2))		#m*3
alpha=0.01
for j in range(10000):
	'''
	cost=0
	for i in range(m):
		h=sigmoid(np.dot(x[i],theta))				#1*1
		temp=-(labels[i]*np.log(h))-((1-labels[i])*np.log(1-h))
		cost+=temp
	cost/=m
	if j%1000==0:
		print(cost)'''
	h=sigmoid(np.dot(x,theta))				#m*1
	error=labels-h
	theta=theta+(alpha/m)*x.transpose()*error
print theta

