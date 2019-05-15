import numpy as np

def sigmoid(x):
	return 1.0/(1+np.exp(-x))
def normalize(x):
	mi=min(x)
	ma=max(x)
	for i in range(len(x)):
		x[i]=(x[i]-mi)/float(ma-mi)
	return x
labels=[]
m1=[]
m2=[]
with open("/home/jenit1/Desktop/Manas/AndrewNg/Assignments/Week3/ex2/ex2data2.txt") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		m1.append(float(x[0]))
		m2.append(float(x[1]))
		labels.append(int(x[2]))
theta=np.ones((5,1))
m=len(labels)
m3=[]
m4=[]
for i in m1:
	m3.append(i**2)
for i in m2:
	m4.append(i**2)
'''m1=normalize(m1)
m2=normalize(m2)
m3=normalize(m3)
m4=normalize(m4)
'''
labels=np.mat(labels).transpose()
m1=np.mat(m1).transpose()
m2=np.mat(m2).transpose()
m3=np.mat(m3).transpose()
m4=np.mat(m4).transpose()
x=np.hstack((np.ones((m,1)),m1))
x=np.hstack((x,m2))		#m*5
x=np.hstack((x,m3))		#m*5
x=np.hstack((x,m4))		#m*5
alpha=0.01
l=0.1
for j in range(1500):
	'''
	cost=0
	for i in range(m):
		h=sigmoid(np.dot(x[i],theta))				#1*1
		temp=-(labels[i]*np.log(h))-((1-labels[i])*np.log(1-h))
		cost+=temp
	term=theta[1]*theta[1]+theta[2]*theta[2]
	term*=(l/(2*m))
	cost/=m
	cost+=term
	if j%100==0:
		print(cost)
	'''
	h=sigmoid(np.dot(x,theta))				#m*1
	error=labels-h
	theta=theta+(alpha/m)*x.transpose()*error
	theta[1:]=theta[1:]+ (l*theta[1:])/m
print theta
