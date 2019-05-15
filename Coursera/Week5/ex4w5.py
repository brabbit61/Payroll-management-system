#one vs all logistic regression with regularization
import numpy as np
import math
def convertlabels(labels):
	x=[]
	for i in range(len(labels)):
		y=[0]*10
		y[labels[i]]=1
		x.append(y)
	return x
	
def normalize(x):
	for i in range(len(x)):
		x[i]=int(x[i])
	s=sum(x)
	avg=s/len(x)
	avg=np.tile(avg,(len(x),1))
	x=np.array(x)
	diff=(x-avg)**2
	var=np.sum(diff)/len(x)
	x=(x-avg)/math.sqrt(var+0.000001)
	return x
	
def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def sigmoid_prime(x):
	return np.multiply(sigmoid(x),(1.0-sigmoid(x)))

info=[]
truelabels=[]

with open("/home/jenit1/Desktop/Assignments/Week5/train.csv") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		truelabels.append(int(x[0]))
		info.append(x[1:])						#grayscale image of a 28*28 picture hence there are 784 pixels for 1 image
		if len(info)>500:
			break
			
for x in range(len(info)):
	info[x]=normalize(info[x])

info=np.mat(info)
m=len(info)
labels=convertlabels(truelabels)
labels=np.mat(labels)
x=np.ones((m,1))
x=np.hstack((x,info))
one=np.ones((1,1))
alpha=0.01
lamb=0.1
theta1=np.random.rand(25,785)					#25*785
theta2=np.random.rand(10,26)					#10*26
DELTA1=np.zeros((25,785))						#25*785
DELTA2=np.zeros((10,26))						#10*26
for j in range(1):
	for i in range(len(labels)):
		print(i)
		a1=x[i].transpose()						#785*1
		z2=np.dot(theta1,a1)					#25*1
		a2=sigmoid(z2)							#25*1
		a2=np.vstack((one,z2))					#26*1
		z3=np.dot(theta2,a2)					#10*1
		a3=sigmoid(z3)							#10*1
		#backpropogation
		delta3=a3-labels[i].T					#10*1
		temp1=np.dot(theta2[:,:25].transpose(),delta3)	#25*1
		temp2=sigmoid_prime(z2)					#25*1			
		delta2=np.multiply(temp1,temp2)			#25*1
		DELTA1=np.dot(delta2,a1.transpose())	#25*785
		DELTA2=np.dot(delta3,a2.transpose())	
		#for layer 1
		for k in range(25):
			for l in range(785):
				c=list(np.array(DELTA1[k]).flat)
				d=c[l]
				if l ==0:
					d/=m
				else:
					d/=m
					d=d+lamb*theta1[k][l]
				theta1[k][l]-=((alpha/m)*d)
				
		#for layer 2
		for k in range(10):
			for l in range(26):
				count=0
				for g in np.array(DELTA2[k]).flat:
					if count==l:
						d=g
						break
					count+=1
				if l==0:
					d/=m
				else:
					d/=m
					d=d+lamb*theta2[k][l]	
				theta2[k][l]-=((alpha/m)*d)
'''
error=0
for i in  range(len(truelabels)):
	a1=x[i].transpose()					#785*1
	z2=np.dot(theta1,a1)					#25*1
	a2=sigmoid(z2)							#25*1
	a2=np.vstack((one,z2))						#26*1
	z3=np.dot(theta2,a2)					#10*1
	a3=sigmoid(z3)							#10*1
	mp=np.max(a3)							#max probability
	index=0
	for j in np.array(a3).flat:
		if j==mp:
				break
		index+=1
	if index!=truelabels[i]:
		error+=1
print(error)
'''
