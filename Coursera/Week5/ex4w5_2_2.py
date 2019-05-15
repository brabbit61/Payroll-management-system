import numpy as np
import math
def convertlabels(labels):
	x=[]
	for i in range(len(labels)):
		y=[0]*10
		y[labels[i]]=1
		x.append(y)
	return x
def norm(x):
	y=[]	
	for i in x:
		temp=[]
		for j in i:
			if int(j)>0:
				temp.append(1)
			else:
				temp.append(0)
		y.append(temp)
	return y
		
def normalize(x):
	n=len(x)
	for i in range(n):
		x[i]=int(x[i])
	s=sum(x)
	avg=s/n
	avg=np.tile(avg,(1,n))
	x=np.mat(x)
	diff=np.multiply((x-avg),(x-avg))
	var=np.sum(diff)/n
	x=(x-avg)/math.sqrt(var+0.000001)
	return x
	
def relu(x):
	return np.maximum(np.zeros_like(x),x)

def relu_prime(x):
	n=len(x)
	y=[0.01]*n
	for i in range(n):
		if x[i]>0.01:
			y[i]=1
	return np.mat(y)
	
def softmax(x):
	s=0
	n=len(x)
	y=[]
	for i in range(n):
		s+=np.exp(x[i])
	
	j=np.exp(x[0])
	j/=s
	y=np.mat(j)
	for i in range(1,n):
		j=np.exp(x[i])
		y=np.vstack((y,(j/s)))
	return np.mat(y)

info=[]
truelabels=[]
test=[]
testlabels=[]
with open("/home/jenit1/Desktop/Assignments/Week5/train.csv") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		if len(info)<100:
			truelabels.append(int(x[0]))
			info.append(x[1:])						#grayscale image of a 28*28 picture hence there are 784 pixels for 1 image
		else:
			if len(test)<3000:
				testlabels.append(int(x[0]))
				test.append(x[1:])
			else:
				break	
data=norm(info)
data=np.mat(data)
'''
data=normalize(info[0])
for x in range(1,len(info)):
	data=np.vstack((data,normalize(info[x])))
'''
m=len(info)
labels=convertlabels(truelabels)
labels=np.mat(labels)
#x=np.ones((m,1))
x=data											#np.hstack((x,data))
#one=np.ones((1,1))
alpha=0.01
l=0.001
'''
theta1=np.random.rand(25,785)/np.sqrt(25/2)				#25*785
theta2=np.random.rand(10,26)/np.sqrt(5)					#10*26
DELTA1=np.zeros((25,785))						#25*785
DELTA2=np.zeros((10,26))	
'''
theta1=np.random.rand(25,784)/np.sqrt(25/2)				#25*784
theta2=np.random.rand(10,25)/np.sqrt(5)					#10*25
for i in range(200):
	print(i)
	DELTA1=np.zeros((25,784))								#25*784
	DELTA2=np.zeros((10,25))								#10*25
	for j in range(m):
		a1=x[j]									#1*784
		z2=np.dot(theta1,a1.transpose())		#25*1
		a2=relu(z2)								#25*1
		#aa2=np.vstack((one,a2))				#26*1
		z3=np.dot(theta2,a2)					#10*1
		scores=np.exp(z3-z3.max(axis=1))
		correct=scores[np.arange(scores.shape[0]),labels[j]]
		scores_total=np.sum(scores,axis=1)
		loss=np.sum(-np.log(np.divide(correct,scores_total)))/scores.shape[0]
		loss+=0.5*l*(np.sum(theta1*theta1)+np.sum(theta2*theta2))
		#a3=softmax(z3)
		#crs_ent=-np.dot(labels[j],np.log(a3))-np.dot((1-labels[j]),np.log(1-a3))
		# a3 goes into crs ent -- crt_ent(a3, labels[i]) --> scalar
		#backpropogation
		dw1=np.zeros_like(theta1)
		dw2=np.zeros_like(theta2)
		grad=(scores.T/scores_total)
		grad[y,np.arange(scores.shape[0])]-=1.0
		dw2=np.dot(grad,a2).T/m+l*theta2
		drelu1=theta2.dot(grad)
		dz2=(z2>0)*drelu1.T
		dw1=a1.T.dot(dz2)+l*theta1
		DELTA1+=dw1
		DELTA2+=dw2
		'''
		delta3=(a3-labels[j].T)					#10*1
		temp1=np.dot(theta2.transpose(),delta3)	#25*1
		temp2=relu_prime(z2)					#25*1	
		delta2=np.multiply(temp1,temp2.transpose())			#25*1
		DELTA1+=np.dot(delta2,a1)				#25*785
		DELTA2+=np.dot(delta3,a2.transpose())	
		'''
	alph=alpha/(i+1)
	theta1-=(alph/m)*DELTA1-(l/m)*theta1
	theta2-=(alph/m)*DELTA2-(l/m)*theta2
	


error=0
for i in  range(len(truelabels)):
	a1=x[i]									#1*785
	z2=np.dot(theta1,a1.transpose())		#25*1
	a2=relu(z2)								#25*1
	z3=np.dot(theta2,a2)					#10*1
	a3=softmax(z3)
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

#test data
data=normalize(test[0])
for x in range(1,len(test)):
	data=np.vstack((data,normalize(test[x])))
m=len(test)
labels=convertlabels(testlabels)
labels=np.mat(labels)
x=data	
error=0
for i in  range(len(test)):
	a1=x[i]									#1*785
	z2=np.dot(theta1,a1.transpose())		#25*1
	a2=relu(z2)								#25*1
	#aa2=np.vstack((one,a2))					#26*1
	z3=np.dot(theta2,a2)					#10*1
	a3=softmax(z3)
	mp=np.max(a3)							#max probability
	index=0
	for j in np.array(a3).flat:
		if j==mp:
				break
		index+=1
	if index!=testlabels[i]:
		error+=1
print(error)
'''
