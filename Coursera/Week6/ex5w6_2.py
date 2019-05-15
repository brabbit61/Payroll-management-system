#polynomial regression
import numpy as np
import scipy.io as spio

def normalize(x):
	ma=np.max(x)
	mi=np.min(x)
	for i in range(len(x)):
		x[i]=(x[i]-mi)/float(ma-mi)
	return x
info=spio.loadmat("/home/jenit1/Desktop/ex5/ex5data1.mat")
trainx=np.mat(info["X"])
trainy=np.mat(info["y"])
crossx=np.mat(info["Xval"])
crossy=np.mat(info["yval"])
testx=np.mat(info["Xtest"])
testy=np.mat(info["ytest"])
cost=0
m=len(trainx)
theta=np.random.rand(9,1)#2*1
#print(theta)
theta=np.mat(theta)
lamb=0.01
alpha=0.001
x=trainx
x2=normalize(np.multiply(x,x))
x3=normalize(np.multiply(x,x2))
x4=normalize(np.multiply(x,x3))
x5=normalize(np.multiply(x,x4))
x6=normalize(np.multiply(x,x5))
x7=normalize(np.multiply(x,x6))
x8=normalize(np.multiply(x,x7))
x=np.hstack((x,x2,x3,x4,x5,x6,x7,x8))
for j in range(1000):
	pd=[0]*9
	X=np.hstack((np.ones((m,1)),x))
	X=np.dot(X,theta)
	temp1=(X-trainy)
	pd[0]=sum(temp1)
	pd[0]*=(alpha/m)
	temp2=np.dot(x.transpose(),temp1)
	for i in range(1,len(pd)):
		pd[i]=temp2[i-1]
	for g in range(1,len(pd)):
		pd[g]+=lamb*theta[g]
		pd[g]*=(alpha/m)
	for g in range(len(pd)):
		theta[g]=theta[g]-pd[g]
#cost for training data
cost=0
for i in range(m):
	X=x[i]
	X=np.hstack((np.ones((1,1)),X))
	X=np.dot(X,theta)
	temp=(X-trainy[i])
	temp=np.multiply(temp,temp)
	cost+=temp
for i in range(1,9):
	cost+=0.01*(theta[i]**2)
cost/=(2*m)
print("cost of training data: "+str(cost))

#cross validation, to find the regularizing coefficient
m=len(crossx)
all_costs=[]
x=crossx
x2=normalize(np.multiply(x,x))
x3=normalize(np.multiply(x,x2))
x4=normalize(np.multiply(x,x3))
x5=normalize(np.multiply(x,x4))
x6=normalize(np.multiply(x,x5))
x7=normalize(np.multiply(x,x6))
x8=normalize(np.multiply(x,x7))
x=np.hstack((x,x2,x3,x4,x5,x6,x7,x8))
for j in range(2000):
	cost=0
	for i in range(m):
		X=x[i]
		X=np.hstack((np.ones((1,1)),X))
		X=np.dot(X,theta)
		temp=(X-crossy[i])
		temp=np.multiply(temp,temp)
		cost+=temp
	for i in range(1,9):
		cost+=(j/1000.0)*(theta[i]**2)
	cost/=(2*m)
	all_costs.append(cost)
cost=min(all_costs)
index=all_costs.index(cost)
lamb=index*0.001
print("cost of cross validation data: "+str(cost)+" and the lambda is: "+str(lamb))


#cost for test data
m=len(testx)
x=testx
x2=normalize(np.multiply(x,x))
x3=normalize(np.multiply(x,x2))
x4=normalize(np.multiply(x,x3))
x5=normalize(np.multiply(x,x4))
x6=normalize(np.multiply(x,x5))
x7=normalize(np.multiply(x,x6))
x8=normalize(np.multiply(x,x7))
x=np.hstack((x,x2,x3,x4,x5,x6,x7,x8))
cost=0
for i in range(m):
	X=x[i]
	X=np.hstack((np.ones((1,1)),X))
	X=np.dot(X,theta)
	temp=(X-testy[i])
	temp=np.multiply(temp,temp)
	cost+=temp
for i in range(1,9):
	cost+=0.001*(theta[i]**2)
cost/=(2*m)
print("cost of test data: "+str(cost))


