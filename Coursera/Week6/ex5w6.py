#linear regression
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
info=spio.loadmat("/home/jenit1/Desktop/ex5/ex5data1.mat")
trainx=info["X"]
trainy=info["y"]
crossx=info["Xval"]
crossy=info["yval"]
testx=info["Xtest"]
testy=info["ytest"]
m=len(trainx)
theta=np.random.rand(2,1)#2*1
theta=np.mat(theta)
lamb=0.01
alpha=0.001
xaxis=trainx
yaxis=trainy
trainx=np.mat(trainx)
trainy=np.mat(trainy)
for j in range(1000):
	pd0=0
	pd1=0
	x=trainx
	X=np.hstack((np.ones((m,1)),x))
	X=np.dot(X,theta)
	temp1=(X-trainy)
	pd0=sum(temp1)
	temp2=np.multiply(temp1,x) + lamb*theta[1]
	pd1=sum(temp2)
	pd0*=(alpha/m)
	pd1*=(alpha/m)
	theta[0]=theta[0]-pd0
	theta[1]=theta[1]-pd1

cost=0
for i in range(m):
	x=trainx[i]
	x=np.vstack((np.ones((1,1)),x))
	x=np.dot(theta.transpose(),x)
	temp=((x-trainy[i])**2)
	cost+=temp
cost+=0.01*(theta[1]**2)
cost/=(2*m)
print("cost of training data: "+str(cost))



#cross validation, to find the regularizing coefficient
m=len(crossx)
all_costs=[]
for j in range(2000):
	cost=0
	for i in range(m):
		x=crossx[i]
		x=np.vstack((np.ones((1,1)),x))
		x=np.dot(theta.transpose(),x)
		temp=((x-crossy[i])**2)
		cost+=temp
	cost+=(j/1000.0)*(theta[1]**2)
	cost/=(2*m)
	all_costs.append(cost)
cost=min(all_costs)
index=all_costs.index(cost)
lamb=index*0.001
print("cost of cross validation data: "+str(cost)+" and the lambda is: "+str(lamb))



#test validation
m=len(testx)
x=testx
X=np.hstack((np.ones((m,1)),x))
X=np.dot(X,theta)
temp1=(X-testy)
#cost
cost=0
for i in range(m):
	x=testx[i]
	x=np.vstack((np.ones((1,1)),x))
	x=np.dot(theta.transpose(),x)
	temp=((x-testy[i])**2)
	cost+=temp
cost+=lamb*(theta[1]**2)
cost/=(2*m)
print("cost of test validation data: "+str(cost))
