#one vs all logistic regression with regularization
import numpy as np

def convertlabels(labels,i):
	x=[]
	for j in labels:
		if j==i:
			x.append(1)
		else:
			x.append(0)
	return x

def normalize(x):
	if int(x)>0:
		return 1
	return 0
def relu(x):
	return np.maximum(0,x)
info=[]
labels=[]
with open("/home/jenit1/Desktop/Assignments/Week5/train.csv") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		labels.append(int(x[0]))
		info.append(x[1:])
		if len(info)>5000:
			break
for x in info:
	for i in range(len(x)):
		x[i]=normalize(x[i])
info=np.mat(info)
m=len(info)
x=np.ones((m,1))
x=np.hstack((x,info))

alpha=0.01
l=0.1
templabels=[]
for i in range(10):
	temp=convertlabels(labels,i)
	templabels.append(temp)
all_theta=np.ones((10,785))
for j in range(1500):
	print(i)
	h=relu(np.dot(x,all_theta.T)).T
	beta=[]
	for i in range(10):
		beta=templabels[i]-h[i]
		pd=np.dot(x.T,beta.T)
		all_theta[i]=all_theta[i]+(alpha/m)*pd.T
		all_theta[i][1:]=all_theta[i][1:]+ (l*all_theta[i][1:])/m
error=0
for i in  range(len(labels)):
	probability=np.dot(x[i],all_theta.transpose())
	mp=np.max(probability)											#max probability
	index=0
	for j in np.array(probability).flat:
		if j==mp:
				break
		index+=1
	if index!=labels[i]:
		error+=1
print(error)

