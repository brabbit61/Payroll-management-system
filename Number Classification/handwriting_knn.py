from numpy import *
import operator
from itertools import * 
def classify(inx, dataset, labels, k):  #the actual KNN computation
	datasetsize=dataset.shape[0]#number of rows
	diffmat=tile(inx,(datasetsize,1))-dataset
	sqdiffmat=diffmat**2
	sqdistances=sqdiffmat.sum(axis=1)#row-wise sum
	distances=sqdistances**0.5
	sorteddistindices=distances.argsort()
	classcount={}
	for i in range(k):
		voteilabel=labels[sorteddistindices[i]]
		classcount[voteilabel]=classcount.get(voteilabel,0)+1
	sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedclasscount[0][0]


b=[]
with open("train.csv") as f:
        x=f.read()
mat=zeros((42000,784))
b=x.split(",")
i=0
l=[b[0]]
for x in b:
        if i%784==0 and i!=0:
                b[i]=x[0]
                l.append(x[-1])
        i+=1
x=0
c=[]
for i in b:
        if int(i)>=1:
                c.append(1)
        else:
                c.append(0)
        if len(c)==784:
                mat[x]=c
                c=[]
                x+=1

m=mat.shape[0]
number=15000
error=0
for i in range(number):
        x=classify(mat[i],mat[number:],l[number:],10)
        if x!=l[i]:
                error+=1
print("the error is= "+str(error))

            


















        
        
