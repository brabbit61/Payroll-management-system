import timeit
start=timeit.default_timer()
from numpy import *
import time
def greater(x):
    if int(x)>0:    return 1
    return 0
def loaddataset(filename):
    datamat=[]
    labelmat=[]
    with open(filename) as fr:
        for x in fr.readlines():
            x=x.strip().split(',')
            labelmat.append(int(x[0]))
            data=list(map(greater,x))
            datamat.append(data[1:])
    return datamat,labelmat
def trainNB(trainmat,traincat):
    n=len(trainmat)
    pnumerator=ones((10,784)) # the matrix to store the number of occurence of 1's and 0's for each number 
    pdenominator=ones(10)+1
    for x in range(n):
        pnumerator[traincat[x]]+=trainmat[x]
        pdenominator[traincat[x]]+=sum(trainmat[x])
    pvect=[]
    for i in range(10):
        pvect.append(log(pnumerator[i])-log(pdenominator[i]))
    '''p=[exp(i) for i in pvect]
    sum_exp=sum(p)
    x = [around(i /p,3) for i in pvect]'''
    return pvect

def classify(mat,pnum,pvect):
    prob=[(sum(mat*pvect[i])+ log(pnum[i])) for i in range(10)]
    return prob.index(max(prob))
datamat,labelmat=loaddataset("/home/jenit1/Desktop/train.csv")
dict={}
for x in labelmat:
    if x not in dict.keys():
        dict[x]=1
    dict[x]+=1
pnum=[dict[x]/len(datamat) for x in dict] #probabilty of a number occuring 
pvect=trainNB(datamat[:31500],labelmat[:31500])
error=0
for i in range(31500,42000):
    num=classify(datamat[i],pnum,pvect)
    if num!=labelmat[i]:
        error+=1
    if i%2000==0:
        print("error: {0}".format(error))
stop=timeit.default_timer()
print(stop-start)
