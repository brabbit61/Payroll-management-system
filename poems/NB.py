import pandas as pd
from math import log
#Love = 0, Mythology & Folklore = 2, Nature = 1
def setofWordsToVec(string, period):
	global words
	w = list(words)
	vec = [0]*len(w)
	p = [0,0]
	s = string.replace('\n',' ').split(' ')
	for i in range(len(s)):
		for j in range(len(w)):
			if s[i] == w[j]:
				vec[j] += 1
	if period == 'Renaissance':
		p[0] = 1
	else:
		p[1] = 1
	vec = vec + p 	
	return vec


dataset = pd.read_csv("all.csv")
train_data = dataset.iloc[:430,[0,1]]
train_labels = dataset.iloc[:430,[-1]]
test_data = dataset.iloc[430:,[0,1]]
test_labels = dataset.iloc[430:,[-1]]

words = set()

train_data_0 = train_data.iloc[:,0].tolist()
test_data_0 = test_data.iloc[:,0].tolist()
 
train_labels = train_labels.iloc[:,0].tolist()
test_labels = test_labels.iloc[:,0].tolist()

for i in range(len(train_data_0)):
	words = words | set(train_data_0[i].replace('\n',' ').split(' '))

for i in range(len(test_data_0)):
	words = words | set(test_data_0[i].replace('\n',' ').split(' '))

x = []
for i in range(len(train_data_0)):
	x.append(setofWordsToVec(train_data_0[i],train_data.iloc[i,1]))
train_data = x

x = []
for i in range(len(test_data_0)):
	x.append(setofWordsToVec(test_data_0[i],test_data.iloc[i,1]))
test_data = x


pDenom = [2,2,2]
pNum = [[1]*len(words)]*3

p = [0,0,0]	
for i in range(len(train_data_0)):
	if train_labels[i] == 'Love':
		pNum[0] = [sum(x) for x in zip(train_data[i],pNum[0])]
		pDenom[0] += sum(train_data[i])
		p[0] += 1
	elif train_labels[i] == 'Nature':
		pNum[1] = [sum(x) for x in zip(train_data[i],pNum[1])]
		pDenom[1] += sum(train_data[i])
		p[1] += 1
	else:
		pNum[2] = [sum(x) for x in zip(train_data[i],pNum[2])]
		pDenom[2] += sum(train_data[i])
		p[2] += 1

p = [i/len(train_data_0) for i in p]
pvect = [0,0,0]
for i in range(3):
	pvect[i] = [log(j/pDenom[i]) for j in pNum[i]]

correct = 0

for i in range(len(test_data)):
	probs = [0,0,0] 
	probs[0] = sum([x*y for x,y in zip(pvect[0],test_data[i])]) + log(p[0])
	probs[1] = sum([x*y for x,y in zip(pvect[1],test_data[i])]) + log(p[1])
	probs[2] = sum([x*y for x,y in zip(pvect[2],test_data[i])]) + log(p[2])
	if max(probs) == probs[0] and test_labels[i] == 'Love':
		correct+=1
	elif max(probs) == probs[1] and test_labels[i] == 'Nature':
		correct+=1
	elif max(probs) == probs[2] and test_labels[i] == 'Mythology & Folklore':
		correct+=1

print("accuracy: "+str(correct/len(test_data)))
