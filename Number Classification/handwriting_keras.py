import numpy as np
import csv
from keras.optimizers import SGD
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
def greater(x):
	for i in x:
		if i>0:
			i=1
	return x

def loaddataset(filename):      #loads the data set and returns the datamatrix and the labelmatrix
    datamat=[]
    labelmat=[]
    with open(filename) as fr:
        for x in fr.readlines():
            x=x.strip().split(',')
            labelmat.append(int(x[0]))
            data=list(map(greater,x))
            datamat.append(data[1:])
    return datamat,labelmat,labelmat

def predict(model):
	global data,labels
	data=np.array(data)
	x=model.predict(data[31500:])
	error=0
	total=0
	for i in range(len(x)):
		total+=1
		y=x[i].tolist()
		if labels[i+31500]!= y.index(max(y)):
			error+=1
	print(total)
	print(error)



def train():
	global data,labels,lossiter
	epochs=6000
	nb_epochs=0
	batch_size=1000
	data=np.array(greater(data))
	m,n=np.shape(data)
	le=LabelEncoder()
	labels=le.fit_transform(labels)
	labels=np_utils.to_categorical(labels,10)
	(traindata,testdata,trainlabels,testlabels)=train_test_split(data,labels,test_size=0.25,random_state=42)
	model=load_model('hidden_5_6000.hdf5')
	'''model=Sequential()
	model.add(Dense(units=400, activation='relu', input_dim=784,init='uniform',use_bias=True))
	model.add(Dropout(0.2))
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	sgd = SGD(lr=0.005, decay=1e-5, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
	'''
	while epochs:
		print( "\n\nEpoch: ",nb_epochs)
		model.fit(traindata,trainlabels, batch_size=batch_size, verbose=1, epochs=1)
		nb_epochs+=1
		epochs-=1
		if nb_epochs%1000==0:
			model.save('hidden_5_{}.hdf5'.format(nb_epochs))
		if nb_epochs%5==0:
			(loss,accuracy)=model.evaluate(testdata,testlabels,verbose=1)
			lossiter.append([loss,nb_epochs])
			print("loss= "+str(loss)+" accuracy: "+str(accuracy*100))
	
data,_,labels=loaddataset("train.csv")
lossiter=[]
#train()
model=load_model('hidden_5_3000.hdf5')
predict(model)
