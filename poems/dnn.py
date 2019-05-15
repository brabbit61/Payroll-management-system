from keras.optimizers import SGD
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import save_model, load_model
from keras.preprocessing.text import one_hot,Tokenizer
import pandas as pd
import numpy as np

dataset = pd.read_csv("all.csv")
train_data = dataset.iloc[:350,[0,1]]
train_labels = dataset.iloc[:350,[-1]]
val_data = dataset.iloc[350:420,[0,1]]
val_labels = dataset.iloc[350:420,[-1]]
test_data = dataset.iloc[420:,[0,1]]
test_labels = dataset.iloc[420:,[-1]]

t = Tokenizer(lower=True, split = ' ')
t.fit_on_texts(dataset.iloc[:,0])
n = len(t.word_index)

test_data_0 = t.texts_to_matrix(test_data.iloc[:,0], mode='count')
train_data_0 = t.texts_to_matrix(train_data.iloc[:,0], mode='count')
val_data_0 = t.texts_to_matrix(val_data.iloc[:,0], mode='count')

del dataset

labelencoder_X_1 = LabelEncoder()
train_data_1 = labelencoder_X_1.fit_transform(train_data.iloc[:,1])

labelencoder_X_1 = LabelEncoder()
val_data_1 = labelencoder_X_1.fit_transform(val_data.iloc[:,1])

labelencoder_X_1 = LabelEncoder()
test_data_1 = labelencoder_X_1.fit_transform(test_data.iloc[:,1])

test_data_1 = test_data_1.reshape(-1,1)
val_data_1 = val_data_1.reshape(-1,1)
train_data_1 = train_data_1.reshape(-1,1)

train_data = np.hstack((train_data_0,train_data_1))
test_data = np.hstack((test_data_0,test_data_1))
val_data = np.hstack((val_data_0,val_data_1))


onehotencoder = OneHotEncoder()
val_labels = val_labels.iloc[:,0].values.reshape(-1,1)
val_labels = onehotencoder.fit_transform(val_labels).toarray()

onehotencoder = OneHotEncoder()
train_labels = train_labels.iloc[:,0].values.reshape(-1,1)
train_labels = onehotencoder.fit_transform(train_labels).toarray()

onehotencoder = OneHotEncoder()
test_labels = test_labels.iloc[:,0].values.reshape(-1,1)
test_labels = onehotencoder.fit_transform(test_labels).toarray()

'''
classifier = Sequential()
classifier.add(Dense(2048, kernel_initializer = 'uniform', activation = 'relu', input_dim = train_data.shape[1]))
classifier.add(Dense(2048, kernel_initializer = 'uniform',activation = 'relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.2))
classifier.add(Dense(1024, kernel_initializer = 'uniform',activation = 'relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.25))
classifier.add(Dense(1024, kernel_initializer = 'uniform',activation = 'relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.25))
classifier.add(Dense(512, kernel_initializer = 'uniform',activation = 'relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.3))
classifier.add(Dense(512, kernel_initializer = 'uniform',activation = 'relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.3))
classifier.add(Dense(3, kernel_initializer = 'uniform'))
classifier.add(Activation('softmax'))
'''
classifier = load_model("final.h5")

print(classifier.summary())

'''
sgd = SGD(lr = 0.00005, momentum = 0.93, decay=0.000001, nesterov= True)
classifier.compile(optimizer = sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs = 1000
train_losses = []
train_acc = []
test_losses = []
test_acc = []
iters = []
(loss,accuracy) = classifier.evaluate(test_data, test_labels, verbose = 1)
test_losses.append(loss)
test_acc.append(accuracy)
(loss,accuracy) = classifier.evaluate(train_data, train_labels, verbose = 1)
train_losses.append(loss)
train_acc.append(accuracy)
iters.append(1000-epochs)
save_model(classifier, "model"+str(1000-epochs)+".h5")

while epochs > 0:
	classifier.fit(train_data, train_labels, batch_size = 8,verbose = 1, validation_data = (val_data,val_labels), shuffle=True, epochs = 1)
	(loss,accuracy) = classifier.evaluate(test_data, test_labels, verbose = 1)
	test_losses.append(loss)
	test_acc.append(accuracy)
	(loss,accuracy) = classifier.evaluate(train_data, train_labels, verbose = 1)
	train_losses.append(loss)
	train_acc.append(accuracy)
	iters.append(1000-epochs)
	if epochs%100==0:
		save_model(classifier,"model"+str(1000-epochs)+".h5")
	epochs -= 1

save_model(classifier, "final"+".h5")
'''


#####################################################################
# Replace the test_data and the test_labels numpy arrays with your test data and labels in the appropriate format
(loss,accuracy) = classifier.evaluate(test_data, test_labels, verbose = 1)
print("Test loss: "+str(loss))
print("Test accuracy: "+str(accuracy))
#####################################################################

