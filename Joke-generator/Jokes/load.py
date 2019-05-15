from __future__ import print_function
import numpy as np
from keras.models import load_model
	
'''
def sample(preds, temperature=1.0):
	print(preds.shape)
	preds = np.asarray(preds).astype('float64')
	print(preds.shape)
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds/temperature)
	preds = exp_preds / np.sum(exp_preds)
	return np.argmax(preds)

def generate_text_posttraining(model,vocablen,ix_to_char,char_to_ix):
	global temperature
	word=str(raw_input("Enter a word to start the joke: "))
	prefixes=['A ','The ','All the ', 'Did you know a ']
	index=np.random.randint(0,len(prefixes))
	word= prefixes[index]+word
	length=np.random.randint(50,150)
	print(length)
	y_char=[word]
	ix=[char_to_ix[y_char[0][-1]]]
	X = np.zeros((1, length, vocablen))
	for i in range(length):
		X[0, i, :][ix[-1]] = 1						
		preds = model.predict(X[:, :i+1, :])[0]
        ix = sample(preds, temperature)
        next_char = ix_to_char[ix[-1]]
        print(model.predict(X[:, :i+1, :])[0].shape)
        y_char.append(next_char)
	y_char.append(" end")
	return ('').join(y_char)	
'''

def generate_text(model, vocablen, ix_to_char):
	length=np.random.randint(70,200)
	ix = [np.random.randint(vocablen)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocablen))
	for i in range(length):
		X[0, i, :][ix[-1]] = 1
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

chars=(['\n', ' ', "'", ')', '(', '+', '-', ',', '/', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', ':', '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])

vocablen=len(chars)
ix_to_char = {0: '\n', 1: ' ', 2: "'", 3: ')', 4: '(', 5: '+', 6: '-', 7: ',', 8: '/', 9: '.', 10: '1', 11: '0', 12: '3', 13: '2', 14: '5', 15: '4', 16: '7', 17: '6', 18: '9', 19: '8', 20: ':', 21: '?', 22: 'A', 23: 'C', 24: 'B', 25: 'E', 26: 'D', 27: 'G', 28: 'F', 29: 'I', 30: 'H', 31: 'K', 32: 'J', 33: 'M', 34: 'L', 35: 'O', 36: 'N', 37: 'Q', 38: 'P', 39: 'S', 40: 'R', 41: 'U', 42: 'T', 43: 'W', 44: 'V', 45: 'Y', 46: 'X', 47: 'Z', 48: 'a', 49: 'c', 50: 'b', 51: 'e', 52: 'd', 53: 'g', 54: 'f', 55: 'i', 56: 'h', 57: 'k', 58: 'j', 59: 'm', 60: 'l', 61: 'o', 62: 'n', 63: 'q', 64: 'p', 65: 's', 66: 'r', 67: 'u', 68: 't', 69: 'w', 70: 'v', 71: 'y', 72: 'x', 73: 'z'}

char_to_ix = {'\n': 0, ' ': 1, "'": 2, ')': 3, '(': 4, '+': 5, '-': 6, ',': 7, '/': 8, '.': 9, '1': 10, '0': 11, '3': 12, '2': 13, '5': 14, '4': 15, '7': 16, '6': 17, '9': 18, '8': 19, ':': 20, '?': 21, 'A': 22, 'C': 23, 'B': 24, 'E': 25, 'D': 26, 'G': 27, 'F': 28, 'I': 29, 'H': 30, 'K': 31, 'J': 32, 'M': 33, 'L': 34, 'O': 35, 'N': 36, 'Q': 37, 'P': 38, 'S': 39, 'R': 40, 'U': 41, 'T': 42, 'W': 43, 'V': 44, 'Y': 45, 'X': 46, 'Z': 47, 'a': 48, 'c': 49, 'b': 50, 'e': 51, 'd': 52, 'g': 53, 'f': 54, 'i': 55, 'h': 56, 'k': 57, 'j': 58, 'm': 59, 'l': 60, 'o': 61, 'n': 62, 'q': 63, 'p': 64, 's': 65, 'r': 66, 'u': 67, 't': 68, 'w': 69, 'v': 70, 'y': 71, 'x': 72, 'z': 73}


model=load_model('final3.hdf5')
sentence=generate_text(model, vocablen, ix_to_char)
print(sentence)
