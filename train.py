from model import NeuralNetworkModel 
import random as r

def get_training_data(n):
	data = []
	for _ in range(n):
		d = [float(r.randint(0,99))]
		d.append(d[0]%2)
		data.append(d)
	
	return data

LAYOUT = [1, 3, 4, 1]
EPOCH = int(1)
LEARNING_RATE = 0.01
BATCH_SIZE = 3

nnm = NeuralNetworkModel(LAYOUT)
training_data = get_training_data(6)

#res = nnm.predict(training_data)

#print(f"\nModel prediction value(s):\n{res}")

nnm.train(EPOCH, BATCH_SIZE, LEARNING_RATE, training_data)
