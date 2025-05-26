from model import NeuralNetworkModel 

def get_training_data():
	return [[24, 1], [65, 0]] #is this an even number? yes (1) no (0)

LAYOUT = [1, 3, 4, 1]
EPOCH = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 10

nnm = NeuralNetworkModel(LAYOUT)
training_data = get_training_data()

res = nnm.predict(training_data)

print(f"\nModel prediction value(s):\n{res}")
