import os
import random as r
from layer import HLayer

LAYERS = [1, 3, 4, 1]
EPOCH = 1

# create file, if doesn't exist, to store weights and biases
if not os.path.isfile("weights.txt") and not os.path.isfile("biases.txt"):
	weights = [[str(round(r.uniform(-0.5, 0.5), 2)) for _ in range(LAYERS[n]*LAYERS[n+1])] for n in range(len(LAYERS)-1)]
	biases = [["0" for _ in range(n)] for n in LAYERS[1:-1]]

	print("Weights and Biases generated and saved in weights.txt and biases.txt respectively")
	print(f"Weights: {weights}")
	print(f"Biases: {biases}")

	with open("weights.txt", "w") as f:
		weights = [",".join(i) for i in weights]
		weights = "\n".join(weights)
		f.write(weights)
	
	with open("biases.txt", "w") as f:
		biases = [",".join(i) for i in biases]
		biases = "\n".join(biases)
		print("biases:")
		print(biases)
		f.write(biases)

def get_weights_biases():
	w = [i.split(",") for i in open("weights.txt", "r").read().split("\n")]
	b = [i.split(",") for i in open("biases.txt", "r").read().split("\n")]

	print("weights")
	print(w)
	print("biases")
	print(b)
	w = [[float(j) for j in i] for i in w]
	b = [[float(j) for j in i] for i in b]
	return w, b

w, b = get_weights_biases()
h_layers = []

for i in range(len(LAYERS)-1):
	if i == len(LAYERS)-2:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], [w[i]], None))
	else:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], [w[i]], [b[i]]))	

for e in range(EPOCH):
	curr_l_output = [[2]] #temp input value
	for l in h_layers:
		l.forward(curr_l_output)

print(f"Model prediction value: {curr_l_output}")
