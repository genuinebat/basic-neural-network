import os
import ast
import random as r
from layer import HLayer

LAYERS = [1, 3, 4, 1]
EPOCH = 1

# create file, if doesn't exist, to store weights and biases
if not os.path.isfile("weights.txt") and not os.path.isfile("biases.txt"):
	weights = []
	for i in range(len(LAYERS)-1):
		weights.append([[str(round(r.uniform(-0.5, 0.5), 2)) for _ in range(LAYERS[i+1])] for _ in range(LAYERS[i])])
	biases = [["0" for _ in range(n)] for n in LAYERS[1:-1]]

	print("Weights and Biases generated and saved in weights.txt and biases.txt respectively")
	print(f"Weights: {weights}")
	print(f"Biases: {biases}")

	with open("weights.txt", "w") as f:
		f.write(str(weights))
	
	with open("biases.txt", "w") as f:
		biases = [",".join(i) for i in biases]
		biases = "\n".join(biases)
		f.write(biases)

def get_weights_biases():
	w = ast.literal_eval(open("weights.txt", "r").read())
	w = [[[float(k) for k in j] for j in i] for i in w]
	b = [i.split(",") for i in open("biases.txt", "r").read().split("\n")]

	return w, b

w, b = get_weights_biases()
h_layers = []

for i in range(len(LAYERS)-1):
	if i == len(LAYERS)-2:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], w[i], None))
	else:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], w[i], [b[i]]))	

for e in range(EPOCH):
	curr_l_output = [[2]] #temp input value
	for l in h_layers:
		curr_l_output = l.forward(curr_l_output)

print(f"Model prediction value: {curr_l_output}")
