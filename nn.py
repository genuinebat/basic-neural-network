import os
import random as r
from layer import HLayer

LAYERS = [1, 3, 4, 1]
EPOCH = 1

# create file, if doesn't exist, to store weights and biases
if not os.path.isfile("weights_biases.txt"):
	weights = [[round(r.uniform(-0.5, 0.5), 2) for _ in range(LAYERS[n]*LAYERS[n+1])] for n in range(len(LAYERS)-1)]
	biases = [[0 for _ in range(n)] for n in LAYERS[1:-1]]

	print("Weights and Biases generated and saved in weights_biases.txt")
	print(f"Weights: {weights}")
	print(f"Biases: {biases}")

	with open("weights_biases.txt", "w") as f:
		 f.write(str(weights) + "\n" + str(biases))

def get_weights_biases():
	with open("weights_biases.txt", "r") as f:
		content = f.read()
		content = content.split("\n")
		return content[0], content[1]

w, b = get_weights_biases()
h_layers = []

for i in range(len(LAYERS)-1):
	if i == len(LAYERS)-1:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], w[i], null))
	else:
		h_layers.append(HLayer(LAYERS[i], LAYERS[i+1], w[i], b[i]))	

for e in EPOCH:
	curr_l_output = 2 #temp input value
	for l in h_layers:
		curr_l_output.forward(curr_l_output)

print(f"Model prediction value: {curr_l_output}")
