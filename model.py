import math
import os
import ast
import random as r

class NeuralNetworkModel():
	def __init__(self, _layout):
		self.layout = _layout 
		self.h_layers = []

		self.set_weights()
		self.create_h_layers()

	def set_weights(self):
		# create file, if doesn't exist, to store weights and biases
		if not os.path.isfile("./saved_data/weights.txt") and not os.path.isfile("./saved_data/biases.txt"):
			weights = []
			for i in range(len(self.layout)-1):
				weights.append([[str(round(r.uniform(-0.5, 0.5), 2)) for _ in range(self.layout[i+1])] for _ in range(self.layout[i])])

			biases = [[["0" for _ in range(n)]] for n in self.layout[1:-1]]

			print("New Weights and Biases generated and saved in saved_data folder")

			with open("./saved_data/weights.txt", "w") as f:
				f.write(str(weights))
			
			with open("./saved_data/biases.txt", "w") as f:
				f.write(str(biases))

	def get_weights_biases(self):
		w = ast.literal_eval(open("./saved_data/weights.txt", "r").read())
		w = [[[float(k) for k in j] for j in i] for i in w]

		b = ast.literal_eval(open("./saved_data/biases.txt", "r").read())
		b = [[[float(k) for k in j] for j in i] for i in b]

		return w, b
	
	def create_h_layers(self):
		w, b = self.get_weights_biases()

		print(f"Weights retrieved from saved_data: {w}")
		print(f"Biases retrieved from saved_data: {b}")

		for i in range(len(self.layout)-1):
			if i == len(self.layout)-2:
				self.h_layers.append(self.HLayer(self.layout[i], self.layout[i+1], w[i], None))
			else:
				self.h_layers.append(self.HLayer(self.layout[i], self.layout[i+1], w[i], b[i]))	

	def train(self, epoch, data):
		res = [{"predicted value": 0, "expected value": 0, "cost": 0} for _ in range(epoch)]

		for e in range(epoch):
			curr_l_output = [[data[e][0]]]
			for l in self.h_layers:
				curr_l_output = l.forward(curr_l_output)
				
			res[e]["predicted value"] = curr_l_output[0][0]
			res[e]["expected value"] = data[e][1]
			res[e]["cost"] = abs(data[e][1] - curr_l_output[0][0])

		return res

	class HLayer():
		def __init__(self, _n_in, _n_out, _weights, _biases):
			self.n_in = _n_in 
			self.n_out = _n_out
			self.weights = _weights
			self.biases = _biases

		def add_matrix(self, a, b):
			if not b:
				return a
			return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

		def multiply_matrix(self, a, b):
			res = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
			
			for i in range(len(a)):
				for j in range(len(b[0])):
					for k in range(len(b)):
						res[i][j] = res[i][j] + (a[i][k] * b[k][j])
					res[i][j] = round(res[i][j], 2)

			return res
		
		def forward(self, inputs):
			print("\n")
			print("input:")
			print(inputs)
			print("weights:")
			print(self.weights)
			res = self.multiply_matrix(inputs, self.weights)
			print("after mulitplication:")
			print(res)
			
			print("biases:")
			print(self.biases)
			res = self.add_matrix(res, self.biases)
			print("after adding biases")
			print(res)

			res = self.activate(res)
			print("after activation")
			print(res)

			return res

		def activate(self, r):
			return r

