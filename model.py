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
		if not os.path.isfile("./model_parameters/weights.txt") and not os.path.isfile("./model_parameters/biases.txt"):
			weights = []
			for i in range(len(self.layout)-1):
				weights.append([[str(round(r.uniform(-0.5, 0.5), 5)) for _ in range(self.layout[i+1])] for _ in range(self.layout[i])])

			biases = [[["0.01" for _ in range(n)]] for n in self.layout[1:-1]]

			print("New Weights and Biases generated and saved in model_parameters folder")

			with open("./model_parameters/weights.txt", "w") as f:
				f.write(str(weights))
			
			with open("./model_parameters/biases.txt", "w") as f:
				f.write(str(biases))

	def get_weights_biases(self):
		w = ast.literal_eval(open("./model_parameters/weights.txt", "r").read())
		w = [[[float(k) for k in j] for j in i] for i in w]

		b = ast.literal_eval(open("./model_parameters/biases.txt", "r").read())
		b = [[[float(k) for k in j] for j in i] for i in b]

		return w, b
	
	def create_h_layers(self):
		w, b = self.get_weights_biases()

		print(f"Weights retrieved from model_parameters: {w}")
		print(f"Biases retrieved from model_parameters: {b}\n")

		for i in range(len(self.layout)-1):
			if i == len(self.layout)-2:
				self.h_layers.append(self.HLayer(w[i], None))
			else:
				self.h_layers.append(self.HLayer(w[i], b[i]))	

	def train(self, epoch, bs, lr, data):
		for e in range(epoch):
			for i in range(0, len(data), bs):
				batch_outputs = [[] for _ in range(bs)]
				for b in range(bs):
					curr_l_output = [[data[i+b][0]]]
					for l in self.h_layers:	
						curr_l_output = l.forward(curr_l_output)
						batch_outputs[b].append(curr_l_output)
				self.modify_weights(batch_outputs)

	def predict(self, data):
		res = [{"input value": 0, "predicted value": 0, "expected value": 0, "cost": 0} for _ in data]

		for d in range(len(data)):
			curr_l_output = [[data[d][0]]]
			for l in self.h_layers:
				curr_l_output = l.forward(curr_l_output)
				
			res[d]["input value"] = data[d][0]
			res[d]["predicted value"] = curr_l_output[0][0]
			res[d]["expected value"] = data[d][1]
			res[d]["cost"] = abs(res[d]["predicted value"] - res[d]["expected value"])

		return res

	def modify_weights(self, batch_outputs):
		print("batch outputs:")
		print(batch_outputs)	

	def save_weights(self):
		pass

	class HLayer():
		def __init__(self,_weights, _biases):
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
					res[i][j] = res[i][j]

			return res
		
		def forward(self, inputs):
			print("inputs")
			print(inputs)
			print("weights")
			print(self.weights)
			print("biases")
			print(self.biases)
			res = self.multiply_matrix(inputs, self.weights)
			print("after mult")
			print(res)
			if self.biases != None:
				res = self.add_matrix(res, self.biases)
			print("before activ")
			print(res)
			res = self.relu(res)
			print("after activ")
			print(res)
			print("\n")

			return res

		def sigmoid(self, r):
			return [[round(1/(1+math.exp(-j)), 6) for j in i] for i in r]

		def relu(self, r):
			return r if r[0][0] > 0 else [[0 for _ in r[0]]]

