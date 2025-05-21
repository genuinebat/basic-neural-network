import math

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
		print("input:")
		print(inputs)
		print("weights:")
		print(self.weights)
		print("after multiplication:")
		print(self.multiply_matrix(inputs, self.weights))
		print("biases")
		print(self.biases)
		print("final res")
		print(self.add_matrix(self.multiply_matrix(inputs, self.weights), self.biases))

		res = self.multiply_matrix(inputs, self.weights)
		res = self.add_matrix(res, self.biases)
		res = self.activate(res)

		return res

	def activate(self, r):
		return r
