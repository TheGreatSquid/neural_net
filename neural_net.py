

import math
import numpy as np
import random as rd
from collections import namedtuple as nt
import datetime

def sigmoid(v):
	return 1 / (1 + np.exp(-v))
	
def dsigmoid(y):
	''' the 'x' values already underwent sigmoid '''
	return y * (1 - y)


class NeuralNetwork:
	def __init__(self, num_in, num_hid, num_out):
		self.num_in = num_in
		self.num_hid = num_hid
		self.num_out = num_out
		
		self.weights_in_hid = np.random.rand(self.num_hid, self.num_in)
		self.weights_hid_out = np.random.rand(self.num_out, self.num_hid)
		
		self.bias_hid = np.random.rand(self.num_hid, 1)
		self.bias_out = np.random.rand(self.num_out, 1)
		
		self.learning_rate = 0.5
		
	def feed_forward(self, input_arr):
		''' send inputs through NN and returns output values '''
		# generate hidden layer neuron values
		
		input_mat = np.reshape(input_arr, [self.num_in,1])
		hidden = self.weights_in_hid.dot(input_mat)
		hidden += self.bias_hid
		hidden = sigmoid(hidden)
		# generate output neuron values
		output = self.weights_hid_out.dot(hidden)
		output += self.bias_out
		output = sigmoid(output)
		# return output in array form
		return output.flatten()
		
	def train(self, input_mat, target_mat):
		# generate hidden layer neuron values
		hidden = self.weights_in_hid.dot(input_mat)
		hidden += self.bias_hid
		hidden = sigmoid(hidden)
		# generate output neuron values
		output_mat = self.weights_hid_out.dot(hidden)
		output_mat += self.bias_out
		# activation function
		output_mat = sigmoid(output_mat)
		
		# calculate output errors (targets - outputs)
		out_errors = target_mat - output_mat
		# calculate gradient
		gradients = dsigmoid(output_mat)
		gradients *= out_errors
		gradients *= self.learning_rate
		# calculate deltas
		hidden_t = hidden.T
		weights_hid_out_delta = gradients.dot(hidden_t)
		# apply deltas (just gradients for bias)
		self.weights_hid_out += weights_hid_out_delta
		self.bias_out += gradients
		
		# calculate hidden errors (sum of incoming errors scaled by weights)
		weights_hid_out_t = self.weights_hid_out.T
		hid_errors = weights_hid_out_t.dot(out_errors)
		# calculate hidden gradients
		hidden_gradients = dsigmoid(hidden)
		hidden_gradients *= hid_errors
		hidden_gradients *= self.learning_rate
		# calculate hidden deltas
		inputs_t = input_mat.T
		weights_in_hid_delta = hidden_gradients.dot(inputs_t)
		# apply hidden deltas (just hidden gradients for bias)
		self.weights_in_hid += weights_in_hid_delta
		self.bias_hid += hidden_gradients
		

class Datum:
	def __init__(self, inputs, targets):
		self.inputs = np.reshape(inputs, [len(inputs), 1])
		self.targets = np.reshape(targets, [len(targets), 1])

def main():
	brain = NeuralNetwork(2, 50, 1)
	data_s = datetime.datetime.now()
	training_data = [
		Datum([0, 0], [0]), 
		Datum([0, 1], [1]), 
		Datum([1, 0], [1]), 
		Datum([1, 1], [0]), 
	]
	data_e = datetime.datetime.now()
	print(data_e - data_s)
	
	start = datetime.datetime.now()
	
	training_matrix = np.array([[
								[[[0], [0]], [[0]]], 
								[[[0], [1]], [[1]]], 
								[[[1], [0]], [[1]]], 
								[[[1], [1]], [[0]]], 
								]])

	training_matrix = np.repeat(training_matrix, 100, axis=0).reshape(400, 2)

	input_mat = training_matrix[...,0]	
	target_mat = training_matrix[...,1]
	
	
	#brain.train(input_mat, target_mat)

	for _ in range(50000):
		datum = rd.choice(training_data)		
		brain.train(datum.inputs, datum.targets)
			
	
	print(brain.feed_forward([0, 0]))
	print(brain.feed_forward([1, 0]))
	print(brain.feed_forward([0, 1]))
	print(brain.feed_forward([1, 1]))
	
	'''
	d = Datum(None, None)
	print(d.targets)
	print(brain.feed_forward(list(d.inputs.ravel())))	
	'''
	
	end = datetime.datetime.now()
	print(end - start)
	

if __name__ == '__main__': main()
