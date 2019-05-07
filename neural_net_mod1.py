
import numpy as np
import random as rd


def sigmoid(v):
	return 1 / (1 + np.exp(-v))
	
def dsigmoid(y):
	''' the 'x' values already underwent sigmoid '''
	return y * (1 - y)

def tanh(v):
	return np.tanh(v)

def dtanh(y):
	return 1 - y*y


class NeuralNetwork:
	def __init__(self, num_in, num_hid, num_out):
		self.num_in = num_in
		self.num_hid = num_hid
		self.num_out = num_out
		
		self.weights_in_hid = np.random.rand(self.num_hid, self.num_in) / 10
		self.weights_hid_out = np.random.rand(self.num_out, self.num_hid) / 10
		
		self.bias_hid = np.random.rand(self.num_hid, 1) / 10
		self.bias_out = np.random.rand(self.num_out, 1) / 10
		
		self.learning_rate = .1
	
	def prepare(self, arr):
		return np.reshape(arr, (len(arr), 1))
	
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

