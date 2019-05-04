
from collections import namedtuple as nt
from PIL import Image, ImageDraw
import math
import numpy as np
import random as rd
import time
import os


def log_time(func):
	def wrapper(*args, **kwargs):
		start = time.time()
		out = func(*args, **kwargs)
		end = time.time() - start
		print(f'Time to run "{func}": {end}')
		return out
		
	return wrapper

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
		
		self.learning_rate = 1
		
	def feed_forward(self, input_arr):
		''' send inputs through NN and returns output values '''
		# generate hidden layer neuron values
		# print(input_arr)		
		input_mat = np.reshape(input_arr, [self.num_in,1])
		# print('weights in hidden')
		# print(self.weights_in_hid)
		hidden = self.weights_in_hid.dot(input_mat)
		hidden += self.bias_hid
		# print('before sigmoid')
		# print(hidden)
		hidden = sigmoid(hidden)
		# print('after sigmoid')
		# print(hidden)
		# generate output neuron values
		output = self.weights_hid_out.dot(hidden)
		output += self.bias_out
		# print('out before sig')
		# print(output)
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
	brain = NeuralNetwork(784, 50, 10)
	data_s = time.time()
	
	img_data = []
	for subdir, dirs, files in os.walk('training_images/'):
		# just for testing purposes
		if subdir[-1] == '/':
			continue
		if int(subdir[-1]) > 6:
			continue
		for file in files:
			if file == 'readme.txt':
				continue
			p = os.path.join(subdir, file)
			img = Image.open(p)
			inputs = np.array(img.getdata()) / 255
			targets = [0 for _ in range(10)]
			targets[int(subdir[-1])] = 1
			d = Datum(inputs, targets)
			img_data.append(d)
	
	data_e = time.time()
	print(data_e - data_s)
	
	start = time.time()
	# train
	for _ in range(5000):
		datum = rd.choice(img_data)
		brain.train(datum.inputs, datum.targets)
	
	# test		
	test_img = Image.open('test_1.png')
	test_in = list(np.array(test_img.getdata()) / 255)
	print('test 1')
	print(brain.feed_forward(test_in))
	test_img = Image.open('test_6.png')
	test_in = list(np.array(test_img.getdata()) / 255)	
	print('test 6')
	print(brain.feed_forward(test_in))
	
	end = time.time()
	print(end - start)
	

if __name__ == '__main__': main()
