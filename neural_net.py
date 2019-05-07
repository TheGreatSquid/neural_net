
from collections import namedtuple as nt
from PIL import Image, ImageDraw
import neural_net_mod as nn
import math
import numpy as np
import random as rd
import time
import os
		

class Datum:
	def __init__(self, inputs, targets):
		self.inputs = np.reshape(inputs, [len(inputs), 1])
		self.targets = np.reshape(targets, [len(targets), 1])

def main():
	brain = nn.NeuralNetwork(784, 50, 10)
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
			inputs = brain.prepare(inputs)
			targets = [0 for _ in range(10)]
			targets[int(subdir[-1])] = 1
			targets = brain.prepare(targets)
			d = (inputs, targets)
			img_data.append(d)
	
	data_e = time.time()
	print(data_e - data_s)
	
	start = time.time()
	# train
	for _ in range(10000):
		datum = rd.choice(img_data)
		brain.train(datum[0], datum[1])
	
	# test		
	test_img = Image.open('test_1.png')
	test_in = list(np.array(test_img.getdata()) / 255)
	print('test 1')
	out = brain.feed_forward(test_in)
	prediction = np.where(out == max(out))
	print(prediction[0])
	
	test_img = Image.open('test_6.png')
	test_in = list(np.array(test_img.getdata()) / 255)	
	print('test 6')
	out = brain.feed_forward(test_in)
	prediction = np.where(out == max(out))
	print(prediction[0])
	
	end = time.time()
	print(end - start)
	

if __name__ == '__main__': main()
