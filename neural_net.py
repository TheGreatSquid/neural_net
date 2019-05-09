
from collections import namedtuple as nt
from PIL import Image, ImageDraw
import neural_net_mod as nn
import math
import numpy as np
import random as rd
import time
import os
		

Datum = nt('Datum', 'inputs, targets')


def test(brain, target):
	try:
		test_img = Image.open(f'test_{target}.png')
	except:
		print(f'Test image for {target} does not exist.')
	
	test_in = list(np.array(test_img.getdata()) / 255)
	print(f'Testing number: {target}')
	out = brain.feed_forward(test_in)
	prediction = np.where(out == max(out))
	print(f'Brain thinks this is a: {prediction[0]}')


def main():
	brain = nn.NeuralNetwork(784, 50, 10)
	data_start = time.time()
	
	img_data = []
	for subdir, dirs, files in os.walk('training_images/'):
		last_char = subdir[-1]
		# just for testing purposes
		if last_char == '/':
			continue
		if int(last_char) > 6:
			continue
		for file in files:
			if file == 'readme.txt':
				continue
			p = os.path.join(subdir, file)
			img = Image.open(p)
			inputs = np.array(img.getdata()) / 255
			targets = [0] * 10
			targets[int(last_char)] = 1
			d = Datum(brain.prepare(inputs), brain.prepare(targets))
			img_data.append(d)
	
	data_end = time.time()
	print(f'Time to parse data: {data_end - data_start}')
	
	train_start = time.time()
	# train
	for _ in range(10000):
		datum = rd.choice(img_data)
		brain.train(datum.inputs, datum.targets)
	
	train_end = time.time()
	print(f'Time to train: {train_end - train_start}')
	# test
	test(brain, 1)
	test(brain, 6)
	

if __name__ == '__main__': 
	main()
