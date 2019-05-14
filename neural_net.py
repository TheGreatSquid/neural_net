
from collections import namedtuple as nt
from PIL import Image
import neural_net_mod as nn
import math
import numpy as np
import random as rd
import argparse
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
	print(f'Brain thinks this is a: {prediction[0][0]}')
	print(out.astype(float))


def main(args):
	brain = nn.NeuralNetwork(784, 64, 10)
	brain.learning_rate = args.learningrate
	data_start = time.time()
	
	img_data = []
	for subdir, dirs, files in os.walk('training_images/'):
		last_char = subdir[-1]
		# just for testing purposes
		if last_char == '/':
			continue
		#if int(last_char) is not 0 and int(last_char) is not 2:
			#continue
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
	# train

	train_start = time.time()
	'''
	for i in range(100):
		brain.train_epoch(img_data)
	'''
	for _ in range(args.iterations):

		datum = rd.choice(img_data)
		brain.train(datum.inputs, datum.targets)
	
	train_end = time.time()
	print(f'Time to train: {train_end - train_start}')
	# test
	testing = True
	
	np.set_printoptions(suppress=True)
	for i in range(10):
		test(brain, i)
	
	while testing:
		i = input('Number to test: ')
		
		if i == 'QUIT':
			testing = False
			
		test(brain, int(i))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learningrate', '-l', type=float, default=0.3, help='Brain learning rate; should be (0, 1)')
	parser.add_argument('--iterations', '-i', type=int, default=10000, help='Number of iterations to train')	
	
	args = parser.parse_args()
	main(args)
