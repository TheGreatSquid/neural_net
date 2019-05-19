import struct
import gzip
import numpy as np
from PIL import Image

def read_idx(filename):
	with gzip.open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def read_idx_lbl(filename):
	with gzip.open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		print(zero, data_type, dims)
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


img_arr = read_idx('train-images-idx3-ubyte.gz')
print(img_arr[0])
mask_g = img_arr > 100
mask_l = img_arr <= 100
img_arr[mask_g] = 255
img_arr[mask_l] = 0
print(img_arr[0])
lbl_arr = read_idx_lbl('train-labels-idx1-ubyte.gz')
np.save('training_images/mnist_images.npy', img_arr)
np.save('training_images/mnist_labels.npy', lbl_arr)
'''
np.set_printoptions(threshold=10000)
arr = np.load('training_images/mnist_images.npy')
img_arr = arr[0]
print(img_arr)
img = Image.fromarray(img_arr, mode='L')
img.show()
'''
