import numpy as np
import tkinter as tk
import random as rd
import count_training_data as ctd
from PIL import Image
from functools import partial, partialmethod


WIDTH, HEIGHT = 1000, 800


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


class GUI (object):
	def __init__(self, brain=None):
		self.buttons = []
		self.matrix = np.zeros((28,28))
		self.mode_switch = dict(training='testing', testing='training')
		if brain:
			self.mode = 'testing'
			self.brain = brain
		else:
			self.mode = 'training'
			self.brain = None		
		
		root = self.root = tk.Tk()
		root.protocol("WM_DELETE_WINDOW", self.update_readmes)
	
		self.cv = tk.Canvas(root, width=WIDTH, height=HEIGHT, bd=5, bg='gray')
		self.cv.place(x=0, y=0, relwidth=1, relheight=1)
		
		self.draw_area = tk.Canvas(root, bg='white')
		self.draw_area.place(relx=.3, rely=.3, width=280, height=280)
		self.draw_area.bind("<B1-Motion>", self.draw_pt)
		
		for i in range(10):
			b = tk.Button(root, text=f'{i}', bg='gray', command=lambda i=i: self.save_image(i))
			b.place(relx=.7, rely=.09*i, relwidth=.2, relheight=.1)	
			self.buttons.append(b)
		
		self.mode_button = tk.Button(root, text=f'Saving to:\n{self.mode}', bg='red', command=self.change_mode)
		self.mode_button.place(relx=.2, rely=.7, relwidth=.2, relheight=.1)
		self.buttons.append(self.mode_button)


	def update_readmes(self):
		ctd.count_training_data()
		print('counting data')
		self.root.destroy()	
	
	def change_mode(self):
		if self.brain:
			print('Cannot change mode right now.')
			return
		
		self.mode = self.mode_switch[self.mode]
		self.mode_button['text'] = f'Saving to:\n{self.mode}'
	
	def save_image(self, id):
		tag = rd.random()
		self.matrix = (self.matrix * 255).astype(np.uint8)
		img = Image.fromarray(self.matrix, mode='L')
		img = img.convert('1')
		img.show()
		print(tag)
		if self.mode == 'training':
			img.save(f'training_images/{id}/{tag}.png')
			print(f'Saved to training_images/{id}.')
		elif self.mode == 'testing':
			img.save(f'test_{id}.png')
			print('Saved test image for {id}.')
			# if called from neural_net.py, make a prediction
			if self.brain:
				test(self.brain, int(id))
	
		self.draw_area.delete("all")
		self.matrix = np.zeros_like(self.matrix)	
	
	def draw_pt(self, event):
		x, y = event.x, event.y
		w = event.widget
		
		o = w.create_oval(x-5, y-5, x+5, y+5, fill='black')
		
		p_x, p_y = x/10, y/10
		
		m_0, m_1 = int(p_y), int(p_x)
		self.matrix[m_0][m_1] = 1


def main(brain):
	gui = GUI(brain)	
	gui.root.mainloop()


if __name__ == '__main__':
	main(None)
