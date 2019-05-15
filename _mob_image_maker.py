
from scene import *
from ui import Path, get_window_size
from PIL import Image, ImageDraw
import count_training_data as ctd
import math
import os
import sys
import numpy as np
import random as rd
import draw_ext as de


WIDTH, HEIGHT = get_window_size()


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


class Button (ShapeNode):
	def __init__(self, id, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.id = id
		
		if type(id) is int:
			self.label = LabelNode(text=f'{id}', font=('<System>', min(WIDTH, HEIGHT)/20), position=(0,0), parent=self)
		elif id == 'training':
			self.label = LabelNode(text='Saving to:\ntraining', font=('<System>', min(WIDTH, HEIGHT)/30), position=(0,0), parent=self)
		elif id == 'testing':
			self.label = LabelNode(text='Saving to:\ntesting', font=('<System>', min(WIDTH, HEIGHT)/30), position=(0,0), parent=self)
		elif id == 'close':
			self.label = LabelNode(text='Close and update', font=('<System>', min(WIDTH, HEIGHT)/30), position=(0,0), parent=self)


class Display (Scene):	
	def __init__(self, brain, *args, **kwargs):
		self.brain = brain
		super().__init__(*args, **kwargs)
	
	def setup(self):
		de.rect_mode = de.CORNER
		self.draw_area = de.Rect(WIDTH*.4, HEIGHT*.4, 280, 280, fill_color='transparent', stroke_color='red')
		self.add_child(self.draw_area)
		
		self.mode_switch = dict(training='testing', testing='training')
		if self.brain:
			self.mode = 'testing'
		else:
			self.mode = 'training'		
		
		p = Path.rounded_rect(0, 0, WIDTH*.2, HEIGHT*.09, 5)
		self.buttons = [Button(i, path=p, position=(WIDTH*.85, (i+.5)*(HEIGHT/10)), fill_color='green', parent=self) for i in range(10)]
		p = Path.rounded_rect(0, 0, WIDTH*.2, HEIGHT*.1, 5)
		self.mode_button = Button(f'{self.mode}', path=p, position=(WIDTH*.15, HEIGHT*.15), fill_color='#1e6cff', parent=self)
		self.buttons.append(self.mode_button)
		self.close_button = Button('close', path=p, position=(WIDTH*.15, HEIGHT*.75), fill_color='#ff0f0f', parent=self)
		self.buttons.append(self.close_button)
				
		self.matrix = np.zeros((28, 28))
		self.img = None
		self.dots = []		
	
	def save_image(self, id):
		tag = rd.random()
		self.matrix = (self.matrix * 255).astype(np.uint8)
		self.img = Image.fromarray(self.matrix, mode='L')
		self.img = self.img.convert('1')
		self.img.show()
		if self.mode == 'training':
			self.img.save(f'training_images/{id}/{tag}.png')
			print(f'Saved to training/{id}.')
		elif self.mode == 'testing':
			self.img.save(f'test_{id}.png')
			print(f'Saved test image for {id}.')
			
			if self.brain:
				test(self.brain, int(id))
		#np.save(f'training_images/{id}/{tag}.npy', self.matrix)
			
	
	def change_mode(self):
		if self.brain:
			print('Cannot change mode right now.')
			return
			
		self.mode = self.mode_switch[self.mode]	
		self.mode_button.label.text = f'Saving to:\n{self.mode}'
	
	def close(self):
		ctd.count_training_data()
		print('Updated training image counts')
		#os._exit(os.EX_OK)
		self.view.close()

	def touch_began(self, touch):
		loc = touch.location
		for btn in self.buttons:
			if loc in btn.frame:
				if btn is self.mode_button:
					self.change_mode()
					break
				elif btn is self.close_button:
					self.close()
					break
				self.save_image(btn.id)
				self.matrix = np.zeros_like(self.matrix)
				for d in self.dots:
					d.remove_from_parent()
				self.dots = []					
		
	def touch_moved(self, touch):
		loc = touch.location
		if loc in self.draw_area.frame:
			p = de.Point(*loc, 10, parent=self)
			self.dots.append(p)
			#t_x, t_y = loc
			pt = self.draw_area.point_from_scene(loc) / 10
			# backwards, because matrix row is the y-coord
			m_pt = (28 - int(pt.y))-1, int(pt.x)
			self.matrix[m_pt] = 1
		
	def touch_ended(self, touch):
		pass


def main(brain):
	run(Display(brain), show_fps=True)


if __name__ == '__main__': 
	main(None)
