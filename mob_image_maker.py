
from scene import *
from ui import Path, get_window_size
from PIL import Image, ImageDraw
import math
import numpy as np
import random as rd
import draw_ext as de


WIDTH, HEIGHT = get_window_size()


class Button (ShapeNode):
	def __init__(self, id, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.id = id


class Display (Scene):
	def setup(self):
		self.draw_area = de.Rect(WIDTH/2-140, HEIGHT/2-140, 280, 280, fill_color='transparent', stroke_color='red')
		self.add_child(self.draw_area)
		
		p = Path.rounded_rect(0, 0, 220, 60, 5)
		self.buttons = [Button(i, path=p, position=(WIDTH-200, (i+.5)*100), fill_color='green', parent=self) for i in range(10)]
		self.labels = [LabelNode(text=f'{i}', font=('<System>', 50), position=(WIDTH-200, (i+.5)*100), parent=self) for i in range(10)]
		
		self.matrix = np.zeros((28, 28))
		self.img = None
		self.dots = []
	
	def save_image(self, id):
		tag = rd.random()
		self.matrix = (self.matrix * 255).astype(np.uint8)
		self.img = Image.fromarray(self.matrix, mode='L')
		self.img = self.img.convert('1')
		self.img.show()
		self.img.save(f'training_images/{id}/{tag}.png')
		# self.img.save(f'test_{id}.png')
		#np.save(f'training_images/{id}/{tag}.npy', self.matrix)
			
				
	def touch_began(self, touch):
		loc = touch.location
		for btn in self.buttons:
			if loc in btn.frame:
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

def main():
	run(Display(), show_fps=True)

if __name__ == '__main__': main()
