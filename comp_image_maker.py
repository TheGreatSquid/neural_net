import numpy as np
import tkinter as tk
import random as rd
import count_training_data as ctd
from PIL import Image
from functools import partial


WIDTH, HEIGHT = 1000, 800
buttons = []
matrix = np.zeros((28,28))
mode_switch = dict(training='testing', testing='training')
mode = 'training'

root = Tk.tk()


def update_readmes():
	ctd.count_training_data()
	root.destroy()	

def change_mode():
	global mode
	mode = mode_switch[mode]

def save_image(draw_area, id):
	global matrix
	tag = rd.random()
	matrix = (matrix * 255).astype(np.uint8)
	img = Image.fromarray(matrix, mode='L')
	img = img.convert('1')
	img.show()
	print(tag)
	if mode is 'training':
		img.save(f'training_images/{id}/{tag}.png')
		print(f'Saved to training_images/{id}.')
	elif mode is 'testing':
		img.save(f'test_{id}.png')
		print('Saved test image for {id}.')

	draw_area.delete("all")
	matrix = np.zeros_like(matrix)


def draw_pt(event):
	x, y = event.x, event.y
	w = event.widget
	
	o = w.create_oval(x-5, y-5, x+5, y+5, fill='black')
	
	global matrix
	p_x, p_y = x/10, y/10
	
	m_0, m_1 = int(p_y), int(p_x)
	matrix[m_0][m_1] = 1

def main():
	global root
	root.protocol("WM_DELETE_WINDOW", update_readmes)
	
	cv = tk.Canvas(root, width=WIDTH, height=HEIGHT, bd=5, bg='gray')
	cv.place(x=0, y=0, relwidth=1, relheight=1)
	
	draw_area = tk.Canvas(root, bg='white')
	draw_area.place(relx=.3, rely=.3, width=280, height=280)
	draw_area.bind("<B1-Motion>", draw_pt)
	
	global buttons
	for i in range(10):
		with_arg = partial(save_image, draw_area, i)
		b = tk.Button(root, text=f'{i}', bg='gray', command=with_arg)
		b.place(relx=.7, rely=.09*i, relwidth=.2, relheight=.1)	
		buttons.append(b)
	
	global mode	
	mode_button = tk.Button(root, text=f'Saving to:\n{mode}', bg='red', command=change_mode)
	mode_button.place(relx=.7, rely=.2, relwidth=.2, relheight=.1)
	buttons.append(mode_button)
	
	root.mainloop()


if __name__ == '__main__':
	main()


