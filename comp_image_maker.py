import numpy as np
import tkinter as tk
import random as rd
from PIL import Image


WIDTH, HEIGHT = 1000, 800
buttons = []
matrix = np.zeros((28,28))



def save_image(id):
	global matrix
	tag = rd.random()
	matrix = (matrix * 255).astype(np.uint8)
	img = Image.fromarray(matrix, mode='L')
	img = img.convert('1')
	img.show()
	img.save(f'training_images/{id}/{tag}.png')
	print(tag)
	
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
	root = tk.Tk()
	cv = tk.Canvas(root, width=WIDTH, height=HEIGHT, bd=5, bg='gray')
	cv.place(x=0, y=0, relwidth=1, relheight=1)
	
	draw_area = tk.Canvas(root, bg='white')
	draw_area.place(relx=.3, rely=.3, width=280, height=280)
	draw_area.bind("<Button-1>", draw_pt)
	
	global buttons
	for i in range(10):
		b = tk.Button(root, text=f'{i}', bg='gray', command=lambda: save_image(id=i))
		b.place(relx=.7, rely=.09*i, relwidth=.2, relheight=.1)
		#b.bind("<Button-1>", command=lambda: save_image(id=i))
	
	
	root.mainloop()


if __name__ == '__main__':
	main()


