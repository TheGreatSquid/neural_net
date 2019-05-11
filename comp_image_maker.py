import numpy as np
import tkinter as tk


WIDTH, HEIGHT = 1000, 800
buttons = []


def drawtest(event):
	x, y = event.x, event.y
	w = event.widget
	o = w.create_oval(x-5, y-5, x+5, y+5, fill='black')
	print(f'clicked at {(x, y)}')

def main():
	root = tk.Tk()
	cv = tk.Canvas(root, width=WIDTH, height=HEIGHT, bd=5, bg='gray')
	cv.place(x=0, y=0, relwidth=1, relheight=1)
	
	draw_area = tk.Canvas(root, bg='white')
	draw_area.place(relx=.3, rely=.3, width=280, height=280)
	draw_area.bind("<Button-1>", drawtest)
	
	global buttons
	for i in range(10):
		b = tk.Button(root, text=f'{i}', bg='gray')
		b.place(relx=.7, rely=.09*i, relwidth=.2, relheight=.1)
	
	
	root.mainloop()


if __name__ == '__main__':
	main()


