import numpy as np
import tkinter as tk

WIDTH, HEIGHT = 800, 600
buttons = []

def main():
	root = tk.Tk()
	cv = tk.Canvas(root, width=WIDTH, height=HEIGHT, bd=5, bg='gray')
	cv.pack()
	
	draw_area = tk.Canvas(root, relx=.1, rely=.1, width=280, height=280, bg='white')
	
	global buttons
	for i in range(10):
		b = tk.Button(root, relx=.7, rely=.09*i, relwidth=.2, relheight=.1, bg='gray')
	
	
	root.mainLoop()


if __name__ == '__main__':
	main()
