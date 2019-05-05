import numpy as np
import tkinter as tk




def main():
	root = tk.Tk()
	cv = tk.Canvas(root, bd=5, bg='gray')
	cv.pack()
	
	draw_area = tk.Canvas(root)
	
	
	root.mainLoop()


if __name__ == '__main__':
	main()
